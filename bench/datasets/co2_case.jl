# The JutulDarcy "CO2 injection in saline aquifer with storage inventory"
# example (examples/workflow/co2_sloped.jl), without plotting, and with two
# scenario knobs for the ensemble: the injection rate and the permeability.
# rate_factor = perm_factor = 1 reproduces the published example.

include("common.jl")

function co2_case(; rate_factor = 1.0, perm_factor = 1.0, nx = 100, nz = 50, nstep = 25, nstep_shut = 475)
    Darcy, bar, day = si_units(:darcy, :bar, :day)
    mesh = reservoir_mesh((nx, 1, nz), (1000.0, 1.0, 50.0))
    points = mesh.node_points
    for (i, pt) in enumerate(points)
        x, y, z = pt
        x_u = 2*π*x/1000.0
        dz = 0.05*x + 0.05*abs(x - 500.0) + 0.2*(30*sin(2.0*x_u) + 20*sin(5.0*x_u))
        points[i] = pt + [0, 0, dz]
    end
    trajectory = [645.0 0.5 75; 660.0 0.5 85; 710.0 0.5 100.0]
    wc = Jutul.find_enclosing_cells(mesh, trajectory)

    nc = number_of_cells(mesh)
    perm = zeros(3, nc); poro = zeros(nc); region = zeros(Int, nc)
    for cell in 1:nc
        I, J, K = cell_ijk(mesh, cell)
        reg, permxy, phi = K < 0.3*nz ? (1, 0.3Darcy, 0.2) : K < 0.7*nz ? (2, 1.2Darcy, 0.35) : (3, 0.1Darcy, 0.1)
        perm[1, cell] = perm[2, cell] = perm_factor*permxy
        perm[3, cell] = 0.5*perm_factor*permxy
        poro[cell] = phi
        region[cell] = reg
    end
    domain = reservoir_domain(mesh, permeability = perm, porosity = poro, temperature = convert_to_si(60.0, :Celsius))
    injector = setup_well(domain, wc, name = :Injector, simple_well = true)
    model = setup_reservoir_model(domain, :co2brine, wells = injector,
        salt_names = ["NaCl", "KCl", "CaSO4", "CaCl2", "MgSO4", "MgCl2"],
        salt_mole_fractions = [0.01, 0.005, 0.005, 0.001, 0.0002, 1e-5],
        co2_physics = :kvalue)

    so = range(0, 1, 10)
    krog = PhaseRelativePermeability(so, so.^2, label = :og)
    sg = range(0, 1, 50)
    krg_drain = PhaseRelativePermeability(sg, JutulDarcy.brooks_corey_relperm.(sg, n = 2, residual = 0.1), label = :g)
    krg_imb = PhaseRelativePermeability(sg, JutulDarcy.brooks_corey_relperm.(sg, n = 3, residual = 0.25), label = :g)
    relperm = JutulDarcy.ReservoirRelativePermeabilities(g = (krg_drain, krg_imb), og = krog,
        hysteresis_g = JutulDarcy.KilloughHysteresis())
    replace_variables!(model, RelativePermeabilities = relperm)
    JutulDarcy.add_relperm_parameters!(model)

    depth = domain[:cell_centroids][3, :]
    p0 = @. 160bar + depth*Jutul.gravity_constant*1000.0
    state0 = setup_reservoir_state(model, Pressure = p0, OverallMoleFractions = [1.0, 0.0])
    parameters = setup_parameters(model)
    boundary = [c for c in 1:nc if cell_ijk(mesh, c)[1] in (1, nx)]
    bc = flow_boundary_condition(boundary, domain, p0[boundary], fractional_flow = [1.0, 0.0])

    dt_inject = fill(365.0day, nstep)
    inj_rate = rate_factor*0.075*sum(pore_volume(model, parameters))/sum(dt_inject)
    controls = Dict(:Injector => InjectorControl(TotalRateTarget(inj_rate), [0.0, 1.0], density = 630.0))
    forces = vcat(fill(setup_reservoir_forces(model, control = controls, bc = bc), nstep),
                  fill(setup_reservoir_forces(model, bc = bc), nstep_shut))
    dt = vcat(dt_inject, fill(365.0day, nstep_shut))

    rmodel = reservoir_model(model)
    push!(rmodel.output_variables, :RelativePermeabilities)
    push!(rmodel.output_variables, :PhaseViscosities)
    result = simulate_reservoir(state0, model, dt, parameters = parameters, forces = forces,
        max_timestep = 90day, info_level = -1)

    ws, states, t = result
    inv = co2_inventory(model, ws, states, t)
    inventory = Dict(String(k) => [step[k] for step in inv] for k in keys(first(inv)))
    extra = Dict{String, Any}(
        "co2_inventory_kg" => inventory,
        "region" => region,
        # K = 1 is the top layer (z is depth, positive down).
        "region_names" => Dict(1 => "upper layer (K < 15): 0.3 D, porosity 0.2",
                               2 => "middle high-permeability layer (15 <= K < 35): 1.2 D, porosity 0.35",
                               3 => "lower low-permeability layer (K >= 35): 0.1 D, porosity 0.1"),
        "initial_pressure_Pa" => p0,
        "boundary_cells" => boundary,
        "injector_cells" => collect(wc),
        "injection_rate_m3_per_s" => inj_rate,
        "injection_end_time_s" => sum(dt_inject),
    )
    meta = Dict{String, Any}(
        "source" => "JutulDarcy examples/workflow/co2_sloped.jl (kvalue CO2-brine, Killough hysteresis)",
        "rate_factor" => rate_factor, "perm_factor" => perm_factor,
        "description" => "CO2 injected for $nstep years into a sloped 2D saline aquifer, then $nstep_shut years of migration with constant-pressure side boundaries",
    )
    return (; result, model, parameters, extra, meta)
end
