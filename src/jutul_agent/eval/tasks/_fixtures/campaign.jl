using JutulDarcy, Jutul
using Statistics

const campaign_cases = Dict(
    "run-cedar" => (porosity = 0.16, injection_rate = 650.0),
    "run-falcon" => (porosity = 0.18, injection_rate = 1100.0),
    "run-lotus" => (porosity = 0.20, injection_rate = 850.0),
    "run-maple" => (porosity = 0.22, injection_rate = 1250.0),
    "run-raven" => (porosity = 0.25, injection_rate = 950.0),
    "run-sable" => (porosity = 0.28, injection_rate = 1400.0),
)

function run_campaign_case(name::AbstractString)
    config = campaign_cases[name]
    Darcy, bar, kg, meter, day = si_units(:darcy, :bar, :kilogram, :meter, :day)
    grid = CartesianMesh((8, 8, 3), (500.0, 500.0, 20.0))
    domain = reservoir_domain(
        grid;
        permeability = 0.2 * Darcy,
        porosity = config.porosity,
    )
    injector = setup_vertical_well(domain, 1, 1, name = :Injector)
    producer = setup_vertical_well(domain, 8, 8, name = :Producer)
    system = ImmiscibleSystem(
        (AqueousPhase(), LiquidPhase());
        reference_densities = [1000.0, 850.0] .* kg / meter^3,
    )
    model, parameters = setup_reservoir_model(
        domain,
        system;
        wells = [injector, producer],
        extra_out = true,
    )
    parameters[:Reservoir][:PhaseViscosities][1, :] .= 1e-3
    parameters[:Reservoir][:PhaseViscosities][2, :] .= 5e-3
    reference_pressure = 101325.0
    compressibility = 1e-10
    density = ConstantCompressibilityDensities(
        system, reference_pressure, [1000.0, 850.0], compressibility,
    )
    replace_variables!(model; PhaseMassDensities = density)
    state0 = setup_reservoir_state(
        model;
        Pressure = 150 * bar,
        Saturations = [0.0, 1.0],
    )
    timesteps = fill(60.0 * day, 12)
    injector_control = InjectorControl(
        TotalRateTarget(config.injection_rate * meter^3 / day),
        [1.0, 0.0];
        density = 1000.0 * kg / meter^3,
    )
    producer_control = ProducerControl(BottomHolePressureTarget(100 * bar))
    forces = setup_reservoir_forces(
        model;
        control = Dict(:Injector => injector_control, :Producer => producer_control),
    )
    wells, states, _ = simulate_reservoir(
        state0,
        model,
        timesteps;
        parameters = parameters,
        forces = forces,
        info_level = -1,
        output_substates = true,
        tol_cnv = 1e-6,
        tol_mb = 1e-10,
        tol_cnv_well = 1e-7,
        tol_mb_well = 1e-10,
    )

    isapprox(wells.time[end], sum(timesteps); rtol = 1e-10) || error("incomplete simulation")
    pore_volumes = pore_volume(model, parameters)
    initial_oil_mass = sum(pore_volumes) * 850.0 * exp(
        (150 * bar - reference_pressure) * compressibility,
    )
    elapsed = diff(vcat(0.0, wells.time))
    produced_oil_mass = cumsum(-wells[:Producer, :Liquid_mass_rate] .* elapsed)
    recovery_history = produced_oil_mass ./ initial_oil_mass
    peak_pressure = max(150.0, maximum(maximum(state[:Pressure]) / bar for state in states))
    history = String["time_days,recovery_factor,max_pressure_bar"]
    for (step, state) in enumerate(states)
        recovery = recovery_history[step]
        pressure = state[:Pressure] ./ bar
        push!(history, join((wells.time[step] / day, recovery, maximum(pressure)), ','))
        println(
            "run=$name step=$step recovery=$(round(recovery; digits=8)) " *
            "avg_pressure_bar=$(round(mean(pressure); digits=5)) " *
            "max_pressure_bar=$(round(maximum(pressure); digits=5))",
        )
    end
    final_state = states[end]
    final_oil_mass = sum(final_state[:TotalMasses][2, :])
    balance_error = abs(initial_oil_mass - final_oil_mass - produced_oil_mass[end]) / initial_oil_mass
    balance_error < 1e-3 || error("oil mass balance error exceeds 0.1%: $balance_error")
    recovery = recovery_history[end]
    mkpath("diagnostics")
    write(joinpath("diagnostics", "$name-history.csv"), join(history, '\n') * "\n")
    write(
        joinpath("diagnostics", "$name.csv"),
        "initial_oil_mass_kg,produced_oil_mass_kg,final_oil_mass_kg,mass_balance_relative_error,peak_pressure_bar\n" *
        join((initial_oil_mass, produced_oil_mass[end], final_oil_mass, balance_error, peak_pressure), ',') * "\n",
    )
    pressure = final_state[:Pressure] ./ bar
    water = final_state[:Saturations][1, :]
    average_pressure = mean(pressure)
    maximum_pressure = maximum(pressure)
    rows = String[
        "run_id,porosity,injection_rate_m3_day,recovery_factor,avg_pressure_bar,max_pressure_bar,cell_id,cell_pressure_bar,water_saturation",
    ]
    for cell in eachindex(pressure)
        push!(
            rows,
            join(
                (
                    name,
                    config.porosity,
                    config.injection_rate,
                    recovery,
                    average_pressure,
                    maximum_pressure,
                    cell,
                    pressure[cell],
                    water[cell],
                ),
                ',',
            ),
        )
    end
    mkpath("results")
    artifact = joinpath("results", "$name.csv")
    write(artifact * ".tmp", join(rows, '\n') * "\n")
    mv(artifact * ".tmp", artifact; force = true)
    summary = (
        run_id = name,
        porosity = config.porosity,
        injection_rate_m3_day = config.injection_rate,
        recovery_factor = recovery,
        avg_pressure_bar = average_pressure,
        max_pressure_bar = maximum_pressure,
    )
    println("summary=$summary")
    return summary
end

function run_campaign_case_with_ledger(name::AbstractString)
    ledger = "campaign-index.csv"
    if isfile(ledger) && any(row -> first(split(row, ',')) == name, readlines(ledger)[2:end])
        error("completed case already recorded: $name")
    end
    summary = run_campaign_case(name)
    artifact = joinpath("results", "$(summary.run_id).csv")
    if !isfile(ledger)
        write(
            ledger,
            "run_id,porosity,injection_rate_m3_day,recovery_factor," *
            "avg_pressure_bar,max_pressure_bar,artifact_path\n",
        )
    end
    open(ledger, "a") do io
        println(
            io,
            join(
                (
                    summary.run_id,
                    summary.porosity,
                    summary.injection_rate_m3_day,
                    summary.recovery_factor,
                    summary.avg_pressure_bar,
                    summary.max_pressure_bar,
                    artifact,
                ),
                ',',
            ),
        )
    end
    println(
        "recorded run_id=$(summary.run_id) ledger_path=$ledger " *
        "artifact_path=$artifact",
    )
    return merge(summary, (ledger_path = ledger, artifact_path = artifact))
end

const AUTONOMOUS_CAMPAIGN_CASES = (
    "run-cedar",
    "run-falcon",
    "run-lotus",
    "run-maple",
    "run-raven",
    "run-sable",
)

function run_autonomous_campaign_case(name::AbstractString)
    summary = run_campaign_case_with_ledger(name)
    ledger_rows = readlines(summary.ledger_path)[2:end]
    completed_names = Set(first(split(row, ',')) for row in ledger_rows)
    completed_count = count(run_id -> run_id in completed_names, AUTONOMOUS_CAMPAIGN_CASES)
    println(
        "campaign_progress completed_count=$completed_count " *
        "expected_count=$(length(AUTONOMOUS_CAMPAIGN_CASES))",
    )
    if completed_count == length(AUTONOMOUS_CAMPAIGN_CASES)
        println(
            "Campaign complete. Read $(summary.ledger_path), then grep the selected result file.",
        )
    end
    return merge(summary, (completed_count = completed_count,))
end
