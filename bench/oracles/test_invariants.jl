# Physics invariants on the saved datasets and the oracles built on them.
# Usage: julia --project=bench bench/oracles/test_invariants.jl
using Test
include("run_oracles.jl")

@testset "CO2 storage" begin
    d = load_dataset("co2_storage"); inv = co2_inventory(d); inj = inv["injected"][end]
    # The inventory categories add up to what is in the domain...
    @test all(isapprox.(inv["residual"] .+ inv["mobile"] .+ inv["dissolved"], inv["domain_total"]; rtol = 1e-6))
    # ...and in the domain + left + produced is what was injected.
    @test all(abs.(inv["domain_total"] .+ inv["outside_domain"] .+ inv["produced"] .- inv["injected"]) .<= 1e-6 * inj)
    # The in-domain CO2 in the inventory is the CO2 mass summed over cells.
    @test vec(sum(d.states["TotalMasses"][CO2, :, :], dims = 1)) ≈ inv["domain_total"] rtol = 1e-8
    @test issorted(inv["injected"])
    # Nothing flows back in through the boundaries (brine-only inflow). The
    # inventory's outside-domain figure is injected minus in-place, so it
    # carries bookkeeping noise of up to ~0.07% of the injected mass (largest
    # when injection stops); that is far below the 1% leakage-onset threshold.
    @test all(diff(inv["outside_domain"]) .>= -2e-3 * inj)
    @test all(0 .<= d.states["Saturations"] .<= 1 + 1e-9)
    @test all(sum(d.states["Saturations"], dims = 1) .≈ 1)
    # Injected mass = rate x injection time.
    @test inj ≈ d.wells["Injector"]["CO2_mass_rate"][1] * d.extra["injection_end_time_s"] rtol = 1e-6
end

@testset "EGG waterflood" begin
    d = load_dataset("egg_waterflood")
    @test EGG_ORACLES["egg-mass-conserved"](d)["max_relative_error"] < 1e-4
    @test all(0 .<= d.wells[w]["wcut"][i] .<= 1 for w in PRODUCERS for i in eachindex(d.report_time_s))
    @test all(sum(d.states["Saturations"], dims = 1) .≈ 1)
    @test all(all(d.wells[w]["orat"] .<= 0) for w in PRODUCERS)
    @test all(all(d.wells[w]["wrat"] .>= 0) for w in INJECTORS)
    rf = EGG_ORACLES["egg-recovery-factor"](d)["value"]
    @test 0 < rf < 100
end

@testset "CO2 ensemble" begin
    ens = load_ensemble()
    # The nominal scenario reproduces the base case.
    base = load_dataset("co2_storage")
    @test stored_t(by_factors(ens, 1.0, 1.0)) ≈ stored_t(base) rtol = 1e-6
    # More injection means more pressure rise, at every permeability.
    for k in (0.5, 1.0, 2.0)
        @test issorted([peak_rise_bar(by_factors(ens, r, k)) for r in (0.5, 1.0, 1.5, 2.0)])
    end
    # The pressure limit splits the ensemble (the question is not degenerate).
    n = ENSEMBLE_ORACLES["ens-scenarios-within-pressure-limit"](ens)["value"]
    @test 0 < n < 12
end

@testset "every oracle answers" begin
    for (dataset, oracles) in ORACLES
        d = dataset == "co2_ensemble" ? load_ensemble() : load_dataset(dataset)
        for (id, oracle) in oracles
            a = oracle(d)
            @test haskey(a, "choice") || (haskey(a, "value") && isfinite(a["value"]))
        end
    end
end
