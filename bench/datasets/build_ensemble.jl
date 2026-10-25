# Scenario ensemble: 12 variants of the CO2 case (injection rate x permeability).
# Each variant saves pressure, saturations and the CO2 inventory only, to keep
# the ensemble small on disk.
# Usage: julia --project=bench bench/datasets/build_ensemble.jl
include("co2_case.jl")
const RATES = (0.5, 1.0, 1.5, 2.0)
const PERMS = (0.5, 1.0, 2.0)
for (i, (r, k)) in enumerate(Iterators.product(RATES, PERMS))
    name = "co2_ensemble/scenario_$(lpad(i, 2, '0'))"
    isfile(joinpath(BENCH_DATA, "$name.jld2")) && (println("skip $name"); continue)
    t0 = time()
    case = try
        co2_case(rate_factor = r, perm_factor = k)
    catch err
        println("FAILED $name rate_factor=$r perm_factor=$k: ", sprint(showerror, err)); continue
    end
    println("$name rate_factor=$r perm_factor=$k simulated in $(round(time() - t0, digits = 1)) s")
    save_dataset(name; case.result, case.model, case.parameters, case.extra, case.meta,
        keep = (:Pressure, :Saturations))
end
