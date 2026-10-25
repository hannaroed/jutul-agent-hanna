# Usage: julia --project=bench bench/datasets/build_co2.jl
include("co2_case.jl")
t0 = time()
case = co2_case()
println("simulated in $(round(time() - t0, digits = 1)) s")
save_dataset("co2_storage"; case.result, case.model, case.parameters, case.extra, case.meta)
