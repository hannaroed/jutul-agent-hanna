# Compute every oracle answer and write bench/answers.json.
# Usage: julia --project=bench bench/oracles/run_oracles.jl
include("common.jl")
include("co2_storage.jl")
include("egg_waterflood.jl")
isfile(joinpath(@__DIR__, "co2_ensemble.jl")) && include("co2_ensemble.jl")
using JSON

const ORACLES = Dict(
    "co2_storage" => CO2_ORACLES,
    "egg_waterflood" => EGG_ORACLES,
    (@isdefined(ENSEMBLE_ORACLES) ? ("co2_ensemble" => ENSEMBLE_ORACLES,) : ())...,
)

if abspath(PROGRAM_FILE) == @__FILE__
    answers = Dict{String, Any}()
    for (dataset, oracles) in ORACLES
        d = dataset == "co2_ensemble" ? load_ensemble() : load_dataset(dataset)
        for id in sort(collect(keys(oracles))); oracle = oracles[id]
            answers[id] = merge(oracle(d), Dict("dataset" => dataset))
            println(rpad(id, 44), answers[id])
        end
    end
    open(joinpath(@__DIR__, "..", "answers.json"), "w") do io
        JSON.print(io, answers, 2)
    end
end
