# Load a saved dataset as a NamedTuple of plain arrays and dicts.
#   include("bench/datasets/load.jl"); d = load_dataset("co2_storage")
using JLD2

const BENCH_DATA_DIR = joinpath(@__DIR__, "..", "data")

load_dataset(name::AbstractString) = (; (Symbol(k) => v for (k, v) in load(joinpath(BENCH_DATA_DIR, "$name.jld2")))...)

list_datasets() = sort([relpath(joinpath(r, f), BENCH_DATA_DIR)[1:end-5]
                        for (r, _, fs) in walkdir(BENCH_DATA_DIR) for f in fs if endswith(f, ".jld2")])

if abspath(PROGRAM_FILE) == @__FILE__
    for name in list_datasets()
        t = @elapsed d = load_dataset(name)
        println(rpad(name, 32), round(t, digits = 2), " s  ", length(d.report_time_s), " steps, ",
                d.grid["number_of_cells"], " cells, fields ", join(sort(collect(keys(d.states))), ","))
    end
end
