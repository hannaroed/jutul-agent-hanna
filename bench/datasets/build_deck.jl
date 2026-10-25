# A published input-deck case from the GeoEnergyIO test data, for the RQ3
# scaling datasets: SPE9 (black-oil, 9000 cells) or Norne (real field geometry,
# ~44k active cells). Substates are saved, as in build_egg.jl, so summing
# rate * dt is exact; the OPM reference curves are not stored with the output.
# Usage: julia --project=bench bench/datasets/build_deck.jl spe9|norne
include("common.jl")

const DECKS = Dict(
    "spe9" => ("SPE9", "SPE9.DATA", "spe9_blackoil", "SPE9 three-phase black-oil benchmark, 1 water injector, 25 producers"),
    "norne" => ("NORNE_NOHYST", "NORNE_NOHYST.DATA", "norne_field", "Norne field model (no hysteresis), real field geometry and well schedule"),
)

key = only(ARGS)
dir, file, name, description = DECKS[key]
case = setup_case_from_data_file(joinpath(JutulDarcy.GeoEnergyIO.test_input_file_path(dir), file))
t0 = time()
result = simulate_reservoir(case, info_level = -1, output_substates = true)
println("$key simulated in $(round(time() - t0, digits = 1)) s")
rsys = reservoir_model(case.model).system
state0 = case.state0[:Reservoir]
extra = Dict{String, Any}(
    "initial_state" => Dict(String(k) => Array(v) for (k, v) in pairs(state0) if v isa AbstractArray{<:Real}),
    "phases" => string.(JutulDarcy.get_phases(rsys)),
    "components" => collect(string.(JutulDarcy.component_names(rsys))),
)
meta = Dict{String, Any}("source" => "GeoEnergyIO test data $dir/$file", "description" => description)
save_dataset(name; result, case.model, case.parameters, extra, meta)
