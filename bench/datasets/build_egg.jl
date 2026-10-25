# The EGG validation model (JutulDarcy examples/validation/2_validation_egg.jl):
# first member of the EGG ensemble, two-phase oil-water, 8 injectors and 4
# producers on fixed controls.
# Usage: julia --project=bench bench/datasets/build_egg.jl
include("common.jl")
using DelimitedFiles

egg_dir = JutulDarcy.GeoEnergyIO.test_input_file_path("EGG")
case = setup_case_from_data_file(joinpath(egg_dir, "EGG.DATA"))
t0 = time()
# Substates, as in the published example: with backward Euler the rate at the
# end of each ministep is the rate over it, so summing rate * dt is exact.
result = simulate_reservoir(case, info_level = -1, output_substates = true)
println("simulated in $(round(time() - t0, digits = 1)) s")
ref, header = readdlm(joinpath(egg_dir, "REFERENCE.CSV"), ',', header = true)
state0 = case.state0[:Reservoir]
extra = Dict{String, Any}(
    "initial_state" => Dict(String(k) => Array(v) for (k, v) in pairs(state0) if v isa AbstractArray{<:Real}),
    "phase_order" => "Saturations and TotalMasses rows: 1 = water, 2 = oil",
    "opm_reference" => Dict(String(h) => ref[:, i] for (i, h) in enumerate(vec(header))),
    "opm_reference_note" => "OPM Flow reference from the EGG test data; columns as in REFERENCE.CSV (time in days, field units as given there)",
)
meta = Dict{String, Any}(
    "source" => "GeoEnergyIO test data EGG/EGG.DATA, as in JutulDarcy validation example 2",
    "description" => "Channelised two-phase oil-water waterflood, 8 water injectors, 4 producers",
)
save_dataset("egg_waterflood"; result, case.model, case.parameters, extra, meta)
