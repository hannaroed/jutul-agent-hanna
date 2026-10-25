# Shared helpers for building and saving the benchmark datasets.
#
# Every dataset is one JLD2 file under bench/data/ holding plain arrays and
# dicts (no Jutul types), so it loads without rebuilding a model. All values
# are SI: Pa, K, kg, m, s, m^3/s, kg/s.

using Jutul, JutulDarcy, JLD2, Dates

const BENCH_DATA = joinpath(@__DIR__, "..", "data")

"Stack per-step state dicts into one array per field, time as the last dimension."
function stack_states(states; keep = missing)
    fields = Dict{String, Any}()
    for key in keys(first(states))
        ismissing(keep) || key in keep || continue
        v0 = first(states)[key]
        v0 isa AbstractArray{<:Real} || continue
        fields[String(key)] = cat((Array(s[key]) for s in states)...; dims = ndims(v0) + 1)
    end
    return fields
end

"Well results as name => quantity => vector over report steps."
function plain_wells(ws)
    return Dict(String(w) => Dict(String(k) => collect(v) for (k, v) in pairs(r)) for (w, r) in pairs(ws.wells))
end

"Per report step: ministeps, Newton iterations and wall time (s), where reported."
function solver_summary(reports)
    ministeps = Int[]; newton = Int[]; wall = Float64[]
    for r in reports
        ms = get(r, :ministeps, [])
        push!(ministeps, length(ms))
        push!(newton, sum((length(get(m, :steps, [])) for m in ms); init = 0))
        push!(wall, Float64(get(r, :total_time, NaN)))
    end
    return Dict("ministeps" => ministeps, "newton_iterations" => newton, "wall_time_s" => wall)
end

function grid_info(model, parameters)
    domain = reservoir_domain(reservoir_model(model))
    g = physical_representation(domain)
    nc = number_of_cells(domain)
    info = Dict{String, Any}(
        "cell_centroids" => Array(domain[:cell_centroids]),
        "volumes" => Array(domain[:volumes]),
        "porosity" => Array(domain[:porosity]),
        "permeability" => Array(domain[:permeability]),
        "pore_volume" => Array(pore_volume(model, parameters)),
        "number_of_cells" => nc,
    )
    try
        info["cell_ijk"] = permutedims(reduce(hcat, [collect(cell_ijk(g, c)) for c in 1:nc]))
    catch
    end
    return info
end

function save_dataset(name; result::ReservoirSimResult, model, parameters, extra = Dict{String, Any}(), meta = Dict{String, Any}(), keep = missing)
    path = joinpath(BENCH_DATA, "$name.jld2")
    mkpath(dirname(path))
    ws, states, t = result
    meta = merge(Dict{String, Any}(
        "name" => name,
        "created_utc" => string(now(UTC)),
        "jutuldarcy_version" => string(pkgversion(JutulDarcy)),
        "units" => "SI (Pa, K, kg, m, s; rates in m^3/s or kg/s, negative = production)",
        "time_axis" => "report_time_s[i] is the end of report step i; states and well results are given there",
    ), meta)
    jldsave(path; meta, grid = grid_info(model, parameters), report_time_s = collect(t),
        states = stack_states(states; keep), wells = plain_wells(ws),
        solver = solver_summary(result.result.reports), extra)
    println("saved $path ($(round(filesize(path) / 2^20, digits = 1)) MiB)")
    return path
end
