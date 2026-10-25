# Oracle helpers. Every oracle takes a loaded dataset and returns a Dict with
# "value" and "unit" (a quantity), or "choice" (a categorical answer), or both.
# Units are the ones bench/grader.py knows; values are in that unit.

include(joinpath(@__DIR__, "..", "datasets", "load.jl"))

const DAY = 86400.0
const YEAR = 365.0 * DAY  # the CO2 case uses 365-day report steps
const BAR = 1e5

quantity(value, unit) = Dict{String, Any}("value" => Float64(value), "unit" => unit)
choice(name) = Dict{String, Any}("choice" => String(name))

"Index of the first report step whose end time reaches `t` seconds."
step_at(d, t) = findfirst(>=(t - 1.0), d.report_time_s)

"Durations of the report steps, in seconds."
step_lengths(d) = diff(vcat(0.0, d.report_time_s))
