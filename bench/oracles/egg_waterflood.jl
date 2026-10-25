# Oracles for the EGG waterflood (bench/data/egg_waterflood.jld2).
# Rows of Saturations / TotalMasses: 1 = water, 2 = oil. Well rates are surface
# volume rates in m^3/s, negative for production.

const WATER, OIL = 1, 2
const PRODUCERS = ("PROD1", "PROD2", "PROD3", "PROD4")
const INJECTORS = Tuple("INJECT$i" for i in 1:8)
# Breakthrough: the water cut first exceeds this. The 1-2% water cut on day 1
# is a start-up transient that falls back to zero, not breakthrough.
const BREAKTHROUGH_WCUT = 0.10

cumulative(d, w, key) = sum(d.wells[w][key] .* step_lengths(d))
breakthrough_step(d, w) = findfirst(>(BREAKTHROUGH_WCUT), d.wells[w]["wcut"])
oil_in_place(d) = sum(d.grid["pore_volume"] .* d.extra["initial_state"]["Saturations"][OIL, :])

const EGG_ORACLES = Dict{String, Function}(
    "egg-number-of-wells" => d -> quantity(length(d.wells), "1"),
    "egg-prod2-final-water-cut" => d -> quantity(100 * d.wells["PROD2"]["wcut"][end], "%"),
    "egg-max-injector-bhp" => d -> quantity(maximum(maximum(d.wells[w]["bhp"]) for w in INJECTORS) / BAR, "bar"),
    "egg-field-oil-rate-final" => d -> quantity(-sum(d.wells[w]["orat"][end] for w in PRODUCERS) * DAY, "m3/day"),

    "egg-first-breakthrough-well" =>
        d -> choice(PRODUCERS[argmin([d.report_time_s[breakthrough_step(d, w)] for w in PRODUCERS])]),
    "egg-first-breakthrough-time" =>
        d -> quantity(minimum(d.report_time_s[breakthrough_step(d, w)] for w in PRODUCERS) / DAY, "day"),
    "egg-cumulative-oil" => d -> quantity(-sum(cumulative(d, w, "orat") for w in PRODUCERS), "m3"),
    "egg-top-oil-producer" =>
        d -> choice(PRODUCERS[argmax([-cumulative(d, w, "orat") for w in PRODUCERS])]),
    # Trap: the reference is initial oil in place in reservoir volume; EGG oil
    # is dead and incompressible enough that surface and reservoir volumes are
    # close but not equal, so the definition must be stated.
    "egg-recovery-factor" => function (d)
        quantity(100 * -sum(cumulative(d, w, "orat") for w in PRODUCERS) / oil_in_place(d), "%")
    end,
    "egg-average-pressure-final" => function (d)
        pv = d.grid["pore_volume"]
        quantity(sum(pv .* d.states["Pressure"][:, end]) / sum(pv) / BAR, "bar")
    end,

    # Hard: closed boundaries, so the change in fluid mass in place must match
    # the net mass the wells moved. Total mass (water + oil): per phase the
    # balance is off by up to ~2%, because the well bores' own contents change
    # composition (water displaces oil inside the producers), and that is not
    # a conservation error. Measured from the first saved step: the saved
    # initial state has no evaluated masses. Relative to the mass injected.
    "egg-mass-conserved" => function (d)
        m = vec(sum(d.states["TotalMasses"], dims = (1, 2)))
        dt = step_lengths(d)
        rate(w) = d.wells[w]["Aqueous_mass_rate"] .+ get(d.wells[w], "Liquid_mass_rate", zero(dt))
        net = cumsum(sum(rate(w) for w in (INJECTORS..., PRODUCERS...)) .* dt)
        inj = cumsum(sum(rate(w) for w in INJECTORS) .* dt)
        err = maximum(abs((m[i] - m[1]) - (net[i] - net[1])) for i in eachindex(dt)) / (inj[end] - inj[1])
        merge(choice(err < 1e-3 ? "yes" : "no"), Dict("max_relative_error" => err))
    end,
)
