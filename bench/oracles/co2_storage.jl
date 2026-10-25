# Oracles for the CO2 storage dataset (bench/data/co2_storage.jld2).
# Rows of Saturations / TotalMasses: 1 = brine, 2 = CO2 (the initial state puts
# all mass in the first component). K = 1 is the top layer.

const CO2 = 2

co2_inventory(d) = d.extra["co2_inventory_kg"]
injection_end(d) = step_at(d, d.extra["injection_end_time_s"])

const CO2_ORACLES = Dict{String, Function}(
    # Easy: one lookup and at most a unit conversion.
    "co2-bhp-end-of-injection" => d -> quantity(d.wells["Injector"]["bhp"][injection_end(d)] / BAR, "bar"),
    "co2-total-injected" => d -> quantity(co2_inventory(d)["injected"][end] / 1e3, "t"),
    "co2-injection-rate" => d -> quantity(d.wells["Injector"]["CO2_mass_rate"][1] * DAY, "kg/day"),
    "co2-max-gas-saturation-end-of-injection" =>
        d -> quantity(maximum(d.states["Saturations"][CO2, :, injection_end(d)]), "1"),

    # Medium: 2-4 operations (time integral, region, domain definition).
    "co2-dissolved-fraction-end-of-injection" => function (d)
        inv = co2_inventory(d); k = injection_end(d)
        quantity(100 * inv["dissolved"][k] / inv["injected"][k], "%")
    end,
    "co2-residual-fraction-final" => function (d)
        inv = co2_inventory(d)
        quantity(100 * inv["residual"][end] / inv["injected"][end], "%")
    end,
    # Trap: peak vs final. The largest rise over the injection period, anywhere.
    "co2-peak-pressure-rise-during-injection" => function (d)
        p = d.states["Pressure"][:, 1:injection_end(d)]
        quantity(maximum(p .- d.extra["initial_pressure_Pa"]) / BAR, "bar")
    end,
    "co2-mass-in-upper-layer-end-of-injection" => function (d)
        cells = findall(==(1), d.extra["region"])
        quantity(sum(d.states["TotalMasses"][CO2, cells, injection_end(d)]) / 1e3, "t")
    end,
    "co2-mobile-fraction-final" => function (d)
        inv = co2_inventory(d)
        quantity(100 * inv["mobile"][end] / inv["injected"][end], "%")
    end,

    # Hard: several steps, a threshold, or a physics check.
    "co2-leakage-fraction-final" => function (d)
        inv = co2_inventory(d)
        quantity(100 * inv["outside_domain"][end] / inv["injected"][end], "%")
    end,
    # When the CO2 that has left the domain first exceeds 1% of the injected total.
    "co2-leakage-onset" => function (d)
        inv = co2_inventory(d)
        k = findfirst(i -> inv["outside_domain"][i] > 0.01 * inv["injected"][end], eachindex(inv["outside_domain"]))
        quantity(d.report_time_s[k] / YEAR, "yr")
    end,
)
