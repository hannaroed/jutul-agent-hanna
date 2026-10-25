# Oracles for the 12-scenario CO2 ensemble (bench/data/co2_ensemble/).
# Scenario i has meta["rate_factor"] and meta["perm_factor"]; each file holds
# Pressure, Saturations and the CO2 inventory.

load_ensemble() = [load_dataset("co2_ensemble/scenario_$(lpad(i, 2, '0'))") for i in 1:12]

scenario_name(i) = "scenario_$(lpad(i, 2, '0'))"
peak_rise_bar(d) = maximum(d.states["Pressure"] .- d.extra["initial_pressure_Pa"]) / BAR
stored_t(d) = d.extra["co2_inventory_kg"]["domain_total"][end] / 1e3
leaked_pct(d) = 100 * d.extra["co2_inventory_kg"]["outside_domain"][end] / d.extra["co2_inventory_kg"]["injected"][end]
by_factors(ens, r, k) = only(d for d in ens if d.meta["rate_factor"] == r && d.meta["perm_factor"] == k)

# Allowed pressure rise above initial, in any cell at any time. Chosen from the
# spread of peak rises across the ensemble so that the limit excludes some
# scenarios (set before any agent run; see bench/questions/co2_ensemble.yaml).
const PRESSURE_RISE_LIMIT_BAR = 0.5

const ENSEMBLE_ORACLES = Dict{String, Function}(
    "ens-most-stored-within-pressure-limit" => function (ens)
        ok = [i for (i, d) in enumerate(ens) if peak_rise_bar(d) <= PRESSURE_RISE_LIMIT_BAR]
        choice(scenario_name(ok[argmax([stored_t(ens[i]) for i in ok])]))
    end,
    "ens-most-leakage-scenario" => ens -> choice(scenario_name(argmax(leaked_pct.(ens)))),
    "ens-scenarios-within-pressure-limit" =>
        ens -> quantity(count(d -> peak_rise_bar(d) <= PRESSURE_RISE_LIMIT_BAR, ens), "1"),
    # Two runs at nominal permeability: half vs twice the nominal rate. (Nominal
    # vs twice differs by only ~0.15 points, too small to grade at 1%.)
    "ens-rate-effect-on-leakage" => function (ens)
        quantity(leaked_pct(by_factors(ens, 0.5, 1.0)) - leaked_pct(by_factors(ens, 2.0, 1.0)), "percentage_point")
    end,
)
