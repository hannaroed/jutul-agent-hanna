---
name: output-semantics
description: Read saved simulation output correctly - SI units, sign conventions, peak vs final, rates vs cumulatives, CO2 inventory terms, and how to keep large data out of the conversation
---

# Reading saved simulation output

## When to use

Use this whenever a question is answered from simulation output that is already
saved (states, well results, reports), rather than by running a new simulation.

## Keep the data in Julia

The saved arrays are far too large to read. Load them in `run_julia`, compute the
one number the question needs there, and print only that number (and a short check).
Never print whole arrays, tables or dictionaries of arrays. Look at the structure
first (keys, sizes, metadata) and plan the computation before printing anything.

## Units and signs

- Saved values are SI: pressure in Pa (1 bar = 1e5 Pa), mass in kg, volume in m^3,
  time in s, rates per second (m^3/s or kg/s). Convert to the unit asked for at the end.
- 1 day = 86400 s. State which year length you use if it matters (365 or 365.25 days).
- Well rates are positive for injection and negative for production. Report a
  production rate or volume as a positive number.
- Water cut is a fraction (0-1) unless converted; say percent explicitly.
- A *percentage point* difference (30% vs 20% = 10 points) is not a percent change
  (50%). Answer in the one the question asks for.

## Time: peak, final, at a time

- "At the end of X" means the saved step at the end of that period, not the maximum.
- "Largest", "peak" or "maximum over a period" means the maximum over every saved
  step in that period (and, for a field, over every cell).
- A "rise above initial" is a difference from the initial value, cell by cell, not
  an absolute level.
- Saved time is the *end* of each step. Find a step by its time, not by assuming
  a step length.

## Rates vs cumulatives

- A cumulative amount is the time integral of a rate: sum rate * step length over
  the saved steps, with each step's length = its end time minus the previous end
  time (the first from time 0). With fully implicit time stepping the rate saved at
  the end of a step is the rate over that step.
- Check that a cumulative never decreases and that it matches any amount the output
  states directly.

## Space: regions and layers

- In a Cartesian grid, layer K = 1 is normally the top (depth increases with K);
  check the cell depths if unsure.
- A quantity "in a region" is summed over that region's cells only. Weight averages
  by pore volume (or bulk volume) when asked for an average over cells.
- Multi-component arrays have one row per phase or component. Find the order from
  metadata or the initial state; do not assume it.

## CO2 storage terms

- *Injected*: total mass injected so far. *In the domain*: mass currently in the cells.
  *Left the domain*: injected minus in the domain minus produced (through open boundaries).
- Of the CO2 in the domain: *dissolved* (in brine), *residual* (trapped gas, immobile
  at the current saturation), *mobile* (gas that can still flow). These three add up to
  the mass in the domain; a percentage "of the injected CO2" divides by injected.

## Checks before answering

- Does the order of magnitude make sense for the model's size and dimensions?
- Does a conservation total close (in - out = change in place)?
- Is the unit the one asked for?
