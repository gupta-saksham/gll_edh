# Copyright 2026 ewz - Zurich Municipal Electric Utility.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regenerate the tariff-bank incidence and marginal-exposure numbers.

    uv run python scripts/measure_tariff_incidence.py

Two claims are measured here, and both are quoted in
``CONTROLLER_FRAMEWORK_PLAN.md`` section 8 and in ``sandbox/tariff_family.py``.

**A tariff's headline rate is not its incentive.** Every scenario in the bank
is re-settled over one fixed fair-LEG trajectory -- behaviour held fixed,
which is the condition :func:`sandbox.metrics.revenue_adequate` uses -- and
reported three ways: the weekly pool it moves, each household type's cost per
kWh of its own load against fair LEG, and what one extra exported kWh costs
the connection point that exported it. The marginal probe runs twice, because
the two answers differ by an order of magnitude for some mechanisms: once for
a sustained increase in every interval, and once for a single-interval spike
measured on the point's whole-episode settlement, which is where a ratchet's
intertemporal cost shows up.

**The voltage-sensitivity estimate does not recover electrical distance.** A
locational price should charge the sensitivity of the binding quantity to a
point's own injection rather than the voltage level, and the bank's
``voltage_sensitivity`` scenario estimates it in the carry from own voltage
against own energy while controlling for the substation flow. The second
table compares that estimate against the physical sensitivity implied by each
connection point's Thevenin impedance to the slack. They do not agree, which
is why the scenario is labelled a measured negative result rather than a
recommendation.

Re-run this whenever the bank's parameters, the population or the weather
model changes. It takes a few minutes: the marginal probes re-settle the week
once per scenario per probe.
"""

import jax
import numpy as np

from sandbox.controller import base_controller
from sandbox.experiments import grid_views, household_load_kwh, resettle
from sandbox.metrics import HOUSEHOLD_TYPES, REVENUE_TOLERANCE
from sandbox.rollout import rollout
from sandbox.scenarios import (
    EPISODE_STEPS,
    _self_impedance_to_slack,
    grid_arrays,
    reference_scenario,
    step_duration_h,
)
from sandbox.tariff_family import (
    TARIFF_BANK_NAMES,
    _sensitivity_index,
    family_tariff,
    init_family_carry,
    scenario_params,
)

#: One extra exported kWh is the probe. A quarter of a kWh in a 15-minute
#: interval is 1 kW, small enough to stay inside every activation ramp.
PROBE_KWH = 0.25

#: Two connection points to probe: the furthest ``large_flex`` and a
#: mid-feeder ``pv_battery``. Both export heavily, which is what the export
#: charges in the bank are written against.
PROBE_POINTS = (13, 8)

#: How many of the week's most strained intervals the sustained probe averages
#: over -- one day's worth, so it reads the price where it is meant to bite.
STRESSED_INTERVALS = 96


def settle(trajectory, population, scenario):
    """The whole episode re-settled under one scenario, carry threaded."""
    return np.asarray(resettle(trajectory, population, scenario_params(scenario)).settlement_chf)


def marginal_chf_per_kwh(trajectory, population, scenario, stressed, worst):
    """Sustained and spike marginal exposure, per probe point."""
    reference = settle(trajectory, population, scenario)
    exposure = {}
    for point in PROBE_POINTS:
        sustained = trajectory.replace(
            e_grid_kwh=trajectory.e_grid_kwh.at[:, point].add(PROBE_KWH)
        )
        spike = trajectory.replace(
            e_grid_kwh=trajectory.e_grid_kwh.at[worst, point].add(PROBE_KWH)
        )
        every = settle(sustained, population, scenario)[:, point]
        once = settle(spike, population, scenario)[:, point]
        exposure[point] = (
            float(np.mean((reference[:, point] - every)[stressed]) / PROBE_KWH),
            float((reference[:, point].sum() - once.sum()) / PROBE_KWH),
        )
    return reference, exposure


def incidence(settlement, loads, masks):
    """What a kWh of own load cost each household type. Negative means earned."""
    return {
        name: -settlement.sum(0)[mask].sum() / loads[mask].sum() for name, mask in masks.items()
    }


def charged_sensitivity(trajectory, population):
    """The index the bank actually charges, after a full episode of learning."""
    params = scenario_params("voltage_sensitivity")

    def accumulate(carry, view):
        _, carry = family_tariff(view, carry, params)
        return carry, None

    carry, _ = jax.lax.scan(
        accumulate, init_family_carry(population.num_pq), grid_views(trajectory, population)
    )
    return np.asarray(_sensitivity_index(carry))


def physical_sensitivity():
    """dV/dE in pu per kWh at each connection point, from its Thevenin impedance.

    The classic reading of voltage rise off the bus admittance matrix: the
    self-impedance to the slack, in pu on the grid's own base, times the power
    one kWh in an interval represents. Not available to a tariff, which is the
    whole point of the comparison.
    """
    base_s_kw = float(grid_arrays()["base_s_mva"]) * 1000.0
    return _self_impedance_to_slack() / step_duration_h() / base_s_kw


def report_sensitivity(trajectory, population):
    charged = charged_sensitivity(trajectory, population)
    physical = physical_sensitivity()
    rank = np.asarray(population.distance_rank)
    print("\nVoltage sensitivity: what the carry estimates against what the network does")
    print(f"{'point':>6}{'type':>12}{'rank':>6}{'dV/dE pu/kWh':>14}{'charged index':>15}")
    print("-" * 53)
    for point in np.argsort(rank):
        print(
            f"{point:>6}{population.type_of_pq[point]:>12}{rank[point]:>6}"
            f"{physical[point]:>14.5f}{charged[point]:>15.2f}"
        )
    print(
        f"  physical sensitivity spans {physical.min():.5f} to {physical.max():.5f} pu/kWh, "
        "monotone in distance by construction"
    )
    print(
        f"  charged index correlates {np.corrcoef(rank, charged)[0, 1]:+.2f} with distance rank "
        f"and {np.corrcoef(physical, charged)[0, 1]:+.2f} with the physical sensitivity"
    )
    zeroed = np.flatnonzero(charged <= 0.0)
    print(
        f"  {zeroed.size} connection points are charged nothing at all: "
        f"{[population.type_of_pq[i] for i in zeroed]}"
    )


def main() -> None:
    population = reference_scenario()
    trajectory = rollout(
        base_controller(), population, jax.random.PRNGKey(0), n_steps=EPISODE_STEPS
    )
    loads = household_load_kwh(trajectory, population)
    masks = {name: np.asarray(population.mask_for(name)) for name in HOUSEHOLD_TYPES}
    fair = np.asarray(trajectory.settlement_chf)
    reference = incidence(fair, loads, masks)

    transformer = np.asarray(trajectory.transformer_kw)
    stressed = np.argsort(transformer)[:STRESSED_INTERVALS]
    worst = int(np.argmin(transformer))

    header = f"{'scenario':<28}{'CHF':>8}{'pool':>7}"
    header += "".join(f"{name[:9]:>10}" for name in HOUSEHOLD_TYPES)
    header += "".join(f"{label:>8}" for label in ("sust13", "spike13", "sust8", "spike8"))
    print("Incidence and marginal exposure, behaviour held fixed, one week")
    print(header)
    print("-" * len(header))
    print(
        f"{'fair_leg (absolute)':<28}{fair.sum():>8.1f}{0.0:>7.1f}"
        + "".join(f"{reference[name]:>10.3f}" for name in HOUSEHOLD_TYPES)
    )
    for scenario in TARIFF_BANK_NAMES:
        settlement, exposure = marginal_chf_per_kwh(
            trajectory, population, scenario, stressed, worst
        )
        pool = float(np.abs(settlement - fair).sum() / 2.0)
        cost = incidence(settlement, loads, masks)
        row = f"{scenario:<28}{settlement.sum():>8.1f}{pool:>7.1f}"
        row += "".join(f"{cost[name] - reference[name]:>+10.3f}" for name in HOUSEHOLD_TYPES)
        row += "".join(f"{value:>8.3f}" for point in PROBE_POINTS for value in exposure[point])
        print(row)
    print(
        f"\n  CHF is the community total; fair LEG collects {fair.sum():.1f} and the gate allows"
        f" {REVENUE_TOLERANCE:.0%}.\n  Household-type columns are the change in CHF per kWh of that"
        " type's own load.\n  sust/spike are CHF per extra exported kWh at connection points"
        f" {PROBE_POINTS}, sustained and single-interval."
    )
    report_sensitivity(trajectory, population)


if __name__ == "__main__":
    main()
