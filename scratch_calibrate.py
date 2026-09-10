"""Scratch: incidence and marginal exposure of every bank scenario.

Re-settles one fair-LEG trajectory (behaviour held fixed) under each
scenario -- exactly the condition the revenue-adequacy gate uses -- and
reports the pooled transfer, per-group incidence in CHF per kWh of own load
(the quantity the experiment's screens use), and two marginal probes:

* sustained: 1 kWh/interval more export at one point, all intervals;
* spike: 1 kWh more export in the single worst export interval, measured on
  the point's WHOLE-episode settlement so a ratchet's intertemporal cost is
  included.
"""

import jax
import numpy as np

from sandbox.controller import base_controller
from sandbox.experiments import household_load_kwh, resettle
from sandbox.rollout import rollout
from sandbox.scenarios import reference_scenario
from sandbox.tariff_family import TARIFF_BANK_NAMES, scenario_params

pop = reference_scenario()
tr = rollout(base_controller(), pop, jax.random.PRNGKey(0), n_steps=672)
loads = household_load_kwh(tr, pop)
fair = np.asarray(tr.settlement_chf)
GROUPS = ("tenant", "pv_only", "pv_battery", "large_flex")
masks = {k: np.asarray(pop.mask_for(k)) for k in GROUPS}

DELTA = 0.25
PROBES = {13: "large_flex", 8: "pv_battery"}
transformer = np.asarray(tr.transformer_kw)
worst = int(np.argmin(transformer))
stressed = np.argsort(transformer)[:96]


def cost_per_load(settled):
    return {k: -settled.sum(0)[m].sum() / loads[m].sum() for k, m in masks.items()}


def probes(name):
    params = scenario_params(name)
    base = np.asarray(resettle(tr, pop, params).settlement_chf)
    out = {}
    for point in PROBES:
        sustained = tr.replace(e_grid_kwh=tr.e_grid_kwh.at[:, point].add(DELTA))
        spike = tr.replace(e_grid_kwh=tr.e_grid_kwh.at[worst, point].add(DELTA))
        more = np.asarray(resettle(sustained, pop, params).settlement_chf)[:, point]
        once = np.asarray(resettle(spike, pop, params).settlement_chf)[:, point]
        out[point] = (
            float(np.mean((base[:, point] - more)[stressed]) / DELTA),
            float((base[:, point].sum() - once.sum()) / DELTA),
        )
    return base, out


header = f"{'scenario':<28}{'total':>8}{'pool':>7}"
for group in GROUPS:
    header += f"{group[:9]:>10}"
header += f"{'sus13':>8}{'spk13':>8}{'sus8':>8}{'spk8':>8}"
print(header)
reference = cost_per_load(fair)
row = f"{'fair_leg (absolute)':<28}{fair.sum():>8.1f}{0.0:>7.1f}"
for group in GROUPS:
    row += f"{reference[group]:>10.3f}"
print(row)

for name in TARIFF_BANK_NAMES:
    base, marginal = probes(name)
    pool = float(np.abs(base - fair).sum() / 2.0)
    cost = cost_per_load(base)
    row = f"{name:<28}{base.sum():>8.1f}{pool:>7.1f}"
    for group in GROUPS:
        row += f"{cost[group] - reference[group]:>+10.3f}"
    for point in PROBES:
        row += f"{marginal[point][0]:>8.3f}{marginal[point][1]:>8.3f}"
    print(row)
