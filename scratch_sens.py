"""Scratch: can own-bus voltage sensitivity be estimated from available fields?

Compares three estimators against distance rank and against a physical
finite-difference sensitivity from marginal_audit-style perturbations.
"""

import jax
import numpy as np

from sandbox.controller import base_controller
from sandbox.rollout import rollout
from sandbox.scenarios import reference_scenario

pop = reference_scenario()
tr = rollout(base_controller(), pop, jax.random.PRNGKey(0), n_steps=672)

e = np.asarray(tr.e_grid_kwh)
v = np.asarray(tr.voltage_pu)[:, np.asarray(pop.pq_bus_id())]
t = np.asarray(tr.transformer_kw)
rank = np.asarray(pop.distance_rank)

simple = np.array([np.polyfit(e[:, i], v[:, i], 1)[0] for i in range(18)])

# Partial slope of own voltage on own energy, holding transformer flow fixed.
partial = []
for i in range(18):
    X = np.column_stack([e[:, i], t, np.ones_like(t)])
    beta = np.linalg.lstsq(X, v[:, i], rcond=None)[0]
    partial.append(beta[0])
partial = np.array(partial)

# Same, but with the feeder aggregate proxied by the sum of everyone's energy
# (a tariff sees e_grid_kwh, so this needs no extra field either).
agg = e.sum(1)
partial_agg = []
for i in range(18):
    X = np.column_stack([e[:, i], agg, np.ones_like(agg)])
    beta = np.linalg.lstsq(X, v[:, i], rcond=None)[0]
    partial_agg.append(beta[0])
partial_agg = np.array(partial_agg)

print("type            rank  simple   partial(tk)  partial(sum e)")
for i in range(18):
    print(f"{pop.type_of_pq[i]:<14} {rank[i]:>4}  {simple[i]:+.5f}   {partial[i]:+.5f}      {partial_agg[i]:+.5f}")

for name, est in (("simple", simple), ("partial_tk", partial), ("partial_agg", partial_agg)):
    owners = np.array([pop.type_of_pq[i] != "tenant" for i in range(18)])
    c_all = np.corrcoef(rank, est)[0, 1]
    c_own = np.corrcoef(rank[owners], est[owners])[0, 1]
    print(f"{name}: corr with distance rank all={c_all:+.3f} owners={c_own:+.3f}")
