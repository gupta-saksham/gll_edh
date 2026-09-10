"""Scratch: reference-trajectory statistics for calibrating tariff magnitudes."""

import jax
import numpy as np

from sandbox.controller import base_controller
from sandbox.rollout import rollout
from sandbox.scenarios import reference_scenario, step_duration_h

pop = reference_scenario()
tr = rollout(base_controller(), pop, jax.random.PRNGKey(0), n_steps=672)

e = np.asarray(tr.e_grid_kwh)
tk = np.asarray(tr.transformer_kw)
tq = np.asarray(tr.transformer_kvar)
loss = np.asarray(tr.losses_kw)
v = np.asarray(tr.voltage_pu)[:, np.asarray(pop.pq_bus_id())]
s = np.asarray(tr.settlement_chf)

print("step_h", step_duration_h())
print("transformer_kw: min %.1f max %.1f mean %.1f" % (tk.min(), tk.max(), tk.mean()))
print("kva: max %.1f p99 %.1f mean %.1f" % (np.hypot(tk, tq).max(), np.percentile(np.hypot(tk, tq), 99), np.hypot(tk, tq).mean()))
print("transformer_kvar: min %.1f max %.1f" % (tq.min(), tq.max()))
print("losses_kw: mean %.3f max %.3f  total kWh %.1f" % (loss.mean(), loss.max(), loss.sum() * step_duration_h()))
print("e_grid_kwh per point: min %.2f max %.2f" % (e.min(), e.max()))
print("export kwh total %.1f import total %.1f" % (np.maximum(e, 0).sum(), np.maximum(-e, 0).sum()))
print("abs e sum %.1f" % np.abs(e).sum())
print("fair leg total CHF %.2f" % s.sum())
print("fair leg per interval mean %.4f" % s.sum(1).mean())
print("voltage pq: min %.4f max %.4f mean %.4f" % (v.min(), v.max(), v.mean()))
print("per point mean voltage:", np.round(v.mean(0), 4))
print("per point max |e|:", np.round(np.abs(e).max(0), 2))
print("per point export kwh:", np.round(np.maximum(e, 0).sum(0), 1))
print("per point import kwh:", np.round(np.maximum(-e, 0).sum(0), 1))
print("type_of_pq", pop.type_of_pq)
print("distance_rank", pop.distance_rank)

# reverse flow share and hour distribution of extremes
day = np.asarray(tr.day_step) * step_duration_h()
print("reverse share %.3f" % (tk < 0).mean())
print("hour of export peak %.2f" % day[np.argmin(tk)])
print("hour of draw peak %.2f" % day[np.argmax(tk)])
exp_stress_hours = day[tk < -30]
print("hours where export > 30kW:", np.percentile(exp_stress_hours, [0, 25, 50, 75, 100]) if exp_stress_hours.size else "none")
imp = day[tk > 30]
print("hours where draw > 30kW:", np.percentile(imp, [0, 25, 50, 75, 100]) if imp.size else "none")

# own-voltage vs own-energy slope, per point (the sensitivity proxy)
slopes = []
for i in range(e.shape[1]):
    slope = np.polyfit(e[:, i], v[:, i], 1)[0]
    slopes.append(slope)
print("slope pu per kWh:", np.round(np.array(slopes), 5))
