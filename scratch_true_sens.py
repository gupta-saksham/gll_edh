"""Scratch: the physical own-injection voltage sensitivity, for reference only."""

import numpy as np

from sandbox.scenarios import _self_impedance_to_slack, grid_arrays, reference_scenario

pop = reference_scenario()
arrays = grid_arrays()
z = _self_impedance_to_slack()
base_s_kw = float(arrays["base_s_mva"]) * 1000.0
# 1 kWh in a 15-minute interval is 4 kW.
dv_per_kwh = z * 4.0 / base_s_kw
rank = np.asarray(pop.distance_rank)
print("base_s_mva", arrays["base_s_mva"])
for i in np.argsort(rank):
    print(f"{pop.type_of_pq[i]:<12} rank {rank[i]:>2}  z_ii {z[i]:.4f} pu  dV/dE {dv_per_kwh[i]:.5f} pu/kWh")
print("spread of true dV/dE: %.5f to %.5f pu/kWh" % (dv_per_kwh.min(), dv_per_kwh.max()))
