# Voltage-price tariff: four-cell comparison

The selected family response is policy 5 (`stagger_11_1h`). Values are means over the independent test weeks. Group prices are CHF per actual behind-meter load kWh; parentheses show the difference from `fair_leg/base`, where positive is worse.

| Cell | Export peak kW | Import peak kW | Ramp kW | Curtailment | Community CHF/episode | Tenant CHF/kWh | PV-only CHF/kWh | PV+battery CHF/kWh | Large-flex CHF/kWh |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fair_leg/base` | 65.30 | 7.90 | 15.28 | 1.50% | 294.28 | +0.223 (+0.000) | -0.263 (+0.000) | -0.421 (+0.000) | -0.148 (+0.000) |
| `fair_leg/selected_family` | 61.09 | 8.24 | 11.73 | 0.42% | 304.06 | +0.223 (+0.000) | -0.260 (+0.003) | -0.431 (-0.010) | -0.155 (-0.007) |
| `voltage_price/tuned_base` | 65.23 | 8.23 | 47.61 | 0.65% | 300.22 | +0.036 (-0.187) | -0.302 (-0.039) | -0.352 (+0.068) | -0.103 (+0.046) |
| `voltage_price/selected_family` | 61.09 | 8.24 | 11.73 | 0.42% | 304.06 | +0.035 (-0.188) | -0.299 (-0.037) | -0.357 (+0.064) | -0.105 (+0.043) |

The tariff energy price is `clip(0.15 - 1.5 * (voltage_pu - 1.0), 0.05, 0.25)` CHF/kWh. An equal lump-sum balance reconciles every interval to fair LEG's total.

This is a diagnostic voltage-level tariff. Local voltage measures exposure, not a household's marginal contribution to the voltage condition.
