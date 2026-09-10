# Local-voltage price tariff: experiment and results

## Tariff

The tariff assigns every connection point a local energy price that decreases
as its solved bus voltage increases:

```text
price_chf_per_kwh = clip(0.15 - 1.5 * (voltage_pu - 1.0), 0.05, 0.25)
raw_settlement_chf = price_chf_per_kwh * e_grid_kwh
```

Positive `e_grid_kwh` is export and earns the local price; negative energy is
import and pays it. For example, the unclipped price is 0.225 CHF/kWh at
0.95 pu, 0.150 CHF/kWh at 1.00 pu, and 0.075 CHF/kWh at 1.05 pu.

After calculating the raw settlements, every connection receives the same
lump-sum balance so that the interval total equals fair LEG's interval total.
This makes the tariff exactly revenue-neutral on a fixed physical trajectory.
The twenty-test-week fixed-behaviour revenue difference was less than
0.00001 CHF.

This is intentionally the simplest voltage-level price. It prices exposure to
a local voltage condition, not the connection's marginal contribution to that
condition, and should be treated as a diagnostic design.

## Controller selection and evaluation

All 32 family policies and all 16 installed-base parameter combinations were
tuned on four weather weeks using root 11003. Selection maximised mean
settlement per inverter household. The winning family policy was **ID 31,
`instant_slow_stagger_1h`**: solar charging begins around 11:00 with a
persistent random household delay of up to one hour, uses half the available
charging rate, and adds a linear 10 kW/pu response to the latest observed local
voltage relative to 1.00 pu. The installed-base best response used no export
cap and delayed charging until 11:00. This supersedes the earlier 22-policy run,
which could not pair direct voltage response with the winning schedule.

The four cells below were evaluated on twenty independent weather weeks using
root 33013. They mirror the quickstart convention. Group prices are CHF per
actual behind-meter load kWh; negative values mean net earnings. Parentheses
show the difference from `fair_leg/base`, and positive differences are worse.

| Cell | Export peak kW | Import peak kW | Ramp kW | Curtailment | Community CHF/week | Tenant CHF/kWh | PV-only CHF/kWh | PV+battery CHF/kWh | Large-flex CHF/kWh |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fair_leg/base` | 65.30 | 7.90 | 15.28 | 1.50% | 294.28 | +0.223 (+0.000) | -0.263 (+0.000) | -0.421 (+0.000) | -0.148 (+0.000) |
| `fair_leg/selected_family` | 59.20 | 8.26 | 6.85 | 0.38% | 304.86 | +0.223 (+0.000) | -0.260 (+0.003) | -0.431 (-0.011) | -0.155 (-0.007) |
| `voltage_price/tuned_base` | 65.23 | 8.23 | 47.61 | 0.65% | 300.22 | +0.036 (-0.187) | -0.302 (-0.039) | -0.352 (+0.068) | -0.103 (+0.046) |
| `voltage_price/selected_family` | 59.20 | 8.26 | 6.85 | 0.38% | 304.86 | +0.036 (-0.188) | -0.299 (-0.036) | -0.358 (+0.063) | -0.105 (+0.043) |

## Interpretation

The selected family controller lowers export peak by 6.11 kW (9.3%), lowers
maximum ramp by 8.43 kW (55.1%), and reduces curtailment by 1.11 percentage
points relative to `fair_leg/base`. Community settlement rises by CHF 10.58
per week. Most of that physical improvement is the controller choice: the
`fair_leg/selected_family` and
`voltage_price/selected_family` physical metrics are identical because the
controller cannot see live settlement within an episode.

The voltage tariff changes which controller is privately attractive across
episodes and substantially changes incidence. Relative to `fair_leg/base`, the
combined voltage-price/family cell improves the tenant price by 0.188 CHF/kWh
and the PV-only price by 0.036 CHF/kWh, but worsens the PV+battery price by
0.063 CHF/kWh and the large-flex price by 0.043 CHF/kWh. The selected response
is therefore not a participation-safe winner for all household types.

The installed-base best response is especially unattractive operationally: it
leaves export peak almost unchanged and raises the maximum ramp from 15.28 to
47.61 kW. This illustrates the main limitation of a raw voltage-level price:
it can reward timing or location without accurately pricing causal congestion
relief.

## Reproduce

```bash
uv run python -m scripts.run_voltage_tariff_experiment \
  --output results/voltage_price_tariff_linear_voltage_20260910
```

The output includes `controller_tuning.csv`, the full
`four_cell_comparison.csv`, a rendered `four_cell_comparison.md`, `manifest.json`,
and source snapshots.
