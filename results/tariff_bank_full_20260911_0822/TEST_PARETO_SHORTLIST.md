# Exploratory test-weather Pareto shortlist

This analysis uses the 29 already-tested `tuned_family` tariff/controller pairs. It does not evaluate the other credible validation policies on test weather. The Pareto mask minimizes transformer export peak, transformer draw peak, maximum ramp, curtailment, and the four household-group price differences from `fair_leg/tuned_family`.

This is a post-hoc diagnostic. It does not replace the validation frontier, revise controller selection, or change `selection.json`.

## Result

Twenty of the 29 selected pairs lie on the exploratory test frontier. No revenue-adequate pair both improves system outcomes and lowers the load-normalized price for all four household groups.

The only pair that lowers all four group prices is `subsidy_ablation/delay_13`. It is excluded because the revenue screen fails and its physical trajectory is identical to tuned fair LEG, so it does not improve the system.

## Shortlist

| Rank | Tariff | Controller | System result vs tuned fair LEG | Customer-price result vs tuned fair LEG | Status |
|---:|---|---|---|---|---|
| 1 | `kva_peak_ratchet` | 31: `instant_slow_stagger_1h` | Export peak −9.97%, import peak −6.60%, ramp −30.10%; curtailment +0.096 percentage points | Tenant −0.0121 CHF/kWh; PV-only +0.0010; PV+battery +0.0058; large-flex +0.0022 | Revenue adequate, passed validation screens, declared finalist |
| 2 | `loss_share_quadratic` | 31: `instant_slow_stagger_1h` | Same physical improvements as rank 1; curtailment +0.096 percentage points | Tenant −0.0326 CHF/kWh; PV-only effectively flat at +0.0001; PV+battery +0.0141; large-flex +0.0068 | Revenue adequate; failed the declared validation screens |
| 3 | `two_part_fixed` | 7: `slow_stagger` | Export peak −2.77% and ramp −36.84%; import peak +14.56%; curtailment effectively unchanged | PV-only −0.0136 CHF/kWh; PV+battery −0.0159; large-flex −0.0087; tenant +0.0487 | Revenue adequate; failed the declared validation screens |

## Interpretation

`kva_peak_ratchet` is the strongest balanced candidate because it materially improves three system metrics and keeps the largest household-group price increase below 0.006 CHF/kWh. It is also the only shortlisted pair that passed the predeclared validation screens.

`loss_share_quadratic` produces the same physical outcome and gives tenants more relief, but shifts more cost to flexible owners. It is useful as a fairness trade-off, not a replacement for the accepted finalist.

`two_part_fixed` is the strongest owner-price candidate: three inverter-owner groups pay less and ramp falls substantially. The tenant price increase and higher import peak make it unsuitable without redesigning the allocation rule.

Other frontier points were not shortlisted because they combine larger group-price increases, materially higher curtailment, higher ramps, or failure of the revenue screen. In particular, the strongest export-stress cases reduce peaks by curtailing a large share of solar, and `voltage_level_exposure` raises the large-flex price by about 0.048 CHF/kWh.

## Files

- `cross_tariff_test_pareto.csv` contains all 20 exploratory test-front pairs.
- `test_pareto_shortlist.csv` contains the three shortlisted pairs with the same metrics and flags.

