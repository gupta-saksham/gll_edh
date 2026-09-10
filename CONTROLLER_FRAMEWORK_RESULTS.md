# Controller framework: implementation and measured results

The framework is implemented and the run is complete. **None of the six tested
tariffs passes the declared acceptance screens.** The experiment successfully
exposes the central failure mode: a large stress charge makes export caps and
solar curtailment privately attractive, while a shared-policy optimum is not
stable against individual household deviations.

This is a useful experimental framework, not a recommended production tariff.

## What was implemented

- A fixed bank of 22 local JAX policies with scheduled charging, charging-rate
  control, evening reserves, persistent staggering, export caps, smoothing,
  optional grid charging/battery export, and voltage-trend feedback.
- Voltage response uses last interval's local voltage minus a six-hour EWMA,
  a 0.005 pu deadband, gains of 25 or 75 kW/pu, and a 1 kW correction limit.
  The first observation initialises the EWMA; mode and feasibility bounds apply.
- Fair LEG plus bounded directional transformer stress charges, equal rebates,
  and optional tenant penalty exemption. The initial sweep prices export only.
- A reproducible runner with fair-LEG settlement replay verified against live
  tariff rollouts, source/configuration snapshots, all-household incidence,
  explicit tuned fair-LEG comparison, and matched voltage-off runs.
- Independent unilateral-response, battery-state, and one-interval marginal
  settlement diagnostics, plus an executed results notebook.

The Sol subagents implemented the controller, tariff, and diagnostic modules.
The parent integrated, ran, reviewed, and corrected the final experiment.

## Experiment design

Every tariff searches the same 22-policy bank and the installed base's existing
16 parameter combinations. Training uses four full weather weeks (root 11003),
validation eight separate weeks (22007), and final testing twenty further weeks
(33013). Each week has 672 fifteen-minute intervals. All test power-flow solves
converged. Candidate comparisons share weather keys within each split.

The six tariffs cross export thresholds 30/45 kW with maximum rates
0.05/0.15/0.30 CHF/kWh. The rate rises linearly from zero at the threshold to its
maximum 20 kW above it. Each exporting connection pays on its own export energy;
all 18 connections receive an equal share of the collected penalties.

The declared screens limit additional curtailment to 2 percentage points,
additional import peak to 2 kW, and cost increases for each household group to
0.01 CHF per actual load kWh, relative to the tuned fair-LEG controller.
**No candidate passed all three on validation.** Under the runner's declared
fallback rule, the lowest-export-peak candidate was tested as a diagnostic:
30 kW threshold, 0.30 CHF/kWh maximum rate. It must not be called an accepted winner.

## Twenty-week independent test results

| Controller / settlement | Export peak kW | Ramp kW | Curtailment | Community CHF/week |
|---|---:|---:|---:|---:|
| Installed base / fair LEG | 65.30 | 15.28 | 1.50% | 294.28 |
| Tuned family / fair LEG | 65.76 | 9.81 | 0.29% | 305.19 |
| Tuned installed base / diagnostic tariff | 39.04 | 24.61 | 19.11% | 187.04 |
| Tuned family / diagnostic tariff | 37.88 | 7.14 | 18.68% | 191.96 |
| Voltage-enabled family / diagnostic tariff | 37.30 | 6.85 | 19.23% | 188.68 |

Positive community settlement is money received by households, not operator
profit. The diagnostic tariff conserves the fair-LEG total at fixed behaviour;
behavioural changes still reduce the community's energy earnings.

Against tuned fair LEG, the diagnostic tariff's tuned family reduces mean export
peak by **42.4%**, but spills **18.68%** of available solar and reduces community
settlement by **CHF 113.23/week (37.1%)**. It also raises coincidence from about
0.781 to 0.861. Lower peak is being bought with discarded energy and private loss.

The selected family policy is ID 11: charging from roughly 11:00 with a 4 kW
export cap and voltage gain zero. The best voltage-enabled policy is ID 17,
which adds 25 kW/pu feedback to the same schedule/cap. Voltage feedback lowers
peak by a further **0.58 kW**, but increases curtailment by **0.56 percentage
points** and reduces settlement by **CHF 3.28/week**. Turning feedback off in
policy 17 exactly reproduces policy 11's results. Thus voltage feedback works,
but does not rescue this excessively strong tariff.

## The more promising voltage experiment

On the eight validation weeks, the **45 kW / 0.15 CHF/kWh** tariff selected
policy 15, voltage-enabled staggering, on household settlement itself. It gave:

- Export peak **59.06 kW**, versus **63.93 kW** for tuned fair LEG: **7.6% lower**.
- Curtailment **0.49%**, versus **0.23%**.
- Community settlement **CHF 296.73/week**, versus **CHF 297.90/week**.
- Ramp **12.65 kW**, versus **10.19 kW**.

It still failed the fairness screen: battery-household cost rose from
−0.4547 to −0.4246 CHF per actual load kWh, a loss of about **3.01 Rp./kWh**.
These are validation results, not an independently confirmed recommendation.
A reasonable next experiment is weaker charges and a different rebate allocation
that preserves participation incentives for the households providing flexibility.
Do not simply force a voltage policy if households earn more using another one.

## Household incidence and the denominator correction

Costs below are CHF per **actual behind-meter load kWh**; more negative means
more net earnings relative to load.

| Group | Fair LEG, tuned family | Diagnostic tariff, tuned family |
|---|---:|---:|
| tenant | 0.2235 | 0.1518 |
| pv_only | -0.2597 | -0.1756 |
| pv_battery | -0.4323 | -0.2838 |
| large_flex | -0.1551 | -0.0933 |

Tenants gain, while every owner group loses. Budget neutrality does not imply
participation fairness.

During review, the official metric labelled “CHF/kWh consumed” was found to
divide by **grid imports**, which become nearly zero for some battery policies.
It can therefore produce values in the millions. Official jury metrics were
left unchanged. The experimental incidence metric reconstructs load from
`realised inverter energy − net grid energy`, includes all households, and is
tested for invariance to battery dispatch. Selection was rerun using that
corrected denominator. The earlier output is retained separately under
`results/controller_framework_import_normalization` and is superseded.

## Response and storage audits

Holding neighbours on policy 11, each inverter household tried policies 1, 4,
and 15 on two additional paired weather weeks (root 44021). **11 of 12** could
improve their own mean weekly settlement by more than CHF 0.01. The largest
measured gain was **CHF 10.18/week**. This is strong evidence that the shared-policy
response is not an individual equilibrium. The audit tests three alternatives,
not every possible controller.

Four additional diagnostic weeks (55001) showed total initial stored energy
averaging 79.02 kWh; final energy was 64.79 kWh for tuned fair LEG, 83.56 kWh for
the diagnostic tariff policy, and 88.18 kWh for its voltage variant. Endpoint
energy therefore differs materially. The reported settlement objective includes
neither battery wear nor terminal-energy value; these remain limitations.

Only policy 11 was within CHF 0.10 per agent-week of the diagnostic tariff's
training winner, so the near-optimal audit had one qualifying policy. A separate
one-day audit records 192 feasible positive/negative 0.1 kW action perturbations,
with neighbours' requests fixed and physics re-solved; its marginal settlements
include the whole tariff, rebates, and fair-LEG matching.

The unchanged official four-cell score was also run and passed revenue adequacy.
Its seed conventions and untuned fair-LEG submitted cell differ from this
experiment; use the twenty-week table above for the paired tariff comparison.

## Verification

All 60 collected tests passed across the full-suite run and subsequent targeted
runs for the added or revised diagnostics. Ruff checks passed. The results
notebook executed successfully. Settlement replay, actual-load reconstruction,
voltage direction/ablation, asset bounds, unilateral deviations, and advancing
state in marginal counterfactuals have dedicated checks.

## Reproduce and inspect

```sh
uv run python -m sandbox.experiments --output results/controller_framework
uv run python -m scripts.run_framework_audits
uv run jupyter lab notebooks/01_tariff_controller_experiments.ipynb
```

The current `sandbox/my_idea.py` adapter exposes the **diagnostic** 30 kW /
0.30 CHF/kWh tariff, with voltage-enabled policy 17 as its untuned default.
`score()` is free to select a different policy from the complete bank; here
it selects the cap-based policy without voltage. This default is a reproducible
failed candidate, not a recommended tariff.

Results include `metrics.csv`, `household_settlements.csv`, `tuning.csv`,
`paired_deltas.json`, `selection.json`, `official_score.txt`,
`unilateral_deviations.csv`, `storage_diagnostics.csv`, and
`marginal_settlements.csv`. The notebook contains test charts and group tables.

## Expanded direct-voltage policy rerun

The controller bank was subsequently expanded from 22 to 32 policies without
changing IDs 0–21. Policies 22–31 add bounded linear response to the latest
observed local voltage level, matched one- and two-hour stagger variants, a
1.01 pu reference variant, a level/trend blend, and a slow-charge combination.
The complete comparison was rerun with the original train, validation, and test
roots under `results/controller_framework_linear_voltage_20260910`.

The new policies became household-settlement winners for three milder stress
tariffs: policy 23 (`instant_10`) for 30 kW / 0.15 CHF/kWh and policy 31
(`instant_slow_stagger_1h`) for both 45 kW / 0.05 and 0.15 CHF/kWh. Policy 31
was also the best voltage-enabled response under fair LEG. The strong 30 kW /
0.30 CHF/kWh diagnostic still selected policy 11 (`cap_4`), produced the same
failed test outcome reported above, and remained the fallback because no tariff
passed all validation screens. The expanded response space therefore improves
the milder candidates but does not rescue the rejected tariff design.

The fairness-aware credible-response table contains 31 tariff/controller rows.
All are non-dominated across the four network and four group-price objectives;
use the individual columns rather than interpreting membership as acceptance.
The expanded unilateral audit tested policies 1, 4, 15, 23, and 31 against the
diagnostic baseline. All twelve inverter households had at least one profitable
tested deviation, with a maximum measured gain of CHF 10.48 per week, so the
stability conclusion is stronger than in the original shortlist audit.
