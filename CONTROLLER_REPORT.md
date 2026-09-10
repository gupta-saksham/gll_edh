# Controller report

This report describes the household controllers currently used by the tariff
and controller framework. It covers the two reference controllers in
`sandbox/controller.py`, the 32 complete policies in
`sandbox/controller_family.py`, and the way those policies are selected by the
experiment runner and submission adapter.

## 1. Controller contract

Every controller acts on one household for one 15-minute interval and returns
one scalar: requested inverter active power, `p_inv_kw`, in kW. Positive power
means that the inverter is producing. The household's metered grid exchange is

```text
p_grid_kw = p_inv_kw - p_load_kw
```

The controller therefore does not command the battery directly. Solar and the
battery share one inverter, solar is dispatched first, and the requested
inverter power implies whether the battery charges or discharges. The harness
clips the request to the inverter's interval-specific limits, and the simulator
then projects it onto the physically feasible set.

Each household receives only local information: the time, its own previous
voltage and grid exchange, current energy state, upcoming load and PV, battery
limits, and inverter limits. It cannot see the live tariff, its neighbours, or
a multi-hour forecast. Parameters are shared across households. Persistent
per-household differences, such as staggered start times, are stored in local
controller memory.

The six tenant connection points have no controllable inverter. All policies
still run for them, but their feasible request is zero.

## 2. Reference controllers

The reference controllers are separate from the 32-policy family. They provide
the installed-base and no-storage comparison cases.

| Controller | Implementation | Load signal | Battery use | Tunable characteristics | Main role |
|---|---|---|---|---|---|
| `passive` | `passive_controller()` | None | Never intentionally uses the battery | None in practice | Do-nothing economic anchor: sends all available PV through the inverter |
| `self_consumption` | `base_controller()` | Previous interval's `p_load_kw` | Greedily absorbs surplus and serves load shortfall | Export cap: 1000, 8, 4, or 2 kW; charging start: 00:00, 09:00, 11:00, or 13:00 | Installed-base reference and 16-candidate baseline tuning grid |

The reference `self_consumption` controller deliberately uses the previous
interval's measured load. The family controller with the same name (policy 1)
uses the upcoming load forecast and is therefore similar in intent but not an
exact duplicate. The reference controller also uses a hard charging-time gate;
the family uses a smooth transition.

## 3. Building blocks shared by the policy family

The policy family expresses each strategy as a complete row of parameters. A
single `policy_id` selects the row, so every tariff is evaluated against the
same response set.

### Solar charging and ordinary discharge

Unless a row says otherwise, the controller:

- begins solar charging immediately, at 100% of the available charging rate;
- uses only local PV surplus for charging;
- discharges to cover the upcoming local load shortfall;
- does not deliberately charge from the grid or export stored battery energy;
- has no meaningful local export cap (`1000 kW` is the disabled setting); and
- follows the newly calculated target fully (`exchange_mix = 1.0`).

A charging start time is implemented with a half-hour logistic transition, not
a hard switch. A stagger is a household-specific non-negative random offset,
drawn once from `[0, spread_h)` and kept for the entire episode.

### Evening reserve

A reserve holds a fraction of nominal battery capacity until 17:00. Before
17:00, discharge is reduced according to how much state of charge remains above
that reserve. At and after 17:00, the reserve is released. It is a controller
preference, not a separate simulator constraint.

### Export cap

The cap limits the individual household's requested grid export, not total
transformer export. The controller first asks the battery to absorb excess PV;
if storage and inverter headroom are exhausted, enforcing the cap can curtail
solar. Lower caps are therefore stronger but carry a larger curtailment risk.

### Voltage feedback

The original voltage-enabled policies compare the previous interval's local
voltage with a six-hour exponentially weighted moving average. The first sample
initializes the average and produces no trend correction. Subsequent trend
corrections use a `0.005 pu` deadband and are limited to `1 kW` in magnitude.

Policies 22–31 add a direct voltage-level term. It is linear relative to a
1.00 or 1.01 pu reference, has no deadband, acts on the first available sample,
and shares the same 1 kW action bound. “Instant” in the policy name means an
immediate response to the latest controller observation; simulator observations
remain unchanged, so that voltage is still from the preceding interval. A
blend policy adds the direct level and EWMA-trend terms before applying the
bound.

A rising voltage trend increases charging or reduces export; a falling trend
increases discharge toward local load. Voltage feedback remains inside the
selected policy's mode permissions: it cannot cause grid charging or battery
export unless that complete policy explicitly enables the mode. The global
`voltage_enabled` parameter can turn the feedback off without changing the
rest of a policy, which is used for matched ablations.

### Exchange smoothing and explicit grid modes

`exchange_mix = 0.5` moves halfway from the previous realized grid exchange to
the current target, reducing abrupt changes while respecting the scheduled
battery permissions. The `grid_charge` policy separately permits overnight
grid charging. The `battery_export` policy separately permits deliberate export
from storage during its evening window.

## 4. Complete controller-family catalogue

In the table, “none” under cap means the implemented `1000 kW` setting. Charge
times are the centers of the smooth transition. `U(0,n)` denotes the persistent
per-household start delay drawn uniformly from zero up to `n` hours. Reserves
apply until 17:00. Voltage entries distinguish EWMA-trend gain from direct-level
gain in kW/pu. Trend policies use the `0.005 pu` deadband; direct-level policies
are linear around their stated reference. All use the 1 kW correction limit.

| ID | Policy name | Solar charge rule | Reserve | Local export cap | Voltage | Smoothing / special mode | Character and intended use |
|---:|---|---|---:|---:|---:|---|---|
| 0 | `passive_pv` | Off | — | — | — | Battery inactive | Family-level no-battery anchor; requests available PV |
| 1 | `self_consumption` | Immediate, 100% | 0% | None | 0 | Full response | Forecast-based greedy self-consumption baseline |
| 2 | `delay_09` | 09:00, 100% | 0% | None | 0 | Full response | Leaves morning battery headroom for later PV |
| 3 | `delay_11` | 11:00, 100% | 0% | None | 0 | Full response | Shifts charging toward the midday/afternoon export period |
| 4 | `delay_13` | 13:00, 100% | 0% | None | 0 | Full response | Strongest pure charging delay; preserves headroom longest |
| 5 | `stagger_11_1h` | 11:00 + `U(0,1)`, 100% | 0% | None | 0 | Full response | Spreads charging starts across one hour to reduce herding |
| 6 | `stagger_11_2h` | 11:00 + `U(0,2)`, 100% | 0% | None | 0 | Full response | Wider two-hour desynchronization |
| 7 | `slow_stagger` | 11:00 + `U(0,2)`, 50% | 0% | None | 0 | Full response | Combines desynchronization with slower charging |
| 8 | `reserve_25` | 11:00, 100% | 25% | None | 0 | Full response | Retains a quarter of capacity for the evening |
| 9 | `reserve_50` | 11:00, 100% | 50% | None | 0 | Full response | Stronger evening protection with less daytime discharge freedom |
| 10 | `cap_8` | 11:00, 100% | 0% | 8 kW | 0 | Full response | Mild household export limiting |
| 11 | `cap_4` | 11:00, 100% | 0% | 4 kW | 0 | Full response | Strong export limiting; selected under the diagnostic tariff |
| 12 | `cap_2` | 11:00, 100% | 0% | 2 kW | 0 | Full response | Most aggressive standalone cap and highest curtailment risk |
| 13 | `voltage_25` | 11:00, 100% | 0% | None | Trend 25 | Full response | Moderate local voltage-trend correction |
| 14 | `voltage_75` | 11:00, 100% | 0% | None | Trend 75 | Full response | More sensitive voltage response; still bounded to 1 kW |
| 15 | `voltage_stagger` | 11:00 + `U(0,2)`, 100% | 0% | None | Trend 25 | Full response | Combines voltage feedback with persistent desynchronization |
| 16 | `voltage_reserve` | 11:00, 100% | 25% | None | Trend 25 | Full response | Couples voltage relief with an evening energy reserve |
| 17 | `voltage_cap` | 11:00, 100% | 0% | 4 kW | Trend 25 | Full response | Adds bounded voltage relief to the 4 kW export cap |
| 18 | `smooth` | 11:00 + `U(0,1)`, 100% | 0% | None | 0 | 50% exchange mix | Dampens changes while staggering start times |
| 19 | `grid_charge` | Immediate, 100% | 0% | None | 0 | Grid charge 01:00–05:00 at up to 50%, targeting 75% SoC | Tests explicit off-peak charging capability |
| 20 | `battery_export` | Immediate, 100% | 25% | None | 0 | Battery export 17:00–20:00 at 50% of spare discharge capability | Tests deliberate evening export from stored energy |
| 21 | `joint` | 11:00 + `U(0,2)`, 50% | 25% | 4 kW | Trend 25 | 50% exchange mix | Combined policy: slow/staggered charge, reserve, cap, voltage, and smoothing |
| 22 | `instant_5` | 11:00, 100% | 0% | None | Level 5 @ 1.00 pu | Full response | Low-gain linear voltage-level response without staggering |
| 23 | `instant_10` | 11:00, 100% | 0% | None | Level 10 @ 1.00 pu | Full response | Medium-gain linear voltage-level response |
| 24 | `instant_25` | 11:00, 100% | 0% | None | Level 25 @ 1.00 pu | Full response | High-gain linear voltage-level response, bounded to 1 kW |
| 25 | `instant_stagger_1h_5` | 11:00 + `U(0,1)`, 100% | 0% | None | Level 5 @ 1.00 pu | Full response | Matched low-gain voltage extension of policy 5 |
| 26 | `instant_stagger_1h_10` | 11:00 + `U(0,1)`, 100% | 0% | None | Level 10 @ 1.00 pu | Full response | Matched medium-gain voltage extension of policy 5 |
| 27 | `instant_stagger_1h_25` | 11:00 + `U(0,1)`, 100% | 0% | None | Level 25 @ 1.00 pu | Full response | Matched high-gain voltage extension of policy 5 |
| 28 | `instant_stagger_2h_10` | 11:00 + `U(0,2)`, 100% | 0% | None | Level 10 @ 1.00 pu | Full response | Matched direct-level voltage extension of policy 6 |
| 29 | `instant_stagger_1h_10_ref101` | 11:00 + `U(0,1)`, 100% | 0% | None | Level 10 @ 1.01 pu | Full response | Tests a higher neutral-voltage reference |
| 30 | `blend_stagger_1h` | 11:00 + `U(0,1)`, 100% | 0% | None | Trend 25 + level 10 @ 1.00 pu | Full response | Combines persistent level response with short-term trend correction |
| 31 | `instant_slow_stagger_1h` | 11:00 + `U(0,1)`, 50% | 0% | None | Level 10 @ 1.00 pu | Full response | Combines linear voltage response, staggering, and slower charging |

## 5. Which controllers are used where

| Context | Controller or selection | Meaning |
|---|---|---|
| `base_controller()` default | Reference `self_consumption`, immediate charging and no effective cap | Installed-base comparison before tuning |
| Reference baseline tuning | All 16 combinations of four start times and four export caps | Gives each tariff a separately tuned installed-base response |
| `family_controller()` default | Policy 15, `voltage_stagger` | General family default and a voltage-enabled starting point |
| Experiment family tuning | All IDs 0–31 | Household settlement chooses a shared-policy response for each tariff |
| Tuned fair LEG in the completed run | Policy 4, `delay_13` | Training winner; IDs 4–7 were within CHF 0.10 per agent-week |
| 45 kW / 0.15 CHF/kWh validation candidate | Policy 15, `voltage_stagger` | More promising physical result, but it failed the participation screen |
| 30 kW / 0.30 CHF/kWh diagnostic tariff | Policy 11, `cap_4` | Lowest-peak fallback after every tariff failed acceptance; not a winner |
| Best voltage-enabled match for that diagnostic | Policy 17, `voltage_cap` | Reduced peak slightly further but increased curtailment and reduced settlement |
| `sandbox/my_idea.py` current fixed default | Policy 17 with voltage enabled | Reproduces the failed diagnostic candidate before `score()` retunes the bank |
| Expanded voltage-price experiment | Policy 31, `instant_slow_stagger_1h` | Current settlement winner for the local-voltage tariff; combines linear level response, one-hour staggering, and half-rate charging |

The distinction between fixed defaults and tuned selections is important.
`check()` runs the adapter's fixed policy 17. `score()` may choose any family
policy through `TUNE_OVER`. The experiment runner also retunes each tariff, but
uses independent train, validation, and test weather splits and reports the
chosen policy explicitly.

### Credible-response frontier

For each tariff, the runner now retains every controller whose training
settlement is within CHF 0.10 per inverter household and episode of the best
shared-policy response. It evaluates this credible-response set on validation
weather and computes a Pareto frontier by minimising these outcomes:

- transformer export peak;
- transformer import peak;
- maximum transformer ramp; and
- curtailed solar share;
- tenant price change relative to tuned fair LEG;
- PV-only household price change relative to tuned fair LEG;
- PV-and-battery household price change relative to tuned fair LEG; and
- large-flex household price change relative to tuned fair LEG.

Prices are CHF per actual behind-the-meter load kWh. A positive difference
means that group pays more, or earns less, than under tuned fair LEG. Keeping
the four differences separate makes distributional tradeoffs visible instead
of hiding them in a single average or spread.

`credible_response_candidates.csv` compiles all qualifying tariff/controller
pairs, their full controller characteristics, tariff parameters, training
settlement gap, validation metrics, actual group prices, fair-LEG reference
prices, price differences, and an `is_pareto` flag.
`credible_response_frontier.csv` contains only the non-dominated rows. The
frontier is constructed without final-test weather so that the holdout remains
available for an independently selected design.

## 6. Interpretation and limitations

These are interpretable, bounded local heuristics, not forecasts of a unique
market equilibrium. Tuning maximizes average settlement across the 12 inverter
households with one shared policy; tenants are not part of that tuning
objective. A shared-policy winner can still invite profitable unilateral
deviations, as the completed audit found for policy 11.

The family intentionally does not change simulator physics, controller
observations, feasibility projection, or official metrics. It also does not
include battery degradation or terminal stored-energy value in the household
objective. Export caps may improve feeder peaks by curtailing energy, reserves
can shift rather than eliminate stress, and grid charging or battery export can
create new peaks if their timing is poorly chosen. Controller quality should
therefore be judged jointly on household settlement, all-household incidence,
curtailment, peak, ramp, coincidence, and endpoint battery energy.

## 7. Source of truth

- Controller interface and reference implementations: `sandbox/controller.py`
- Complete policy bank and shared execution logic: `sandbox/controller_family.py`
- Current submission default: `sandbox/my_idea.py`
- Tuning and split selection: `sandbox/experiments.py`
- Completed measurements and warnings: `CONTROLLER_FRAMEWORK_RESULTS.md`

This catalogue reflects the current bank order. IDs are array indices, so this
report must be reviewed if policies are inserted, removed, or reordered.
