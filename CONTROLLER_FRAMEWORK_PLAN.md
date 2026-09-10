# A reusable controller for tariff experiments

Use one fixed, interpretable controller family, and re-tune its parameters for
each tariff. Start with scheduled battery charging, an evening energy reserve,
an export cap, and persistent household staggering. Add local voltage feedback
as an optional module. This gives tariffs several plausible responses to
induce without requiring a new controller implementation for every experiment.

The objective is a **credible bounded response**, not an optimal controller or
a proven market equilibrium. This document is an implementation plan; the
proposed controller and experiments have not yet been implemented or measured.

## 1. What the existing sandbox permits

The design follows [README.md](README.md),
[TARIFF_COOKBOOK.md](TARIFF_COOKBOOK.md),
[CONTROLLER_COOKBOOK.md](CONTROLLER_COOKBOOK.md), and the reference experiment,
observation inspection, and exports in
[00_quickstart.ipynb](notebooks/00_quickstart.ipynb).

The implementation also matters in three places:

- [observation.py](sandbox/observation.py): a controller sees only its own
  state, clock, upcoming PV and load, and previous interval's voltage/exchange.
  There is no live tariff, neighbour state, or multi-hour forecast input.
- [tuning.py](sandbox/tuning.py): parameters are shared across households.
  Selection maximises mean episode settlement across the 12 inverter agents;
  the six tenants are excluded from this objective. Positive settlement means
  money received, so maximising settlement corresponds to reducing net cost.
- [evaluate.py](sandbox/evaluate.py): submitted-tariff cells are re-tuned, but
  fair-LEG cells use the controllers' supplied parameters. The revenue check
  holds behaviour fixed. These semantics must remain visible in comparisons.

Consequently, a tariff changes behaviour through **offline adaptation**:

```text
Choose tariff parameters φ
    → search controller parameters θ using household settlement
    → freeze θ and evaluate on unseen weather
    → inspect network performance and all households' bills
    → revise φ; repeat with the same controller family and search budget
```

The current search is a population-level shared-policy approximation. It does
not test a household changing its policy while neighbours keep theirs fixed.
Call its output a “tuned shared-policy response”, even where the existing
harness calls it a best response.

## 2. Keep the tariff modular

Write the interval settlement for household i as:

`settlement_i = fair_leg_i + reward_i − penalty_i + rebate_i`.

Separate four decisions for every tariff:

| Component | Decision to record |
|---|---|
| System stress | Export/import transformer power, apparent power, ramp, losses, or voltage-related constraint |
| Activation | Threshold, smooth transition, time window, and any averaging period |
| Attribution | Which controllable household action increases or relieves the stress? |
| Funding | Who pays for rewards; how charges are rebated; treatment of tenants |

Start with **directional transformer stress**. Household export is positive,
whereas transformer export is negative:

`export_stress = max(−grid.transformer_kw − export_threshold_kw, 0)`

`import_stress = max(grid.transformer_kw − import_threshold_kw, 0)`.

A simple first family charges exporting energy during export stress and
importing energy during import stress. For example, use bounded rates
`r_export = strength × clip(export_stress / stress_scale_kw, 0, 1)` and
the analogous import rate, then compute:

`penalty_i = r_export × max(e_i, 0) + r_import × max(−e_i, 0)`.

Rates are CHF/kWh, `e_i` is `grid.e_grid_kwh`, and penalties are CHF. Choose
thresholds from reference traces and an explicit engineering hypothesis;
an experimental threshold is not a verified transformer rating.

For the initial experiment set `reward_i = 0` and redistribute penalties with
fixed nonnegative weights summing to one:

`rebate_i = weight_i × sum_j(penalty_j)`.

This preserves the fair-LEG total on the same trajectory. Equal connection-point
weights are a transparent starting point, not a guarantee of fair incidence.
If explicit relief rewards are added, fund the net transfers consistently:
`rebate_i = weight_i × sum_j(penalty_j − reward_j)`. A negative rebate then
becomes a charge; report that clearly and reconsider tenant protection.

Test export and import stress separately before combining them. An import
charge can harm tenants who cannot respond. Any tenant floor must include its
funding source and be evaluated against fair LEG under changed behaviour too.

Later tariff modules can price ramp contribution, peak increments, or estimated
locational sensitivity. Raw voltage level is exposure, not household causation.
Apparent-power stress can be measured, but households control active power only;
do not assume they can independently reduce reactive power.

For each candidate, measure the **net marginal settlement** from small feasible
action changes in representative states, including the changed stress rate,
rebate, and fair-LEG matching. Headline rates alone do not establish an incentive.
Keep neighbours' requests fixed and re-solve physics for these diagnostics;
this requires an offline diagnostic helper, not extra controller observations.

## 3. Controller family: scheduled storage with local correction

Use the existing signature `controller(obs, carry, params, key)`. Internally
describe battery intent, then translate it to the inverter request. Define
`b_kw > 0` as desired charging and `b_kw < 0` as desired discharging:

`p_inv_request_kw = obs.pv_available_kw − b_kw`.

This is a planning coordinate, not a new simulator action. The simulator's
PV-first dispatch, power limits, and feasibility projection decide the actual
battery flow and curtailment. The final return is always inverter kW.

### A. Timing and energy reservation

Use a smooth charging gate around a tunable start hour. Draw one household
offset from `key` on the first interval and retain it in the carry; do not
reroll the start time every interval. A 30–60 minute smooth transition avoids
an artificial fleet-wide switching edge.

Before that gate, allow little or no PV charging, leaving room for the export
peak. Afterwards, absorb available surplus up to a tunable fraction of the
available charging limit. Without a charge request, available PV is exported,
subject to the equipment limits.

Preserve a tunable fraction of stored energy before an evening release hour.
After release, use it to cover load. Estimate usable energy scale locally as
`soc_kwh + soc_headroom_kwh`; guard the zero-capacity case. Convert reserve
energy to a discharge-power bound using the configured interval duration and
verified efficiency convention. This reserve is a soft policy preference;
the simulator remains the physical authority.

### B. Optional local correction

Add a bounded correction to charging intent from the previous local voltage:

`voltage_trend = obs.voltage_pu − carry.voltage_ewma_pu`

`b_corrected = b_scheduled + voltage_gain × deadband(voltage_trend)`.

A positive trend increases absorption or reduces discharge. Use a deadband and
saturation, and initialise the EWMA from the first observed voltage so that
the initial value does not create a spurious response. This signal is delayed;
it cannot react to the interval currently being settled.

Start with the trend. If using a fitted voltage residual later, fit only on
training episodes and align voltage with the previous interval's PV/load/time,
retaining those features in the carry. Upcoming PV paired with past voltage
is not a correctly aligned residual. Revalidate the fit after behaviour changes.

### C. Explicit permissions for additional behaviours

Allow two optional scheduled capabilities, disabled initially:

- Grid charging toward a target state of charge in a declared time window.
- Battery export above household load in a declared discharge window.

These are necessary for broader time-of-use or import-relief tariffs. Without
them, a failed tariff may simply be asking for behaviour the policy cannot
express. Enable them through the same declared candidate space for every tariff
in a comparison round. They must respect local reserve and equipment bounds.

### D. Smoothing, export cap, and final feasibility

Use this explicit action order:

1. Compute scheduled battery intent from surplus, reserve, and allowed modes.
2. Add bounded voltage correction; apply policy charging/discharging limits.
3. Translate to inverter power, then desired grid exchange using the upcoming
   load forecast.
4. Optionally smooth desired grid exchange toward `obs.p_grid_kw`, the last
   realised exchange. A tunable mixing fraction of 1 disables smoothing.
5. Apply the chosen export cap, convert back by adding the upcoming load,
   and call `clip_to_feasible`.
6. Update memory. On the next interval use measured exchange to see what
   actually happened, rather than assuming the requested setpoint was realised.

The export cap can force extra charging and eventually curtailment; it therefore
overrides the soft schedule. Keep it as an explicit tunable choice so the
experiment can reveal that spilling solar is the privately attractive response.
Log cap activation and curtailed energy. Smoothing is also a preference, not a
promise that physical limits permit a particular ramp.

No-inverter households return zero. PV-only households can export or curtail
but cannot shift energy. No division by battery capacity is allowed without a
zero-capacity guard. Do not assume `large_flex` means the household's exogenous
load itself can be scheduled: the exposed action remains inverter active power.

### Initial knobs

These values are proposed search brackets, not measured optima.

| Parameter | Initial candidates / treatment | Purpose |
|---|---|---|
| Charge start | 0, 9, 11, 13 h | Leave capacity for midday |
| Start spread | 0, 1, 2 h | Persistent household differentiation |
| Charge fraction | 0.5, 1.0 of available limit | Spread absorption through time |
| Pre-evening reserve | 0, 0.25, 0.5 of usable capacity | Retain evening flexibility |
| Evening release | Initially fixed at 17 h; later 16, 18, 20 h | Control discharge timing |
| Export cap | Disabled, 8, 4, 2 kW | Reveal curtailment incentive |
| Voltage gain | 0, 25, 75 kW/pu | Optional local feedback |
| Voltage deadband | Initially fixed at 0.005 pu | Ignore small variations; test sensitivity |
| Exchange mixing fraction | 1.0, 0.5 | Optional smoothing |
| Grid charge / battery export | Initially disabled | Extend coverage when required |

Voltage changes of 0.01 pu produce 0.25 or 0.75 kW before deadband and clipping
at the proposed gains. Inspect observed correction magnitudes before widening
the search. Use fixed-shape, fixed-dtype JAX carry fields for offset, EWMA,
interval count, and any optional lagged features.

## 4. Search in small stages, then freeze the comparison space

Do not take the Cartesian product of every knob above. The existing `TUNE_OVER`
interface evaluates every combination, and cost grows quickly.

First establish a compact policy bank with at most roughly 32–64 complete
parameter sets. Include passive PV pass-through, exact installed-base behaviour
as a separate benchmark, forecast-based self-consumption, delayed charging,
staggered charging, reserve behaviour, and cap-based responses. Include all-off
module settings so the search can reject unhelpful complexity.

During development, search timing and cap first, retain several good candidates,
then vary reserve/charge fraction and finally staggering/feedback around them.
Retaining several candidates reduces premature commitment, but this staged
search is still approximate. Add a few jointly varied candidates to expose
interactions that one-block-at-a-time tuning misses.

For a formal tariff comparison, freeze the resulting policy bank and evaluate
the **same bank, training weather, and budget under every tariff**, including
fair LEG. If a later experiment adds a capability, rerun all shortlisted tariffs.
Do not give only a preferred tariff a specially expanded controller space.

For compatibility, a `policy_id` parameter can select a row from a fixed JAX
parameter table, with `TUNE_OVER = {"policy_id": [...]}`. Cast the selected ID
to an integer because the existing tuner stacks parameters as float32. The
table contains complete shared policies, never per-household parameter arrays.
Alternatively add a separate explicit-candidate tuner for exploration; do not
silently change the official scoring machinery.

Rank policies by the existing settlement objective. Do not put transformer
peak or fairness into the household objective: those belong to tariff selection.
As an additional labelled sensitivity analysis, include battery wear cost and
terminal-energy treatment; these are not currently part of the tuner objective.
Check final stored energy and use longer episodes to identify policies whose
apparent gain comes from depleting initial storage or exploiting the endpoint.

## 5. Evaluation that separates tariff quality from controller quality

Preserve the official four-cell report. Add an experimental comparison with:

| Run | What it answers |
|---|---|
| Fair LEG + default installed base | Existing reference |
| New tariff + the same fixed behaviour | Redistribution and revenue check |
| New tariff + tuned installed base | What current firmware can express |
| Fair LEG + tuned reusable family | What the richer controller achieves without the new tariff |
| New tariff + tuned reusable family | Incremental tariff effect with comparable adaptation |

The last two are the most useful tariff comparison. The official fair-LEG
submitted cell does not automatically provide the fourth run because it skips
tuning. Run `tune(..., tariff=None)` explicitly for that experimental benchmark.

Use separate deterministic training, validation, and final-test PRNG roots.
The default tuner and scorer both start from key 0, so do not assume default
calls establish an independent holdout. A practical initial budget is four
training weeks, eight validation weeks, and twenty final-test weeks. Share
weather keys across candidates within each split for paired comparisons.
Use validation to choose tariff parameters and test only frozen finalists.

Record metrics per weather episode before summarising; do not average power
trajectories before computing peaks. Show mean, dispersion, and paired changes.

Track:

- Transformer export/import peaks, coincidence, peak-to-average, ramp, losses,
  voltage exceedances, and battery synchrony.
- Curtailment, self-consumption, community settlement, household CHF and
  CHF/kWh consumed, tenant import share, and worst household/group changes.
- Charging onset, battery fullness during peak export, reserve usage, cap and
  projection activation, realised versus requested exchange, and final energy.
- Chosen policy, return gaps between candidates, and sensitivity to nearby
  parameters. Inspect all policies within a small declared CHF tolerance of
  the best settlement: near-equal private outcomes can have very different
  network consequences. Do not select a favourable tie by grid score and call
  it the household's response.

Use the existing all-18-household dataframe for incidence; `reward_chf` alone
cannot reveal tenant harm. Some controller diagnostics need additional logging
or local reconstruction; do not assume all are already trajectory fields.

Before claiming broad adoption, add a separate offline unilateral-deviation
audit: hold neighbours' policies fixed and try alternative policies for one
household, measuring its own settlement. This needs an experimental rollout
wrapper because the standard harness broadcasts one policy. Report profitable
deviations rather than labelling the shared-policy result an equilibrium.
Mixed adoption and repeated response rounds are later robustness experiments.

## 6. A practical experiment sequence

1. **Establish baselines.** Reproduce the quickstart and official score; store
   reference traces and household incidence. Inspect the existing default
   tariff's curtailment failure before changing its strength.
2. **Build the minimal controller.** Implement timing, charge fraction, reserve,
   cap, and persistent staggering. Verify tenants, PV-only, empty/full storage,
   midnight, and eager/compiled agreement on short deterministic rollouts.
3. **Test one system parameter.** Try export stress thresholds and a small
   strength ladder, always including zero. Tune the frozen controller bank for
   each tariff. Find where behaviour moves from timing changes to curtailment.
4. **Inspect incentives and incidence.** Check marginal settlement, policy
   return gaps, and who funded the improvement. Revise attribution/rebates
   separately from strength so the effect of each change remains interpretable.
5. **Expand only when diagnosed.** If batteries still fill too early, expand
   timing/charge-rate coverage. If tariff stress has locally predictable
   variation, test voltage feedback. If a tariff rewards a missing behaviour,
   add the relevant mode and rerun the shortlist.
6. **Freeze and validate.** Compare on unseen weeks; audit near-optimal policies,
   final energy, and individual deviations for the finalists. Run the official
   score as an additional submission-compatible report.

Choose tariff finalists on a Pareto comparison rather than hiding tradeoffs in
one weighted score. Predeclare acceptable limits for curtailment, tenant/group
harm, and import-peak deterioration. These are experiment acceptance criteria,
separate from the official revenue gate; set their numerical values from the
baseline and the intended tariff proposition before viewing finalist results.

## 7. Proposed implementation deliverables

| Proposed file | Responsibility |
|---|---|
| `sandbox/controller_family.py` | JAX policy, carry, parameter bank, diagnostic calculations |
| `sandbox/tariff_family.py` | Fair-LEG adjustment modules and explicit funding rules |
| `sandbox/experiments.py` | Parameter sweeps, separate seed roots, tuned fair-LEG benchmark, persisted results |
| `notebooks/01_tariff_controller_experiments.ipynb` | Tariff-response tables, paired plots, household incidence, and diagnostic traces |
| `sandbox/my_idea.py` | Thin adapter exposing the chosen controller, tariff, and declared search space |

Keep physics, observations, and jury metrics intact. Stateful tariffs must be
wired with `tariff_from_settlement(..., init_carry=...)`; the current
`my_tariff_factory()` does not automatically discover a custom tariff carry.

Save tariff parameters, controller-family version, candidate bank, selected
policy, seed roots, episode length, repository revision, per-seed metrics, and
household settlements for every experiment. The working question then becomes
concrete: **did this tariff make a useful behaviour privately attractive, and
does that result survive other weather and other nearly equivalent responses?**

## 8. A tariff scenario bank, built the same way

Sections 2 and 4 leave the two seams asymmetric: the controller side gets a
bank of 22 complete policies behind one `policy_id`, and the tariff side gets
one mechanism with a strength knob. A comparison between six strengths of one
mechanism cannot distinguish "this price is the wrong size" from "this price
is charged on the wrong quantity". So give the tariff the same shape: a
`TARIFF_BANK` of complete named scenarios behind one `scenario_id`, one
stacked parameter table, one compiled `family_tariff`.

Populate it along the four decisions section 2 already separates — system
stress, activation, attribution, funding — and require every entry to answer
two questions in one line each: **which real network cost does this
internalise**, and **what signal could a household anticipate it from**. An
entry that cannot answer both is not a tariff, it is a transfer. Include
deliberate ablations, exactly as the policy bank includes `passive_pv` and
the zero-voltage-gain entries: a pass-through control, the exposure-pricing
anti-pattern, an active-power ablation of an apparent-power charge, a
marginal-signal-removed two-part tariff, and one entry that is deliberately
not revenue adequate so the gate can be seen working.

### Three constraints the tariff side does not share with the controller side

**One union carry.** The scenario is selected by a traced id, so a
per-scenario carry type is not available at trace time, and the harness has to
declare the carry's shape before the episode runs. One fixed-shape carry must
therefore cover every scenario's state — running peaks, exponential averages,
regression accumulators — and every field is updated every interval whatever
is selected. A field's meaning then depends on the scenario, which is the
price of the single shape and worth stating in the module rather than
discovering later.

**Replay must thread that carry.** Physical trajectories are reused across
tariffs, and re-settling one is only equivalent to settling it live if the
carry is threaded sequentially over the interval axis. Mapping the tariff over
intervals with a fresh carry each time is correct for a stateless mechanism
and silently wrong for every stateful one, in a way no shape check catches.
Pin it with a test that compares replay against live settlement for a
stateful scenario, and against the per-interval-reset version to show the two
differ.

**Budget neutrality by construction.** Pool every charge and redistribute it
inside the same interval, so the interval total is fair LEG's total exactly
and revenue adequacy holds structurally. Then screen it empirically anyway,
with behaviour held fixed — re-settling one fixed trajectory is exactly the
comparison `revenue_adequate` makes, and it needs no extra rollouts.

### Size the price against marginal exposure, not the headline

Measure, for every entry, on a fixed trajectory: the pooled transfer per
week, each group's cost per kWh of its own load, and the change in one
connection point's own settlement per extra kWh it exports — separately for a
sustained increase and for a one-interval spike, because a ratchet's cost of
a spike is intertemporal and averages away in the first probe. Two results
from doing that here are worth recording as plan-level cautions:

- A charge on own directional energy passes almost its whole headline rate to
  the margin (the equal rebate returns only 1/18 of it), so the shipped
  0.15 CHF/kWh stress term is already competing with the 0.14 CHF/kWh feed-in
  rate. A charge allocated pro rata over an aggregate excess does not.
- The predeclared incidence screen of 0.01 CHF/kWh per group is a far tighter
  constraint than the revenue gate. It admits only mechanisms whose weekly
  pool is roughly under 40 CHF, which rules out most redistribution-heavy
  designs and favours ratchets, where the pool is small and the marginal rate
  at the moment of a new record is large. Where the pool has to be larger,
  return it inside the class that paid it rather than equally over all
  eighteen points: an equal rebate hands the six connection points that paid
  nothing a windfall, and that windfall, not the charge, is what usually
  breaks the incidence screen.

### What the seam cannot supply

A locational price should charge the sensitivity of the binding quantity to a
point's own injection, not the voltage level. Estimating that sensitivity from
the published fields — own voltage against own energy, controlling for the
substation flow, accumulated over an episode — does not work on this feeder:
the physical sensitivity spans 0.0003 to 0.0228 pu/kWh monotone in electrical
distance, and the estimate correlates with distance rank at about -0.19 and
turns negative at two connection points. Keep the scenario as a measured
negative result, and record the missing field: a per-point voltage
sensitivity, per-branch flows, or the power-flow Jacobian. The same gap
applies to funding a rate-class floor in proportion to consumption, which
would need behind-meter load rather than net exchange.
