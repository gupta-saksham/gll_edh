# Working on the tariff/controller framework

## Repository instructions

- Do not edit `README.md` further unless the user explicitly requests it.
  Record implementation and experiment guidance here instead. Preserve existing
  README changes; this instruction does not authorise reverting them.
- Preserve user changes to notebooks and other files. Use the dedicated
  experiment notebook rather than rewriting `notebooks/00_quickstart.ipynb`.
- Keep simulator physics, controller observations, feasibility projection, and
  official jury metrics unchanged when experimenting with policies or tariffs.
- Run commands below from the repository root. Use `uv sync` if dependencies
  are missing. With the existing environment, `.venv/bin/python` can replace
  `uv run python`.

## Implemented changes

| File | Responsibility |
|---|---|
| `sandbox/controller_family.py` | A bank of 22 complete local JAX policies, selected by `policy_id` |
| `sandbox/tariff_family.py` | Fair LEG plus directional transformer stress charges and equal rebates |
| `sandbox/my_idea.py` | Adapter exposing controller, tariff, carry, and tuning bank to `check()` / `score()` |
| `sandbox/experiments.py` | Paired policy/tariff experiments with independent weather splits and saved results |
| `sandbox/response_audit.py` | Unilateral policy deviations and battery/voltage diagnostics |
| `sandbox/marginal_audit.py` | One-interval feasible action perturbations with neighbours' requests held fixed |
| `scripts/run_framework_audits.py` | Runs response, storage, and near-optimal-policy audits for a saved experiment |
| `notebooks/01_tariff_controller_experiments.ipynb` | Loads saved results and displays comparison plots and incidence tables |
| `CONTROLLER_FRAMEWORK_PLAN.md` | Design rationale and proposed iteration process |
| `CONTROLLER_FRAMEWORK_RESULTS.md` | Completed first experiment, tradeoffs, and limitations |

The controller family supports scheduled charging, charging-rate control,
evening reserves, persistent household staggering, export caps, exchange
smoothing, optional grid charging/battery export, and bounded voltage feedback.
It returns **inverter active power in kW**, not net grid exchange.

Voltage feedback uses the previous interval's local voltage minus a six-hour
EWMA. The first observation initialises the EWMA to avoid a false initial
response. Current bank settings include a 0.005 pu deadband, gains of 25 or
75 kW/pu, and a 1 kW correction cap. Mode permissions and physical limits apply.
The carry has fixed shape/dtype; staggering is drawn once per household.

`family_controller()` defaults to policy 15 (voltage-enabled staggering).
The submission adapter currently defaults to **policy 17** (voltage feedback
plus an export cap), with `voltage_enabled=1`, a 30 kW export threshold, and
0.30 CHF/kWh maximum export charge. This reproduces a **failed diagnostic
candidate**, not an accepted tariff. `score()` may choose another policy.

## Run experiments

### Short wiring check

```bash
uv run python -m sandbox.experiments \
  --output results/controller_framework_smoke \
  --steps 96 --train-seeds 1 --validation-seeds 2 --test-seeds 2
```

This is one day per episode. Do not use it to claim weekly performance.

### Full comparison

```bash
uv run python -m sandbox.experiments --output results/controller_framework
```

Defaults: 672 fifteen-minute intervals per episode; four training weather weeks,
eight validation weeks, and twenty final-test weeks. Separate PRNG roots are
11003, 22007, and 33013. Every policy/tariff within a split sees paired weather.
The initial sweep compares fair LEG with six export-stress tariffs: thresholds
30/45 kW crossed with maximum rates 0.05/0.15/0.30 CHF/kWh, using a 20 kW
activation ramp. Import charges are disabled in this sweep.

Use a fresh output directory for each new experiment; reusing one overwrites
files. `results/` is ignored by Git. Preserve meaningful findings in the results
document and retain run manifests/source snapshots alongside data.

### Open results and run follow-up audits

```bash
uv run jupyter lab notebooks/01_tariff_controller_experiments.ipynb
uv run python -m scripts.run_framework_audits --results results/controller_framework
```

The notebook reads saved outputs; it does not rerun training. Its `RESULTS`
variable defaults to `results/controller_framework`; change that variable when
inspecting another run. The audit script reads the supplied directory's manifest
and selected policy. It runs unilateral deviations, storage diagnostics, and
near-optimal-policy comparisons. It does **not** run the marginal-action audit.

Example marginal audit, explicitly using the submission adapter's current tariff:

```python
import jax
import pandas as pd
from sandbox.controller_family import family_controller
from sandbox.marginal_audit import audit_marginals
from sandbox.my_idea import TARIFF_PARAMS
from sandbox.scenarios import reference_scenario
from sandbox.tariff_family import tariff_factory

rows = audit_marginals(
    family_controller(17), reference_scenario(), tariff_factory(TARIFF_PARAMS),
    n_steps=96, key=jax.random.PRNGKey(77041), sample_every=12, delta_kw=0.1,
)
pd.DataFrame(rows).to_csv(
    "results/controller_framework/marginal_settlements.csv", index=False
)
```

For a different saved run, obtain the tariff and policy from its `manifest.json`
and `selected.json`; do not assume they match `sandbox/my_idea.py`.

### Existing submission checks

```bash
uv run python -c 'from sandbox.my_idea import check; check()'
uv run python -c 'from sandbox.my_idea import check; check(fast=False)'
uv run python -c 'from sandbox.my_idea import score; score()'
```

`check()` holds parameters fixed and cannot demonstrate tariff-induced physical
changes. `score()` uses the unchanged official evaluation. It re-tunes submitted
tariff cells but does not tune the submitted controller under fair LEG. Its
default weather keys do not establish an independent train/test split. The
experiment runner supplies a separately tuned fair-LEG comparator and separate
weather roots.

## Change the experiment deliberately

- **Controller behaviour:** edit `POLICY_BANK` / policy logic in
  `sandbox/controller_family.py`. `TUNE_OVER` searches the whole declared bank.
  IDs are array indices; inspect names before using hard-coded IDs in scripts.
  Changing bank order requires reviewing audit-script and adapter IDs.
- **Matched voltage ablation:** use the same policy with
  `controller.replace(params={**controller.params, "voltage_enabled": 0.0})`.
  Keep schedule, tariff, and weather fixed. The runner already compares its
  best voltage-enabled candidate against this matched ablation.
- **Tariff formula/defaults:** edit `sandbox/tariff_family.py`.
- **Official submission tariff:** change `TARIFF_PARAMS` in `sandbox/my_idea.py`.
- **Experiment tariff sweep:** change `tariff_candidates` in
  `sandbox/experiments.py`. This sweep does **not** read the adapter's
  `TARIFF_PARAMS`; changing only `my_idea.py` does not change the sweep.
- **Acceptance rules:** change `criteria` in the runner before viewing finalist
  test results. Current limits are +2 percentage points curtailment, +2 kW
  import peak, and +0.01 CHF per actual load kWh for each household group versus
  tuned fair LEG. If no tariff passes, the runner tests the lowest-peak tariff
  as a diagnostic and writes `accepted: false`; do not call it a winner.

Freeze the same policy bank and search budget for every tariff in a comparison.
If a new capability is added, rerun the shortlist against the expanded bank.
Choose policies by household settlement, not network score; evaluate network
benefits and participation separately. Do not repeatedly tune on final-test
weather and continue calling it a holdout.

## Outputs and interpretation

- `manifest.json`, `source/`: settings, seed roots, policy bank, and code snapshot.
- `tuning.csv`, `selected.json`: household returns, winning IDs, near-optimal IDs.
- `selection.json`: validation choice and whether it passed acceptance criteria.
- `metrics.csv`, `summary.csv`, `paired_deltas.json`: per-weather metrics and
  summaries; compute peaks per episode before averaging.
- `household_settlements.csv`: all 18 households, actual load, and incidence.
- `*_household_trace.csv`, `*_feeder_trace.csv`: illustrative test traces.
- Audit outputs: `unilateral_deviations.csv`, `storage_diagnostics.csv`,
  `near_optimal_metrics.csv`, `audit_manifest.json`, and diagnostic NPZ files.
- `official_score.txt` and `marginal_settlements.csv` are separately generated;
  the main experiment command does not create them.

The controller sees no live price or neighbours. Reusing fixed-policy physical
trajectories across tariffs is valid here because settlement does not feed back
into observations. The current replay helper is specifically for the stateless
stress tariff and **fair-LEG input trajectories**. Do not apply it twice or use
it unchanged for a stateful tariff.

The tuner maximises mean settlement across 12 inverter agents with shared
parameters. Tenants are absent from that objective. This is a shared-policy
approximation, not proof of individual equilibrium; use unilateral audits.
Positive settlement means money received by a household.

Official `*_cost_per_kwh_chf` metrics divide by **grid imports**, despite their
consumption wording. Near-zero imports can create extreme values. Preserve the
official metrics, but use added `*_cost_per_load_kwh_chf` fields for experimental
incidence and selection. Actual load is reconstructed from realised inverter
energy minus net grid energy. Inspect raw household bills as well.

The first completed experiment rejected all six tariffs. The strong diagnostic
tariff reduced peak by 42.4% against tuned fair LEG, but curtailed 18.7% of solar
and reduced community settlement 37.1%. Eleven of twelve inverter households
could profit from tested unilateral deviations. A milder voltage-enabled tariff
looked better physically but still failed the participation screen. See the
results document for validation-versus-test distinctions. The older
`results/controller_framework_import_normalization` run is superseded.

Battery wear and terminal stored-energy value are not included in the current
household objective. Record endpoint energy and avoid attributing its value to
tariff quality. Revenue neutrality alone does not establish fairness.

## Verification

```bash
uv run pytest -q
uv run ruff check sandbox tests scripts/run_framework_audits.py
git diff --check
```

Use targeted tests during iteration, especially `test_controller_family.py`,
`test_tariff_family.py`, `test_experiments.py`, `test_response_audit.py`, and
`test_marginal_audit.py`. They cover voltage/mode bounds, carry persistence,
settlement replay, load reconstruction, unilateral deviations, and advancing
environment state in marginal counterfactuals.

If cache writes are restricted, set `MPLCONFIGDIR=/tmp/gll-mpl` and
`XDG_CACHE_HOME=/tmp/gll-cache` for a run. Executing a notebook needs a local
Jupyter kernel port; follow the environment's approval process if it is blocked.
