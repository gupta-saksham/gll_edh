Execute this overnight experiment now. Do not ask questions. Do not wait. Implement the code change, run the full comparison, then write the comparison artifacts.

# Goal

Test **every** tariff in the family bank in full — not only the runner’s finalist — using the existing controller-family tuning and scoring pipeline. Collect all results, compare them, and write Pareto fronts.

# Constraints

- Do not edit simulator physics, controller observations, feasibility projection, or official jury metrics.
- Do not edit `README.md`. Do not rewrite `notebooks/00_quickstart.ipynb`.
- Do not call official `score()` / `evaluate()` in a per-tariff loop, and do not patch `sandbox/my_idea.py` to do that.
- Do not use `--steps 96` or any one-day smoke length. Use the runner defaults: 672 steps, 4 train / 8 validation / 20 test weeks, seed roots 11003 / 22007 / 33013.
- Use one Python process for JAX rollouts. Do not spawn one process per tariff.
- Choose policies by household settlement, not network score. Do not use final-test weather to select controllers or to build/revise either Pareto set.
- Use a fresh output directory. Never reuse or overwrite an older results folder.
- From the repository root. Use `.venv/bin/python` (or `uv sync` then `uv run python` if the venv is missing).
- If cache writes fail:
  ```bash
  export MPLCONFIGDIR=/tmp/gll-mpl
  export XDG_CACHE_HOME=/tmp/gll-cache
  ```

# 1. Make every tariff get a full test

Edit `sandbox/experiments.py` so the **test-week** evaluation currently applied only to the finalist is applied to **every** entry in `tariff_candidates`, including `fair_leg`.

Keep the existing train/validation/revenue/Pareto-on-validation path. Change the test phase:

- Evaluate these cells for **each** tariff `T`, with paired weather:
  - `fair_leg/default_base`
  - `T/fixed_base` (installed-base defaults, settled under T)
  - `T/tuned_base` (that tariff’s selected base params)
  - `fair_leg/tuned_family` (fair LEG’s selected family, fair LEG settlement)
  - `T/tuned_family` (that tariff’s selected family)
  - `T/voltage_family` (that tariff’s best voltage-enabled family policy)
  - `T/voltage_off_matched` (same policy with `voltage_enabled=0`)
- Cache physical trajectories. Do not re-roll the same controller params / split / weather.
- `fair_leg/default_base` and `fair_leg/tuned_family` must be computed once and reused, not once per tariff.
- Write household and feeder traces for every cell, with tariff-safe filenames (include the tariff name).
- Write per-tariff paired deltas of `T/tuned_family` minus `fair_leg/tuned_family` on test weather (extend `paired_deltas.json` or write `paired_deltas_by_tariff.json`).
- Keep `selection.json` as the declared-screen finalist. Full testing of every tariff does not make a failing tariff a winner.
- Add or extend tests in `tests/test_experiments.py` only if needed to lock the “every tariff is tested” contract. Run `uv run pytest -q tests/test_experiments.py tests/test_controller_family.py tests/test_tariff_family.py` before the long job.
- Record the new test-cell contract in `AGENTS.md` (implementation guidance only).

# 2. Run the full comparison

```bash
.venv/bin/python -m sandbox.experiments \
  --output results/tariff_bank_full_$(date +%Y%m%d_%H%M)
```

If the command fails, fix the cause and rerun into a **new** fresh directory. Do not shrink seeds, steps, or the tariff list to make it finish.

# 3. Collect, compare, Pareto

When the run exits 0, in that same output directory:

1. Confirm `manifest.json` records revision, tariff bank, policy bank, seeds, episode length, settlement tolerance, Pareto objectives, acceptance criteria, and that every tariff was fully tested.
2. Keep the runner’s per-tariff validation files:
   - `credible_response_candidates.csv`
   - `credible_response_frontier.csv`
3. Write `cross_tariff_pareto.csv` by applying `sandbox.experiments.pareto_efficient_mask` **once across all validation rows** in `credible_response_candidates.csv` (not grouped by tariff). This is an engineering front of economically credible shared-policy responses, not an equilibrium proof. Do not rebuild it from test weather.
4. Write `OVERNIGHT_COMPARISON.md` with:
   - selected family and base controller per tariff (id + name)
   - which tariffs pass the declared screens vs tuned fair LEG
   - the runner finalist and whether `accepted` is true
   - test-week means for every tariff’s full cell set (export peak, import peak, ramp, curtailment, community settlement, four group load-normalised prices)
   - parenthesised group-price differences vs `fair_leg/tuned_family`; positive means worse
   - the per-tariff Pareto set and the cross-tariff Pareto set
   - limitations: shared-policy approximation, no live prices, frontiers from validation only, official `score()` was not the per-tariff scorer, `accepted: false` is not a recommended tariff

Do not commit results. `results/` is gitignored. Do not push.

# Success

- Code change: every tariff, not just the finalist, has full test cells.
- One complete results directory with tuning, metrics, both Pareto CSVs, `cross_tariff_pareto.csv`, per-tariff test deltas, and `OVERNIGHT_COMPARISON.md`.
- Short wrap-up: finalist, whether it passed screens, how many (tariff, controller) pairs sit on the cross-tariff front, and the output path.
