"""Reproducible tariff/controller comparisons with independent weather splits.

Run ``python -m sandbox.experiments --output results/controller_framework``.
Physical trajectories can be reused across tariffs because neither observations
nor controller decisions depend on settlement. Replay threads the tariff's carry
sequentially over the interval axis, so stateful scenarios -- a demand-charge
ratchet, a smoothed congestion signal -- replay exactly as they settle live;
both cases are checked against live settlement in tests.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from sandbox.controller import TUNING_GRID, base_controller
from sandbox.controller_family import POLICY_BANK, family_controller, has_voltage_response
from sandbox.export import feeder_dataframe, to_dataframe
from sandbox.metrics import REVENUE_TOLERANCE, score
from sandbox.observation import GridView
from sandbox.rollout import build_env, rollout_seeds
from sandbox.scenarios import EPISODE_STEPS, reference_scenario, step_duration_h
from sandbox.tariff_family import (
    TARIFF_BANK,
    TARIFF_BANK_NAMES,
    family_tariff,
    init_family_carry,
    scenario_params,
)
from sandbox.tuning import parameter_grid

CREDIBLE_RESPONSE_TOLERANCE_CHF_PER_AGENT_WEEK = 0.10
PARETO_NETWORK_OBJECTIVES = (
    "transformer_export_peak_kw",
    "transformer_draw_peak_kw",
    "max_ramp_kw",
    "curtailed_share",
)
FAIRNESS_GROUPS = ("tenant", "pv_only", "pv_battery", "large_flex")
PARETO_FAIRNESS_OBJECTIVES = tuple(
    f"{group}_cost_delta_vs_fair_leg_chf_per_load_kwh" for group in FAIRNESS_GROUPS
)
CREDIBLE_PARETO_OBJECTIVES = PARETO_NETWORK_OBJECTIVES + PARETO_FAIRNESS_OBJECTIVES
FULL_TEST_CELL_ROLES = (
    "reference_default_base",
    "fixed_base",
    "tuned_base",
    "reference_tuned_family",
    "tuned_family",
    "voltage_family",
    "voltage_off_matched",
)


def pareto_efficient_mask(frame, objectives=CREDIBLE_PARETO_OBJECTIVES):
    """Mark rows not dominated on the supplied minimisation objectives.

    Equal rows remain on the frontier. The caller is responsible for applying
    any behavioural-credibility filter before using this engineering frontier.
    """
    if frame.empty:
        return np.zeros(0, dtype=bool)
    values = frame.loc[:, list(objectives)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Pareto objectives must all be finite")
    efficient = np.ones(len(values), dtype=bool)
    for index, candidate in enumerate(values):
        weakly_better = np.all(values <= candidate, axis=1)
        strictly_better = np.any(values < candidate, axis=1)
        efficient[index] = not np.any(weakly_better & strictly_better)
    return efficient


def full_test_plan(tariff_candidates, selected, default_base_params):
    """Build the unique test rollouts and each tariff's seven-cell contract.

    The fair-LEG reference cells are shared by every tariff.  For fair LEG
    itself, the reference and tariff-specific tuned-family roles deliberately
    point to the same run; the physical and settlement cell is evaluated once.
    """
    specs = {}
    cells_by_tariff = {}

    def add(tariff_name, role, kind, params, settlement, run_label):
        cells_by_tariff[tariff_name][role] = run_label
        if run_label not in specs:
            specs[run_label] = {
                "kind": kind,
                "params": params,
                "settlement": settlement,
                "run": run_label,
            }

    for name, tariff in tariff_candidates.items():
        cells_by_tariff[name] = {}
        add(
            name,
            "reference_default_base",
            "base",
            default_base_params,
            None,
            "fair_leg/default_base",
        )
        add(
            name,
            "fixed_base",
            "base",
            default_base_params,
            tariff,
            f"{name}/fixed_base",
        )
        add(
            name,
            "tuned_base",
            "base",
            selected[name]["base"],
            tariff,
            f"{name}/tuned_base",
        )
        add(
            name,
            "reference_tuned_family",
            "family",
            selected["fair_leg"]["family"],
            None,
            "fair_leg/tuned_family",
        )
        add(
            name,
            "tuned_family",
            "family",
            selected[name]["family"],
            tariff,
            f"{name}/tuned_family",
        )
        add(
            name,
            "voltage_family",
            "family",
            selected[name]["voltage_family"],
            tariff,
            f"{name}/voltage_family",
        )
        add(
            name,
            "voltage_off_matched",
            "family",
            {**selected[name]["voltage_family"], "voltage_enabled": 0.0},
            tariff,
            f"{name}/voltage_off_matched",
        )

    for tariff, cells in cells_by_tariff.items():
        if tuple(cells) != FULL_TEST_CELL_ROLES:
            raise AssertionError(f"Incomplete full-test contract for {tariff}: {tuple(cells)}")
    return list(specs.values()), cells_by_tariff


def grid_views(trajectory, population):
    """The tariff's view of every interval of a trajectory, time-major.

    The same fields :func:`sandbox.observation.to_grid_view` builds live, so a
    replay prices exactly what the live tariff priced -- including the clock,
    which a time-of-use scenario replays wrongly if it is off by an interval.
    """
    inverter_ids = jnp.asarray(population.inverter_id, dtype=jnp.int32)
    mask = jnp.zeros(population.num_pq, dtype=bool).at[inverter_ids].set(True)
    return GridView(
        e_grid_kwh=trajectory.e_grid_kwh,
        p_grid_kw=trajectory.e_grid_kwh / step_duration_h(),
        q_grid_kvar=trajectory.q_grid_kvarh / step_duration_h(),
        voltage_pu=trajectory.voltage_pu[:, jnp.asarray(population.pq_bus_id())],
        transformer_kw=trajectory.transformer_kw,
        transformer_kvar=trajectory.transformer_kvar,
        losses_kw=trajectory.losses_kw,
        hour=trajectory.day_step * step_duration_h(),
        fair_leg_chf=trajectory.settlement_chf,
        has_inverter=jnp.broadcast_to(mask, trajectory.e_grid_kwh.shape),
    )


def resettle(trajectory, population, tariff_params):
    """Re-settle a FAIR-LEG trajectory under one tariff scenario.

    The carry is threaded sequentially with ``lax.scan`` over the interval
    axis, not reset per interval. Mapping the tariff over intervals
    independently is only correct for a stateless scenario and silently
    produces wrong numbers for a ratchet, a running peak or a smoothed
    congestion signal; ``tests/test_experiments.py`` pins replay against live
    settlement for both a stateless and a stateful scenario, and against the
    per-interval-reset version to show the two differ.
    """
    if tariff_params is None:
        return trajectory
    inverter_ids = jnp.asarray(population.inverter_id, dtype=jnp.int32)
    views = grid_views(trajectory, population)

    def settle(carry, view):
        settlement, carry = family_tariff(view, carry, tariff_params)
        return carry, settlement

    _, settlement = jax.lax.scan(settle, init_family_carry(population.num_pq), views)
    return trajectory.replace(settlement_chf=settlement, reward_chf=settlement[:, inverter_ids])


def trajectories(controller, candidates, population, root, seeds, n_steps):
    """Batch a fixed candidate bank over paired weather; always fair LEG."""
    env = build_env(population, time_limit=n_steps)
    keys = jax.random.split(jax.random.PRNGKey(root), seeds)
    stacked = {
        name: jnp.asarray([entry[name] for entry in candidates], dtype=jnp.float32)
        for name in candidates[0]
    }
    result = jax.vmap(
        lambda params: rollout_seeds(
            controller,
            population,
            keys,
            n_steps=n_steps,
            params={**controller.params, **params},
            env=env,
        )
    )(stacked)
    jax.block_until_ready(result)
    return result


def take(tree, index):
    return jax.tree_util.tree_map(lambda value: value[index], tree)


def returns_for(batch, population, tariff_params):
    settled = jax.vmap(jax.vmap(lambda tr: resettle(tr, population, tariff_params)))(batch)
    return np.asarray(settled.reward_chf.sum(axis=2).mean(axis=2))


def household_load_kwh(trajectory, population):
    """Recover actual behind-meter load, not imports, from realised power.

    p_grid = p_inverter - p_load. Tenants have no inverter. This diagnostic
    deliberately does not alter the official import-normalised jury metrics.
    """
    inverter_energy = np.zeros_like(np.asarray(trajectory.e_grid_kwh))
    inverter_energy[:, np.asarray(population.inverter_id)] = (
        np.asarray(trajectory.p_inv_realized_kw) * step_duration_h()
    )
    return np.maximum(inverter_energy - np.asarray(trajectory.e_grid_kwh), 0.0).sum(axis=0)


def score_batch(batch, population, tariff_params, label, split):
    rows, households = [], []
    for seed in range(batch.valid.shape[0]):
        tr = resettle(take(batch, seed), population, tariff_params)
        row = {"run": label, "split": split, "seed_index": seed, **score(tr, population).to_dict()}
        row["valid_share"] = float(np.asarray(tr.valid).mean())
        row["projection_mae_kw"] = float(
            np.abs(np.asarray(tr.p_inv_realized_kw - tr.p_inv_set_kw)).mean()
        )
        loads = household_load_kwh(tr, population)
        settlements = np.asarray(tr.settlement_chf).sum(axis=0)
        for kind in ("tenant", "pv_only", "pv_battery", "large_flex"):
            mask = np.asarray(population.mask_for(kind))
            row[f"{kind}_cost_per_load_kwh_chf"] = float(
                -settlements[mask].sum() / max(loads[mask].sum(), 1e-6)
            )
        rows.append(row)
        frame = to_dataframe(tr, population)
        for connection, group in frame.groupby("connection"):
            households.append(
                {
                    "run": label,
                    "split": split,
                    "seed_index": seed,
                    "connection": int(connection),
                    "household": group.household.iloc[0],
                    "settlement_chf": float(group.settlement_chf.sum()),
                    "load_kwh": float(loads[connection]),
                    "cost_per_load_kwh_chf": float(
                        -settlements[connection] / max(loads[connection], 1e-6)
                    ),
                }
            )
    return rows, households


def revenue_screen(batch, population, candidates, reference="fair_leg"):
    """Community settlement per tariff with behaviour held fixed.

    The official gate (:func:`sandbox.metrics.revenue_adequate`) is checked
    this way and not on re-tuned cells: a tariff that makes households export
    less collects less, and that is the tariff working rather than the tariff
    printing money. Re-settling one fixed trajectory is exactly that
    comparison, and it needs no further rollouts.
    """
    totals = {}
    for name, tariff in candidates.items():
        settle = lambda trajectory, tariff=tariff: resettle(trajectory, population, tariff)
        settled = jax.vmap(settle)(batch)
        totals[name] = float(np.asarray(settled.settlement_chf).sum(axis=(1, 2)).mean())
    target = abs(totals[reference])
    return {
        "settlement_chf": totals,
        "adequate": {
            name: bool(abs(total - totals[reference]) <= REVENUE_TOLERANCE * target)
            for name, total in totals.items()
        },
    }


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def _safe_test_filename(run_label, suffix):
    safe = re.sub(r"[^A-Za-z0-9._-]+", "__", run_label).strip("._-")
    return f"{safe}_{suffix}.csv"


def _write_overnight_comparison(
    path,
    selected,
    selection,
    revenue,
    test_frame,
    cells_by_tariff,
    credible,
    cross_tariff,
):
    """Write the full-bank comparison from validation choices and test means."""
    test_means = test_frame.groupby("run").mean(numeric_only=True)
    reference = test_means.loc["fair_leg/tuned_family"]
    lines = [
        "# Overnight tariff-bank comparison",
        "",
        "All controllers were selected on training settlement and screened on independent "
        "validation weather. Test weather is used only for the reported held-out means.",
        "",
        "## Selected controllers",
        "",
        "| Tariff | Family controller | Base controller |",
        "|---|---|---|",
    ]
    for tariff, choice in selected.items():
        family_id = choice["family_candidate_id"]
        family_name = choice["family_name"]
        base_id = choice["base_candidate_id"]
        base_name = choice["base_name"]
        base_params = json.dumps(choice["base"], sort_keys=True)
        lines.append(
            f"| {tariff} | {family_id}: {family_name} | "
            f"{base_id}: {base_name} `{base_params}` |"
        )

    lines.extend(
        [
            "",
            "## Declared validation screens",
            "",
            f"Runner finalist: **{selection['finalist']}**. "
            f"Accepted: **{str(selection['accepted']).lower()}**.",
            "",
            "| Tariff | Screen result vs tuned fair LEG | Revenue adequate |",
            "|---|---:|---:|",
        ]
    )
    for tariff in selected:
        screen = "reference" if tariff == "fair_leg" else (
            "pass" if tariff in selection["eligible"] else "fail"
        )
        lines.append(
            f"| {tariff} | {screen} | {str(revenue['adequate'][tariff]).lower()} |"
        )

    lines.extend(
        [
            "",
            "## Held-out test-week means",
            "",
            "Group prices are CHF per actual load kWh. Parentheses show the difference "
            "from `fair_leg/tuned_family`; positive is worse for that group.",
            "",
            "| Tariff | Cell role | Run | Export peak kW | Import peak kW | Max ramp kW | "
            "Curtailment | Community settlement CHF | Tenant price | PV-only price | "
            "PV+battery price | Large-flex price |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for tariff, cells in cells_by_tariff.items():
        for role, run_label in cells.items():
            row = test_means.loc[run_label]
            prices = []
            for group in FAIRNESS_GROUPS:
                field = f"{group}_cost_per_load_kwh_chf"
                value = float(row[field])
                delta = value - float(reference[field])
                prices.append(f"{value:.4f} ({delta:+.4f})")
            lines.append(
                f"| {tariff} | {role} | `{run_label}` | "
                f"{row['transformer_export_peak_kw']:.3f} | "
                f"{row['transformer_draw_peak_kw']:.3f} | "
                f"{row['max_ramp_kw']:.3f} | {row['curtailed_share']:.4f} | "
                f"{row['community_settlement_chf']:.3f} | "
                + " | ".join(prices)
                + " |"
            )

    lines.extend(
        [
            "",
            "## Per-tariff validation Pareto sets",
            "",
            "These are the runner's tariff-grouped engineering frontiers within each "
            "economically credible shared-policy response set.",
            "",
            "| Tariff | Policy id | Controller | Training settlement gap CHF/agent |",
            "|---|---:|---|---:|",
        ]
    )
    for row in credible.query("is_pareto").itertuples():
        lines.append(
            f"| {row.tariff} | {row.controller_policy_id} | {row.controller_name} | "
            f"{row.train_settlement_gap_chf_per_agent:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Cross-tariff validation Pareto set",
            "",
            "The mask was applied once across all validation rows in "
            "`credible_response_candidates.csv`, without grouping by tariff.",
            "",
            "| Tariff | Policy id | Controller | Training settlement gap CHF/agent |",
            "|---|---:|---|---:|",
        ]
    )
    for row in cross_tariff.itertuples():
        lines.append(
            f"| {row.tariff} | {row.controller_policy_id} | {row.controller_name} | "
            f"{row.train_settlement_gap_chf_per_agent:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Controller tuning is a shared-policy approximation, not proof of an "
            "individual equilibrium.",
            "- Controllers observe neither live prices nor neighbours; tariff effects enter "
            "through ex-ante policy tuning and settlement replay.",
            "- Both Pareto sets use validation weather only. Test weather did not select "
            "controllers or construct or revise either frontier.",
            "- The official `score()`/`evaluate()` submission path was not used as a "
            "per-tariff scorer. The runner retained the unchanged jury metrics while using "
            "its controller-family experiment pipeline.",
            "- `accepted: false` identifies a diagnostic finalist, not a recommended tariff.",
            "- Battery wear and terminal stored-energy value are absent from the household "
            "objective; revenue neutrality alone does not establish fairness.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def run(output, n_steps=EPISODE_STEPS, train_seeds=4, validation_seeds=8, test_seeds=20):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    population = reference_scenario()
    roots = {"train": 11003, "validation": 22007, "test": 33013}
    family_candidates = [{"policy_id": i} for i in range(len(POLICY_BANK))]
    base_candidates = parameter_grid(TUNING_GRID)
    # Predeclared screening criteria, relative to tuned family under fair LEG.
    criteria = {
        "curtailment_increase_max": 0.02,
        "draw_peak_increase_max_kw": 2.0,
        "group_cost_increase_max_chf_per_kwh": 0.01,
        "revenue_tolerance": REVENUE_TOLERANCE,
    }
    # The whole scenario bank, one scalar id each, plus fair LEG unchanged as
    # the reference. `fair_leg_passthrough` is the bank's own null hypothesis
    # and should reproduce `fair_leg` exactly.
    tariff_candidates = {"fair_leg": None}
    tariff_candidates.update({name: scenario_params(name) for name in TARIFF_BANK_NAMES})
    manifest = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "note": "Working tree implementation; source snapshot saved with results.",
        "roots": roots,
        "n_steps": n_steps,
        "seeds": {"train": train_seeds, "validation": validation_seeds, "test": test_seeds},
        "policy_bank": POLICY_BANK,
        "tariff_bank": TARIFF_BANK,
        "tariffs": tariff_candidates,
        "criteria": criteria,
        "near_optimal_tolerance_chf_per_agent_week": (
            CREDIBLE_RESPONSE_TOLERANCE_CHF_PER_AGENT_WEEK
        ),
        "settlement_tolerance_chf_per_agent_week": (
            CREDIBLE_RESPONSE_TOLERANCE_CHF_PER_AGENT_WEEK
        ),
        "credible_response_pareto_objectives": list(CREDIBLE_PARETO_OBJECTIVES),
        "full_test_evaluation": {
            "complete": False,
            "tariffs": list(tariff_candidates),
            "cell_roles": list(FULL_TEST_CELL_ROLES),
        },
    }
    _write_json(output / "manifest.json", manifest)
    snapshot = output / "source"
    snapshot.mkdir(exist_ok=True)
    for name in ("controller_family.py", "tariff_family.py", "experiments.py"):
        (snapshot / name).write_text((Path(__file__).parent / name).read_text())
    print(
        f"Training {len(family_candidates)} family and {len(base_candidates)} base policies",
        flush=True,
    )
    family_train = trajectories(
        family_controller(), family_candidates, population, roots["train"], train_seeds, n_steps
    )
    base_train = trajectories(
        base_controller(), base_candidates, population, roots["train"], train_seeds, n_steps
    )
    selected, tuning_rows = {}, []
    for name, tariff in tariff_candidates.items():
        selected[name] = {}
        for kind, batch, candidates in (
            ("family", family_train, family_candidates),
            ("base", base_train, base_candidates),
        ):
            values = returns_for(batch, population, tariff)
            means = values.mean(axis=1)
            winner = int(means.argmax())
            selected[name][kind] = candidates[winner]
            selected[name][f"{kind}_candidate_id"] = winner
            selected[name][f"{kind}_name"] = (
                POLICY_BANK[winner]["name"] if kind == "family" else base_controller().name
            )
            if kind == "family":
                voltage_ids = [
                    i
                    for i, policy in enumerate(POLICY_BANK)
                    if has_voltage_response(policy)
                ]
                voltage_winner = max(voltage_ids, key=lambda i: means[i])
                selected[name]["voltage_family"] = candidates[voltage_winner]
                selected[name]["voltage_family_candidate_id"] = voltage_winner
                selected[name]["voltage_family_name"] = POLICY_BANK[voltage_winner]["name"]
                selected[name]["near_optimal_policy_ids"] = [
                    int(i)
                    for i in np.flatnonzero(
                        means.max() - means
                        <= CREDIBLE_RESPONSE_TOLERANCE_CHF_PER_AGENT_WEEK
                    )
                ]
            for idx, mean in enumerate(means):
                tuning_rows.append(
                    {
                        "tariff": name,
                        "controller": kind,
                        "candidate": idx,
                        "mean_return_chf": float(mean),
                        "gap_chf": float(means.max() - mean),
                        "selected": idx == winner,
                    }
                )
        print(f"  {name}: {selected[name]}", flush=True)
    tuning = pd.DataFrame(tuning_rows)
    tuning.to_csv(output / "tuning.csv", index=False)
    _write_json(output / "selected.json", selected)
    del family_train, base_train

    rows, household_rows = [], []
    cache = {}

    def batch_for(kind, params, split, seeds):
        identity = (split, kind, tuple(sorted(params.items())))
        if identity not in cache:
            controller = family_controller() if kind == "family" else base_controller()
            cache[identity] = take(
                trajectories(controller, [params], population, roots[split], seeds, n_steps), 0
            )
        return cache[identity]

    def evaluate_one(kind, params, tariff, label, split, seeds):
        batch = batch_for(kind, params, split, seeds)
        scored, homes = score_batch(batch, population, tariff, label, split)
        rows.extend(scored)
        household_rows.extend(homes)
        return batch

    print("Screening revenue adequacy with behaviour held fixed", flush=True)
    fixed_behaviour = take(
        trajectories(
            base_controller(),
            [{key: float(value) for key, value in base_controller().params.items()}],
            population,
            roots["validation"],
            validation_seeds,
            n_steps,
        ),
        0,
    )
    revenue = revenue_screen(fixed_behaviour, population, tariff_candidates)
    _write_json(output / "revenue_adequacy.json", revenue)
    failed = [name for name, ok in revenue["adequate"].items() if not ok]
    print(f"  revenue adequacy fails for: {failed or 'nothing'}", flush=True)
    del fixed_behaviour

    credible_policy_ids = sorted(
        {
            policy_id
            for tariff_selection in selected.values()
            for policy_id in tariff_selection["near_optimal_policy_ids"]
        }
    )
    credible_validation = trajectories(
        family_controller(),
        [{"policy_id": policy_id} for policy_id in credible_policy_ids],
        population,
        roots["validation"],
        validation_seeds,
        n_steps,
    )
    for index, policy_id in enumerate(credible_policy_ids):
        identity = ("validation", "family", (("policy_id", policy_id),))
        cache[identity] = take(credible_validation, index)
    del credible_validation

    print("Validating all tariffs on independent weather", flush=True)
    for name, tariff in tariff_candidates.items():
        evaluate_one(
            "family", selected[name]["family"], tariff, name, "validation", validation_seeds
        )
    validation = pd.DataFrame(rows)
    avg = validation.groupby("run").mean(numeric_only=True)
    reference = avg.loc["fair_leg"]
    group_fields = [f"{kind}_cost_per_load_kwh_chf" for kind in FAIRNESS_GROUPS]
    eligible = []
    for name in tariff_candidates:
        if name == "fair_leg":
            continue
        row = avg.loc[name]
        acceptable = (
            row.curtailed_share <= reference.curtailed_share + criteria["curtailment_increase_max"]
            and row.transformer_draw_peak_kw
            <= reference.transformer_draw_peak_kw + criteria["draw_peak_increase_max_kw"]
            and all(
                row[field] <= reference[field] + criteria["group_cost_increase_max_chf_per_kwh"]
                for field in group_fields
            )
            and row.valid_share == 1.0
            and revenue["adequate"][name]
        )
        if acceptable:
            eligible.append(name)
    finalist = (
        min(eligible, key=lambda name: avg.loc[name, "transformer_export_peak_kw"])
        if eligible
        else min(
            (name for name in tariff_candidates if name != "fair_leg"),
            key=lambda name: avg.loc[name, "transformer_export_peak_kw"],
        )
    )
    selection = {
        "finalist": finalist,
        "eligible": eligible,
        "accepted": finalist in eligible,
        "screens_by_tariff": {
            name: (None if name == "fair_leg" else name in eligible)
            for name in tariff_candidates
        },
        "revenue_inadequate": failed,
        "rule": "Lowest validation export peak among candidates passing declared screens; "
        "if none pass, lowest peak is tested as a diagnostic, not a recommendation.",
    }
    _write_json(output / "selection.json", selection)
    print(f"Finalist: {finalist}; passes screens: {selection['accepted']}", flush=True)

    print("Evaluating credible-response Pareto candidates", flush=True)
    credible_rows = []
    family_tuning = tuning.query("controller == 'family'").set_index(["tariff", "candidate"])
    fair_leg_reference_policy_id = selected["fair_leg"]["family"]["policy_id"]
    for name, candidate_tariff in tariff_candidates.items():
        for policy_id in selected[name]["near_optimal_policy_ids"]:
            params = {"policy_id": float(policy_id)}
            batch = batch_for("family", params, "validation", validation_seeds)
            scored, _ = score_batch(
                batch,
                population,
                candidate_tariff,
                f"{name}/policy_{policy_id}",
                "validation",
            )
            validation_metrics = (
                pd.DataFrame(scored)
                .drop(columns=["run", "split", "seed_index"])
                .mean(numeric_only=True)
                .to_dict()
            )
            fairness_comparison = {}
            fairness_deltas = []
            for group in FAIRNESS_GROUPS:
                cost_field = f"{group}_cost_per_load_kwh_chf"
                reference_cost = float(reference[cost_field])
                delta = float(validation_metrics[cost_field] - reference_cost)
                fairness_comparison[
                    f"fair_leg_reference_{group}_cost_per_load_kwh_chf"
                ] = reference_cost
                fairness_comparison[
                    f"{group}_cost_delta_vs_fair_leg_chf_per_load_kwh"
                ] = delta
                fairness_deltas.append(delta)
            tuning_row = family_tuning.loc[(name, policy_id)]
            policy = POLICY_BANK[policy_id]
            row = {
                "tariff": name,
                "tariff_parameters": json.dumps(candidate_tariff, sort_keys=True),
                "controller_policy_id": policy_id,
                "controller_name": policy["name"],
                "controller_parameters": json.dumps(params, sort_keys=True),
                "split": "validation",
                "validation_seed_count": validation_seeds,
                "train_mean_settlement_chf_per_agent": float(tuning_row["mean_return_chf"]),
                "train_settlement_gap_chf_per_agent": float(tuning_row["gap_chf"]),
                "credible_response_tolerance_chf_per_agent": (
                    CREDIBLE_RESPONSE_TOLERANCE_CHF_PER_AGENT_WEEK
                ),
                "fair_leg_reference_controller_policy_id": fair_leg_reference_policy_id,
                "fair_leg_reference_controller_name": POLICY_BANK[
                    fair_leg_reference_policy_id
                ]["name"],
                "tariff_passes_acceptance_screens": (
                    None if name == "fair_leg" else name in eligible
                ),
                "tariff_is_finalist": name == finalist,
                "pareto_objectives": ";".join(CREDIBLE_PARETO_OBJECTIVES),
                **{
                    f"tariff_{key}": value
                    for key, value in (candidate_tariff or {}).items()
                },
                **{
                    f"controller_{key}": value
                    for key, value in policy.items()
                    if key != "name"
                },
                **validation_metrics,
                **fairness_comparison,
                "worst_group_cost_increase_vs_fair_leg_chf_per_load_kwh": max(
                    fairness_deltas
                ),
            }
            credible_rows.append(row)

    credible = pd.DataFrame(credible_rows)
    credible["is_pareto"] = False
    for _, indices in credible.groupby("tariff", sort=False).groups.items():
        credible.loc[indices, "is_pareto"] = pareto_efficient_mask(credible.loc[indices])
    credible = credible.sort_values(
        ["tariff", "is_pareto", "train_settlement_gap_chf_per_agent"],
        ascending=[True, False, True],
    )
    credible.to_csv(output / "credible_response_candidates.csv", index=False)
    credible.query("is_pareto").to_csv(output / "credible_response_frontier.csv", index=False)

    cross_tariff = credible.loc[pareto_efficient_mask(credible)].copy()
    cross_tariff["is_cross_tariff_pareto"] = True
    cross_tariff = cross_tariff.sort_values(
        ["tariff", "train_settlement_gap_chf_per_agent"], ascending=[True, True]
    )
    cross_tariff.to_csv(output / "cross_tariff_pareto.csv", index=False)

    default_base_params = {
        key: float(value) for key, value in base_controller().params.items()
    }
    test_specs, cells_by_tariff = full_test_plan(
        tariff_candidates, selected, default_base_params
    )
    manifest["full_test_evaluation"]["cells_by_tariff"] = cells_by_tariff
    manifest["full_test_evaluation"]["unique_runs"] = [
        spec["run"] for spec in test_specs
    ]
    _write_json(output / "manifest.json", manifest)

    # Native Python scalars make cache keys hashable for default JAX params.
    cache.clear()
    for spec in test_specs:
        kind = spec["kind"]
        payment = spec["settlement"]
        label = spec["run"]
        params = spec["params"]
        params = {key: float(value) for key, value in params.items()}
        print(f"Testing {label}", flush=True)
        batch = evaluate_one(kind, params, payment, label, "test", test_seeds)
        tr = resettle(take(batch, 0), population, payment)
        to_dataframe(tr, population, label).to_csv(
            output / _safe_test_filename(label, "household_trace"), index=False
        )
        feeder_dataframe(tr).to_csv(
            output / _safe_test_filename(label, "feeder_trace"), index=False
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(household_rows).to_csv(output / "household_settlements.csv", index=False)
    frame.groupby(["split", "run"]).agg(
        {field: ["mean", "std"] for field in score(tr, population).to_dict()}
    ).to_csv(output / "summary.csv")
    paired = frame.query("split == 'test'").pivot(index="seed_index", columns="run")
    deltas_by_tariff = {}
    for tariff_name, cells in cells_by_tariff.items():
        deltas_by_tariff[tariff_name] = {}
        for metric in score(tr, population).to_dict():
            delta = (
                paired[metric][cells["tuned_family"]]
                - paired[metric][cells["reference_tuned_family"]]
            )
            deltas_by_tariff[tariff_name][metric] = {
                "mean_delta": float(delta.mean()),
                "std_delta": float(delta.std()),
                "se_delta": float(delta.std() / np.sqrt(test_seeds)),
            }
    _write_json(output / "paired_deltas_by_tariff.json", deltas_by_tariff)
    _write_overnight_comparison(
        output / "OVERNIGHT_COMPARISON.md",
        selected,
        selection,
        revenue,
        frame.query("split == 'test'"),
        cells_by_tariff,
        credible,
        cross_tariff,
    )
    manifest["full_test_evaluation"].update(
        {
            "complete": True,
            "test_seed_count": test_seeds,
            "paired_deltas_file": "paired_deltas_by_tariff.json",
            "cross_tariff_pareto_file": "cross_tariff_pareto.csv",
            "cross_tariff_pareto_pair_count": len(cross_tariff),
        }
    )
    _write_json(output / "manifest.json", manifest)
    print(
        frame.query("split == 'test'")
        .groupby("run")[
            [
                "transformer_export_peak_kw",
                "max_ramp_kw",
                "curtailed_share",
                "community_settlement_chf",
                "tenant_cost_per_kwh_chf",
            ]
        ]
        .mean()
        .to_string(),
        flush=True,
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/controller_framework")
    parser.add_argument("--steps", type=int, default=EPISODE_STEPS)
    parser.add_argument("--train-seeds", type=int, default=4)
    parser.add_argument("--validation-seeds", type=int, default=8)
    parser.add_argument("--test-seeds", type=int, default=20)
    args = parser.parse_args()
    for name in ("steps", "train_seeds", "validation_seeds", "test_seeds"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    run(args.output, args.steps, args.train_seeds, args.validation_seeds, args.test_seeds)


if __name__ == "__main__":
    main()
