"""Reproducible tariff/controller comparisons with independent weather splits.

Run ``python -m sandbox.experiments --output results/controller_framework``.
Physical trajectories can be reused across tariffs because neither observations
nor controller decisions depend on settlement. Replay is specific to the
stateless stress tariff and is checked against live settlement in tests.
"""

import argparse
import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from sandbox.controller import TUNING_GRID, base_controller
from sandbox.controller_family import POLICY_BANK, family_controller, has_voltage_response
from sandbox.export import feeder_dataframe, to_dataframe
from sandbox.metrics import score
from sandbox.observation import GridView
from sandbox.rollout import build_env, rollout_seeds
from sandbox.scenarios import EPISODE_STEPS, reference_scenario, step_duration_h
from sandbox.tariff import init_tariff_memory
from sandbox.tariff_family import DEFAULT_TARIFF_PARAMS, stress_tariff
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


def resettle(trajectory, population, tariff_params):
    """Apply the stateless stress adjustment to a FAIR-LEG trajectory only."""
    if tariff_params is None:
        return trajectory
    inverter_ids = jnp.asarray(population.inverter_id, dtype=jnp.int32)
    mask = jnp.zeros(population.num_pq, dtype=bool).at[inverter_ids].set(True)
    views = GridView(
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
    settlement = jax.vmap(lambda view: stress_tariff(view, init_tariff_memory(), tariff_params)[0])(
        views
    )
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


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def run(output, n_steps=EPISODE_STEPS, train_seeds=4, validation_seeds=8, test_seeds=20):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    population = reference_scenario()
    roots = {"train": 11003, "validation": 22007, "test": 33013}
    family_candidates = [{"policy_id": i} for i in range(len(POLICY_BANK))]
    base_candidates = parameter_grid(TUNING_GRID)
    # Predeclared screening criteria, relative to tuned family under fair LEG.
    criteria = {
        "curtailment_increase_max": 0.02,
        "draw_peak_increase_max_kw": 2.0,
        "group_cost_increase_max_chf_per_kwh": 0.01,
    }
    tariff_candidates = {"fair_leg": None}
    for threshold in (30.0, 45.0):
        for strength in (0.05, 0.15, 0.30):
            tariff_candidates[f"export_{threshold:g}kw_{strength:g}"] = {
                **DEFAULT_TARIFF_PARAMS,
                "export_threshold_kw": threshold,
                "export_strength_chf_per_kwh": strength,
            }
    manifest = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "note": "Working tree implementation; source snapshot saved with results.",
        "roots": roots,
        "n_steps": n_steps,
        "seeds": {"train": train_seeds, "validation": validation_seeds, "test": test_seeds},
        "policy_bank": POLICY_BANK,
        "tariffs": tariff_candidates,
        "criteria": criteria,
        "near_optimal_tolerance_chf_per_agent_week": (
            CREDIBLE_RESPONSE_TOLERANCE_CHF_PER_AGENT_WEEK
        ),
        "credible_response_pareto_objectives": list(CREDIBLE_PARETO_OBJECTIVES),
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
            if kind == "family":
                voltage_ids = [
                    i
                    for i, policy in enumerate(POLICY_BANK)
                    if has_voltage_response(policy)
                ]
                voltage_winner = max(voltage_ids, key=lambda i: means[i])
                selected[name]["voltage_family"] = candidates[voltage_winner]
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

    tariff = tariff_candidates[finalist]
    test_specs = [
        ("base", dict(base_controller().params), None, "fair_leg/default_base"),
        ("base", dict(base_controller().params), tariff, "tariff/fixed_base"),
        ("base", selected[finalist]["base"], tariff, "tariff/tuned_base"),
        ("family", selected["fair_leg"]["family"], None, "fair_leg/tuned_family"),
        ("family", selected[finalist]["family"], tariff, "tariff/tuned_family"),
        ("family", selected[finalist]["voltage_family"], tariff, "tariff/voltage_family"),
        (
            "family",
            {**selected[finalist]["voltage_family"], "voltage_enabled": 0.0},
            tariff,
            "tariff/voltage_off_matched",
        ),
    ]
    # Native Python scalars make cache keys hashable for default JAX params.
    cache.clear()
    for kind, params, payment, label in test_specs:
        params = {key: float(value) for key, value in params.items()}
        print(f"Testing {label}", flush=True)
        batch = evaluate_one(kind, params, payment, label, "test", test_seeds)
        tr = resettle(take(batch, 0), population, payment)
        to_dataframe(tr, population, label).to_csv(
            output / (label.replace("/", "_") + "_household_trace.csv"), index=False
        )
        feeder_dataframe(tr).to_csv(
            output / (label.replace("/", "_") + "_feeder_trace.csv"), index=False
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(household_rows).to_csv(output / "household_settlements.csv", index=False)
    frame.groupby(["split", "run"]).agg(
        {field: ["mean", "std"] for field in score(tr, population).to_dict()}
    ).to_csv(output / "summary.csv")
    paired = frame.query("split == 'test'").pivot(index="seed_index", columns="run")
    deltas = {}
    for metric in score(tr, population).to_dict():
        delta = paired[metric]["tariff/tuned_family"] - paired[metric]["fair_leg/tuned_family"]
        deltas[metric] = {
            "mean_delta": float(delta.mean()),
            "std_delta": float(delta.std()),
            "se_delta": float(delta.std() / np.sqrt(test_seeds)),
        }
    _write_json(output / "paired_deltas.json", deltas)
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
