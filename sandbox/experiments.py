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
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from sandbox.controller import TUNING_GRID, base_controller
from sandbox.controller_family import POLICY_BANK, family_controller
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
        "near_optimal_tolerance_chf_per_agent_week": 0.10,
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
                    if policy["voltage_gain_kw_per_pu"] > 0
                ]
                voltage_winner = max(voltage_ids, key=lambda i: means[i])
                selected[name]["voltage_family"] = candidates[voltage_winner]
                selected[name]["near_optimal_policy_ids"] = [
                    int(i) for i in np.flatnonzero(means.max() - means <= 0.10)
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
    pd.DataFrame(tuning_rows).to_csv(output / "tuning.csv", index=False)
    _write_json(output / "selected.json", selected)
    del family_train, base_train

    rows, household_rows = [], []
    cache = {}

    def evaluate_one(kind, params, tariff, label, split, seeds):
        identity = (split, kind, tuple(sorted(params.items())))
        if identity not in cache:
            controller = family_controller() if kind == "family" else base_controller()
            cache[identity] = take(
                trajectories(controller, [params], population, roots[split], seeds, n_steps), 0
            )
        batch = cache[identity]
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

    print("Validating all tariffs on independent weather", flush=True)
    for name, tariff in tariff_candidates.items():
        evaluate_one(
            "family", selected[name]["family"], tariff, name, "validation", validation_seeds
        )
    validation = pd.DataFrame(rows)
    avg = validation.groupby("run").mean(numeric_only=True)
    reference = avg.loc["fair_leg"]
    group_fields = [
        f"{kind}_cost_per_load_kwh_chf"
        for kind in ("tenant", "pv_only", "pv_battery", "large_flex")
    ]
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
        "revenue_inadequate": failed,
        "rule": "Lowest validation export peak among candidates passing declared screens; "
        "if none pass, lowest peak is tested as a diagnostic, not a recommendation.",
    }
    _write_json(output / "selection.json", selection)
    print(f"Finalist: {finalist}; passes screens: {selection['accepted']}", flush=True)
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
