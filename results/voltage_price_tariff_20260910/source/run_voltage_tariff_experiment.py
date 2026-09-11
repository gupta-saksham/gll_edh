"""Select a controller for the voltage-price tariff and score four cells.

The four cells mirror the quickstart/evaluator convention. Controller choices
are made on training weather and the displayed table uses independent test
weather. Run from the repository root.
"""

import argparse
import json
import subprocess
from pathlib import Path

import jax
import numpy as np
import pandas as pd

from sandbox.controller import TUNING_GRID, base_controller
from sandbox.controller_family import POLICY_BANK, TUNE_OVER, family_controller
from sandbox.experiments import score_batch
from sandbox.rollout import build_env, rollout_seeds
from sandbox.scenarios import EPISODE_STEPS, reference_scenario
from sandbox.tuning import parameter_grid, tune
from sandbox.voltage_tariff import DEFAULT_VOLTAGE_TARIFF_PARAMS, voltage_tariff_factory

GROUPS = ("tenant", "pv_only", "pv_battery", "large_flex")
TRAIN_ROOT = 11003
TEST_ROOT = 33013


def _plain_params(params):
    return {key: float(np.asarray(value)) for key, value in params.items()}


def _score_cell(label, controller, tariff, population, n_steps, seeds):
    env = build_env(population, time_limit=n_steps, tariff=tariff)
    trajectories = rollout_seeds(
        controller,
        population,
        jax.random.split(jax.random.PRNGKey(TEST_ROOT), seeds),
        n_steps=n_steps,
        env=env,
    )
    scored, _ = score_batch(trajectories, population, None, label, "test")
    mean = (
        pd.DataFrame(scored)
        .drop(columns=["run", "split", "seed_index"])
        .mean(numeric_only=True)
        .to_dict()
    )
    return {"cell": label, **mean}


def _markdown_table(frame, selected_policy):
    columns = [
        ("cell", "Cell"),
        ("transformer_export_peak_kw", "Export peak kW"),
        ("transformer_draw_peak_kw", "Import peak kW"),
        ("max_ramp_kw", "Ramp kW"),
        ("curtailed_share", "Curtailment"),
        ("community_settlement_chf", "Community CHF/episode"),
        ("tenant_cost_per_load_kwh_chf", "Tenant CHF/kWh"),
        ("pv_only_cost_per_load_kwh_chf", "PV-only CHF/kWh"),
        ("pv_battery_cost_per_load_kwh_chf", "PV+battery CHF/kWh"),
        ("large_flex_cost_per_load_kwh_chf", "Large-flex CHF/kWh"),
    ]
    lines = [
        "# Voltage-price tariff: four-cell comparison",
        "",
        "The selected family response is policy "
        f"{selected_policy['policy_id']} (`{selected_policy['name']}`). Values are means over "
        "the independent test weeks. Group prices are CHF per actual behind-meter load kWh; "
        "parentheses show the difference from `fair_leg/base`, where positive is worse.",
        "",
        "| " + " | ".join(label for _, label in columns) + " |",
        "|" + "|".join("---" if key == "cell" else "---:" for key, _ in columns) + "|",
    ]
    for _, row in frame.iterrows():
        values = []
        for key, _ in columns:
            value = row[key]
            if key == "cell":
                values.append(f"`{value}`")
            elif key == "curtailed_share":
                values.append(f"{value:.2%}")
            elif key.endswith("_cost_per_load_kwh_chf"):
                group = key.removesuffix("_cost_per_load_kwh_chf")
                delta = row[
                    f"{group}_cost_delta_vs_fair_leg_base_chf_per_load_kwh"
                ]
                values.append(f"{value:+.3f} ({delta:+.3f})")
            else:
                values.append(f"{value:.2f}")
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "The tariff energy price is `clip(0.15 - 1.5 * (voltage_pu - 1.0), 0.05, 0.25)` "
            "CHF/kWh. An equal lump-sum balance reconciles every interval to fair LEG's total.",
            "",
            "This is a diagnostic voltage-level tariff. Local voltage measures exposure, not a "
            "household's marginal contribution to the voltage condition.",
            "",
        ]
    )
    return "\n".join(lines)


def run(output, n_steps=EPISODE_STEPS, train_seeds=4, test_seeds=20):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    population = reference_scenario()
    tariff_params = dict(DEFAULT_VOLTAGE_TARIFF_PARAMS)
    tariff = voltage_tariff_factory(tariff_params)

    family_params, family_returns = tune(
        family_controller(),
        population,
        TUNE_OVER,
        tariff=tariff,
        n_steps=n_steps,
        seeds=train_seeds,
        key=jax.random.PRNGKey(TRAIN_ROOT),
    )
    base_params, base_returns = tune(
        base_controller(),
        population,
        TUNING_GRID,
        tariff=tariff,
        n_steps=n_steps,
        seeds=train_seeds,
        key=jax.random.PRNGKey(TRAIN_ROOT),
    )
    policy_id = int(family_params["policy_id"])
    selected_policy = {"policy_id": policy_id, "name": POLICY_BANK[policy_id]["name"]}
    family = family_controller(policy_id)
    base = base_controller()
    tuned_base = base.replace(params={**base.params, **base_params})

    tuning_rows = [
        {
            "controller_family": "family",
            "candidate": index,
            "policy_id": index,
            "policy_name": POLICY_BANK[index]["name"],
            "parameters": json.dumps({"policy_id": index}),
            "train_mean_settlement_chf_per_agent": float(value),
            "selected": index == policy_id,
        }
        for index, value in enumerate(family_returns)
    ]
    base_entries = parameter_grid(TUNING_GRID)
    tuning_rows.extend(
        {
            "controller_family": "base",
            "candidate": index,
            "policy_id": np.nan,
            "policy_name": "self_consumption",
            "parameters": json.dumps(entry, sort_keys=True),
            "train_mean_settlement_chf_per_agent": float(value),
            "selected": entry == base_params,
        }
        for index, (entry, value) in enumerate(zip(base_entries, base_returns, strict=True))
    )
    pd.DataFrame(tuning_rows).to_csv(output / "controller_tuning.csv", index=False)

    specs = (
        ("fair_leg/base", base, None),
        ("fair_leg/selected_family", family, None),
        ("voltage_price/tuned_base", tuned_base, tariff),
        ("voltage_price/selected_family", family, tariff),
    )
    rows = [
        {
            **_score_cell(label, controller, payment, population, n_steps, test_seeds),
            "controller_parameters": json.dumps(_plain_params(controller.params), sort_keys=True),
        }
        for label, controller, payment in specs
    ]
    frame = pd.DataFrame(rows)
    reference = frame.loc[frame.cell == "fair_leg/base"].iloc[0]
    for group in GROUPS:
        field = f"{group}_cost_per_load_kwh_chf"
        frame[f"fair_leg_base_{field}"] = reference[field]
        frame[f"{group}_cost_delta_vs_fair_leg_base_chf_per_load_kwh"] = (
            frame[field] - reference[field]
        )
    frame["community_settlement_delta_vs_fair_leg_base_chf"] = (
        frame.community_settlement_chf - reference.community_settlement_chf
    )
    frame.to_csv(output / "four_cell_comparison.csv", index=False)
    (output / "four_cell_comparison.md").write_text(
        _markdown_table(frame, selected_policy)
    )

    fixed_voltage = _score_cell(
        "voltage_price/fixed_base_revenue_check",
        base,
        tariff,
        population,
        n_steps,
        test_seeds,
    )
    manifest = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "note": "Working tree implementation; source snapshot saved with results.",
        "tariff": "local_voltage_price",
        "tariff_params": tariff_params,
        "price_formula": "clip(0.15 - 1.5 * (voltage_pu - 1.0), 0.05, 0.25)",
        "balance": "Equal per-connection transfer matching fair LEG's interval total.",
        "n_steps": n_steps,
        "train_root": TRAIN_ROOT,
        "train_seeds": train_seeds,
        "test_root": TEST_ROOT,
        "test_seeds": test_seeds,
        "selected_family": selected_policy,
        "selected_base_params": _plain_params(base_params),
        "fixed_behaviour_revenue_delta_chf": (
            fixed_voltage["community_settlement_chf"]
            - reference["community_settlement_chf"]
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    snapshot = output / "source"
    snapshot.mkdir(exist_ok=True)
    for path in (
        Path("sandbox/voltage_tariff.py"),
        Path("sandbox/controller_family.py"),
        Path("scripts/run_voltage_tariff_experiment.py"),
    ):
        (snapshot / path.name).write_text(path.read_text())

    print(f"Selected family policy {policy_id}: {selected_policy['name']}", flush=True)
    print(f"Selected base parameters: {_plain_params(base_params)}", flush=True)
    print(frame.to_string(index=False), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/voltage_price_tariff")
    parser.add_argument("--steps", type=int, default=EPISODE_STEPS)
    parser.add_argument("--train-seeds", type=int, default=4)
    parser.add_argument("--test-seeds", type=int, default=20)
    args = parser.parse_args()
    for name in ("steps", "train_seeds", "test_seeds"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    run(args.output, args.steps, args.train_seeds, args.test_seeds)


if __name__ == "__main__":
    main()
