"""Run follow-up response and storage diagnostics for a saved experiment.

From the repository root: ``python -m scripts.run_framework_audits``.
These are diagnostic runs after selection, not additional tariff tuning.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np
import pandas as pd

from sandbox.controller_family import family_controller
from sandbox.experiments import score_batch, take, trajectories
from sandbox.response_audit import audit_deviations, diagnose_policy
from sandbox.scenarios import reference_scenario
from sandbox.tariff_family import tariff_factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/controller_framework")
    args = parser.parse_args()
    output = Path(args.results)
    manifest = json.loads((output / "manifest.json").read_text())
    selected = json.loads((output / "selected.json").read_text())
    finalist = json.loads((output / "selection.json").read_text())["finalist"]
    tariff = manifest["tariffs"][finalist]
    n_steps = manifest["n_steps"]
    population = reference_scenario()
    policy_id = selected[finalist]["family"]["policy_id"]
    controller = family_controller(policy_id)
    candidate_ids = sorted({1, 4, 11, 15} - {policy_id})
    print(f"Unilateral audit: baseline {policy_id}, alternatives {candidate_ids}", flush=True)
    records = audit_deviations(
        controller,
        population,
        [{"policy_id": float(i)} for i in candidate_ids],
        tariff=tariff_factory(tariff),
        n_steps=n_steps,
        seeds=2,
        key=jax.random.PRNGKey(44021),
    )
    pd.DataFrame([asdict(record) for record in records]).to_csv(
        output / "unilateral_deviations.csv", index=False
    )
    diagnostics = []
    keys = jax.random.split(jax.random.PRNGKey(55001), 4)
    for label, policy in (
        ("fair_leg_family", selected["fair_leg"]["family"]["policy_id"]),
        ("tariff_family", policy_id),
        ("voltage_family", selected[finalist]["voltage_family"]["policy_id"]),
    ):
        print(f"Storage diagnostics: {label}", flush=True)
        for seed, key in enumerate(keys):
            result = diagnose_policy(
                family_controller(policy), population, n_steps=n_steps, key=key
            )
            export = -np.asarray(result.trajectory.transformer_kw)
            peak = int(export.argmax())
            row = {
                "run": label,
                "seed_index": seed,
                "policy_id": policy,
                "initial_energy_kwh": float(result.initial_stored_energy_kwh.sum()),
                "final_energy_kwh": float(result.final_stored_energy_kwh.sum()),
                "energy_at_export_peak_kwh": float(result.soc_kwh[peak].sum()),
                "headroom_at_export_peak_kwh": float(result.soc_headroom_kwh[peak].sum()),
                "batteries_full_at_export_peak": int(result.battery_full[peak].sum()),
                "voltage_trend_abs_mean_pu": float(np.abs(result.voltage_trend_pu).mean()),
            }
            diagnostics.append(row)
            if seed == 0:
                np.savez_compressed(
                    output / f"{label}_diagnostics.npz",
                    soc_kwh=result.soc_kwh,
                    headroom_kwh=result.soc_headroom_kwh,
                    voltage_trend_pu=result.voltage_trend_pu,
                    transformer_kw=result.trajectory.transformer_kw,
                )
    pd.DataFrame(diagnostics).to_csv(output / "storage_diagnostics.csv", index=False)
    ids = selected[finalist]["near_optimal_policy_ids"]
    print(f"Near-optimal response audit: {ids}", flush=True)
    batch = trajectories(
        family_controller(), [{"policy_id": i} for i in ids], population, 66029, 4, n_steps
    )
    rows = []
    for i, policy in enumerate(ids):
        scored, _ = score_batch(
            take(batch, i), population, tariff, f"policy_{policy}", "response_audit"
        )
        rows.extend(scored)
    pd.DataFrame(rows).to_csv(output / "near_optimal_metrics.csv", index=False)
    (output / "audit_manifest.json").write_text(
        json.dumps(
            {
                "baseline_policy": policy_id,
                "candidate_ids": candidate_ids,
                "deviation_root": 44021,
                "deviation_seeds": 2,
                "storage_root": 55001,
                "storage_seeds": 4,
                "near_optimal_root": 66029,
                "near_optimal_seeds": 4,
                "n_steps": n_steps,
            },
            indent=2,
        )
        + "\n"
    )
    print("Audits saved", flush=True)


if __name__ == "__main__":
    main()
