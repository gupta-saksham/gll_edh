import math

import jax

from sandbox.controller import passive_controller
from sandbox.marginal_audit import audit_marginals
from sandbox.scenarios import reference_scenario
from sandbox.tariff_family import tariff_factory


def test_one_interval_audits_both_directions_with_finite_results():
    population = reference_scenario()
    rows = audit_marginals(
        passive_controller(),
        population,
        tariff_factory(),
        n_steps=1,
        key=jax.random.PRNGKey(4),
        sample_every=1,
        delta_kw=0.1,
    )

    assert len(rows) == 2 * population.num_agents
    assert {row["direction"] for row in rows} == {-1, 1}
    assert {row["agent_id"] for row in rows} == set(range(population.num_agents))
    numeric = {key for key in rows[0] if key not in {"agent_id", "pq_id", "step", "direction"}}
    assert all(math.isfinite(row[key]) for row in rows for key in numeric)
    assert all(abs(row["requested_delta_kw"]) <= 0.1 + 1e-6 for row in rows)

    # More inverter output raises connection-point export energy and reduces
    # transformer import (or increases transformer export); the reverse move
    # has the opposite signs whenever the feasibility bounds allow movement.
    for row in rows:
        if abs(row["requested_delta_kw"]) > 1e-6:
            assert row["realized_energy_delta_kwh"] * row["direction"] > 0.0
            assert row["transformer_delta_kw"] * row["direction"] < 0.0


def test_sampled_baselines_advance_with_environment_state():
    import numpy as np

    from sandbox.rollout import build_env, rollout

    population = reference_scenario()
    key = jax.random.PRNGKey(7)
    tariff = tariff_factory()
    rows = audit_marginals(
        passive_controller(), population, tariff, n_steps=3, key=key, sample_every=1
    )
    trajectory = rollout(
        passive_controller(),
        population,
        key,
        n_steps=3,
        env=build_env(population, time_limit=3, tariff=tariff),
    )
    # Float32 batched power-flow solves differ below a few watts from scan.
    for row in rows:
        np.testing.assert_allclose(
            row["baseline_transformer_kw"], trajectory.transformer_kw[row["step"]], atol=2e-3
        )
        np.testing.assert_allclose(
            row["baseline_settlement_chf"],
            trajectory.settlement_chf[row["step"], row["pq_id"]],
            atol=2e-5,
        )
