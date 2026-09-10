"""Settlement replay must reproduce actual tariff rollouts, including tenants."""

import jax
import numpy as np

from sandbox.controller import base_controller, passive_controller
from sandbox.experiments import household_load_kwh, resettle
from sandbox.rollout import build_env, rollout
from sandbox.scenarios import reference_scenario
from sandbox.tariff_family import DEFAULT_TARIFF_PARAMS, tariff_factory


def test_replay_matches_live_tariff():
    population = reference_scenario()
    params = {
        **DEFAULT_TARIFF_PARAMS,
        "export_threshold_kw": 0.0,
        "import_threshold_kw": 0.0,
        "import_strength_chf_per_kwh": 0.1,
    }
    controller = base_controller()
    key = jax.random.PRNGKey(83)
    original = rollout(controller, population, key, n_steps=8)
    live = rollout(
        controller,
        population,
        key,
        n_steps=8,
        env=build_env(population, time_limit=8, tariff=tariff_factory(params)),
    )
    replay = resettle(original, population, params)
    np.testing.assert_allclose(replay.settlement_chf, live.settlement_chf, atol=2e-6)
    np.testing.assert_allclose(replay.reward_chf, live.reward_chf, atol=2e-6)
    np.testing.assert_allclose(original.e_grid_kwh, live.e_grid_kwh, atol=2e-6)
    np.testing.assert_allclose(
        replay.settlement_chf.sum(axis=1), original.settlement_chf.sum(axis=1), atol=2e-6
    )


def test_fair_leg_replay_is_identity():
    sentinel = object()
    assert resettle(sentinel, None, None) is sentinel


def test_actual_load_is_independent_of_battery_dispatch():
    population = reference_scenario()
    key = jax.random.PRNGKey(49)
    base = rollout(base_controller(), population, key, n_steps=8)
    passive = rollout(passive_controller(), population, key, n_steps=8)
    load = household_load_kwh(base, population)
    np.testing.assert_allclose(load, household_load_kwh(passive, population), atol=2e-6)
    assert np.all(load > 0)
