"""Settlement replay must reproduce actual tariff rollouts, including tenants."""

import jax
import numpy as np
import pandas as pd

from sandbox.controller import base_controller, passive_controller
from sandbox.experiments import (
    CREDIBLE_PARETO_OBJECTIVES,
    household_load_kwh,
    pareto_efficient_mask,
    resettle,
    revenue_screen,
)
from sandbox.rollout import build_env, rollout, rollout_seeds
from sandbox.scenarios import reference_scenario
from sandbox.tariff_family import family_tariff_factory, scenario_params

#: Long enough to reach the midday export window and the evening draw window,
#: so a replay error in the clock or in the carry has somewhere to show up.
STEPS = 88

#: One stateless scenario and three that carry state: an aggregate ratchet, a
#: per-connection-point average, and a smoothed congestion signal.
STATELESS = "loss_share_linear"
STATEFUL = ("kva_peak_ratchet", "own_deviation_charge", "stress_smoothed")


def _rollouts(scenario):
    population = reference_scenario()
    params = scenario_params(scenario)
    controller = base_controller()
    key = jax.random.PRNGKey(83)
    original = rollout(controller, population, key, n_steps=STEPS)
    live = rollout(
        controller,
        population,
        key,
        n_steps=STEPS,
        env=build_env(
            population, time_limit=STEPS, tariff=family_tariff_factory(params)
        ),
    )
    return population, params, original, live


def _reset_every_interval(trajectory, population, params):
    """Replay the way ``resettle`` used to: a fresh carry for every interval.

    Settling one-interval slices independently is exactly what mapping the
    tariff over the interval axis did, so this reproduces the bug rather than
    describing it.
    """
    one_interval = jax.tree_util.tree_map(lambda leaf: leaf[:, None], trajectory)
    settle = jax.vmap(lambda slice_: resettle(slice_, population, params).settlement_chf[0])
    return np.asarray(settle(one_interval))


def test_replay_matches_live_tariff_for_stateless_and_stateful_scenarios():
    for scenario in (STATELESS, "tou_midday_export", *STATEFUL):
        population, params, original, live = _rollouts(scenario)
        replay = resettle(original, population, params)
        np.testing.assert_allclose(
            replay.settlement_chf, live.settlement_chf, atol=2e-6, err_msg=scenario
        )
        np.testing.assert_allclose(
            replay.reward_chf, live.reward_chf, atol=2e-6, err_msg=scenario
        )
        # A tariff changes no physics, so the trajectory being replayed is the
        # same trajectory the live tariff settled.
        np.testing.assert_allclose(original.e_grid_kwh, live.e_grid_kwh, atol=2e-6)


def test_replaying_a_stateful_tariff_needs_the_carry_threaded():
    """The replay bug, pinned: state makes the per-interval reset wrong."""
    population, params, original, live = _rollouts(STATELESS)
    reset = _reset_every_interval(original, population, params)
    np.testing.assert_allclose(reset, live.settlement_chf, atol=2e-6)

    for scenario in STATEFUL:
        population, params, original, live = _rollouts(scenario)
        reset = _reset_every_interval(original, population, params)
        threaded = np.asarray(resettle(original, population, params).settlement_chf)
        np.testing.assert_allclose(threaded, live.settlement_chf, atol=2e-6, err_msg=scenario)
        assert np.abs(reset - np.asarray(live.settlement_chf)).max() > 1e-4, scenario


def test_fair_leg_replay_is_identity():
    sentinel = object()
    assert resettle(sentinel, None, None) is sentinel


def test_revenue_screen_passes_neutral_scenarios_and_fails_the_subsidy():
    population = reference_scenario()
    batch = rollout_seeds(
        base_controller(), population, jax.random.split(jax.random.PRNGKey(5), 2), n_steps=STEPS
    )
    candidates = {
        "fair_leg": None,
        "fair_leg_passthrough": scenario_params("fair_leg_passthrough"),
        "own_peak_ratchet": scenario_params("own_peak_ratchet"),
        "subsidy_ablation": scenario_params("subsidy_ablation"),
    }
    screen = revenue_screen(batch, population, candidates)
    assert screen["adequate"] == {
        "fair_leg": True,
        "fair_leg_passthrough": True,
        "own_peak_ratchet": True,
        "subsidy_ablation": False,
    }
    np.testing.assert_allclose(
        screen["settlement_chf"]["fair_leg_passthrough"],
        screen["settlement_chf"]["fair_leg"],
        atol=2e-4,
    )


def test_actual_load_is_independent_of_battery_dispatch():
    population = reference_scenario()
    key = jax.random.PRNGKey(49)
    base = rollout(base_controller(), population, key, n_steps=8)
    passive = rollout(passive_controller(), population, key, n_steps=8)
    load = household_load_kwh(base, population)
    np.testing.assert_allclose(load, household_load_kwh(passive, population), atol=2e-6)
    assert np.all(load > 0)


def test_pareto_mask_keeps_tradeoffs_and_equal_rows_but_drops_dominated_rows():
    frame = pd.DataFrame(
        {
            "transformer_export_peak_kw": [10.0, 9.0, 8.0, 9.0],
            "transformer_draw_peak_kw": [5.0, 5.0, 7.0, 5.0],
            "max_ramp_kw": [20.0, 20.0, 19.0, 20.0],
            "curtailed_share": [0.10, 0.10, 0.05, 0.10],
            "tenant_cost_delta_vs_fair_leg_chf_per_load_kwh": [0.0] * 4,
            "pv_only_cost_delta_vs_fair_leg_chf_per_load_kwh": [0.0] * 4,
            "pv_battery_cost_delta_vs_fair_leg_chf_per_load_kwh": [0.0] * 4,
            "large_flex_cost_delta_vs_fair_leg_chf_per_load_kwh": [0.0] * 4,
        }
    )

    np.testing.assert_array_equal(
        pareto_efficient_mask(frame),
        np.array([False, True, True, True]),
    )


def test_pareto_mask_rejects_missing_objective_values():
    frame = pd.DataFrame({objective: [0.0] for objective in CREDIBLE_PARETO_OBJECTIVES})
    frame.loc[0, "max_ramp_kw"] = np.nan

    with np.testing.assert_raises_regex(ValueError, "must all be finite"):
        pareto_efficient_mask(frame)


def test_fairness_tradeoff_keeps_a_network_dominated_controller_on_frontier():
    frame = pd.DataFrame(
        {objective: [0.0, 0.0] for objective in CREDIBLE_PARETO_OBJECTIVES}
    )
    frame.loc[0, "transformer_export_peak_kw"] = 9.0
    frame.loc[1, "transformer_export_peak_kw"] = 10.0
    frame.loc[0, "tenant_cost_delta_vs_fair_leg_chf_per_load_kwh"] = 0.02

    np.testing.assert_array_equal(pareto_efficient_mask(frame), [True, True])
