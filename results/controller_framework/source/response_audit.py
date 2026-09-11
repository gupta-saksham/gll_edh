# Copyright 2026 ewz - Zurich Municipal Electric Utility.
# Licensed under the Apache License, Version 2.0.

"""Offline checks for profitable unilateral controller-policy deviations.

This module is deliberately separate from :mod:`sandbox.rollout`.  The
official rollout broadcasts one parameter tree to every household; an
individual-response audit instead gives one agent candidate parameters while
its neighbours retain the controller's baseline parameters.  Environment
physics, feasibility projection, tariff settlement, and trajectory recording
are reused unchanged.
"""

from dataclasses import dataclass
from typing import Any, Optional

import chex
import jax
import jax.numpy as jnp

from sandbox.controller import Controller
from sandbox.observation import to_action, to_local
from sandbox.rollout import (
    TariffFactory,
    Trajectory,
    _record,
    _tile_carry,
    build_env,
    rollout,
)
from sandbox.scenarios import EPISODE_STEPS, Population


@dataclass(frozen=True)
class DeviationRecord:
    """Mean paired-seed result for one household and candidate policy."""

    agent_id: int
    pq_id: int
    candidate_index: int
    candidate_params: dict[str, Any]
    baseline_settlement_chf: float
    deviation_settlement_chf: float
    improvement_chf: float
    profitable: bool


@dataclass(frozen=True)
class PolicyDiagnostics:
    """Time-aligned local state and actuation diagnostics for one rollout."""

    trajectory: Trajectory
    soc_kwh: chex.Array
    soc_headroom_kwh: chex.Array
    battery_full: chex.Array
    voltage_pu: chex.Array
    voltage_trend_pu: chex.Array
    projection_gap_kw: chex.Array
    initial_stored_energy_kwh: chex.Array
    final_stored_energy_kwh: chex.Array


def _voltage_trend(carry: Any, voltage_pu: chex.Array) -> chex.Array:
    """Return the family's delayed voltage trend, or NaN for other carries."""
    baseline = getattr(carry, "voltage_ewma_pu", None)
    if baseline is None and isinstance(carry, dict):
        baseline = carry.get("voltage_ewma_pu")
    if baseline is None:
        return jnp.full_like(voltage_pu, jnp.nan)
    intervals = getattr(carry, "intervals", None)
    if intervals is None and isinstance(carry, dict):
        intervals = carry.get("intervals")
    if intervals is not None:
        baseline = jnp.where(intervals == 0, voltage_pu, baseline)
    return voltage_pu - baseline


def diagnose_policy(
    controller: Controller,
    population: Population,
    tariff: Optional[TariffFactory] = None,
    n_steps: int = EPISODE_STEPS,
    key: Optional[chex.PRNGKey] = None,
) -> PolicyDiagnostics:
    """Run a policy once and expose endpoint and per-interval local diagnostics.

    ``soc_kwh`` and ``soc_headroom_kwh`` describe the state after each action.
    Initial energy is captured before the first action and final energy after
    the last action.  ``projection_gap_kw`` is realised minus requested inverter
    power.  ``voltage_trend_pu`` is populated when the carry has a
    ``voltage_ewma_pu`` field; otherwise it is NaN.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be at least one")
    env = build_env(population, time_limit=n_steps, tariff=tariff)
    model = env.environment
    num_agents = model.num_agents
    per_agent = jax.vmap(controller.fn, in_axes=(0, 0, None, 0))
    episode_key = key if key is not None else jax.random.PRNGKey(0)
    reset_key, loop_key = jax.random.split(episode_key)
    state, timestep = env.reset(reset_key)
    carry = _tile_carry(controller.init_carry(), num_agents)
    initial_local = to_local(model, timestep.observation, state)

    def body(loop_state: tuple[Any, Any, Any, chex.PRNGKey], _: None):
        state, observation, carry, loop_key = loop_state
        loop_key, decide_key = jax.random.split(loop_key)
        local = to_local(model, observation, state)
        trend = _voltage_trend(carry, local.voltage_pu)
        keys = jax.random.split(decide_key, num_agents)
        actions, carry = per_agent(local, carry, controller.params, keys)
        new_state, new_timestep = env.step(state, to_action(model, actions))
        post_local = to_local(model, new_timestep.observation, new_state)
        record = _record(model, new_state, actions, new_timestep, local.pv_available_kw)
        diagnostics = {
            "soc_kwh": post_local.soc_kwh,
            "soc_headroom_kwh": post_local.soc_headroom_kwh,
            "battery_full": (post_local.soc_headroom_kwh <= 1.0e-4)
            & (post_local.soc_kwh + post_local.soc_headroom_kwh > 1.0e-4),
            "voltage_pu": local.voltage_pu,
            "voltage_trend_pu": trend,
            "projection_gap_kw": record["p_inv_realized_kw"] - actions,
        }
        return (new_state, new_timestep.observation, carry, loop_key), (record, diagnostics)

    (final_state, final_observation, _, _), (records, diagnostics) = jax.lax.scan(
        body, (state, timestep.observation, carry, loop_key), None, length=n_steps
    )
    final_local = to_local(model, final_observation, final_state)
    return PolicyDiagnostics(
        trajectory=Trajectory(**records),
        initial_stored_energy_kwh=initial_local.soc_kwh,
        final_stored_energy_kwh=final_local.soc_kwh,
        **diagnostics,
    )


def _agent_parameter_trees(
    baseline: dict[str, Any], candidate: dict[str, Any], deviating_agent: int, num_agents: int
) -> dict[str, chex.Array]:
    """Stack full parameter trees, replacing exactly one agent's leaves."""
    merged = {**baseline, **candidate}
    try:
        trees = [merged if i == deviating_agent else baseline for i in range(num_agents)]
        return jax.tree_util.tree_map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *trees)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "candidate parameters must preserve the keys and leaf shapes of controller.params"
        ) from exc


def _rollout_with_agent_params(
    controller: Controller,
    population: Population,
    key: chex.PRNGKey,
    params_by_agent: Any,
    n_steps: int,
    env: Any,
) -> Trajectory:
    """Experimental rollout whose only changed contract is per-agent params."""
    model = env.environment
    num_agents = model.num_agents
    per_agent = jax.vmap(controller.fn, in_axes=(0, 0, 0, 0))

    def decide(observation: Any, state: Any, carry: Any, decide_key: chex.PRNGKey):
        local = to_local(model, observation, state)
        keys = jax.random.split(decide_key, num_agents)
        actions, carry = per_agent(local, carry, params_by_agent, keys)
        return actions, carry, local

    reset_key, loop_key = jax.random.split(key)
    state, timestep = env.reset(reset_key)
    carry = _tile_carry(controller.init_carry(), num_agents)

    def body(loop_state: tuple[Any, Any, Any, chex.PRNGKey], _: None):
        state, observation, carry, loop_key = loop_state
        loop_key, decide_key = jax.random.split(loop_key)
        actions, carry, local = decide(observation, state, carry, decide_key)
        new_state, new_timestep = env.step(state, to_action(model, actions))
        record = _record(model, new_state, actions, new_timestep, local.pv_available_kw)
        return (new_state, new_timestep.observation, carry, loop_key), record

    _, records = jax.lax.scan(
        body, (state, timestep.observation, carry, loop_key), None, length=n_steps
    )
    return Trajectory(**records)


def audit_deviations(
    controller: Controller,
    population: Population,
    candidate_params: list[dict[str, Any]],
    tariff: Optional[TariffFactory] = None,
    n_steps: int = EPISODE_STEPS,
    seeds: int = 1,
    key: Optional[chex.PRNGKey] = None,
) -> list[DeviationRecord]:
    """Measure each inverter household's gain from each unilateral deviation.

    Candidate dictionaries override matching entries in ``controller.params``.
    Each comparison uses identical episode keys for the baseline and deviation.
    Settlement is read at the household's connection point, then averaged over
    seeds.  Positive improvement means the deviation paid that household more.
    """
    if not candidate_params:
        raise ValueError("candidate_params must contain at least one candidate")
    if seeds < 1:
        raise ValueError("seeds must be at least one")
    if n_steps < 1:
        raise ValueError("n_steps must be at least one")
    if not isinstance(controller.params, dict):
        raise TypeError("audit_deviations currently requires dict controller.params")

    env = build_env(population, time_limit=n_steps, tariff=tariff)
    episode_keys = jax.random.split(key if key is not None else jax.random.PRNGKey(0), seeds)
    baseline_totals = jax.vmap(
        lambda seed: rollout(
            controller, population, seed, n_steps=n_steps, env=env
        ).settlement_chf.sum(axis=0)
    )(episode_keys)
    variants = [
        _agent_parameter_trees(controller.params, candidate, agent, population.num_agents)
        for agent in range(population.num_agents)
        for candidate in candidate_params
    ]
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *variants)
    totals = jax.vmap(
        lambda params: jax.vmap(
            lambda seed: _rollout_with_agent_params(
                controller, population, seed, params, n_steps, env
            ).settlement_chf.sum(axis=0)
        )(episode_keys)
    )(stacked)

    records: list[DeviationRecord] = []
    for agent_id, pq_id in enumerate(population.inverter_id):
        baseline_mean = float(baseline_totals[:, pq_id].mean())
        for candidate_index, candidate in enumerate(candidate_params):
            variant = agent_id * len(candidate_params) + candidate_index
            deviation_mean = float(totals[variant, :, pq_id].mean())
            improvement = deviation_mean - baseline_mean
            records.append(
                DeviationRecord(
                    agent_id=agent_id,
                    pq_id=pq_id,
                    candidate_index=candidate_index,
                    candidate_params=dict(candidate),
                    baseline_settlement_chf=baseline_mean,
                    deviation_settlement_chf=deviation_mean,
                    improvement_chf=improvement,
                    profitable=improvement > 0.0,
                )
            )
    return records


__all__ = ["DeviationRecord", "PolicyDiagnostics", "audit_deviations", "diagnose_policy"]
