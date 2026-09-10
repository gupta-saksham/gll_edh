"""Offline finite-action marginal settlement diagnostics.

Each counterfactual re-solves the unchanged environment state with one
household's inverter request perturbed.  Neighbour requests, exogenous data,
tariff state, and controller state are therefore identical to the baseline.
"""

from typing import Any, Optional

import chex
import jax
import jax.numpy as jnp
import numpy as np

from sandbox.controller import Controller
from sandbox.observation import to_action, to_local
from sandbox.rollout import TariffFactory, _tile_carry, build_env
from sandbox.scenarios import Population


def _transformer_kw(model: Any, state: Any) -> chex.Array:
    return model.grid.pu_to_kw(
        jnp.real(state.grid_state.bus_power_injection_pu)[model.grid.slack_id[0]]
    )


def audit_marginals(
    controller: Controller,
    population: Population,
    tariff: Optional[TariffFactory],
    n_steps: int = 96,
    key: Optional[chex.PRNGKey] = None,
    sample_every: int = 12,
    delta_kw: float = 0.1,
) -> list[dict[str, Any]]:
    """Audit one-interval settlement effects of feasible inverter perturbations.

    At every sampled interval, the baseline action and all plus/minus
    unilateral perturbations are stepped from the same prior environment
    state.  Only the baseline successor is used to continue the episode.
    Returned rows contain the acting agent and connection point, requested
    and realised changes, own whole-tariff settlement, and signed transformer
    flow plus its directional import/export components.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be at least one")
    if sample_every < 1:
        raise ValueError("sample_every must be at least one")
    if not np.isfinite(delta_kw) or delta_kw <= 0.0:
        raise ValueError("delta_kw must be finite and positive")

    env = build_env(population, time_limit=n_steps, tariff=tariff)
    model = env.environment
    num_agents = model.num_agents
    pq_ids = np.asarray(model.prosumer.inverter_id, dtype=int)
    per_agent = jax.vmap(controller.fn, in_axes=(0, 0, None, 0))

    episode_key = key if key is not None else jax.random.PRNGKey(0)
    reset_key, loop_key = jax.random.split(episode_key)
    state, timestep = env.reset(reset_key)
    carry = _tile_carry(controller.init_carry(), num_agents)
    rows: list[dict[str, Any]] = []

    # All counterfactuals at an interval share the same state.  Batching them
    # keeps a week-long sampled audit practical while preserving exact step
    # semantics for each action vector.
    step_batch = jax.jit(jax.vmap(lambda action: env.step(state, action)))

    for step in range(n_steps):
        loop_key, decide_key = jax.random.split(loop_key)
        local = to_local(model, timestep.observation, state)
        agent_keys = jax.random.split(decide_key, num_agents)
        actions_kw, next_carry = per_agent(local, carry, controller.params, agent_keys)

        if step % sample_every == 0:
            variants = [actions_kw]
            descriptors: list[tuple[int, int, float]] = []
            for agent_id in range(num_agents):
                for direction in (-1.0, 1.0):
                    perturbed = jnp.clip(
                        actions_kw[agent_id] + direction * delta_kw,
                        local.p_inv_min_kw[agent_id],
                        local.p_inv_max_kw[agent_id],
                    )
                    variants.append(actions_kw.at[agent_id].set(perturbed))
                    descriptors.append(
                        (agent_id, -1 if direction < 0 else 1, float(perturbed - actions_kw[agent_id]))
                    )

            normalized = jax.vmap(lambda action: to_action(model, action))(jnp.stack(variants))
            states, timesteps = step_batch(normalized)
            settlements = np.asarray(timesteps.extras["reward"].settlement_chf)
            energies = np.asarray(jnp.real(states.prosumer_state.s_pq_realized_kvah))
            transformer = np.asarray(jax.vmap(lambda s: _transformer_kw(model, s))(states))

            baseline_settlement = settlements[0]
            baseline_energy = energies[0]
            baseline_transformer = float(transformer[0])
            for variant_id, (agent_id, direction, requested_delta) in enumerate(
                descriptors, start=1
            ):
                pq_id = int(pq_ids[agent_id])
                counterfactual_transformer = float(transformer[variant_id])
                own_baseline = float(baseline_settlement[pq_id])
                own_counterfactual = float(settlements[variant_id, pq_id])
                rows.append(
                    {
                        "step": step,
                        "hour": float(local.hour[agent_id]),
                        "agent_id": agent_id,
                        "pq_id": pq_id,
                        "direction": direction,
                        "requested_delta_kw": requested_delta,
                        "baseline_action_kw": float(actions_kw[agent_id]),
                        "counterfactual_action_kw": float(
                            actions_kw[agent_id] + requested_delta
                        ),
                        "baseline_settlement_chf": own_baseline,
                        "counterfactual_settlement_chf": own_counterfactual,
                        "settlement_delta_chf": own_counterfactual - own_baseline,
                        "realized_energy_delta_kwh": float(
                            energies[variant_id, pq_id] - baseline_energy[pq_id]
                        ),
                        "baseline_transformer_kw": baseline_transformer,
                        "counterfactual_transformer_kw": counterfactual_transformer,
                        "transformer_delta_kw": counterfactual_transformer
                        - baseline_transformer,
                        "baseline_export_stress_kw": max(-baseline_transformer, 0.0),
                        "counterfactual_export_stress_kw": max(
                            -counterfactual_transformer, 0.0
                        ),
                        "baseline_import_stress_kw": max(baseline_transformer, 0.0),
                        "counterfactual_import_stress_kw": max(
                            counterfactual_transformer, 0.0
                        ),
                    }
                )

            # Variant zero is the true baseline step used to advance.
            state = jax.tree_util.tree_map(lambda leaf: leaf[0], states)
            timestep = jax.tree_util.tree_map(lambda leaf: leaf[0], timesteps)
        else:
            state, timestep = env.step(state, to_action(model, actions_kw))
        carry = next_carry

    return rows

