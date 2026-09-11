"""Compact, tariff-agnostic family of household battery policies.

The family exposes complete policy sets through one scalar ``policy_id``.  This
keeps tuning cheap and, more importantly, gives every tariff the same response
space.  Voltage feedback uses the previous interval's local voltage and a
persistent EWMA; it is therefore a delayed, bounded correction rather than a
claim to observe current feeder conditions.
"""

from __future__ import annotations

from typing import Any

import chex
import jax
import jax.numpy as jnp

from sandbox.controller import Controller, clip_to_feasible
from sandbox.observation import LocalObservation


EWMA_ALPHA = 1.0 / 24.0


@chex.dataclass(frozen=True)
class FamilyMemory:
    """Fixed-shape local state, including a once-drawn charging offset."""

    offset_h: chex.Array
    voltage_ewma_pu: chex.Array
    intervals: chex.Array


def init_family_memory() -> FamilyMemory:
    return FamilyMemory(
        offset_h=jnp.float32(0.0),
        voltage_ewma_pu=jnp.float32(1.0),
        intervals=jnp.int32(0),
    )


def _policy(
    name: str,
    *,
    active: float = 1.0,
    charge_start_h: float = 0.0,
    spread_h: float = 0.0,
    charge_fraction: float = 1.0,
    reserve_fraction: float = 0.0,
    release_h: float = 17.0,
    export_cap_kw: float = 1.0e3,
    voltage_gain_kw_per_pu: float = 0.0,
    voltage_deadband_pu: float = 0.005,
    voltage_max_kw: float = 1.0,
    exchange_mix: float = 1.0,
    grid_charge_fraction: float = 0.0,
    grid_charge_start_h: float = 0.0,
    grid_charge_end_h: float = 0.0,
    grid_charge_target_soc: float = 0.0,
    battery_export_fraction: float = 0.0,
    battery_export_start_h: float = 0.0,
    battery_export_end_h: float = 0.0,
) -> dict[str, Any]:
    return dict(locals())


# Deliberately small complete-policy bank.  Entries with voltage gain zero are
# explicit ablations; entries 13--18 exercise bounded voltage response.
POLICY_BANK: list[dict[str, Any]] = [
    _policy("passive_pv", active=0.0),
    _policy("self_consumption"),
    _policy("delay_09", charge_start_h=9.0),
    _policy("delay_11", charge_start_h=11.0),
    _policy("delay_13", charge_start_h=13.0),
    _policy("stagger_11_1h", charge_start_h=11.0, spread_h=1.0),
    _policy("stagger_11_2h", charge_start_h=11.0, spread_h=2.0),
    _policy("slow_stagger", charge_start_h=11.0, spread_h=2.0, charge_fraction=0.5),
    _policy("reserve_25", charge_start_h=11.0, reserve_fraction=0.25),
    _policy("reserve_50", charge_start_h=11.0, reserve_fraction=0.5),
    _policy("cap_8", charge_start_h=11.0, export_cap_kw=8.0),
    _policy("cap_4", charge_start_h=11.0, export_cap_kw=4.0),
    _policy("cap_2", charge_start_h=11.0, export_cap_kw=2.0),
    _policy("voltage_25", charge_start_h=11.0, voltage_gain_kw_per_pu=25.0),
    _policy("voltage_75", charge_start_h=11.0, voltage_gain_kw_per_pu=75.0),
    _policy("voltage_stagger", charge_start_h=11.0, spread_h=2.0, voltage_gain_kw_per_pu=25.0),
    _policy("voltage_reserve", charge_start_h=11.0, reserve_fraction=0.25, voltage_gain_kw_per_pu=25.0),
    _policy("voltage_cap", charge_start_h=11.0, export_cap_kw=4.0, voltage_gain_kw_per_pu=25.0),
    _policy("smooth", charge_start_h=11.0, spread_h=1.0, exchange_mix=0.5),
    _policy("grid_charge", grid_charge_fraction=0.5, grid_charge_start_h=1.0,
            grid_charge_end_h=5.0, grid_charge_target_soc=0.75),
    _policy("battery_export", reserve_fraction=0.25, battery_export_fraction=0.5,
            battery_export_start_h=17.0, battery_export_end_h=20.0),
    _policy("joint", charge_start_h=11.0, spread_h=2.0, charge_fraction=0.5,
            reserve_fraction=0.25, export_cap_kw=4.0,
            voltage_gain_kw_per_pu=25.0, exchange_mix=0.5),
]

policy_bank = POLICY_BANK  # convenient lower-case alias for exploratory notebooks
_NUMERIC_KEYS = tuple(k for k in POLICY_BANK[0] if k != "name")
_POLICY_TABLE = {
    k: jnp.asarray([p[k] for p in POLICY_BANK], dtype=jnp.float32)
    for k in _NUMERIC_KEYS
}


def _in_window(hour: chex.Array, start: chex.Array, end: chex.Array) -> chex.Array:
    """Half-open daily window, including windows crossing midnight."""
    ordinary = (hour >= start) & (hour < end)
    wrapped = (hour >= start) | (hour < end)
    return jnp.where(start <= end, ordinary, wrapped)


def family_policy(
    obs: LocalObservation,
    carry: FamilyMemory,
    params: dict[str, chex.Array],
    key: chex.PRNGKey,
) -> tuple[chex.Array, FamilyMemory]:
    """Apply one complete policy selected by the (possibly float) policy ID."""
    policy_id = jnp.clip(jnp.asarray(params["policy_id"], jnp.int32), 0, len(POLICY_BANK) - 1)
    p = {name: jnp.take(values, policy_id) for name, values in _POLICY_TABLE.items()}

    first = carry.intervals == 0
    offset_h = jnp.where(
        first,
        jax.random.uniform(key, (), minval=0.0, maxval=jnp.maximum(p["spread_h"], 1.0e-6)),
        carry.offset_h,
    )
    offset_h = jnp.where(p["spread_h"] > 0.0, offset_h, 0.0)
    voltage_baseline = jnp.where(first, obs.voltage_pu, carry.voltage_ewma_pu)

    capacity_kwh = obs.soc_kwh + obs.soc_headroom_kwh
    has_battery = capacity_kwh > 1.0e-6
    has_inverter = (obs.p_inv_min_kw != 0.0) | (obs.p_inv_max_kw != 0.0)
    surplus_kw = jnp.maximum(obs.pv_available_kw - obs.p_load_forecast_kw, 0.0)
    shortfall_kw = jnp.maximum(obs.p_load_forecast_kw - obs.pv_available_kw, 0.0)

    # A half-hour logistic transition avoids a fleet-wide hard edge.
    gate = jax.nn.sigmoid((obs.hour - p["charge_start_h"] - offset_h) / 0.5)
    solar_charge_ceiling = jnp.minimum(obs.bat_charge_max_kw, surplus_kw)
    charge_kw = p["charge_fraction"] * gate * solar_charge_ceiling

    before_release = obs.hour < p["release_h"]
    reserve_kwh = jnp.where(before_release, p["reserve_fraction"] * capacity_kwh, 0.0)
    # The observation's physical discharge bound already incorporates device
    # efficiency.  This energy bound only prevents crossing the policy reserve
    # within the simulator's documented 15-minute interval.
    # No battery efficiency is exposed to a controller.  Scale the already
    # physical discharge limit by the unreserved share of stored energy.  This
    # is conservative and avoids inventing a conversion efficiency here.
    unreserved_share = jnp.where(
        obs.soc_kwh > 1.0e-6,
        jnp.clip((obs.soc_kwh - reserve_kwh) / obs.soc_kwh, 0.0, 1.0),
        0.0,
    )
    reserve_limited_kw = obs.bat_discharge_max_kw * unreserved_share
    discharge_ceiling = jnp.minimum(obs.bat_discharge_max_kw, reserve_limited_kw)
    discharge_kw = jnp.minimum(shortfall_kw, discharge_ceiling)

    grid_window = _in_window(obs.hour, p["grid_charge_start_h"], p["grid_charge_end_h"])
    target_kwh = p["grid_charge_target_soc"] * capacity_kwh
    target_charge_kw = jnp.maximum(target_kwh - obs.soc_kwh, 0.0) / STEP_H
    grid_charge_kw = jnp.where(
        grid_window,
        jnp.minimum(obs.bat_charge_max_kw * p["grid_charge_fraction"], target_charge_kw),
        0.0,
    )
    charge_kw = jnp.maximum(charge_kw, grid_charge_kw)

    export_window = _in_window(obs.hour, p["battery_export_start_h"], p["battery_export_end_h"])
    export_discharge_kw = jnp.where(
        export_window,
        p["battery_export_fraction"] * jnp.maximum(discharge_ceiling - discharge_kw, 0.0),
        0.0,
    )
    discharge_kw = discharge_kw + export_discharge_kw

    trend_pu = obs.voltage_pu - voltage_baseline
    trend_outside_band = jnp.sign(trend_pu) * jnp.maximum(
        jnp.abs(trend_pu) - p["voltage_deadband_pu"], 0.0
    )
    voltage_kw = jnp.clip(
        p["voltage_gain_kw_per_pu"] * trend_outside_band,
        -p["voltage_max_kw"],
        p["voltage_max_kw"],
    )

    battery_intent_kw = charge_kw - discharge_kw + voltage_kw
    # Voltage may absorb local surplus, but cannot cause grid charging unless
    # that capability is explicitly enabled by the selected complete policy.
    charge_ceiling = jnp.maximum(solar_charge_ceiling, grid_charge_kw)
    allowed_discharge_kw = discharge_kw + export_discharge_kw
    battery_intent_kw = jnp.clip(battery_intent_kw, -allowed_discharge_kw, charge_ceiling)
    battery_intent_kw = jnp.where(has_battery, battery_intent_kw, 0.0)

    desired_grid_kw = obs.pv_available_kw - battery_intent_kw - obs.p_load_forecast_kw
    mixed_grid_kw = obs.p_grid_kw + p["exchange_mix"] * (desired_grid_kw - obs.p_grid_kw)
    # Smoothing remains inside the scheduled battery permissions.  The export
    # cap below is the sole intentional override: it may force extra absorption
    # or, once storage is unavailable, PV curtailment.
    min_scheduled_grid_kw = obs.pv_available_kw - obs.p_load_forecast_kw - charge_ceiling
    max_scheduled_grid_kw = obs.pv_available_kw - obs.p_load_forecast_kw + allowed_discharge_kw
    mixed_grid_kw = jnp.clip(mixed_grid_kw, min_scheduled_grid_kw, max_scheduled_grid_kw)
    capped_grid_kw = jnp.minimum(mixed_grid_kw, p["export_cap_kw"])
    scheduled_inv_kw = capped_grid_kw + obs.p_load_forecast_kw
    passive_inv_kw = obs.pv_available_kw
    request_kw = jnp.where(p["active"] > 0.5, scheduled_inv_kw, passive_inv_kw)
    request_kw = jnp.where(has_inverter, request_kw, 0.0)
    request_kw = clip_to_feasible(request_kw, obs)

    new_memory = FamilyMemory(
        offset_h=offset_h,
        voltage_ewma_pu=(1.0 - EWMA_ALPHA) * voltage_baseline + EWMA_ALPHA * obs.voltage_pu,
        intervals=carry.intervals + jnp.int32(1),
    )
    return request_kw, new_memory


def family_controller(policy_id: float = 15.0) -> Controller:
    """Bundle the family for rollout or tuning with ``policy_id``."""
    return Controller(
        name="controller_family",
        fn=family_policy,
        params={"policy_id": jnp.float32(policy_id)},
        init_carry=init_family_memory,
    )


TUNE_OVER = {"policy_id": [float(i) for i in range(len(POLICY_BANK))]}


__all__ = [
    "FamilyMemory",
    "POLICY_BANK",
    "TUNE_OVER",
    "family_controller",
    "family_policy",
    "init_family_memory",
    "policy_bank",
]
