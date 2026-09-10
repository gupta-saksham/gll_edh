"""Focused contracts for the reusable controller family."""

import jax
import jax.numpy as jnp

from sandbox.controller_family import (
    POLICY_BANK,
    TUNE_OVER,
    FamilyMemory,
    family_controller,
    family_policy,
    init_family_memory,
)
from sandbox.observation import LocalObservation


def _obs(**updates) -> LocalObservation:
    values = dict(
        hour=12.0,
        time_sin=0.0,
        time_cos=-1.0,
        voltage_pu=1.0,
        p_grid_kw=0.0,
        p_load_kw=2.0,
        p_load_forecast_kw=2.0,
        pv_available_kw=6.0,
        soc_kwh=3.0,
        soc_headroom_kwh=3.0,
        bat_charge_max_kw=3.0,
        bat_discharge_max_kw=3.0,
        p_inv_min_kw=-3.0,
        p_inv_max_kw=9.0,
    )
    values.update(updates)
    return LocalObservation(**{k: jnp.float32(v) for k, v in values.items()})


def _run(policy_id, obs=None, carry=None, key=0):
    return family_policy(
        _obs() if obs is None else obs,
        init_family_memory() if carry is None else carry,
        {"policy_id": jnp.float32(policy_id)},
        jax.random.PRNGKey(key),
    )


def test_bank_is_compact_complete_and_contains_voltage_ablation() -> None:
    assert len(POLICY_BANK) <= 32
    assert TUNE_OVER["policy_id"] == [float(i) for i in range(len(POLICY_BANK))]
    keys = set(POLICY_BANK[0])
    assert all(set(policy) == keys for policy in POLICY_BANK)
    assert any(p["voltage_gain_kw_per_pu"] == 0.0 for p in POLICY_BANK)
    assert any(p["voltage_gain_kw_per_pu"] > 0.0 for p in POLICY_BANK)
    assert POLICY_BANK[int(family_controller().params["policy_id"])]["voltage_gain_kw_per_pu"] > 0


def test_policy_id_float_is_jittable_and_cast_to_integer() -> None:
    run = jax.jit(family_policy)
    action_float, _ = run(
        _obs(), init_family_memory(), {"policy_id": jnp.float32(13.9)}, jax.random.PRNGKey(0)
    )
    action_int, _ = run(
        _obs(), init_family_memory(), {"policy_id": jnp.float32(13.0)}, jax.random.PRNGKey(0)
    )
    assert jnp.allclose(action_float, action_int)


def test_voltage_trend_has_bounded_directional_effect() -> None:
    # Use post-initialization memory: the first sample initializes the EWMA and
    # intentionally has no voltage response.
    carry = FamilyMemory(
        offset_h=jnp.float32(0),
        voltage_ewma_pu=jnp.float32(1.0),
        intervals=jnp.int32(1),
    )
    high, _ = _run(14, _obs(voltage_pu=1.02), carry)
    low, _ = _run(14, _obs(voltage_pu=0.98), carry)
    off_high, _ = _run(3, _obs(voltage_pu=1.02), carry)
    off_low, _ = _run(3, _obs(voltage_pu=0.98), carry)
    assert high < low  # rising voltage absorbs more / exports less
    assert jnp.allclose(off_high, off_low)
    assert (low - high) <= 2.0 + 1e-6  # one kW saturation in each direction


def test_voltage_can_be_disabled_on_the_same_policy() -> None:
    carry = FamilyMemory(
        offset_h=jnp.float32(0),
        voltage_ewma_pu=jnp.float32(1.0),
        intervals=jnp.int32(1),
    )
    obs = _obs(voltage_pu=1.02)
    enabled, _ = family_policy(
        obs,
        carry,
        {"policy_id": jnp.float32(14), "voltage_enabled": jnp.float32(1)},
        jax.random.PRNGKey(0),
    )
    disabled, _ = family_policy(
        obs,
        carry,
        {"policy_id": jnp.float32(14), "voltage_enabled": jnp.float32(0)},
        jax.random.PRNGKey(0),
    )
    assert enabled < disabled


def test_first_voltage_sample_has_no_spurious_correction() -> None:
    high, memory = _run(14, _obs(voltage_pu=1.08))
    neutral, _ = _run(14, _obs(voltage_pu=1.0))
    assert jnp.allclose(high, neutral)
    assert jnp.allclose(memory.voltage_ewma_pu, 1.08)


def test_stagger_is_drawn_once_and_persists() -> None:
    _, first = _run(6, key=1)
    _, second = _run(6, carry=first, key=999)
    assert 0.0 <= first.offset_h <= 2.0
    assert jnp.array_equal(second.offset_h, first.offset_h)
    assert second.intervals == 2


def test_zero_asset_household_always_requests_zero() -> None:
    no_assets = _obs(
        pv_available_kw=0.0,
        soc_kwh=0.0,
        soc_headroom_kwh=0.0,
        bat_charge_max_kw=0.0,
        bat_discharge_max_kw=0.0,
        p_inv_min_kw=0.0,
        p_inv_max_kw=0.0,
    )
    for policy_id in (0, 15, 19, 20, 21):
        action, _ = _run(policy_id, no_assets)
        assert action == 0.0


def test_export_cap_works_for_pv_only_household() -> None:
    pv_only = _obs(
        p_load_forecast_kw=1.0,
        pv_available_kw=10.0,
        soc_kwh=0.0,
        soc_headroom_kwh=0.0,
        bat_charge_max_kw=0.0,
        bat_discharge_max_kw=0.0,
        p_inv_min_kw=0.0,
        p_inv_max_kw=10.0,
    )
    action, _ = _run(12, pv_only)
    assert jnp.allclose(action - pv_only.p_load_forecast_kw, 2.0)


def test_reserve_blocks_discharge_until_release() -> None:
    before, _ = _run(9, _obs(hour=16, pv_available_kw=0, p_load_forecast_kw=3,
                               soc_kwh=3, soc_headroom_kwh=3))
    after, _ = _run(9, _obs(hour=18, pv_available_kw=0, p_load_forecast_kw=3,
                              soc_kwh=3, soc_headroom_kwh=3))
    assert before == 0.0  # 50% of the 6 kWh capacity is reserved
    assert after == 3.0


def test_low_voltage_cannot_export_battery_without_permission() -> None:
    carry = FamilyMemory(
        offset_h=jnp.float32(0),
        voltage_ewma_pu=jnp.float32(1.0),
        intervals=jnp.int32(1),
    )
    obs = _obs(voltage_pu=0.97, pv_available_kw=0.0, p_load_forecast_kw=1.0)
    action, _ = _run(14, obs, carry)
    assert action <= obs.p_load_forecast_kw
