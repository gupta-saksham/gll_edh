"""A simple revenue-neutral local-voltage energy price.

The energy price falls linearly as a connection point's solved bus voltage
rises. Import and export use the same signed energy convention: positive grid
energy earns the local price and negative grid energy pays it. A separate equal
lump-sum balance makes the interval total exactly match fair LEG, so the price
experiment changes incidence without creating or destroying community revenue
on a fixed physical trajectory.

This intentionally tests the simplest voltage-level tariff. Voltage exposure is
not the same as marginal contribution, so the result is diagnostic rather than
a recommended production design.
"""

from typing import Any, Mapping

import chex
import jax.numpy as jnp

from sandbox.observation import GridView
from sandbox.tariff import tariff_from_settlement

DEFAULT_VOLTAGE_TARIFF_PARAMS = {
    "reference_voltage_pu": 1.0,
    "reference_price_chf_per_kwh": 0.15,
    "voltage_slope_chf_per_kwh_per_pu": 1.5,
    "minimum_price_chf_per_kwh": 0.05,
    "maximum_price_chf_per_kwh": 0.25,
}


def _params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {**DEFAULT_VOLTAGE_TARIFF_PARAMS, **params}


def voltage_price(
    voltage_pu: chex.Array, params: Mapping[str, Any] | None = None
) -> chex.Array:
    """Return the bounded CHF/kWh price, decreasing in local voltage."""
    p = _params(params or {})
    price = p["reference_price_chf_per_kwh"] - p[
        "voltage_slope_chf_per_kwh_per_pu"
    ] * (jnp.asarray(voltage_pu) - p["reference_voltage_pu"])
    return jnp.clip(
        price,
        p["minimum_price_chf_per_kwh"],
        p["maximum_price_chf_per_kwh"],
    )


def voltage_tariff(
    grid: GridView, carry: Any, params: Mapping[str, Any]
) -> tuple[chex.Array, Any]:
    """Settle signed grid energy at the local price plus an equal balance.

    The balance is independent of a connection point's own interval energy and
    reconciles the raw voltage-price settlement to fair LEG's interval total.
    """
    energy = jnp.asarray(grid.e_grid_kwh)
    raw_settlement_chf = voltage_price(grid.voltage_pu, params) * energy
    balance_chf = (
        jnp.sum(jnp.asarray(grid.fair_leg_chf)) - jnp.sum(raw_settlement_chf)
    ) / energy.size
    return raw_settlement_chf + balance_chf, carry


def voltage_tariff_factory(params: Mapping[str, Any] | None = None):
    """Build the stateless voltage-price tariff for an environment rollout."""
    return tariff_from_settlement(voltage_tariff, _params(params or {}))


__all__ = [
    "DEFAULT_VOLTAGE_TARIFF_PARAMS",
    "voltage_price",
    "voltage_tariff",
    "voltage_tariff_factory",
]
