import jax.numpy as jnp
import numpy as np

from sandbox.observation import GridView
from sandbox.voltage_tariff import (
    DEFAULT_VOLTAGE_TARIFF_PARAMS,
    voltage_price,
    voltage_tariff,
    voltage_tariff_factory,
)


def _grid(energy, voltage, fair_leg):
    energy = jnp.asarray(energy, dtype=jnp.float32)
    n = energy.size
    return GridView(
        e_grid_kwh=energy,
        p_grid_kw=4.0 * energy,
        q_grid_kvar=jnp.zeros(n),
        voltage_pu=jnp.asarray(voltage, dtype=jnp.float32),
        transformer_kw=jnp.float32(0.0),
        transformer_kvar=jnp.float32(0.0),
        losses_kw=jnp.float32(0.0),
        hour=jnp.float32(12.0),
        fair_leg_chf=jnp.asarray(fair_leg, dtype=jnp.float32),
        has_inverter=jnp.ones(n, dtype=bool),
    )


def test_voltage_price_decreases_with_voltage_and_is_bounded():
    price = voltage_price(jnp.asarray([0.80, 0.95, 1.00, 1.05, 1.20]))

    assert np.all(np.diff(price) <= 0.0)
    np.testing.assert_allclose(price, [0.25, 0.225, 0.15, 0.075, 0.05], atol=1e-6)


def test_voltage_tariff_preserves_fair_leg_total_and_carry():
    grid = _grid(
        energy=[2.0, -3.0, 1.0, -1.0],
        voltage=[1.06, 0.96, 1.02, 0.99],
        fair_leg=[0.30, -0.80, 0.10, -0.20],
    )
    settlement, carry = voltage_tariff(grid, "carry", DEFAULT_VOLTAGE_TARIFF_PARAMS)

    np.testing.assert_allclose(settlement.sum(), grid.fair_leg_chf.sum(), atol=1e-7)
    assert carry == "carry"


def test_factory_is_available():
    assert callable(voltage_tariff_factory())
