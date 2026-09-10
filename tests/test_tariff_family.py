import jax.numpy as jnp
import numpy as np

from sandbox.observation import GridView
from sandbox.tariff_family import DEFAULT_TARIFF_PARAMS, stress_tariff, tariff_factory


def _grid(*, energy, transformer_kw, fair_leg=None, has_inverter=None):
    energy = jnp.asarray(energy, dtype=jnp.float32)
    n = energy.size
    return GridView(
        e_grid_kwh=energy,
        p_grid_kw=4.0 * energy,
        q_grid_kvar=jnp.zeros(n),
        voltage_pu=jnp.ones(n),
        transformer_kw=jnp.float32(transformer_kw),
        transformer_kvar=jnp.float32(0.0),
        losses_kw=jnp.float32(0.0),
        hour=jnp.float32(12.0),
        fair_leg_chf=jnp.asarray(
            fair_leg if fair_leg is not None else jnp.zeros(n), dtype=jnp.float32
        ),
        has_inverter=jnp.asarray(
            has_inverter if has_inverter is not None else jnp.ones(n), dtype=bool
        ),
    )


def test_export_stress_charges_only_own_export_kwh_at_bounded_rate():
    grid = _grid(energy=[2.0, -3.0, 1.0], transformer_kw=-60.0)
    params = {
        **DEFAULT_TARIFF_PARAMS,
        "export_threshold_kw": 30.0,
        "export_stress_scale_kw": 20.0,
        "export_strength_chf_per_kwh": 0.2,
    }
    settlement, carry = stress_tariff(grid, "carry", params)

    # The ramp is saturated: penalties are [0.40, 0, 0.20] CHF and each
    # connection receives an equal 0.20 CHF rebate.
    np.testing.assert_allclose(settlement, [-0.2, 0.2, 0.0], atol=1e-7)
    assert carry == "carry"


def test_import_stress_charges_only_own_import_kwh_with_linear_ramp():
    grid = _grid(energy=[-2.0, 1.0, -1.0], transformer_kw=55.0)
    params = {
        **DEFAULT_TARIFF_PARAMS,
        "import_strength_chf_per_kwh": 0.3,
        "import_threshold_kw": 45.0,
        "import_stress_scale_kw": 20.0,
    }
    settlement, _ = stress_tariff(grid, None, params)

    # Half activation gives 0.15 CHF/kWh, hence [0.30, 0, 0.15] penalties.
    np.testing.assert_allclose(settlement, [-0.15, 0.15, 0.0], atol=1e-7)


def test_adjustment_conserves_fair_leg_total_and_rebate_is_equal():
    fair = jnp.asarray([0.7, -0.4, 0.1, -0.2])
    grid = _grid(energy=[3.0, 1.0, -2.0, 0.0], transformer_kw=-40.0, fair_leg=fair)
    settlement, _ = stress_tariff(grid, None, DEFAULT_TARIFF_PARAMS)
    adjustment = settlement - fair

    np.testing.assert_allclose(jnp.sum(settlement), jnp.sum(fair), atol=1e-7)
    # The importer and zero-flow household owe no penalty and get equal rebates.
    np.testing.assert_allclose(adjustment[2], adjustment[3], atol=1e-7)


def test_zero_directional_strength_is_exactly_fair_leg():
    fair = jnp.asarray([0.31, -0.27, 0.08])
    grid = _grid(energy=[5.0, -4.0, 2.0], transformer_kw=-100.0, fair_leg=fair)
    params = {
        **DEFAULT_TARIFF_PARAMS,
        "export_strength_chf_per_kwh": 0.0,
        "import_strength_chf_per_kwh": 0.0,
    }
    settlement, _ = stress_tariff(grid, None, params)
    np.testing.assert_array_equal(settlement, fair)


def test_explicit_tenant_exemption_uses_static_inverter_class():
    grid = _grid(
        energy=[-2.0, -2.0, 1.0],
        transformer_kw=100.0,
        has_inverter=[False, True, True],
    )
    params = {
        **DEFAULT_TARIFF_PARAMS,
        "import_strength_chf_per_kwh": 0.3,
        "tenant_penalty_exempt": True,
    }
    settlement, _ = stress_tariff(grid, None, params)

    # Only the inverter-equipped importer pays 0.60 CHF; all three receive
    # the same 0.20 CHF rebate, including the exempt tenant.
    np.testing.assert_allclose(settlement, [0.2, -0.4, 0.2], atol=1e-7)
    np.testing.assert_allclose(jnp.sum(settlement), 0.0, atol=1e-7)


def test_default_is_export_only_and_factory_is_available():
    grid = _grid(energy=[-3.0, -1.0], transformer_kw=100.0, fair_leg=[-0.4, -0.2])
    settlement, _ = stress_tariff(grid, None, DEFAULT_TARIFF_PARAMS)
    np.testing.assert_array_equal(settlement, grid.fair_leg_chf)
    assert callable(tariff_factory())
