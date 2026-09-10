import jax
import jax.numpy as jnp
import numpy as np

from sandbox.observation import GridView
from sandbox.tariff_family import (
    DEFAULT_TARIFF_PARAMS,
    SCENARIO_IDS,
    SENSITIVITY_WARMUP_INTERVALS,
    TARIFF_BANK,
    TARIFF_BANK_NAMES,
    family_tariff,
    family_tariff_factory,
    init_family_carry,
    scenario_params,
    stress_tariff,
    tariff_factory,
)


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


# ---------------------------------------------------------------------------
# The scenario bank
# ---------------------------------------------------------------------------

STEPS = 24
NUM_PQ = 18
STEP_H = 0.25


def _history(steps=STEPS, num_pq=NUM_PQ):
    """A synthetic day, time-major, so it scans exactly as a rollout settles.

    Six connection points with no inverter drawing a flat load, twelve
    exporting over a daylight bell, and a substation flow consistent with the
    sum of the two -- enough for every mechanism in the bank to fire.
    """
    hour = 6.0 + 0.75 * jnp.arange(steps, dtype=jnp.float32)
    daylight = jnp.clip(jnp.sin((hour - 6.0) / 12.0 * jnp.pi), 0.0, None)
    point = jnp.arange(num_pq, dtype=jnp.float32)
    has_inverter = point >= 6.0
    roof = jnp.where(has_inverter, 0.4 + 0.2 * (point - 6.0), 0.0)

    energy = daylight[:, None] * roof[None, :] - 0.2
    transformer_kw = -jnp.sum(energy, axis=1) / STEP_H
    return GridView(
        e_grid_kwh=energy,
        p_grid_kw=energy / STEP_H,
        q_grid_kvar=0.4 * jnp.abs(transformer_kw)[:, None] / num_pq * jnp.ones((1, num_pq)),
        voltage_pu=1.0 + 0.002 * point[None, :] + 0.03 * daylight[:, None],
        transformer_kw=transformer_kw,
        transformer_kvar=0.4 * jnp.abs(transformer_kw),
        losses_kw=0.03 * jnp.abs(transformer_kw),
        hour=hour,
        fair_leg_chf=0.12 * energy,
        has_inverter=jnp.broadcast_to(has_inverter, (steps, num_pq)),
    )


def _scan(params, views=None, num_pq=NUM_PQ):
    """Settle a whole history with the carry threaded, as the harness does."""
    views = _history() if views is None else views

    def settle(carry, view):
        settlement, carry = family_tariff(view, carry, params)
        return carry, settlement

    return jax.lax.scan(settle, init_family_carry(num_pq), views)


def _interval(views, index=0):
    return jax.tree_util.tree_map(lambda leaf: leaf[index], views)


def _charge_chf(settlement, views):
    """Each connection point's charge, read against one that pays nothing.

    Every scenario rebates its whole pool, so a charge is invisible in the
    total. Connection point 0 has no inverter, never exports and stays below
    the capacity band, so its deviation from fair LEG *is* its rebate -- and
    the difference recovers everybody else's charge.
    """
    deviation = settlement - views.fair_leg_chf
    return deviation[..., :1] - deviation


def test_bank_is_complete_named_and_contiguous():
    assert 18 <= len(TARIFF_BANK) <= 30
    assert len(set(TARIFF_BANK_NAMES)) == len(TARIFF_BANK)
    assert list(SCENARIO_IDS.values()) == list(range(len(TARIFF_BANK)))
    keys = set(TARIFF_BANK[0])
    assert all(set(entry) == keys for entry in TARIFF_BANK)
    assert TARIFF_BANK_NAMES[0] == "fair_leg_passthrough"
    for index, name in enumerate(TARIFF_BANK_NAMES):
        assert scenario_params(name) == scenario_params(index) == {"scenario_id": float(index)}


def test_unknown_scenario_is_refused():
    for missing in ("no_such_tariff", len(TARIFF_BANK)):
        try:
            scenario_params(missing)
        except KeyError:
            continue
        raise AssertionError(f"{missing!r} should not resolve to a scenario")


def test_every_scenario_settles_every_connection_point_with_a_stable_carry():
    views = _history()
    start = init_family_carry(NUM_PQ)
    for name in TARIFF_BANK_NAMES:
        carry, settlement = _scan(scenario_params(name), views)
        assert settlement.shape == (STEPS, NUM_PQ), name
        assert settlement.dtype == jnp.float32, name
        assert bool(jnp.all(jnp.isfinite(settlement))), name
        for field, value in carry.items():
            assert value.shape == start[field].shape, (name, field)
            assert value.dtype == start[field].dtype, (name, field)


def test_every_scenario_is_jittable():
    view = _interval(_history())
    settle = jax.jit(family_tariff)
    for name in TARIFF_BANK_NAMES:
        settlement, _ = settle(view, init_family_carry(NUM_PQ), scenario_params(name))
        assert settlement.shape == (NUM_PQ,), name


def test_bank_is_budget_neutral_by_construction():
    views = _history()
    reference = jnp.sum(views.fair_leg_chf, axis=1)
    for name in TARIFF_BANK_NAMES:
        if name == "subsidy_ablation":
            continue
        _, settlement = _scan(scenario_params(name))
        np.testing.assert_allclose(jnp.sum(settlement, axis=1), reference, atol=2e-5, err_msg=name)


def test_subsidy_ablation_is_deliberately_not_budget_neutral():
    views = _history()
    _, settlement = _scan(scenario_params("subsidy_ablation"), views)
    surplus = jnp.sum(settlement, axis=1) - jnp.sum(views.fair_leg_chf, axis=1)
    np.testing.assert_allclose(surplus, 0.02 * NUM_PQ, atol=1e-6)


def test_passthrough_is_exactly_fair_leg():
    views = _history()
    _, settlement = _scan(scenario_params("fair_leg_passthrough"), views)
    np.testing.assert_array_equal(settlement, views.fair_leg_chf)


def test_stress_entries_reproduce_the_standalone_stress_tariff():
    view = _interval(_history(), index=8)
    for name, threshold, strength in (
        ("stress_export_default", 30.0, 0.15),
        ("stress_export_tuned", 45.0, 0.30),
    ):
        family, _ = family_tariff(view, init_family_carry(NUM_PQ), scenario_params(name))
        standalone, _ = stress_tariff(
            view,
            None,
            {
                **DEFAULT_TARIFF_PARAMS,
                "export_threshold_kw": threshold,
                "export_strength_chf_per_kwh": strength,
            },
        )
        np.testing.assert_allclose(family, standalone, atol=1e-7, err_msg=name)


def test_scenario_id_is_cast_from_float_and_clipped():
    view = _interval(_history(), index=8)
    exact, _ = family_tariff(view, init_family_carry(NUM_PQ), {"scenario_id": 1.0})
    rounded, _ = family_tariff(view, init_family_carry(NUM_PQ), {"scenario_id": 1.9})
    beyond, _ = family_tariff(view, init_family_carry(NUM_PQ), {"scenario_id": 1e6})
    last, _ = family_tariff(view, init_family_carry(NUM_PQ), scenario_params(len(TARIFF_BANK) - 1))
    np.testing.assert_array_equal(exact, rounded)
    np.testing.assert_array_equal(beyond, last)


def test_demand_ratchet_charges_only_a_new_record():
    # One midday interval repeated, so every interval has the same export to
    # allocate a charge over, under a feeder loading that falls monotonically:
    # only the first interval can set a record, and a ratchet must ignore
    # every interval after it.
    midday = _interval(_history(), index=12)
    views = jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(leaf, (STEPS, *leaf.shape)), midday
    )
    falling = -(100.0 - 2.0 * jnp.arange(STEPS, dtype=jnp.float32))
    views = views.replace(transformer_kw=falling, transformer_kvar=jnp.zeros(STEPS))
    _, ratchet = _scan(scenario_params("kva_peak_ratchet"), views)
    _, level = _scan(scenario_params("kva_coincident_charge"), views)

    charged = jnp.sum(_charge_chf(ratchet, views), axis=1)
    np.testing.assert_allclose(charged[0], 1.00 * (100.0 - 60.0), atol=1e-4)
    np.testing.assert_allclose(charged[1:], 0.0, atol=1e-6)
    # The level charge keeps charging the same loading every interval.
    assert float(jnp.min(jnp.sum(_charge_chf(level, views)[:15], axis=1))) > 0.0


def test_smoothed_congestion_is_gentler_than_the_instantaneous_signal():
    views = _history()
    _, instant = _scan(scenario_params("stress_export_default"), views)
    _, smoothed = _scan(scenario_params("stress_smoothed"), views)
    instant_pool = jnp.sum(_charge_chf(instant, views), axis=1)
    smoothed_pool = jnp.sum(_charge_chf(smoothed, views), axis=1)
    assert float(jnp.max(smoothed_pool)) < float(jnp.max(instant_pool))
    assert float(jnp.sum(smoothed_pool)) > 0.0


def test_tenant_floor_is_a_static_class_transfer():
    views = _history()
    _, settlement = _scan(scenario_params("tenant_floor"), views)
    deviation = settlement - views.fair_leg_chf
    np.testing.assert_allclose(deviation[:, :6], 0.002, atol=1e-7)
    np.testing.assert_allclose(deviation[:, 6:], -0.002 * 6 / 12, atol=1e-7)

    # Static means static: the same transfer whatever the meters did.
    busier = views.replace(e_grid_kwh=2.0 * views.e_grid_kwh, p_grid_kw=2.0 * views.p_grid_kw)
    _, doubled = _scan(scenario_params("tenant_floor"), busier)
    np.testing.assert_allclose(doubled - views.fair_leg_chf, deviation, atol=1e-6)


def test_class_neutral_rebate_leaves_the_no_inverter_class_untouched():
    views = _history()
    _, settlement = _scan(scenario_params("stress_class_neutral"), views)
    np.testing.assert_allclose(settlement[:, :6], views.fair_leg_chf[:, :6], atol=1e-7)
    np.testing.assert_allclose(
        jnp.sum(settlement, axis=1), jnp.sum(views.fair_leg_chf, axis=1), atol=2e-5
    )


def test_own_peak_charge_is_local_where_the_aggregate_charge_is_not():
    views = _history()
    louder_energy = views.e_grid_kwh.at[:, 7].multiply(3.0)
    louder = views.replace(
        e_grid_kwh=louder_energy,
        p_grid_kw=louder_energy / STEP_H,
        transformer_kw=-jnp.sum(louder_energy, axis=1) / STEP_H,
    )
    for name, expect_local in (("own_peak_ratchet", True), ("stress_export_default", False)):
        _, quiet = _scan(scenario_params(name), views)
        _, noisy = _scan(scenario_params(name), louder)
        moved = float(
            jnp.max(jnp.abs(_charge_chf(quiet, views)[:, 8] - _charge_chf(noisy, louder)[:, 8]))
        )
        assert (moved < 1e-7) == expect_local, (name, moved)


def test_sensitivity_charge_waits_for_its_warm_up():
    assert SENSITIVITY_WARMUP_INTERVALS > STEPS
    views = _history()
    _, settlement = _scan(scenario_params("voltage_sensitivity"), views)
    np.testing.assert_array_equal(settlement, views.fair_leg_chf)


def test_family_factory_takes_a_name_an_index_or_parameters():
    for scenario in ("own_peak_ratchet", 3, scenario_params("stress_smoothed"), None):
        assert callable(family_tariff_factory(scenario))
