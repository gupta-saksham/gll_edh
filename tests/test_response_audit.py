import jax
import jax.numpy as jnp
import pytest

from sandbox.controller import Controller, clip_to_feasible, init_memory
from sandbox.response_audit import audit_deviations, diagnose_policy
from sandbox.scenarios import reference_scenario


def _constant_controller() -> Controller:
    def constant(obs, carry, params, key):
        del key
        return clip_to_feasible(params["request_kw"], obs), carry

    return Controller(
        name="constant",
        fn=constant,
        params={"request_kw": jnp.float32(0.0)},
        init_carry=init_memory,
    )


def test_baseline_candidate_has_zero_unilateral_improvement() -> None:
    population = reference_scenario()
    records = audit_deviations(
        _constant_controller(),
        population,
        [{"request_kw": 0.0}],
        n_steps=2,
        seeds=1,
        key=jax.random.PRNGKey(7),
    )

    assert len(records) == population.num_agents
    assert [record.pq_id for record in records] == list(population.inverter_id)
    assert all(record.improvement_chf == pytest.approx(0.0, abs=1e-7) for record in records)
    assert not any(record.profitable for record in records)


def test_a_candidate_changes_only_the_deviator_policy_and_is_reproducible() -> None:
    population = reference_scenario()
    kwargs = dict(
        controller=_constant_controller(),
        population=population,
        candidate_params=[{"request_kw": 3.0}],
        n_steps=4,
        seeds=1,
        key=jax.random.PRNGKey(3),
    )
    first = audit_deviations(**kwargs)
    second = audit_deviations(**kwargs)

    assert first == second
    battery_records = [
        record
        for record, kind in zip(first, population.type_of_agent(), strict=True)
        if kind == "pv_battery"
    ]
    assert battery_records
    assert any(abs(record.improvement_chf) > 1e-6 for record in battery_records)


def test_audit_rejects_empty_candidate_list() -> None:
    with pytest.raises(ValueError, match="candidate_params"):
        audit_deviations(_constant_controller(), reference_scenario(), [])


def test_policy_diagnostics_track_local_state_and_endpoints() -> None:
    population = reference_scenario()
    diagnostics = diagnose_policy(
        _constant_controller(), population, n_steps=3, key=jax.random.PRNGKey(11)
    )

    assert diagnostics.soc_kwh.shape == (3, population.num_agents)
    assert diagnostics.projection_gap_kw.shape == (3, population.num_agents)
    assert diagnostics.battery_full.dtype == jnp.bool_
    assert jnp.allclose(diagnostics.final_stored_energy_kwh, diagnostics.soc_kwh[-1])
    assert diagnostics.initial_stored_energy_kwh.shape == (population.num_agents,)
    # The stock Memory exposes an EWMA, so the first trend is initialized to zero.
    assert jnp.allclose(diagnostics.voltage_trend_pu[0], 0.0)
