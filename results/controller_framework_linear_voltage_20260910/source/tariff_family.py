"""Budget-neutral directional transformer-stress tariffs.

The tariff keeps fair LEG as the energy settlement and redistributes a
directional stress penalty within each interval.  Transformer import is
positive while household export is positive, so each stress signal is paired
only with connection-point energy that contributes in the same direction.
"""

from typing import Any, Mapping

import chex
import jax.numpy as jnp

from sandbox.observation import GridView
from sandbox.tariff import tariff_from_settlement

DEFAULT_TARIFF_PARAMS = {
    "export_threshold_kw": 30.0,
    "export_stress_scale_kw": 20.0,
    "export_strength_chf_per_kwh": 0.15,
    "import_threshold_kw": 45.0,
    "import_stress_scale_kw": 20.0,
    "import_strength_chf_per_kwh": 0.0,
    "tenant_penalty_exempt": False,
}


def _params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return defaults with a caller's partial parameter mapping applied."""
    return {**DEFAULT_TARIFF_PARAMS, **params}


def _bounded_rate(
    stress_kw: chex.Array,
    threshold_kw: Any,
    scale_kw: Any,
    strength_chf_per_kwh: Any,
) -> chex.Array:
    """Smooth linear ramp from zero to the configured maximum rate."""
    scale = jnp.maximum(jnp.asarray(scale_kw), jnp.finfo(jnp.float32).eps)
    activation = jnp.clip((stress_kw - threshold_kw) / scale, 0.0, 1.0)
    return jnp.asarray(strength_chf_per_kwh) * activation


def stress_tariff(
    grid: GridView, carry: Any, params: Mapping[str, Any]
) -> tuple[chex.Array, Any]:
    """Settle fair LEG plus a budget-neutral transformer stress transfer.

    Rates are in CHF/kWh and apply to each connection point's own directional
    energy: positive ``e_grid_kwh`` under transformer export stress and
    negative ``e_grid_kwh`` under transformer import stress.  Collected
    penalties are returned as an equal, fixed rebate to every connection
    point.  If ``tenant_penalty_exempt`` is true, the static ``has_inverter``
    class determines penalty eligibility; exempt tenants still receive the
    equal rebate.
    """
    p = _params(params)
    export_rate = _bounded_rate(
        -jnp.asarray(grid.transformer_kw),
        p["export_threshold_kw"],
        p["export_stress_scale_kw"],
        p["export_strength_chf_per_kwh"],
    )
    import_rate = _bounded_rate(
        jnp.asarray(grid.transformer_kw),
        p["import_threshold_kw"],
        p["import_stress_scale_kw"],
        p["import_strength_chf_per_kwh"],
    )

    energy = jnp.asarray(grid.e_grid_kwh)
    penalty_chf = (
        export_rate * jnp.maximum(energy, 0.0)
        + import_rate * jnp.maximum(-energy, 0.0)
    )
    eligible = jnp.logical_or(
        jnp.logical_not(jnp.asarray(p["tenant_penalty_exempt"], dtype=bool)),
        jnp.asarray(grid.has_inverter, dtype=bool),
    )
    penalty_chf = jnp.where(eligible, penalty_chf, 0.0)

    # A fixed equal connection-point rebate is transparent and preserves the
    # fair-LEG total exactly on the same solved trajectory.
    rebate_chf = jnp.sum(penalty_chf) / energy.size
    settlement_chf = jnp.asarray(grid.fair_leg_chf) - penalty_chf + rebate_chf
    return settlement_chf, carry


def tariff_factory(params: Mapping[str, Any] | None = None):
    """Build the environment tariff factory for this stateless family."""
    return tariff_from_settlement(stress_tariff, _params(params or {}))

