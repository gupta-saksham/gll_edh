"""Budget-neutral tariff scenarios, selected by one scalar id.

Two things live here.  :func:`stress_tariff` is the original single mechanism:
fair LEG as the energy settlement plus a redistributed directional
transformer-stress penalty.  Transformer import is positive while household
export is positive, so each stress signal is paired only with
connection-point energy that contributes in the same direction.

:func:`family_tariff` generalises that into a bank of complete, named
scenarios -- the tariff-side counterpart of
:mod:`sandbox.controller_family`'s ``POLICY_BANK``.  Every entry is one
scalar ``scenario_id`` away, so the whole bank is a single compiled function
that a sweep can ``vmap`` across candidates instead of recompiling per
mechanism.

Three decisions worth stating, because they are not the obvious ones.

**One union carry, not one carry per mechanism.**  The harness has to declare
the carry's shape before the episode runs (see ``TARIFF_COOKBOOK.md``,
"Carrying state across intervals"), and the scenario is selected by a traced
id rather than at trace time, so a per-scenario carry type is not available.
:class:`FamilyTariffMemory` therefore covers every scenario's state at once
and every field is updated every interval whatever is selected.  A field's
*meaning* is scenario-dependent -- ``demand_peak_kva`` tracks apparent power
for one scenario and active power for another -- which is the price of the
single fixed shape.

**Budget neutrality by construction, not by hope.**  Every charge a scenario
levies is pooled and redistributed inside the same interval, so the interval
total is fair LEG's total exactly and the revenue-adequacy gate
(:func:`sandbox.metrics.revenue_adequate`) passes structurally rather than
empirically.  ``unfunded_credit_chf`` is the single deliberate exception, and
the only scenario that sets it is labelled an ablation.

**Charges are written against each connection point's own energy.**  A charge
allocated pro rata over everybody contributing to an aggregate excess has a
marginal exposure an order of magnitude below its headline rate, which is the
cookbook's "Scale, and why nothing happened".  Where a scenario does allocate
an aggregate cost pro rata (the demand charges, the loss share) that is
because the cost itself is genuinely joint, and the resulting marginal rate is
reported rather than assumed.
"""

from typing import Any, Mapping, Union

import chex
import jax.numpy as jnp

from sandbox.observation import GridView
from sandbox.scenarios import step_duration_h
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


# ---------------------------------------------------------------------------
# The scenario bank
# ---------------------------------------------------------------------------

#: Interval length in hours, needed to price ``losses_kw`` as energy.
STEP_H = step_duration_h()

#: Averaging window of the smoothed congestion signal, two hours.  A published
#: averaging period, the way a real demand charge has one -- not a filter
#: tuned per scenario, which would make the carry field's meaning vary again.
STRESS_EWMA_ALPHA = 1.0 / 8.0

#: Averaging window of each connection point's own energy, six hours.
OWN_EWMA_ALPHA = 1.0 / 24.0

#: A connection point is charged per 1 % of voltage above the trigger.
VOLTAGE_STEP_PU = 0.01

#: The sensitivity regression is ignored for its first day: with a handful of
#: intervals the normal equations are dominated by whatever the weather did.
SENSITIVITY_WARMUP_INTERVALS = 96

#: No connection point may be charged more than this multiple of the mean
#: estimated sensitivity, so one badly conditioned fit cannot dominate.
SENSITIVITY_INDEX_CAP = 3.0

_EPS = 1.0e-9


@chex.dataclass(frozen=True)
class FamilyTariffMemory:
    """Fixed-shape state covering every scenario in the bank.

    Attributes:
        intervals: Settled intervals so far.
        export_stress_ewma_kw: Smoothed transformer export stress.
        import_stress_ewma_kw: Smoothed transformer draw stress.
        demand_peak_kva: Running peak of the demand-charge base -- apparent
            or active power at the substation, depending on the scenario.
        own_peak_kw: (num_pq,) Running peak of each connection point's own
            absolute exchange, the base of the per-connection demand charge
            and of the published rate class.
        own_energy_ewma_kwh: (num_pq,) Each connection point's own smoothed
            energy, the reference its deviation charge is measured against.
        sensitivity_count: Intervals accumulated into the regression below.
        sum_energy_kwh: (num_pq,) Regression accumulator, own energy.
        sum_voltage_pu: (num_pq,) Own voltage, offset by nominal.
        sum_energy_sq: (num_pq,) Own energy squared.
        sum_energy_voltage: (num_pq,) Own energy times own voltage.
        sum_energy_transformer: (num_pq,) Own energy times transformer power.
        sum_transformer_voltage: (num_pq,) Transformer power times own voltage.
        sum_transformer_kw: Transformer power.
        sum_transformer_sq: Transformer power squared.
    """

    intervals: chex.Array
    export_stress_ewma_kw: chex.Array
    import_stress_ewma_kw: chex.Array
    demand_peak_kva: chex.Array
    own_peak_kw: chex.Array
    own_energy_ewma_kwh: chex.Array
    sensitivity_count: chex.Array
    sum_energy_kwh: chex.Array
    sum_voltage_pu: chex.Array
    sum_energy_sq: chex.Array
    sum_energy_voltage: chex.Array
    sum_energy_transformer: chex.Array
    sum_transformer_voltage: chex.Array
    sum_transformer_kw: chex.Array
    sum_transformer_sq: chex.Array


#: Connection points on the reference feeder.  The carry needs a shape before
#: the episode runs, so it needs this number; :func:`family_tariff_factory`
#: reads it off the prosumer model instead of trusting the default.
NUM_PQ = 18


def init_family_carry(num_pq: int = NUM_PQ) -> FamilyTariffMemory:
    """The starting carry -- one per tariff, with a connection-point axis."""
    zeros = jnp.zeros((num_pq,), dtype=jnp.float32)
    return FamilyTariffMemory(
        intervals=jnp.int32(0),
        export_stress_ewma_kw=jnp.float32(0.0),
        import_stress_ewma_kw=jnp.float32(0.0),
        demand_peak_kva=jnp.float32(0.0),
        own_peak_kw=zeros,
        own_energy_ewma_kwh=zeros,
        sensitivity_count=jnp.float32(0.0),
        sum_energy_kwh=zeros,
        sum_voltage_pu=zeros,
        sum_energy_sq=zeros,
        sum_energy_voltage=zeros,
        sum_energy_transformer=zeros,
        sum_transformer_voltage=zeros,
        sum_transformer_kw=jnp.float32(0.0),
        sum_transformer_sq=jnp.float32(0.0),
    )


def _scenario(
    name: str,
    *,
    fair_leg_weight: float = 1.0,
    export_threshold_kw: float = 30.0,
    export_stress_scale_kw: float = 20.0,
    export_strength_chf_per_kwh: float = 0.0,
    import_threshold_kw: float = 45.0,
    import_stress_scale_kw: float = 20.0,
    import_strength_chf_per_kwh: float = 0.0,
    stress_use_ewma: float = 0.0,
    tou_export_rate_chf_per_kwh: float = 0.0,
    tou_export_start_h: float = 10.0,
    tou_export_end_h: float = 16.0,
    tou_stagger_h: float = 0.0,
    tou_import_rate_chf_per_kwh: float = 0.0,
    tou_import_start_h: float = 17.0,
    tou_import_end_h: float = 21.0,
    demand_rate_chf_per_kva: float = 0.0,
    demand_threshold_kva: float = 60.0,
    demand_use_apparent: float = 1.0,
    demand_ratchet: float = 0.0,
    loss_price_chf_per_kwh: float = 0.0,
    loss_share_exponent: float = 1.0,
    voltage_level_rate_chf_per_kwh: float = 0.0,
    voltage_level_threshold_pu: float = 1.03,
    sensitivity_rate_chf_per_kwh: float = 0.0,
    own_peak_rate_chf_per_kw: float = 0.0,
    own_peak_threshold_kw: float = 3.0,
    own_deviation_rate_chf_per_kwh: float = 0.0,
    reverse_rate_chf_per_kwh: float = 0.0,
    throughput_rate_chf_per_kwh: float = 0.0,
    tenant_floor_chf: float = 0.0,
    rebate_to_tenants: float = 0.0,
    rebate_to_owners: float = 0.0,
    tenant_penalty_exempt: float = 0.0,
    unfunded_credit_chf: float = 0.0,
) -> dict[str, Any]:
    return dict(locals())


# Every entry is a complete tariff.  Each one names the network cost it
# internalises and the signal a household could anticipate it from; a scenario
# that cannot answer both questions does not belong in the bank.  Entries
# marked ABLATION are here to be beaten.
TARIFF_BANK: list[dict[str, Any]] = [
    # ABLATION / control: the bank's own null hypothesis.  No transfer at all.
    _scenario("fair_leg_passthrough"),
    # Transformer thermal duty above a declared export threshold.  Anticipable
    # from the clock and the weather: on this feeder the threshold is crossed
    # between 09:00 and 16:15 on sunny days and never otherwise.
    _scenario("stress_export_default", export_strength_chf_per_kwh=0.15),
    _scenario(
        "stress_export_tuned",
        export_threshold_kw=45.0,
        export_strength_chf_per_kwh=0.30,
    ),
    # ABLATION: the same aggregate mechanism priced to bite.  Expected to buy
    # the export peak with curtailment and a worse coincidence factor -- the
    # failure the cookbook describes, run deliberately.
    _scenario(
        "stress_export_biting",
        export_threshold_kw=20.0,
        export_strength_chf_per_kwh=0.60,
    ),
    # Both directions.  The draw side is universal here, so it also charges
    # the six tenants, who cannot respond: the incidence ablation of the entry
    # below rather than a recommendation.
    _scenario(
        "stress_bidirectional",
        export_strength_chf_per_kwh=0.15,
        import_threshold_kw=5.0,
        import_stress_scale_kw=5.0,
        import_strength_chf_per_kwh=0.15,
    ),
    _scenario(
        "stress_tenant_exempt",
        export_strength_chf_per_kwh=0.15,
        import_threshold_kw=5.0,
        import_stress_scale_kw=5.0,
        import_strength_chf_per_kwh=0.15,
        tenant_penalty_exempt=1.0,
    ),
    # The same congestion cost read through a two-hour average, so the price
    # does not step at one interval and a population tuned against it is not
    # handed a single instant to react to.
    _scenario(
        "stress_smoothed",
        export_strength_chf_per_kwh=0.15,
        stress_use_ewma=1.0,
    ),
    # Published clock.  Midday export window: the most anticipable signal
    # there is, and it stands for the same transformer duty without waiting
    # for the feeder to be in trouble first.
    _scenario("tou_midday_export", tou_export_rate_chf_per_kwh=0.06),
    # Evening draw window, tenant-exempt by rate class: the cost is the
    # evening draw peak, and the households charged for it are the ones with
    # an inverter that could shift into it.
    _scenario(
        "tou_evening_import",
        tou_import_rate_chf_per_kwh=0.06,
        tenant_penalty_exempt=1.0,
    ),
    # Substation apparent-power loading, which is what a transformer's kVA
    # rating actually limits.  Charged whenever the feeder is above the
    # threshold, allocated pro rata to flow in the straining direction.
    _scenario("kva_coincident_charge", demand_rate_chf_per_kva=0.10),
    # The same charge with the pool returned inside the class that pays it.
    # Measured on a fixed trajectory the two rate classes are then left whole
    # -- no group's cost per kWh moves by a rappen -- while the charge still
    # separates the households inside the paying class by contribution.
    _scenario(
        "kva_charge_class_neutral",
        demand_rate_chf_per_kva=0.10,
        rebate_to_owners=1.0,
    ),
    # The same cost as a ratchet: only a new record costs anything, which is
    # the shape a real demand charge has.  1 CHF per kVA of new peak is
    # conservative against a real Swiss demand rate of 5-12 CHF/kW/month.
    _scenario(
        "kva_peak_ratchet",
        demand_rate_chf_per_kva=1.00,
        demand_ratchet=1.0,
    ),
    # ABLATION of the entry above: active power instead of apparent, to test
    # whether pricing kVA rather than kW makes any difference here.
    _scenario(
        "kw_peak_ratchet",
        demand_rate_chf_per_kva=1.00,
        demand_threshold_kva=45.0,
        demand_use_apparent=0.0,
        demand_ratchet=1.0,
    ),
    # Network losses, at the energy price they are bought at.  Caused by
    # everybody together, so shared out pro rata rather than charged to
    # whoever happened to be at the end of the line.
    _scenario("loss_share_linear", loss_price_chf_per_kwh=0.15),
    # Losses are quadratic in flow, so a square-weighted share is closer to
    # each point's marginal contribution than a linear one.
    _scenario(
        "loss_share_quadratic",
        loss_price_chf_per_kwh=0.15,
        loss_share_exponent=2.0,
    ),
    # ABLATION: the tariff the cookbook warns against, run on purpose.  It
    # prices the voltage LEVEL, which is mostly made by other households, so
    # it charges exposure rather than contribution.
    _scenario("voltage_level_exposure", voltage_level_rate_chf_per_kwh=0.02),
    # The locational price done the way the cookbook asks for: charge the
    # SENSITIVITY of the binding quantity to this point's own injection,
    # estimated in the carry from own voltage against own energy while
    # controlling for the feeder aggregate.  See the note on
    # :func:`_sensitivity_index`: on this feeder the estimate does not recover
    # electrical distance, and the scenario is here to show that.
    _scenario(
        "voltage_sensitivity",
        sensitivity_rate_chf_per_kwh=0.06,
    ),
    # Differentiation manufactured where the physics provides none: the same
    # midday window, shifted by up to three hours according to a published
    # rate class read off each connection point's own peak.  Every household
    # can compute its own class from its own meter, and the fleet no longer
    # faces one common signal at one common instant.
    _scenario(
        "staggered_class_tou",
        tou_export_rate_chf_per_kwh=0.06,
        tou_export_start_h=9.0,
        tou_export_end_h=13.0,
        tou_stagger_h=3.0,
    ),
    # Each connection point's own capacity: the service cable and fuse are
    # sized on the individual peak, and the coincidence factor is what links
    # individual peaks to feeder sizing.  Perfectly anticipable -- a household
    # knows its own record without asking anybody.
    _scenario("own_peak_ratchet", own_peak_rate_chf_per_kw=0.30),
    # Own peakiness against own recent average.  Also purely local, and the
    # only entry that charges a household for the shape of its profile rather
    # than for its coincidence with everybody else's.
    _scenario("own_deviation_charge", own_deviation_rate_chf_per_kwh=0.03),
    # Reverse-flow protection: relay coordination and transformer models on an
    # urban LV feeder assume one direction.  Triggered by the sign of the
    # substation flow rather than by a magnitude threshold.
    _scenario("reverse_flow_charge", reverse_rate_chf_per_kwh=0.04),
    # Two-part tariff, volumetric-heavy: a throughput charge on energy in
    # either direction, returned as an equal fixed credit.  Conductor thermal
    # ageing tracks current, not its direction.
    _scenario("two_part_volumetric", throughput_rate_chf_per_kwh=0.02),
    # ABLATION: the same restructuring in reverse -- a per-kWh credit funded
    # by an equal fixed charge.  A network's costs really are mostly fixed,
    # and this is what recovering them fixed does to the marginal signal.
    _scenario("two_part_fixed", throughput_rate_chf_per_kwh=-0.02),
    # A static rate-class floor for the six connection points with no
    # inverter, funded inside the class that has one.  The cost internalised
    # is the cost-base erosion the fairness table measures: self-supply
    # shrinks what the network's charges are recovered from, and what is left
    # is the households that cannot self-supply.
    _scenario("tenant_floor", tenant_floor_chf=0.002),
    # The congestion charge funding that floor instead of being rebated
    # equally: the households causing the strain pay the households that
    # cannot avoid it.
    _scenario(
        "stress_funds_tenant_floor",
        export_strength_chf_per_kwh=0.15,
        rebate_to_tenants=1.0,
    ),
    # The shipped mechanism with only its funding rule changed: the same
    # congestion charge on the same quantity, rebated inside the class that
    # pays it.  Isolates how much of that tariff's incidence is the charge
    # and how much is where the money went.
    _scenario(
        "stress_class_neutral",
        export_strength_chf_per_kwh=0.15,
        rebate_to_owners=1.0,
    ),
    # The composite, in the spirit of the controller bank's `joint`: a
    # congestion charge, a per-connection capacity charge, a loss share, and
    # the tenant floor, each at the magnitude it carries alone.
    _scenario(
        "joint_system_cost",
        export_strength_chf_per_kwh=0.15,
        own_peak_rate_chf_per_kw=0.30,
        loss_price_chf_per_kwh=0.15,
        loss_share_exponent=2.0,
        tenant_floor_chf=0.002,
    ),
    # ABLATION: not budget neutral, on purpose.  An unfunded credit to every
    # connection point every interval -- a subsidy dressed as a price, and the
    # thing the revenue-adequacy gate exists to disqualify.
    _scenario("subsidy_ablation", unfunded_credit_chf=0.02),
]

tariff_bank = TARIFF_BANK  # convenient lower-case alias for exploratory notebooks
TARIFF_BANK_NAMES: tuple[str, ...] = tuple(entry["name"] for entry in TARIFF_BANK)
SCENARIO_IDS: dict[str, int] = {name: i for i, name in enumerate(TARIFF_BANK_NAMES)}
_SCENARIO_KEYS = tuple(k for k in TARIFF_BANK[0] if k != "name")
_SCENARIO_TABLE = {
    k: jnp.asarray([entry[k] for entry in TARIFF_BANK], dtype=jnp.float32) for k in _SCENARIO_KEYS
}


def scenario_params(scenario: Union[str, int]) -> dict[str, float]:
    """The tariff parameters selecting one bank entry, by name or by index."""
    index = SCENARIO_IDS[scenario] if isinstance(scenario, str) else int(scenario)
    if not 0 <= index < len(TARIFF_BANK):
        raise KeyError(f"no tariff scenario {scenario!r}; the bank holds {len(TARIFF_BANK)}")
    return {"scenario_id": float(index)}


def _in_window(hour: chex.Array, start: chex.Array, end: chex.Array) -> chex.Array:
    """Half-open daily window, including windows crossing midnight."""
    ordinary = (hour >= start) & (hour < end)
    wrapped = (hour >= start) | (hour < end)
    return jnp.where(start <= end, ordinary, wrapped)


def _share(weight: chex.Array) -> chex.Array:
    """Normalise nonnegative weights, and hand out nothing when they vanish."""
    total = jnp.sum(weight)
    return jnp.where(total > _EPS, weight / jnp.maximum(total, _EPS), 0.0)


def _sensitivity_index(carry: FamilyTariffMemory) -> chex.Array:
    """Relative own-injection voltage sensitivity, estimated from the carry.

    The partial slope of own bus voltage on own energy, controlling for the
    substation flow, accumulated over the episode and normalised to mean one
    across the points with a positive slope. Controlling for the aggregate is
    what makes it a *partial* effect: the question a locational price needs is
    how much this bus moves per kWh of THIS point's injection, not how much it
    moves when the whole feeder exports at noon.

    **Measured on the reference feeder, it does not work, and that is the
    finding.** The physical sensitivity spans 0.0003 pu/kWh at the connection
    point nearest the transformer to 0.0228 at the furthest, monotone in
    electrical distance. This estimator returns 0.003 to 0.015 pu/kWh with a
    correlation against distance rank of about -0.19, and turns negative at
    two of the four ``large_flex`` points: settled data alone does not
    identify it, because every injection on the feeder moves at once and the
    idiosyncratic variation left over is smaller than the load noise. A
    tariff that wants a genuinely nodal price needs a field the seam does not
    publish -- a per-point voltage sensitivity, per-branch flows, or the
    power-flow Jacobian. The scenario built on this is kept so the experiment
    can measure that failure instead of a paragraph asserting it.
    """
    count = jnp.maximum(carry.sensitivity_count, 1.0)
    mean_energy = carry.sum_energy_kwh / count
    mean_flow = carry.sum_transformer_kw / count
    mean_voltage = carry.sum_voltage_pu / count
    energy_energy = carry.sum_energy_sq - carry.sum_energy_kwh * mean_energy
    energy_flow = carry.sum_energy_transformer - carry.sum_energy_kwh * mean_flow
    flow_flow = carry.sum_transformer_sq - carry.sum_transformer_kw * mean_flow
    energy_voltage = carry.sum_energy_voltage - carry.sum_energy_kwh * mean_voltage
    flow_voltage = carry.sum_transformer_voltage - carry.sum_transformer_kw * mean_voltage

    determinant = energy_energy * flow_flow - energy_flow**2
    slope = jnp.where(
        jnp.abs(determinant) > _EPS,
        (energy_voltage * flow_flow - flow_voltage * energy_flow) / determinant,
        0.0,
    )
    positive = jnp.maximum(slope, 0.0)
    mean = jnp.sum(positive) / jnp.maximum(jnp.sum(positive > 0.0), 1.0)
    index = jnp.clip(positive / jnp.maximum(mean, _EPS), 0.0, SENSITIVITY_INDEX_CAP)
    ready = carry.sensitivity_count >= SENSITIVITY_WARMUP_INTERVALS
    return jnp.where(ready, index, 0.0)


def family_tariff(
    grid: GridView, carry: FamilyTariffMemory, params: Mapping[str, Any]
) -> tuple[chex.Array, FamilyTariffMemory]:
    """Settle one interval under the complete scenario selected by ``scenario_id``.

    Every scenario returns ``fair LEG - charge + rebate``, where the rebate
    redistributes the interval's whole charge pool, so the settlement sums to
    fair LEG's own total by construction. The only exception is the scenario
    that sets ``unfunded_credit_chf``, which is the revenue-adequacy ablation.
    """
    scenario_id = jnp.clip(jnp.asarray(params["scenario_id"], jnp.int32), 0, len(TARIFF_BANK) - 1)
    p = {name: jnp.take(values, scenario_id) for name, values in _SCENARIO_TABLE.items()}

    energy_kwh = jnp.asarray(grid.e_grid_kwh, dtype=jnp.float32)
    power_kw = jnp.asarray(grid.p_grid_kw, dtype=jnp.float32)
    voltage_pu = jnp.asarray(grid.voltage_pu, dtype=jnp.float32)
    transformer_kw = jnp.asarray(grid.transformer_kw, dtype=jnp.float32)
    has_inverter = jnp.asarray(grid.has_inverter, dtype=bool)
    num_pq = energy_kwh.size
    exported_kwh = jnp.maximum(energy_kwh, 0.0)
    imported_kwh = jnp.maximum(-energy_kwh, 0.0)

    # -- directional transformer stress ------------------------------------
    export_stress_kw = jnp.maximum(-transformer_kw, 0.0)
    import_stress_kw = jnp.maximum(transformer_kw, 0.0)
    export_ewma_kw = (
        1.0 - STRESS_EWMA_ALPHA
    ) * carry.export_stress_ewma_kw + STRESS_EWMA_ALPHA * export_stress_kw
    import_ewma_kw = (
        1.0 - STRESS_EWMA_ALPHA
    ) * carry.import_stress_ewma_kw + STRESS_EWMA_ALPHA * import_stress_kw
    smoothed = p["stress_use_ewma"] > 0.5
    export_signal_kw = jnp.where(smoothed, export_ewma_kw, export_stress_kw)
    import_signal_kw = jnp.where(smoothed, import_ewma_kw, import_stress_kw)

    export_rate = _bounded_rate(
        export_signal_kw,
        p["export_threshold_kw"],
        p["export_stress_scale_kw"],
        p["export_strength_chf_per_kwh"],
    )
    import_rate = _bounded_rate(
        import_signal_kw,
        p["import_threshold_kw"],
        p["import_stress_scale_kw"],
        p["import_strength_chf_per_kwh"],
    )
    stress_chf = export_rate * exported_kwh + import_rate * imported_kwh
    export_scale_kw = jnp.maximum(p["export_stress_scale_kw"], _EPS)
    congestion = jnp.clip((export_signal_kw - p["export_threshold_kw"]) / export_scale_kw, 0.0, 1.0)

    # -- published clock ---------------------------------------------------
    # The rate class is each point's own peak so far, so the stagger is a
    # published function of something the household measures itself.
    peak_rank = carry.own_peak_kw / jnp.maximum(jnp.max(carry.own_peak_kw), _EPS)
    offset_h = p["tou_stagger_h"] * peak_rank
    export_window = _in_window(
        grid.hour, p["tou_export_start_h"] + offset_h, p["tou_export_end_h"] + offset_h
    )
    import_window = _in_window(grid.hour, p["tou_import_start_h"], p["tou_import_end_h"])
    tou_chf = jnp.where(export_window, p["tou_export_rate_chf_per_kwh"] * exported_kwh, 0.0)
    tou_chf += jnp.where(import_window, p["tou_import_rate_chf_per_kwh"] * imported_kwh, 0.0)

    # -- substation demand charge ------------------------------------------
    # A transformer is rated in kVA, so the apparent-power base is the
    # faithful one; the active-power base is kept for the ablation.
    demand_base = jnp.where(
        p["demand_use_apparent"] > 0.5,
        jnp.hypot(transformer_kw, jnp.asarray(grid.transformer_kvar, dtype=jnp.float32)),
        jnp.abs(transformer_kw),
    )
    floor_kva = jnp.where(
        p["demand_ratchet"] > 0.5,
        jnp.maximum(carry.demand_peak_kva, p["demand_threshold_kva"]),
        p["demand_threshold_kva"],
    )
    demand_cost_chf = p["demand_rate_chf_per_kva"] * jnp.maximum(demand_base - floor_kva, 0.0)
    straining_kwh = jnp.where(transformer_kw < 0.0, exported_kwh, imported_kwh)
    demand_chf = demand_cost_chf * _share(straining_kwh)

    # -- losses ------------------------------------------------------------
    loss_cost_chf = p["loss_price_chf_per_kwh"] * jnp.asarray(grid.losses_kw) * STEP_H
    loss_chf = loss_cost_chf * _share(jnp.abs(energy_kwh) ** p["loss_share_exponent"])

    # -- voltage -----------------------------------------------------------
    over_voltage = jnp.maximum(voltage_pu - p["voltage_level_threshold_pu"], 0.0) / VOLTAGE_STEP_PU
    voltage_level_chf = p["voltage_level_rate_chf_per_kwh"] * over_voltage * exported_kwh
    sensitivity_chf = (
        p["sensitivity_rate_chf_per_kwh"] * _sensitivity_index(carry) * congestion * exported_kwh
    )

    # -- each connection point's own profile -------------------------------
    own_base_kw = jnp.abs(power_kw)
    own_excess_kw = jnp.maximum(
        own_base_kw - jnp.maximum(carry.own_peak_kw, p["own_peak_threshold_kw"]), 0.0
    )
    own_peak_chf = p["own_peak_rate_chf_per_kw"] * own_excess_kw
    own_deviation_chf = p["own_deviation_rate_chf_per_kwh"] * jnp.abs(
        energy_kwh - carry.own_energy_ewma_kwh
    )

    # -- reverse flow and throughput ---------------------------------------
    reverse_chf = jnp.where(
        transformer_kw < 0.0, p["reverse_rate_chf_per_kwh"] * exported_kwh, 0.0
    )
    throughput_chf = p["throughput_rate_chf_per_kwh"] * jnp.abs(energy_kwh)

    charge_chf = (
        stress_chf
        + tou_chf
        + demand_chf
        + loss_chf
        + voltage_level_chf
        + sensitivity_chf
        + own_peak_chf
        + own_deviation_chf
        + reverse_chf
        + throughput_chf
    )
    exempt = jnp.logical_and(p["tenant_penalty_exempt"] > 0.5, jnp.logical_not(has_inverter))
    charge_chf = jnp.where(exempt, 0.0, charge_chf)

    # -- funding -----------------------------------------------------------
    tenants = jnp.logical_not(has_inverter)
    num_tenants = jnp.sum(tenants)
    num_owners = num_pq - num_tenants
    pool_chf = jnp.sum(charge_chf)
    # Where the pool goes is as much of the mechanism as what it charges.
    # Equally over every connection point is transparent but hands the class
    # that pays nothing a windfall; back inside the paying class leaves the
    # two classes whole and still differentiates within the one that acts.
    to_tenants = jnp.where(num_tenants > 0, p["rebate_to_tenants"], 0.0)
    to_owners = jnp.where(num_owners > 0, p["rebate_to_owners"], 0.0)
    rebate_chf = (
        pool_chf * (1.0 - to_tenants - to_owners) / num_pq
        + jnp.where(tenants, pool_chf * to_tenants / jnp.maximum(num_tenants, 1), 0.0)
        + jnp.where(has_inverter, pool_chf * to_owners / jnp.maximum(num_owners, 1), 0.0)
    )

    # A static rate-class floor, funded inside the class that can respond.
    # Both counts must be nonzero or the transfer would not balance.
    fundable = jnp.logical_and(num_tenants > 0, num_owners > 0)
    floor_chf = jnp.where(fundable, p["tenant_floor_chf"], 0.0)
    class_chf = jnp.where(
        tenants, floor_chf, -floor_chf * num_tenants / jnp.maximum(num_owners, 1)
    )

    settlement_chf = (
        p["fair_leg_weight"] * jnp.asarray(grid.fair_leg_chf, dtype=jnp.float32)
        - charge_chf
        + rebate_chf
        + class_chf
        + p["unfunded_credit_chf"]
    )

    voltage_offset_pu = voltage_pu - 1.0
    new_carry = FamilyTariffMemory(
        intervals=carry.intervals + jnp.int32(1),
        export_stress_ewma_kw=export_ewma_kw,
        import_stress_ewma_kw=import_ewma_kw,
        demand_peak_kva=jnp.maximum(carry.demand_peak_kva, demand_base),
        own_peak_kw=jnp.maximum(carry.own_peak_kw, own_base_kw),
        own_energy_ewma_kwh=(1.0 - OWN_EWMA_ALPHA) * carry.own_energy_ewma_kwh
        + OWN_EWMA_ALPHA * energy_kwh,
        sensitivity_count=carry.sensitivity_count + 1.0,
        sum_energy_kwh=carry.sum_energy_kwh + energy_kwh,
        # Voltage enters offset from nominal: the accumulated cross moments
        # are differences of large similar numbers, and float32 has little to
        # spare.
        sum_voltage_pu=carry.sum_voltage_pu + voltage_offset_pu,
        sum_energy_sq=carry.sum_energy_sq + energy_kwh**2,
        sum_energy_voltage=carry.sum_energy_voltage + energy_kwh * voltage_offset_pu,
        sum_energy_transformer=carry.sum_energy_transformer + energy_kwh * transformer_kw,
        sum_transformer_voltage=carry.sum_transformer_voltage + transformer_kw * voltage_offset_pu,
        sum_transformer_kw=carry.sum_transformer_kw + transformer_kw,
        sum_transformer_sq=carry.sum_transformer_sq + transformer_kw**2,
    )
    return settlement_chf, new_carry


def family_tariff_factory(params: Union[Mapping[str, Any], str, int, None] = None):
    """Build the environment tariff factory for one bank scenario.

    Takes a scenario name, an index, or a parameter mapping carrying
    ``scenario_id``. The carry is sized from the prosumer model rather than
    from :data:`NUM_PQ`, because a tariff has to settle whatever feeder it is
    handed.
    """
    if params is None:
        resolved = scenario_params(0)
    elif isinstance(params, (str, int)):
        resolved = scenario_params(params)
    else:
        resolved = dict(params)

    def build(prosumer):
        num_pq = int(prosumer.num_pq)
        return tariff_from_settlement(
            family_tariff, resolved, init_carry=lambda: init_family_carry(num_pq)
        )(prosumer)

    return build


__all__ = [
    "DEFAULT_TARIFF_PARAMS",
    "SCENARIO_IDS",
    "TARIFF_BANK",
    "TARIFF_BANK_NAMES",
    "FamilyTariffMemory",
    "family_tariff",
    "family_tariff_factory",
    "init_family_carry",
    "scenario_params",
    "stress_tariff",
    "tariff_bank",
    "tariff_factory",
]

