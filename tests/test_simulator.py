import numpy as np

from src.simulator import (
    EVENT_TYPES,
    LOBState,
    StructuralParameters,
    event_intensities,
    simulate_lob,
)


def test_all_eight_intensities_are_nonnegative_and_finite():
    state = LOBState(Qb=4, Qa=6, Pb=100, Pa=103)
    rates = event_intensities(state, StructuralParameters())
    assert len(rates) == 8
    assert np.all(np.isfinite(rates))
    assert np.all(rates >= 0)


def test_inside_spread_events_are_disabled_at_one_tick_spread():
    state = LOBState(Qb=3, Qa=4, Pb=100, Pa=101)
    rates = event_intensities(state, StructuralParameters())
    assert rates[6] == 0.0
    assert rates[7] == 0.0


def test_simulator_is_reproducible_and_preserves_state_constraints():
    params = StructuralParameters()
    state = LOBState()
    first = simulate_lob(100.0, params, 5.0, state, seed=42)
    second = simulate_lob(100.0, params, 5.0, state, seed=42)
    assert first.equals(second)
    if not first.empty:
        assert (first["Qb_after"] >= 1).all()
        assert (first["Qa_after"] >= 1).all()
        assert ((first["Pa_after"] - first["Pb_after"]) >= 1).all()
        assert first["t"].is_monotonic_increasing
        assert set(first["event"]).issubset(set(EVENT_TYPES))


def test_activity_schedule_simulates_positive_event_flow():
    params = StructuralParameters()
    edges = np.array([0.0, 5.0, 10.0])
    multipliers = np.array([0.5, 2.0])
    log = simulate_lob(10.0, params, 5.0, LOBState(), seed=10,
                       activity_edges=edges, activity_multipliers=multipliers)
    assert (log["t"] >= 0).all()
    assert (log["t"] < 10).all()
