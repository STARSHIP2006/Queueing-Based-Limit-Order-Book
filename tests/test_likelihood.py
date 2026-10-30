import numpy as np
import pandas as pd

from src.likelihood import integrated_total_intensity, log_likelihood, state_segments
from src.simulator import LOBState, StructuralParameters, event_intensities


def test_no_event_path_has_one_full_horizon_segment():
    state = LOBState()
    empty = pd.DataFrame(columns=["t", "event", "Qb_before", "Qa_before", "Pb_before",
                                  "Pa_before", "Qb_after", "Qa_after", "Pb_after", "Pa_after"])
    segments = state_segments(empty, 10.0, state)
    assert segments == [(0.0, 10.0, state)]


def test_integrated_intensity_for_no_event_path():
    state = LOBState()
    params = StructuralParameters()
    empty = pd.DataFrame(columns=["t", "event", "Qb_before", "Qa_before", "Pb_before",
                                  "Pa_before", "Qb_after", "Qa_after", "Pb_after", "Pa_after"])
    expected = 10.0 * event_intensities(state, params).sum()
    assert np.isclose(integrated_total_intensity(empty, 10.0, params, state), expected)
    assert np.isclose(log_likelihood(empty, 10.0, params, state), -expected)


def test_piecewise_integrated_intensity_uses_post_event_state():
    initial = LOBState(Qb=2, Qa=2, Pb=99, Pa=101)
    log = pd.DataFrame([{
        "t": 2.0, "event": "buy_limit_bid",
        "Qb_before": 2, "Qa_before": 2, "Pb_before": 99, "Pa_before": 101,
        "Qb_after": 3, "Qa_after": 2, "Pb_after": 99, "Pa_after": 101,
    }])
    params = StructuralParameters()
    expected = (2.0 * event_intensities(initial, params).sum()
                + 3.0 * event_intensities(LOBState(3, 2, 99, 101), params).sum())
    assert np.isclose(integrated_total_intensity(log, 5.0, params, initial), expected)
