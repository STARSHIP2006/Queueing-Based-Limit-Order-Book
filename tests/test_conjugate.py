import numpy as np
import pandas as pd

from src.inference import gamma_posterior, replenishment_posterior
from src.simulator import LOBState


def test_gamma_posterior_shape_rate_update():
    log = pd.DataFrame([
        {"t": 1.0, "event": "market_sell", "Qb_before": 2, "Qa_before": 3,
         "Pb_before": 10, "Pa_before": 12, "Qb_after": 1, "Qa_after": 3,
         "Pb_after": 10, "Pa_after": 12},
        {"t": 3.0, "event": "market_sell", "Qb_before": 1, "Qa_before": 3,
         "Pb_before": 10, "Pa_before": 12, "Qb_after": 4, "Qa_after": 3,
         "Pb_after": 9, "Pa_after": 12},
    ])
    result = gamma_posterior(log, 5.0, "market_sell", LOBState(2, 3, 10, 12), 2.0, 3.0)
    assert result["alpha_posterior"] == 4.0
    assert result["beta_posterior"] == 8.0
    assert np.isclose(result["mean"], 0.5)


def test_replenishment_gamma_poisson_update():
    log = pd.DataFrame({
        "Qb_replenished": [3.0, np.nan, 5.0],
        "Qa_replenished": [np.nan, 2.0, np.nan],
    })
    result = replenishment_posterior(log, alpha=2.0, beta=1.0)
    # Y values are 2, 1, and 4; n=3.
    assert result["n"] == 3
    assert result["sum_y"] == 7.0
    assert result["alpha_posterior"] == 9.0
    assert result["beta_posterior"] == 4.0
