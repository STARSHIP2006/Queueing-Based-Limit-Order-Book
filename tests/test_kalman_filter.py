import numpy as np

from src.kalman_filter import (
    build_count_observations,
    generate_latent_activity,
    run_kalman_filter,
)
from src.simulator import LOBState, StructuralParameters, simulate_lob


def test_kalman_filter_handles_missing_observations():
    z = np.array([0.1, np.nan, 0.4])
    r = np.array([0.2, np.nan, 0.3])
    result = run_kalman_filter(z, r, rho=0.9, q=0.05)
    assert result["filtered_mean"].shape == (3,)
    assert np.all(np.isfinite(result["filtered_mean"]))
    assert result["gain"][1] == 0.0


def test_latent_activity_generation_is_reproducible():
    first = generate_latent_activity(20, rho=0.95, q=0.02, seed=11)
    second = generate_latent_activity(20, rho=0.95, q=0.02, seed=11)
    assert np.allclose(first, second)


def test_count_observations_have_valid_shape_and_variance():
    params = StructuralParameters()
    initial = LOBState()
    horizon = 20.0
    edges = np.linspace(0, horizon, 5)
    log = simulate_lob(horizon, params, 5.0, initial, seed=17)
    observations = build_count_observations(log, horizon, edges, params, initial)
    assert len(observations["counts"]) == 4
    assert len(observations["baseline_exposure"]) == 4
    assert np.all(observations["R"][observations["valid"]] > 0)
