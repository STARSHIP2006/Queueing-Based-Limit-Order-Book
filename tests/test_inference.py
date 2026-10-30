import numpy as np

from src.inference import grid_quadrature, metropolis_2d
from src.simulator import LOBState, StructuralParameters, simulate_lob


def test_metropolis_sampler_returns_expected_shape():
    samples, acceptance = metropolis_2d(
        lambda x: -0.5 * float(np.sum(x ** 2)),
        initial=(0.0, 0.0), n_samples=100, burn_in=20, seed=3,
    )
    assert samples.shape == (100, 2)
    assert 0.0 <= acceptance <= 1.0
    assert np.all(np.isfinite(samples))


def test_grid_quadrature_normalizes_posterior_mass():
    params = StructuralParameters()
    state = LOBState()
    log = simulate_lob(20.0, params, 5.0, state, seed=5)
    result = grid_quadrature(
        log, 20.0, "bid", state,
        np.linspace(-2, 2, 11), np.linspace(-1.5, 1.5, 9),
    )
    assert np.isclose(result["weights"].sum(), 1.0)
    assert np.isfinite(result["summary"]["theta_mean"])
