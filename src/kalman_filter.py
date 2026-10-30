"""Approximate Kalman filtering for a latent log market-activity factor."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .likelihood import state_segments
from .simulator import LOBState, StructuralParameters, event_intensities


def generate_latent_activity(
    n_bins: int,
    rho: float = 0.95,
    q: float = 0.04,
    initial_h: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate h[k] = rho*h[k-1] + Normal(0, q)."""
    if n_bins <= 0 or abs(rho) >= 1 or q < 0:
        raise ValueError("Require n_bins > 0, |rho| < 1, and q >= 0.")
    rng = np.random.default_rng(seed)
    h = np.empty(n_bins, dtype=float)
    previous = float(initial_h)
    for k in range(n_bins):
        previous = rho * previous + float(rng.normal(scale=np.sqrt(q)))
        h[k] = previous
    return h


def generate_switching_activity(
    n_bins: int,
    bin_width: float,
    mean_regime_duration: float = 60.0,
    low_multiplier: float = 0.5,
    high_multiplier: float = 2.0,
    seed: Optional[int] = None,
    initial_high: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a two-regime activity path with geometric regime durations.

    Returns the activity multipliers and latent regime labels (0=low, 1=high).
    Labels should only be used for evaluation, never as filter observations.
    """
    if n_bins <= 0 or bin_width <= 0 or mean_regime_duration <= 0:
        raise ValueError("n_bins, bin_width, and mean duration must be positive.")
    if low_multiplier <= 0 or high_multiplier <= 0:
        raise ValueError("Activity multipliers must be positive.")
    rng = np.random.default_rng(seed)
    probability = min(1.0, bin_width / mean_regime_duration)
    labels = np.empty(n_bins, dtype=int)
    state = int(initial_high)
    for k in range(n_bins):
        if k > 0 and rng.uniform() < probability:
            state = 1 - state
        labels[k] = state
    multipliers = np.where(labels == 1, high_multiplier, low_multiplier).astype(float)
    return multipliers, labels


def make_bin_edges(horizon: float, bin_width: float) -> np.ndarray:
    if horizon <= 0 or bin_width <= 0:
        raise ValueError("horizon and bin_width must be positive.")
    n_bins = int(np.ceil(horizon / bin_width))
    edges = np.minimum(np.arange(n_bins + 1, dtype=float) * bin_width, horizon)
    if edges[-1] != horizon:
        edges = np.append(edges, horizon)
    return np.unique(edges)


def compute_bin_exposures(
    event_log: pd.DataFrame,
    horizon: float,
    bin_edges: np.ndarray,
    baseline_params: StructuralParameters,
    initial_state: LOBState,
) -> np.ndarray:
    """Compute A_k = integral sum_e lambda_e^0(X_t) dt in each bin."""
    edges = np.asarray(bin_edges, dtype=float)
    if len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("bin_edges must be strictly increasing.")
    if not np.isclose(edges[0], 0.0) or not np.isclose(edges[-1], horizon):
        raise ValueError("bin_edges must span [0, horizon].")
    exposures = np.zeros(len(edges) - 1, dtype=float)
    segments = state_segments(event_log, horizon, initial_state)
    i = 0
    for left, right, state in segments:
        while i < len(exposures) and edges[i + 1] <= left:
            i += 1
        j = i
        rate = float(event_intensities(state, baseline_params).sum())
        while j < len(exposures) and edges[j] < right:
            overlap = max(0.0, min(right, edges[j + 1]) - max(left, edges[j]))
            exposures[j] += overlap * rate
            if edges[j + 1] >= right:
                break
            j += 1
        i = min(j, len(exposures) - 1)
    return exposures


def build_count_observations(
    event_log: pd.DataFrame,
    horizon: float,
    bin_edges: np.ndarray,
    baseline_params: StructuralParameters,
    initial_state: LOBState,
) -> dict[str, np.ndarray]:
    """Build count observations and delta-method Gaussian approximations."""
    edges = np.asarray(bin_edges, dtype=float)
    n_bins = len(edges) - 1
    if event_log.empty:
        counts = np.zeros(n_bins, dtype=int)
    else:
        times = event_log["t"].to_numpy(dtype=float)
        indices = np.searchsorted(edges, times, side="right") - 1
        valid = (indices >= 0) & (indices < n_bins)
        counts = np.bincount(indices[valid], minlength=n_bins).astype(int)

    exposures = compute_bin_exposures(
        event_log, horizon, edges, baseline_params, initial_state
    )
    z = np.full(n_bins, np.nan, dtype=float)
    observation_variance = np.full(n_bins, np.nan, dtype=float)
    valid = exposures > 0
    z[valid] = np.log((counts[valid] + 0.5) / exposures[valid])
    observation_variance[valid] = 1.0 / (counts[valid] + 0.5)
    return {
        "counts": counts,
        "baseline_exposure": exposures,
        "z": z,
        "R": observation_variance,
        "valid": valid,
    }


def run_kalman_filter(
    observations: np.ndarray,
    observation_variances: np.ndarray,
    rho: float,
    q: float,
    initial_mean: float = 0.0,
    initial_variance: float = 1.0,
) -> dict[str, np.ndarray]:
    """Run a scalar Kalman filter; NaN observations trigger prediction only."""
    z = np.asarray(observations, dtype=float)
    R = np.asarray(observation_variances, dtype=float)
    if z.ndim != 1 or R.shape != z.shape:
        raise ValueError("observations and observation_variances must be equal-length vectors.")
    if abs(rho) >= 1 or q < 0 or initial_variance < 0:
        raise ValueError("Require |rho| < 1, q >= 0, and initial_variance >= 0.")

    n = len(z)
    predicted_mean = np.empty(n)
    predicted_variance = np.empty(n)
    filtered_mean = np.empty(n)
    filtered_variance = np.empty(n)
    gains = np.zeros(n)

    previous_mean = float(initial_mean)
    previous_variance = float(initial_variance)
    for k in range(n):
        mean_pred = rho * previous_mean
        variance_pred = rho ** 2 * previous_variance + q
        predicted_mean[k] = mean_pred
        predicted_variance[k] = variance_pred

        if np.isfinite(z[k]) and np.isfinite(R[k]) and R[k] > 0:
            gain = variance_pred / (variance_pred + R[k])
            mean_filt = mean_pred + gain * (z[k] - mean_pred)
            variance_filt = max(0.0, (1.0 - gain) * variance_pred)
            gains[k] = gain
        else:
            mean_filt = mean_pred
            variance_filt = variance_pred

        filtered_mean[k] = mean_filt
        filtered_variance[k] = variance_filt
        previous_mean, previous_variance = mean_filt, variance_filt

    return {
        "predicted_mean": predicted_mean,
        "predicted_variance": predicted_variance,
        "filtered_mean": filtered_mean,
        "filtered_variance": filtered_variance,
        "gain": gains,
        "activity_multiplier": np.exp(filtered_mean),
    }
