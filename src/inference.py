"""Bayesian inference: conjugate updates, grid quadrature, and Metropolis-Hastings."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from .likelihood import (
    event_count_and_exposure,
    exposure_function,
    queue_parameter_log_likelihood,
    state_segments,
    scalar_log_rate_log_likelihood,
)
from .simulator import LOBState, StructuralParameters


def gamma_posterior(
    event_log: pd.DataFrame,
    horizon: float,
    event_type: str,
    initial_state: LOBState,
    alpha: float,
    beta: float,
) -> dict[str, float]:
    """Exact Gamma posterior for a rate coefficient (shape-rate form)."""
    if alpha <= 0 or beta <= 0:
        raise ValueError("Gamma prior shape and rate must be positive.")
    count, exposure = event_count_and_exposure(
        event_log, horizon, event_type, initial_state,
        exposure_function(event_type),
    )
    return {
        "alpha_prior": float(alpha),
        "beta_prior": float(beta),
        "count": int(count),
        "exposure": float(exposure),
        "alpha_posterior": float(alpha + count),
        "beta_posterior": float(beta + exposure),
        "mean": float((alpha + count) / (beta + exposure)),
        "variance": float((alpha + count) / (beta + exposure) ** 2),
    }


def replenishment_posterior(
    event_log: pd.DataFrame,
    alpha: float,
    beta: float,
) -> dict[str, float]:
    """Exact Gamma posterior for m under Y_i = Q_new,i - 1 ~ Poisson(m)."""
    if alpha <= 0 or beta <= 0:
        raise ValueError("Gamma prior shape and rate must be positive.")
    values: list[float] = []
    for column in ("Qb_replenished", "Qa_replenished"):
        if column in event_log.columns:
            observed = pd.to_numeric(event_log[column], errors="coerce").dropna().to_numpy()
            values.extend((observed - 1.0).tolist())
    if any(y < 0 or not float(y).is_integer() for y in values):
        raise ValueError("Replenishment observations must be positive integer queue sizes.")
    n = len(values)
    sum_y = float(np.sum(values))
    return {
        "alpha_prior": float(alpha),
        "beta_prior": float(beta),
        "n": int(n),
        "sum_y": sum_y,
        "alpha_posterior": float(alpha + sum_y),
        "beta_posterior": float(beta + n),
        "mean": float((alpha + sum_y) / (beta + n)),
        "variance": float((alpha + sum_y) / (beta + n) ** 2),
    }


def _normal_log_prior(x: np.ndarray, sd: np.ndarray) -> float:
    return float(-0.5 * np.sum((x / sd) ** 2))


def pair_log_posterior(
    x: np.ndarray,
    event_log: pd.DataFrame,
    horizon: float,
    side: str,
    initial_state: LOBState,
) -> float:
    theta, gamma = float(x[0]), float(x[1])
    likelihood = queue_parameter_log_likelihood(
        event_log, horizon, side, theta, gamma, initial_state
    )
    return likelihood + _normal_log_prior(np.array([theta, gamma]), np.array([2.0, 1.0]))


def scalar_log_posterior(
    theta: float,
    event_log: pd.DataFrame,
    horizon: float,
    event_type: str,
    initial_state: LOBState,
) -> float:
    # The specification uses N(0, 2^2) priors for scalar log-intensities.
    return scalar_log_rate_log_likelihood(
        event_log, horizon, event_type, theta, initial_state
    ) - 0.5 * (theta / 2.0) ** 2


def metropolis_2d(
    log_density: Callable[[np.ndarray], float],
    initial: tuple[float, float],
    n_samples: int = 4000,
    burn_in: int = 1000,
    proposal_sd: tuple[float, float] = (0.12, 0.12),
    seed: Optional[int] = None,
    proposal_covariance: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float]:
    """Random-walk Metropolis sampler for a two-dimensional posterior."""
    if n_samples <= 0 or burn_in < 0:
        raise ValueError("n_samples must be positive and burn_in non-negative.")
    rng = np.random.default_rng(seed)
    current = np.asarray(initial, dtype=float)
    scale = np.asarray(proposal_sd, dtype=float)
    if current.shape != (2,) or np.any(scale <= 0):
        raise ValueError("Initial point and proposal scales must be two-dimensional.")
    if proposal_covariance is None:
        proposal_cholesky = np.diag(scale)
    else:
        covariance = np.asarray(proposal_covariance, dtype=float)
        if covariance.shape != (2, 2) or not np.allclose(covariance, covariance.T):
            raise ValueError("proposal_covariance must be a symmetric 2x2 matrix.")
        try:
            proposal_cholesky = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("proposal_covariance must be positive definite.") from exc
    current_lp = float(log_density(current))
    saved = np.empty((n_samples, 2), dtype=float)
    accepted = 0
    total = n_samples + burn_in
    for i in range(total):
        proposal = current + proposal_cholesky @ rng.normal(size=2)
        proposal_lp = float(log_density(proposal))
        if np.isfinite(proposal_lp) and np.log(rng.uniform()) < proposal_lp - current_lp:
            current, current_lp = proposal, proposal_lp
            accepted += 1
        if i >= burn_in:
            saved[i - burn_in] = current
    return saved, accepted / total


def metropolis_1d(
    log_density: Callable[[float], float],
    initial: float,
    n_samples: int = 4000,
    burn_in: int = 1000,
    proposal_sd: float = 0.12,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, float]:
    """Random-walk Metropolis sampler for a scalar posterior."""
    if n_samples <= 0 or burn_in < 0 or proposal_sd <= 0:
        raise ValueError("Invalid MCMC sample count, burn-in, or proposal scale.")
    rng = np.random.default_rng(seed)
    current = float(initial)
    current_lp = float(log_density(current))
    saved = np.empty(n_samples, dtype=float)
    accepted = 0
    total = n_samples + burn_in
    for i in range(total):
        proposal = current + float(rng.normal(scale=proposal_sd))
        proposal_lp = float(log_density(proposal))
        if np.isfinite(proposal_lp) and np.log(rng.uniform()) < proposal_lp - current_lp:
            current, current_lp = proposal, proposal_lp
            accepted += 1
        if i >= burn_in:
            saved[i - burn_in] = current
    return saved, accepted / total


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cumulative = np.cumsum(w) / np.sum(w)
    return float(np.interp(probability, cumulative, v))



def _prepare_pair_statistics(
    event_log: pd.DataFrame, horizon: float, side: str, initial_state: LOBState
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    event_type = "buy_limit_bid" if side == "bid" else "sell_limit_ask"
    q_column = "Qb_before" if side == "bid" else "Qa_before"
    event_q = (event_log.loc[event_log["event"] == event_type, q_column].to_numpy(dtype=int)
               if not event_log.empty else np.array([], dtype=int))
    exposure_q: dict[int, float] = {}
    for start, end, state in state_segments(event_log, horizon, initial_state):
        q = state.Qb if side == "bid" else state.Qa
        exposure_q[q] = exposure_q.get(q, 0.0) + (end - start)
    event_counts: dict[int, int] = {}
    for q in event_q:
        event_counts[int(q)] = event_counts.get(int(q), 0) + 1
    all_q = np.array(sorted(set(exposure_q) | set(event_counts)), dtype=float)
    counts = np.array([event_counts.get(int(q), 0) for q in all_q], dtype=float)
    durations = np.array([exposure_q.get(int(q), 0.0) for q in all_q], dtype=float)
    return all_q, counts, durations


def _pair_log_posterior_from_stats(
    x: np.ndarray, q_values: np.ndarray, counts: np.ndarray, durations: np.ndarray
) -> float:
    theta, gamma = float(x[0]), float(x[1])
    log_q = np.log1p(q_values)
    log_rates = theta + gamma * log_q
    if np.any(log_rates > 700):
        return -np.inf
    with np.errstate(over="ignore", invalid="ignore"):
        event_term = float(np.sum(counts * log_rates))
        compensator = float(np.sum(durations * np.exp(log_rates)))
    if not np.isfinite(compensator):
        return -np.inf
    return event_term - compensator - 0.5 * (theta / 2.0) ** 2 - 0.5 * gamma ** 2


def approximate_pair_proposal_covariance(
    q_values: np.ndarray, counts: np.ndarray, durations: np.ndarray,
    initial: tuple[float, float] = (0.5, 0.0),
) -> np.ndarray:
    """Use the local posterior curvature to construct an efficient MH proposal.

    The proposal remains a symmetric Gaussian random walk.  This numerical
    curvature estimate changes only its scale and orientation, not the target
    posterior distribution.
    """
    x = np.asarray(initial, dtype=float).copy()
    log_q = np.log1p(q_values)
    for _ in range(100):
        log_rates = x[0] + x[1] * log_q
        if np.any(log_rates > 700):
            break
        rates = np.exp(log_rates)
        weighted = durations * rates
        gradient = np.array([
            counts.sum() - weighted.sum() - x[0] / 4.0,
            np.dot(counts, log_q) - np.dot(weighted, log_q) - x[1],
        ])
        precision = np.array([
            [weighted.sum() + 0.25, np.dot(weighted, log_q)],
            [np.dot(weighted, log_q), np.dot(weighted, log_q ** 2) + 1.0],
        ])
        step = np.linalg.solve(precision, gradient)
        x += step
        if np.linalg.norm(step) < 1e-8:
            break
    log_rates = x[0] + x[1] * log_q
    rates = np.exp(np.minimum(log_rates, 700))
    weighted = durations * rates
    precision = np.array([
        [weighted.sum() + 0.25, np.dot(weighted, log_q)],
        [np.dot(weighted, log_q), np.dot(weighted, log_q ** 2) + 1.0],
    ])
    covariance = np.linalg.inv(precision)
    # Roberts-Rosenthal scaling for a two-dimensional random-walk proposal.
    return (2.38 ** 2 / 2.0) * covariance

def grid_quadrature(
    event_log: pd.DataFrame,
    horizon: float,
    side: str,
    initial_state: LOBState,
    theta_grid: Optional[np.ndarray] = None,
    gamma_grid: Optional[np.ndarray] = None,
) -> dict[str, object]:
    """Evaluate and normalize the 2-D posterior on a rectangular grid."""
    theta_grid = np.asarray(theta_grid if theta_grid is not None else np.linspace(-3.0, 3.0, 151))
    gamma_grid = np.asarray(gamma_grid if gamma_grid is not None else np.linspace(-2.5, 2.5, 151))
    if theta_grid.ndim != 1 or gamma_grid.ndim != 1 or len(theta_grid) < 2 or len(gamma_grid) < 2:
        raise ValueError("theta_grid and gamma_grid must be one-dimensional grids with >= 2 values.")

    q_values, event_counts, durations = _prepare_pair_statistics(
        event_log, horizon, side, initial_state
    )
    log_q = np.log1p(q_values)
    theta_mesh, gamma_mesh = np.meshgrid(theta_grid, gamma_grid, indexing="ij")
    log_rates = theta_mesh[:, :, None] + gamma_mesh[:, :, None] * log_q[None, None, :]
    with np.errstate(over="ignore", invalid="ignore"):
        event_term = np.sum(event_counts[None, None, :] * log_rates, axis=2)
        compensator = np.sum(durations[None, None, :] * np.exp(log_rates), axis=2)
    logp = event_term - compensator - 0.5 * (theta_mesh / 2.0) ** 2 - 0.5 * gamma_mesh ** 2
    logp[~np.isfinite(logp)] = -np.inf
    max_logp = float(np.max(logp))
    unnormalized = np.exp(logp - max_logp)

    def trapezoid_weights(grid: np.ndarray) -> np.ndarray:
        result = np.empty_like(grid, dtype=float)
        result[0] = (grid[1] - grid[0]) / 2.0
        result[-1] = (grid[-1] - grid[-2]) / 2.0
        if len(grid) > 2:
            result[1:-1] = (grid[2:] - grid[:-2]) / 2.0
        return result

    integration_weights = trapezoid_weights(theta_grid)[:, None] * trapezoid_weights(gamma_grid)[None, :]
    weighted_mass = unnormalized * integration_weights
    weights = weighted_mass / np.sum(weighted_mass)
    theta_mean = float(np.sum(weights * theta_mesh))
    gamma_mean = float(np.sum(weights * gamma_mesh))
    theta_var = float(np.sum(weights * (theta_mesh - theta_mean) ** 2))
    gamma_var = float(np.sum(weights * (gamma_mesh - gamma_mean) ** 2))
    theta_flat, gamma_flat, weight_flat = theta_mesh.ravel(), gamma_mesh.ravel(), weights.ravel()
    summary = {
        "theta_mean": theta_mean,
        "theta_sd": float(np.sqrt(theta_var)),
        "theta_ci95": [_weighted_quantile(theta_flat, weight_flat, 0.025),
                       _weighted_quantile(theta_flat, weight_flat, 0.975)],
        "gamma_mean": gamma_mean,
        "gamma_sd": float(np.sqrt(gamma_var)),
        "gamma_ci95": [_weighted_quantile(gamma_flat, weight_flat, 0.025),
                       _weighted_quantile(gamma_flat, weight_flat, 0.975)],
    }
    return {"theta_grid": theta_grid, "gamma_grid": gamma_grid,
            "weights": weights, "summary": summary, "log_posterior": logp}


def fit_structural_posterior(
    event_log: pd.DataFrame,
    horizon: float,
    initial_state: LOBState,
    n_samples: int = 4000,
    burn_in: int = 1000,
    n_chains: int = 4,
    seed: int = 2026,
    scalar_proposal_sd: float = 0.12,
) -> dict[str, object]:
    """Sample the factorized posterior of all ten structural parameters.

    The bid and ask queue-dependent parameter pairs are sampled jointly.  The
    six scalar log-rate parameters are sampled with separate one-dimensional
    Metropolis chains.  Conditional factorization is valid because the state
    path is treated as observed, exactly as in the specification.
    """
    if n_chains < 1:
        raise ValueError("n_chains must be at least one.")
    if scalar_proposal_sd <= 0:
        raise ValueError("scalar_proposal_sd must be positive.")
    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = iter(seed_sequence.spawn(8 * n_chains))
    output: dict[str, object] = {"acceptance": {}}

    for side, names, initial in (
        ("bid", ("theta_b", "gamma_b"), (0.5, 0.0)),
        ("ask", ("theta_a", "gamma_a"), (0.5, 0.0)),
    ):
        pair_stats = _prepare_pair_statistics(event_log, horizon, side, initial_state)
        pair_logpost = lambda x, st=pair_stats: _pair_log_posterior_from_stats(x, *st)
        proposal_cov = approximate_pair_proposal_covariance(*pair_stats, initial=initial)
        chains = []
        rates = []
        for _ in range(n_chains):
            chain_seed = int(next(child_seeds).generate_state(1)[0])
            start = np.asarray(initial) + np.random.default_rng(chain_seed).normal(0, 0.15, 2)
            chain, acceptance = metropolis_2d(
                pair_logpost, tuple(start), n_samples, burn_in, seed=chain_seed,
                proposal_covariance=proposal_cov,
            )
            chains.append(chain)
            rates.append(acceptance)
        stacked = np.stack(chains, axis=0)
        output[names[0]] = stacked[:, :, 0]
        output[names[1]] = stacked[:, :, 1]
        output["acceptance"][side] = float(np.mean(rates))

    scalar_events = {
        "theta_mb": "market_sell",
        "theta_ma": "market_buy",
        "theta_cb": "cancel_bid",
        "theta_ca": "cancel_ask",
        "theta_ib": "inside_buy",
        "theta_ia": "inside_sell",
    }
    for parameter, event_type in scalar_events.items():
        count, exposure = event_count_and_exposure(
            event_log, horizon, event_type, initial_state, exposure_function(event_type)
        )
        def scalar_logpost(x: float, n: int = count, a: float = exposure) -> float:
            if a <= 0:
                return -0.5 * (x / 2.0) ** 2 if n == 0 else -np.inf
            if x > 700:
                return -np.inf
            return n * x - np.exp(x) * a - 0.5 * (x / 2.0) ** 2
        chains_1d = []
        rates = []
        for _ in range(n_chains):
            chain_seed = int(next(child_seeds).generate_state(1)[0])
            count = int((event_log["event"] == event_type).sum()) if not event_log.empty else 0
            _, exposure = event_count_and_exposure(
                event_log, horizon, event_type, initial_state, exposure_function(event_type)
            )
            initial = np.log(max(count / exposure, 1e-3)) if exposure > 0 else -2.0
            chain, acceptance = metropolis_1d(
                scalar_logpost, initial, n_samples, burn_in,
                proposal_sd=scalar_proposal_sd, seed=chain_seed,
            )
            chains_1d.append(chain)
            rates.append(acceptance)
        output[parameter] = np.stack(chains_1d, axis=0)
        output["acceptance"][parameter] = float(np.mean(rates))

    output["parameter_names"] = tuple(StructuralParameters.__dataclass_fields__.keys())
    return output


def posterior_summary(samples: dict[str, object]) -> pd.DataFrame:
    """Create mean, SD, and 95% credible interval summaries."""
    rows = []
    for name in StructuralParameters.__dataclass_fields__:
        values = np.asarray(samples[name], dtype=float).ravel()
        rows.append({
            "parameter": name,
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        })
    return pd.DataFrame(rows)
