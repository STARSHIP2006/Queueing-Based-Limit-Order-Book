"""MCMC diagnostics and lightweight plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def split_rhat(chains: np.ndarray) -> float:
    """Compute split-chain Gelman-Rubin R-hat for (chains, draws)."""
    x = np.asarray(chains, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 8:
        return float("nan")
    half = x.shape[1] // 2
    split = np.concatenate((x[:, :half], x[:, -half:]), axis=0)
    n = split.shape[1]
    chain_means = split.mean(axis=1)
    within = split.var(axis=1, ddof=1).mean()
    between = n * chain_means.var(ddof=1)
    if within == 0:
        return 1.0 if between == 0 else float("inf")
    variance_hat = ((n - 1) / n) * within + between / n
    return float(np.sqrt(variance_hat / within))


def effective_sample_size(chains: np.ndarray, max_lag: Optional[int] = None) -> float:
    """Estimate ESS using the initial-positive-sequence autocorrelation sum."""
    x = np.asarray(chains, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] < 3:
        return float("nan")
    n_chains, n_draws = x.shape
    max_lag = min(n_draws - 1, max_lag or n_draws - 1)
    centered = x - x.mean(axis=1, keepdims=True)
    variances = np.mean(centered ** 2, axis=1)
    if np.all(variances == 0):
        return float(n_chains * n_draws)
    autocorrelations = []
    for lag in range(1, max_lag + 1):
        products = np.mean(centered[:, :-lag] * centered[:, lag:], axis=1)
        denom = np.where(variances > 0, variances, 1.0)
        autocorrelations.append(float(np.mean(products / denom)))
    tau = 1.0
    for i in range(0, len(autocorrelations) - 1, 2):
        pair_sum = autocorrelations[i] + autocorrelations[i + 1]
        if pair_sum <= 0:
            break
        tau += 2.0 * pair_sum
    return float(min(n_chains * n_draws, n_chains * n_draws / max(tau, 1.0)))


def summarize_chains(samples: dict[str, object]) -> dict[str, dict[str, float]]:
    result = {}
    for name, value in samples.items():
        if name in {"acceptance", "parameter_names"}:
            continue
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 2:
            result[name] = {
                "rhat": split_rhat(arr),
                "ess": effective_sample_size(arr),
            }
    return result


def save_trace_plots(samples: dict[str, object], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in samples.items():
        if name in {"acceptance", "parameter_names"}:
            continue
        arr = np.asarray(value, dtype=float)
        if arr.ndim != 2:
            continue
        fig, ax = plt.subplots(figsize=(9, 3.5))
        for chain_index, chain in enumerate(arr):
            ax.plot(chain, linewidth=0.65, alpha=0.8, label=f"chain {chain_index + 1}")
        ax.set_title(f"Trace: {name}")
        ax.set_xlabel("Retained iteration")
        ax.set_ylabel(name)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(output / f"trace_{name}.png", dpi=150)
        plt.close(fig)
