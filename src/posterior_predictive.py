"""Posterior draws and posterior predictive LOB simulations."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .inference import replenishment_posterior
from .simulator import LOBState, StructuralParameters, simulate_lob


def draw_parameter_vectors(
    posterior_samples: dict[str, object],
    n_draws: int,
    seed: Optional[int] = None,
) -> list[StructuralParameters]:
    if n_draws <= 0:
        raise ValueError("n_draws must be positive.")
    rng = np.random.default_rng(seed)
    names = tuple(StructuralParameters.__dataclass_fields__.keys())
    flattened = {name: np.asarray(posterior_samples[name], dtype=float).ravel() for name in names}
    if any(values.size == 0 for values in flattened.values()):
        raise ValueError("Posterior sample arrays must not be empty.")
    draws: list[StructuralParameters] = []
    for _ in range(n_draws):
        values = {name: float(arr[rng.integers(0, len(arr))]) for name, arr in flattened.items()}
        draws.append(StructuralParameters.from_dict(values))
    return draws


def posterior_predictive_simulations(
    posterior_samples: dict[str, object],
    event_log: pd.DataFrame,
    horizon: float,
    initial_state: LOBState,
    n_simulations: int = 100,
    m_prior_alpha: float = 2.0,
    m_prior_beta: float = 0.4,
    seed: Optional[int] = None,
) -> list[pd.DataFrame]:
    """Draw structural parameters and m, then simulate replicated paths."""
    rng = np.random.default_rng(seed)
    params_draws = draw_parameter_vectors(posterior_samples, n_simulations, seed=seed)
    m_post = replenishment_posterior(event_log, m_prior_alpha, m_prior_beta)
    m_draws = rng.gamma(
        shape=m_post["alpha_posterior"],
        scale=1.0 / m_post["beta_posterior"],
        size=n_simulations,
    )
    simulations = []
    for i, (params, m_value) in enumerate(zip(params_draws, m_draws)):
        sim_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        simulations.append(simulate_lob(
            horizon, params, float(m_value), initial_state,
            seed=sim_seed, session=i,
        ))
    return simulations


def trajectory_summary(event_log: pd.DataFrame, horizon: float) -> dict[str, float]:
    """Calculate compact descriptive statistics for one simulated trajectory."""
    if event_log.empty:
        return {
            "n_events": 0.0,
            "mean_Qb_before": float("nan"),
            "mean_Qa_before": float("nan"),
            "mean_spread_before": float("nan"),
            "bid_depletions": 0.0,
            "ask_depletions": 0.0,
            "price_changes": 0.0,
            "event_rate": 0.0,
        }
    bid_depletion = event_log["Qb_replenished"].notna().sum() if "Qb_replenished" in event_log else 0
    ask_depletion = event_log["Qa_replenished"].notna().sum() if "Qa_replenished" in event_log else 0
    price_changes = (
        (event_log["Pb_after"] != event_log["Pb_before"])
        | (event_log["Pa_after"] != event_log["Pa_before"])
    ).sum()
    return {
        "n_events": float(len(event_log)),
        "mean_Qb_before": float(event_log["Qb_before"].mean()),
        "mean_Qa_before": float(event_log["Qa_before"].mean()),
        "mean_spread_before": float((event_log["Pa_before"] - event_log["Pb_before"]).mean()),
        "bid_depletions": float(bid_depletion),
        "ask_depletions": float(ask_depletion),
        "price_changes": float(price_changes),
        "event_rate": float(len(event_log) / horizon),
    }
