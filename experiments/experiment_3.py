"""Experiment 3: conjugate posterior and grid-quadrature validation."""

import json

import matplotlib.pyplot as plt
import numpy as np

from src.experiment_utils import get_config, make_model, ROOT, save_figure, save_json
from src.inference import (
    gamma_posterior, replenishment_posterior, grid_quadrature, metropolis_2d,
    _prepare_pair_statistics, _pair_log_posterior_from_stats,
    approximate_pair_proposal_covariance,
)
from src.simulator import simulate_lob
from src.likelihood import event_count_and_exposure, exposure_function


def main() -> None:
    cfg = get_config()
    params, initial_state, m = make_model(cfg)
    horizon = float(cfg["simulation"]["inference_horizon"])
    seed = int(cfg["project"]["random_seed"]) + 3
    log = simulate_lob(horizon, params, m, initial_state, seed=seed)

    prior = cfg["priors"]["conjugate_rate"]
    gamma_results = {}
    for event_type in ("market_sell", "market_buy", "cancel_bid", "cancel_ask", "inside_buy", "inside_sell"):
        gamma_results[event_type] = gamma_posterior(
            log, horizon, event_type, initial_state,
            float(prior["alpha"]), float(prior["beta"]),
        )
    m_prior = cfg["priors"]["replenishment_m"]
    gamma_results["queue_replenishment_m"] = replenishment_posterior(
        log, float(m_prior["alpha"]), float(m_prior["beta"])
    )

    grid_cfg = cfg["grid"]
    theta_grid = np.linspace(grid_cfg["theta_min"], grid_cfg["theta_max"], int(grid_cfg["theta_points"]))
    gamma_grid = np.linspace(grid_cfg["gamma_min"], grid_cfg["gamma_max"], int(grid_cfg["gamma_points"]))
    comparison = {}
    for side in ("bid", "ask"):
        grid = grid_quadrature(log, horizon, side, initial_state, theta_grid, gamma_grid)
        pair_stats = _prepare_pair_statistics(log, horizon, side, initial_state)
        pair_logpost = lambda x, stats=pair_stats: _pair_log_posterior_from_stats(x, *stats)
        proposal_cov = approximate_pair_proposal_covariance(*pair_stats)
        pair_samples, acceptance = metropolis_2d(
            pair_logpost, initial=(0.5, 0.0), n_samples=int(cfg["mcmc"]["samples"]),
            burn_in=int(cfg["mcmc"]["burn_in"]), seed=seed + (0 if side == "bid" else 1),
            proposal_covariance=proposal_cov,
        )
        comparison[side] = {
            "grid_summary": grid["summary"],
            "mcmc_mean": pair_samples.mean(axis=0).tolist(),
            "mcmc_sd": pair_samples.std(axis=0, ddof=1).tolist(),
            "acceptance": acceptance,
        }
        fig, ax = plt.subplots(figsize=(6, 4.5))
        contour = ax.contourf(grid["theta_grid"], grid["gamma_grid"], grid["weights"].T, levels=20)
        ax.scatter(pair_samples[:, 0], pair_samples[:, 1], s=3, alpha=0.15)
        ax.set(xlabel="theta", ylabel="gamma", title=f"{side.title()}-side posterior")
        fig.colorbar(contour, ax=ax, label="Normalized grid mass")
        save_figure(fig, f"experiment_3_{side}_grid_mcmc.png")

    save_json({"conjugate_gamma_posteriors": gamma_results,
               "queue_parameter_grid_mcmc_comparison": comparison,
               "n_events": len(log), "horizon": horizon},
              "experiment_3_validation.json")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
