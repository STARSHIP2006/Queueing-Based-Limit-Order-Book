"""Experiment 6: posterior predictive simulations."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.diagnostics import summarize_chains
from src.experiment_utils import get_config, make_model, ROOT, save_figure, save_json
from src.inference import fit_structural_posterior, posterior_summary
from src.posterior_predictive import posterior_predictive_simulations, trajectory_summary
from src.simulator import simulate_lob


def main() -> None:
    cfg = get_config()
    params, initial_state, m = make_model(cfg)
    horizon = float(cfg["simulation"]["predictive_horizon"])
    seed = int(cfg["project"]["random_seed"]) + 6
    observed = simulate_lob(horizon, params, m, initial_state, seed=seed)

    mcmc = cfg["mcmc"]
    posterior = fit_structural_posterior(
        observed, horizon, initial_state,
        n_samples=int(mcmc["samples"]), burn_in=int(mcmc["burn_in"]),
        n_chains=int(mcmc["chains"]), seed=seed + 1,
        scalar_proposal_sd=float(mcmc["proposal_sd_scalar"]),
    )
    simulations = posterior_predictive_simulations(
        posterior, observed, horizon, initial_state,
        n_simulations=int(cfg["posterior_predictive"]["n_simulations"]),
        m_prior_alpha=float(cfg["priors"]["replenishment_m"]["alpha"]),
        m_prior_beta=float(cfg["priors"]["replenishment_m"]["beta"]),
        seed=seed + 2,
    )

    observed_summary = trajectory_summary(observed, horizon)
    simulated_summaries = [trajectory_summary(log, horizon) for log in simulations]
    summary_frame = pd.DataFrame(simulated_summaries)
    summary_frame.to_csv(ROOT / "results" / "experiment_6_predictive_summaries.csv", index=False)
    posterior_summary(posterior).to_csv(ROOT / "results" / "experiment_6_parameter_posterior.csv", index=False)
    pd.concat([log.assign(simulation=i) for i, log in enumerate(simulations)], ignore_index=True).to_csv(
        ROOT / "data" / "experiment_6_posterior_predictive_logs.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(summary_frame["mean_Qb_before"].dropna(), bins=15, alpha=0.65, label="Predictive bid queue mean")
    ax.axvline(observed_summary["mean_Qb_before"], linestyle="--", label="Observed")
    ax.set(xlabel="Mean bid queue size", ylabel="Number of simulations", title="Posterior predictive queue-size distribution")
    ax.legend()
    save_figure(fig, "experiment_6_predictive_queue.png")

    metrics = {
        "observed_summary": observed_summary,
        "predictive_mean": summary_frame.mean(numeric_only=True).to_dict(),
        "predictive_sd": summary_frame.std(numeric_only=True).to_dict(),
        "mcmc_diagnostics": summarize_chains(posterior),
        "n_simulations": len(simulations),
        "horizon": horizon,
    }
    save_json(metrics, "experiment_6_posterior_predictive.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
