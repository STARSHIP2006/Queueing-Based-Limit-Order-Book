"""Experiment 2: structural parameter recovery at multiple observation durations."""

import json

import pandas as pd

from src.diagnostics import save_trace_plots, summarize_chains
from src.experiment_utils import get_config, make_model, ROOT, save_json
from src.inference import fit_structural_posterior, posterior_summary
from src.simulator import simulate_lob


def main() -> None:
    cfg = get_config()
    params, initial_state, m = make_model(cfg)
    base_horizon = float(cfg["simulation"]["inference_horizon"])
    horizons = [0.5 * base_horizon, base_horizon, 1.5 * base_horizon]
    max_horizon = max(horizons)
    seed = int(cfg["project"]["random_seed"]) + 2

    full_log = simulate_lob(max_horizon, params, m, initial_state, seed=seed)
    full_log.to_csv(ROOT / "data" / "experiment_2_synthetic_log.csv", index=False)

    mcmc = cfg["mcmc"]
    all_summaries = []
    diagnostic_results = {}
    acceptance_results = {}
    for duration_index, horizon in enumerate(horizons):
        log = full_log.loc[full_log["t"] <= horizon].copy().reset_index(drop=True)
        posterior = fit_structural_posterior(
            log, horizon, initial_state,
            n_samples=int(mcmc["samples"]), burn_in=int(mcmc["burn_in"]),
            n_chains=int(mcmc["chains"]), seed=seed + duration_index,
            scalar_proposal_sd=float(mcmc["proposal_sd_scalar"]),
        )
        summary = posterior_summary(posterior)
        summary.insert(0, "horizon", horizon)
        summary.insert(1, "n_events", len(log))
        all_summaries.append(summary)
        diagnostic_results[str(horizon)] = summarize_chains(posterior)
        acceptance_results[str(horizon)] = posterior["acceptance"]

        # Save trace plots for the central observation duration to keep the
        # output directory compact while retaining complete diagnostics for all.
        if duration_index == 1:
            save_trace_plots(posterior, ROOT / "results" / "experiment_2_traces")

    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv(ROOT / "results" / "experiment_2_parameter_recovery.csv", index=False)
    save_json({
        "true_parameters": params.to_dict(),
        "observation_durations": horizons,
        "posterior_summaries": combined.to_dict(orient="records"),
        "diagnostics_by_duration": diagnostic_results,
        "acceptance_by_duration": acceptance_results,
        "note": "Each fit uses a prefix of the same reproducible maximum-horizon synthetic path.",
    }, "experiment_2_parameter_recovery.json")
    print(combined.to_string(index=False))
    print(json.dumps(diagnostic_results, indent=2))


if __name__ == "__main__":
    main()
