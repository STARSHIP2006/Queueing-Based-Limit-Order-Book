"""Experiment 4: Kalman filtering of smoothly varying market activity."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiment_utils import get_config, make_model, ROOT, save_figure, save_json
from src.kalman_filter import (
    build_count_observations,
    generate_latent_activity,
    run_kalman_filter,
)
from src.simulator import simulate_lob


def main() -> None:
    cfg = get_config()
    params, initial_state, m = make_model(cfg)
    kcfg = cfg["kalman"]
    n_bins = int(kcfg["n_bins"])
    bin_width = float(kcfg["bin_width"])
    horizon = n_bins * bin_width
    edges = np.linspace(0.0, horizon, n_bins + 1)
    seed = int(cfg["project"]["random_seed"]) + 4

    h_true = generate_latent_activity(
        n_bins, float(kcfg["rho"]), float(kcfg["q"]),
        float(kcfg["initial_h"]), seed=seed,
    )
    multipliers = np.exp(h_true)
    event_log = simulate_lob(
        horizon, params, m, initial_state, seed=seed + 1,
        activity_edges=edges, activity_multipliers=multipliers,
    )
    observations = build_count_observations(
        event_log, horizon, edges, params, initial_state
    )
    filtered = run_kalman_filter(
        observations["z"], observations["R"],
        float(kcfg["rho"]), float(kcfg["q"]),
        float(kcfg["initial_h"]), float(kcfg["initial_variance"]),
    )
    rmse_h = float(np.sqrt(np.mean((filtered["filtered_mean"] - h_true) ** 2)))
    rmse_multiplier = float(np.sqrt(np.mean((filtered["activity_multiplier"] - multipliers) ** 2)))
    posterior_sd = np.sqrt(filtered["filtered_variance"])
    interval_coverage = float(np.mean(
        (h_true >= filtered["filtered_mean"] - 1.96 * posterior_sd)
        & (h_true <= filtered["filtered_mean"] + 1.96 * posterior_sd)
    ))

    event_log.to_csv(ROOT / "data" / "experiment_4_smooth_activity_log.csv", index=False)
    pd.DataFrame({"bin_start": edges[:-1], "bin_end": edges[1:],
                  "h_true": h_true, "activity_true": multipliers,
                  "count": observations["counts"],
                  "baseline_exposure": observations["baseline_exposure"],
                  "z": observations["z"], "R": observations["R"],
                  "h_filtered": filtered["filtered_mean"],
                  "activity_filtered": filtered["activity_multiplier"],
                  "filtered_variance": filtered["filtered_variance"]}).to_csv(
                      ROOT / "results" / "experiment_4_kalman_path.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(edges[:-1], h_true, label="True log activity")
    ax.plot(edges[:-1], filtered["filtered_mean"], label="Filtered log activity")
    ax.set(xlabel="Time (seconds)", ylabel="Log activity", title="Kalman tracking: smoothly varying activity")
    ax.legend()
    save_figure(fig, "experiment_4_smooth_activity.png")

    metrics = {"baseline_parameter_mode": "known generating parameters in a controlled synthetic experiment",
               "rmse_log_activity": rmse_h,
               "rmse_activity_multiplier": rmse_multiplier,
               "nominal_95_percent_interval_coverage": interval_coverage,
               "n_events": len(event_log), "n_bins": n_bins,
               "bin_width": bin_width, "rho": float(kcfg["rho"]), "q": float(kcfg["q"])}
    save_json(metrics, "experiment_4_kalman_metrics.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
