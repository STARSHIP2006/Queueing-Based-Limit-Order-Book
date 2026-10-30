"""Experiment 5: Kalman-filter stress test under abrupt activity changes."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiment_utils import get_config, make_model, ROOT, save_figure, save_json
from src.kalman_filter import (
    build_count_observations,
    generate_switching_activity,
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
    seed = int(cfg["project"]["random_seed"]) + 5

    multipliers, regimes = generate_switching_activity(
        n_bins, bin_width,
        mean_regime_duration=float(kcfg["stress_mean_regime_duration"]),
        low_multiplier=float(kcfg["stress_low_multiplier"]),
        high_multiplier=float(kcfg["stress_high_multiplier"]),
        seed=seed,
    )
    h_true = np.log(multipliers)
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
    estimated_multiplier = filtered["activity_multiplier"]
    rmse = float(np.sqrt(np.mean((estimated_multiplier - multipliers) ** 2)))
    regime_accuracy = float(np.mean((estimated_multiplier >= 1.0) == (regimes == 1)))
    transition_lags = []
    transition_indices = np.flatnonzero(np.diff(regimes) != 0) + 1
    for transition in transition_indices:
        target_high = bool(regimes[transition] == 1)
        response_indices = np.flatnonzero(
            (estimated_multiplier[transition:] >= 1.0) if target_high
            else (estimated_multiplier[transition:] < 1.0)
        )
        if len(response_indices):
            transition_lags.append(float(response_indices[0] * bin_width))
    mean_transition_lag = float(np.mean(transition_lags)) if transition_lags else float("nan")

    event_log.to_csv(ROOT / "data" / "experiment_5_switching_activity_log.csv", index=False)
    pd.DataFrame({"bin_start": edges[:-1], "bin_end": edges[1:],
                  "regime_for_evaluation_only": regimes,
                  "activity_true": multipliers, "h_true": h_true,
                  "count": observations["counts"], "z": observations["z"],
                  "R": observations["R"], "activity_filtered": estimated_multiplier,
                  "h_filtered": filtered["filtered_mean"]}).to_csv(
                      ROOT / "results" / "experiment_5_stress_path.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(edges[:-1], multipliers, where="post", label="True multiplier")
    ax.plot(edges[:-1], estimated_multiplier, label="Kalman estimate")
    ax.set(xlabel="Time (seconds)", ylabel="Activity multiplier", title="Kalman stress test: abrupt regime switches")
    ax.legend()
    save_figure(fig, "experiment_5_stress_test.png")

    metrics = {"activity_multiplier_rmse": rmse,
               "low_high_classification_accuracy": regime_accuracy,
               "mean_transition_response_lag_seconds": mean_transition_lag,
               "number_of_evaluated_transitions": len(transition_lags),
               "n_events": len(event_log), "n_bins": n_bins,
               "mean_regime_duration_target_seconds": float(kcfg["stress_mean_regime_duration"]),
               "note": "Regime labels are retained only for evaluation; the filter receives only transformed event counts."}
    save_json(metrics, "experiment_5_stress_metrics.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
