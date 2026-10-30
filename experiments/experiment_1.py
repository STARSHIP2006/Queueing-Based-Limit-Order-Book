"""Experiment 1: simulator sanity check."""

import json

import matplotlib.pyplot as plt
import numpy as np

from src.experiment_utils import get_config, make_model, ROOT, save_figure, save_json
from src.posterior_predictive import trajectory_summary
from src.simulator import simulate_lob


def main() -> None:
    cfg = get_config()
    params, initial_state, m = make_model(cfg)
    horizon = float(cfg["simulation"]["sanity_horizon"])
    seed = int(cfg["project"]["random_seed"])
    log = simulate_lob(horizon, params, m, initial_state, seed=seed)
    log.to_csv(ROOT / "data" / "experiment_1_event_log.csv", index=False)

    summary = trajectory_summary(log, horizon)
    bid_depletions = summary["bid_depletions"]
    ask_depletions = summary["ask_depletions"]
    summary.update({"horizon": horizon, "seed": seed,
                    "mean_bid_queue_after": float(log["Qb_after"].mean()) if len(log) else None,
                    "mean_ask_queue_after": float(log["Qa_after"].mean()) if len(log) else None,
                    "estimated_bid_depletion_interval_seconds": horizon / bid_depletions if bid_depletions else None,
                    "estimated_ask_depletion_interval_seconds": horizon / ask_depletions if ask_depletions else None})
    save_json(summary, "experiment_1_sanity_summary.json")

    if not log.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(log["t"], log["Qb_after"], label="Bid queue", linewidth=0.7)
        ax.plot(log["t"], log["Qa_after"], label="Ask queue", linewidth=0.7, alpha=0.8)
        ax.set(xlabel="Time (seconds)", ylabel="Queue size", title="LOB queue sizes")
        ax.legend()
        save_figure(fig, "experiment_1_queue_sizes.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
