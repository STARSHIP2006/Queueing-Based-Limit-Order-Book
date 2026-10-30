"""Shared helpers for experiment scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .config import load_config
from .simulator import LOBState, StructuralParameters

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "default.yaml"


def get_config() -> dict[str, Any]:
    return load_config(CONFIG_PATH)


def make_model(config: dict[str, Any]) -> tuple[StructuralParameters, LOBState, float]:
    model = config["model"]
    params = StructuralParameters.from_dict(model["parameters"])
    state = LOBState(**model["initial_state"])
    return params, state, float(model["m"])


def save_json(data: Any, filename: str) -> Path:
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    path = output / filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    return path


def save_figure(fig: plt.Figure, filename: str) -> Path:
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    path = output / filename
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
