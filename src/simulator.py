"""Continuous-time, event-driven limit order book simulator.

The simulator follows Gillespie's direct method.  Event intensities are
constant between state transitions and, when an activity schedule is supplied,
constant within each activity bin.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

EVENT_TYPES = (
    "buy_limit_bid",
    "sell_limit_ask",
    "market_sell",
    "market_buy",
    "cancel_bid",
    "cancel_ask",
    "inside_buy",
    "inside_sell",
)

REQUIRED_LOG_COLUMNS = [
    "t", "event", "Qb_before", "Qa_before", "Pb_before", "Pa_before",
    "session", "Qb_after", "Qa_after", "Pb_after", "Pa_after",
]
EXTRA_LOG_COLUMNS = ["Qb_replenished", "Qa_replenished"]


@dataclass(frozen=True)
class LOBState:
    Qb: int = 5
    Qa: int = 5
    Pb: int = 99
    Pa: int = 101

    def __post_init__(self) -> None:
        if self.Qb < 1 or self.Qa < 1:
            raise ValueError("Queue sizes must be at least one.")
        if self.Pa - self.Pb < 1:
            raise ValueError("The spread must be at least one tick.")

    @property
    def spread(self) -> int:
        return self.Pa - self.Pb


@dataclass(frozen=True)
class StructuralParameters:
    theta_b: float = 0.7
    gamma_b: float = -0.3
    theta_a: float = 0.6
    gamma_a: float = -0.3
    theta_mb: float = float(np.log(0.8))
    theta_ma: float = float(np.log(0.8))
    theta_cb: float = float(np.log(0.1))
    theta_ca: float = float(np.log(0.1))
    theta_ib: float = 0.0
    theta_ia: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, float]) -> "StructuralParameters":
        return cls(**{k: float(v) for k, v in values.items() if k in cls.__dataclass_fields__})


PARAMETER_NAMES = tuple(StructuralParameters.__dataclass_fields__.keys())


def event_intensities(state: LOBState, params: StructuralParameters) -> np.ndarray:
    """Return intensities in the order specified by ``EVENT_TYPES``."""
    rates = np.array([
        np.exp(params.theta_b + params.gamma_b * np.log1p(state.Qb)),
        np.exp(params.theta_a + params.gamma_a * np.log1p(state.Qa)),
        np.exp(params.theta_mb),
        np.exp(params.theta_ma),
        np.exp(params.theta_cb) * state.Qb,
        np.exp(params.theta_ca) * state.Qa,
        np.exp(params.theta_ib) if state.spread >= 2 else 0.0,
        np.exp(params.theta_ia) if state.spread >= 2 else 0.0,
    ], dtype=float)
    if not np.all(np.isfinite(rates)) or np.any(rates < 0):
        raise FloatingPointError("Non-finite or negative event intensity encountered.")
    return rates


def total_intensity(state: LOBState, params: StructuralParameters) -> float:
    return float(event_intensities(state, params).sum())


def _apply_event(
    state: LOBState,
    event_index: int,
    m: float,
    rng: np.random.Generator,
) -> tuple[LOBState, Optional[int], Optional[int]]:
    """Apply one event and all associated depletion/replenishment effects."""
    qb, qa, pb, pa = state.Qb, state.Qa, state.Pb, state.Pa
    qb_replenished: Optional[int] = None
    qa_replenished: Optional[int] = None

    if event_index == 0:  # buy limit at best bid
        qb += 1
    elif event_index == 1:  # sell limit at best ask
        qa += 1
    elif event_index in (2, 4):  # market sell or cancel bid
        qb -= 1
        if qb == 0:
            pb -= 1
            qb = 1 + int(rng.poisson(m))
            qb_replenished = qb
    elif event_index in (3, 5):  # market buy or cancel ask
        qa -= 1
        if qa == 0:
            pa += 1
            qa = 1 + int(rng.poisson(m))
            qa_replenished = qa
    elif event_index == 6:  # buy limit inside spread
        if pa - pb < 2:
            raise RuntimeError("Inside-spread buy selected when unavailable.")
        pb += 1
        qb = 1
    elif event_index == 7:  # sell limit inside spread
        if pa - pb < 2:
            raise RuntimeError("Inside-spread sell selected when unavailable.")
        pa -= 1
        qa = 1
    else:
        raise ValueError(f"Unknown event index: {event_index}")

    new_state = LOBState(Qb=qb, Qa=qa, Pb=pb, Pa=pa)
    return new_state, qb_replenished, qa_replenished


def _activity_at(t: float, edges: Optional[np.ndarray], multipliers: Optional[np.ndarray]) -> tuple[float, float]:
    """Return the activity multiplier and next schedule boundary."""
    if edges is None or multipliers is None:
        return 1.0, float("inf")
    idx = int(np.searchsorted(edges, t, side="right") - 1)
    idx = min(max(idx, 0), len(multipliers) - 1)
    return float(multipliers[idx]), float(edges[idx + 1])


def simulate_lob(
    horizon: float,
    params: StructuralParameters,
    m: float = 5.0,
    initial_state: LOBState = LOBState(),
    seed: Optional[int] = None,
    session: int = 0,
    activity_edges: Optional[Sequence[float]] = None,
    activity_multipliers: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Simulate a LOB over ``[0, horizon]`` and return its event log.

    ``activity_edges`` must contain one more element than
    ``activity_multipliers``.  The multiplier is piecewise constant on each
    interval ``[edges[k], edges[k+1])`` and scales every event intensity.
    """
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("horizon must be a positive finite number.")
    if not np.isfinite(m) or m < 0:
        raise ValueError("m must be non-negative and finite.")

    edges_arr: Optional[np.ndarray] = None
    mult_arr: Optional[np.ndarray] = None
    if (activity_edges is None) != (activity_multipliers is None):
        raise ValueError("Provide both activity_edges and activity_multipliers.")
    if activity_edges is not None and activity_multipliers is not None:
        edges_arr = np.asarray(activity_edges, dtype=float)
        mult_arr = np.asarray(activity_multipliers, dtype=float)
        if len(edges_arr) != len(mult_arr) + 1 or len(mult_arr) == 0:
            raise ValueError("Activity schedule needs len(edges) = len(multipliers) + 1.")
        if not np.all(np.diff(edges_arr) > 0) or edges_arr[0] > 0 or edges_arr[-1] < horizon:
            raise ValueError("Activity edges must be increasing and cover [0, horizon].")
        if not np.all(np.isfinite(mult_arr)) or np.any(mult_arr <= 0):
            raise ValueError("Activity multipliers must be positive and finite.")

    rng = np.random.default_rng(seed)
    state = initial_state
    t = 0.0
    records: list[dict[str, object]] = []

    while t < horizon:
        activity, boundary = _activity_at(t, edges_arr, mult_arr)
        boundary = min(boundary, horizon)
        if boundary <= t:
            # Guard against a floating-point boundary lookup stall.
            t = np.nextafter(t, float("inf"))
            continue

        rates = event_intensities(state, params) * activity
        total_rate = float(rates.sum())
        if not np.isfinite(total_rate) or total_rate <= 0:
            raise RuntimeError("Total event intensity must be positive and finite.")

        wait = float(rng.exponential(1.0 / total_rate))
        event_time = t + wait
        if event_time >= boundary:
            t = boundary
            continue

        event_index = int(rng.choice(len(EVENT_TYPES), p=rates / total_rate))
        before = state
        state, qb_new, qa_new = _apply_event(state, event_index, m, rng)
        records.append({
            "t": event_time,
            "event": EVENT_TYPES[event_index],
            "Qb_before": before.Qb,
            "Qa_before": before.Qa,
            "Pb_before": before.Pb,
            "Pa_before": before.Pa,
            "session": int(session),
            "Qb_after": state.Qb,
            "Qa_after": state.Qa,
            "Pb_after": state.Pb,
            "Pa_after": state.Pa,
            "Qb_replenished": qb_new,
            "Qa_replenished": qa_new,
        })
        t = event_time

    columns = REQUIRED_LOG_COLUMNS + EXTRA_LOG_COLUMNS
    log = pd.DataFrame.from_records(records, columns=columns)
    log.attrs["horizon"] = float(horizon)
    log.attrs["initial_state"] = initial_state
    log.attrs["seed"] = seed
    return log
