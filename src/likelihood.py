"""Continuous-time event-intensity likelihood and exposure calculations."""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .simulator import EVENT_TYPES, LOBState, StructuralParameters, event_intensities


def _state_from_row(row: pd.Series, suffix: str) -> LOBState:
    return LOBState(
        Qb=int(row[f"Qb_{suffix}"]),
        Qa=int(row[f"Qa_{suffix}"]),
        Pb=int(row[f"Pb_{suffix}"]),
        Pa=int(row[f"Pa_{suffix}"]),
    )


def state_segments(
    event_log: pd.DataFrame,
    horizon: float,
    initial_state: LOBState,
) -> list[tuple[float, float, LOBState]]:
    """Reconstruct intervals on which the observed LOB state is constant.

    Each interval is represented as ``(start, end, state)``.  Events occur at
    interval right endpoints; the post-event state applies immediately after
    that endpoint.
    """
    if horizon <= 0 or not np.isfinite(horizon):
        raise ValueError("horizon must be positive and finite.")
    required = {"t", "Qb_before", "Qa_before", "Pb_before", "Pa_before",
                "Qb_after", "Qa_after", "Pb_after", "Pa_after", "event"}
    missing = required.difference(event_log.columns)
    if missing:
        raise ValueError(f"Event log is missing columns: {sorted(missing)}")

    if event_log.empty:
        return [(0.0, float(horizon), initial_state)]

    ordered = event_log.sort_values("t", kind="stable")
    times = ordered["t"].to_numpy(dtype=float)
    if np.any(~np.isfinite(times)) or np.any(np.diff(times) < 0):
        raise ValueError("Event times must be finite and nondecreasing.")
    if times[0] < 0 or times[-1] > horizon:
        raise ValueError("Event times must lie in [0, horizon].")

    segments: list[tuple[float, float, LOBState]] = []
    current_state = initial_state
    previous_time = 0.0
    for _, row in ordered.iterrows():
        event_time = float(row["t"])
        pre_state = _state_from_row(row, "before")
        if pre_state != current_state:
            raise ValueError(
                "Event log is not a continuous state path: an event's before-state "
                "does not match the previous after-state."
            )
        if event_time > previous_time:
            segments.append((previous_time, event_time, current_state))
        current_state = _state_from_row(row, "after")
        previous_time = event_time
    if horizon > previous_time:
        segments.append((previous_time, float(horizon), current_state))
    return segments


def integrated_total_intensity(
    event_log: pd.DataFrame,
    horizon: float,
    params: StructuralParameters,
    initial_state: LOBState,
) -> float:
    """Compute the exact piecewise-constant compensator integral."""
    total = 0.0
    for start, end, state in state_segments(event_log, horizon, initial_state):
        total += (end - start) * float(event_intensities(state, params).sum())
    return float(total)


def log_likelihood(
    event_log: pd.DataFrame,
    horizon: float,
    params: StructuralParameters,
    initial_state: LOBState,
) -> float:
    """Full log-likelihood for the eight-event marked point process."""
    compensator = integrated_total_intensity(event_log, horizon, params, initial_state)
    event_term = 0.0
    for _, row in event_log.iterrows():
        event = str(row["event"])
        if event not in EVENT_TYPES:
            raise ValueError(f"Unknown event type in log: {event}")
        state = _state_from_row(row, "before")
        rates = event_intensities(state, params)
        rate = float(rates[EVENT_TYPES.index(event)])
        if rate <= 0:
            return -np.inf
        event_term += np.log(rate)
    return float(event_term - compensator)


def exposure_function(event_type: str) -> Callable[[LOBState], float]:
    """Return the state exposure multiplier for a single-rate event."""
    mapping: dict[str, Callable[[LOBState], float]] = {
        "market_sell": lambda s: 1.0,
        "market_buy": lambda s: 1.0,
        "cancel_bid": lambda s: float(s.Qb),
        "cancel_ask": lambda s: float(s.Qa),
        "inside_buy": lambda s: float(s.spread >= 2),
        "inside_sell": lambda s: float(s.spread >= 2),
    }
    if event_type not in mapping:
        raise ValueError(f"No scalar exposure function defined for {event_type!r}.")
    return mapping[event_type]


def event_count_and_exposure(
    event_log: pd.DataFrame,
    horizon: float,
    event_type: str,
    initial_state: LOBState,
    exposure: Callable[[LOBState], float] | None = None,
) -> tuple[int, float]:
    """Return event count and integrated exposure for ``rate * g(state)``."""
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Unknown event type: {event_type}")
    g = exposure if exposure is not None else exposure_function(event_type)
    count = int((event_log["event"] == event_type).sum()) if not event_log.empty else 0
    integrated = sum((end - start) * float(g(state))
                     for start, end, state in state_segments(event_log, horizon, initial_state))
    return count, float(integrated)


def queue_parameter_log_likelihood(
    event_log: pd.DataFrame,
    horizon: float,
    side: str,
    theta: float,
    gamma: float,
    initial_state: LOBState,
) -> float:
    """Event-specific likelihood for the queue-dependent limit-order rate."""
    if side not in {"bid", "ask"}:
        raise ValueError("side must be 'bid' or 'ask'.")
    event_type = "buy_limit_bid" if side == "bid" else "sell_limit_ask"
    q_name = "Qb_before" if side == "bid" else "Qa_before"
    q_values = event_log.loc[event_log["event"] == event_type, q_name].to_numpy(dtype=float)
    event_term = float(np.sum(theta + gamma * np.log1p(q_values)))
    q_col = "Qb" if side == "bid" else "Qa"
    exposure = 0.0
    for start, end, state in state_segments(event_log, horizon, initial_state):
        q = state.Qb if q_col == "Qb" else state.Qa
        exposure += (end - start) * np.exp(theta + gamma * np.log1p(q))
    return float(event_term - exposure)


def scalar_log_rate_log_likelihood(
    event_log: pd.DataFrame,
    horizon: float,
    event_type: str,
    theta: float,
    initial_state: LOBState,
) -> float:
    """Likelihood contribution when lambda(state) = exp(theta) * g(state)."""
    count, exposure = event_count_and_exposure(
        event_log, horizon, event_type, initial_state
    )
    if exposure <= 0:
        return 0.0 if count == 0 else -np.inf
    # The sum of log(g(state)) at event times is independent of theta and may
    # be omitted for posterior sampling of theta.
    return float(count * theta - np.exp(theta) * exposure)
