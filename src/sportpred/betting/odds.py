"""Odds mathematics (decimal odds everywhere).

Definitions used throughout the project
---------------------------------------
implied probability      = 1 / decimal odds                      (includes the bookmaker margin)
overround (booksum)      = Σ 1/odds over all outcomes of a market (e.g. 1.05 = 5% margin)
margin                   = overround − 1
fair (no-vig) probability = implied probabilities with the margin removed (see ``devig``)
expected value (EV)      = p_model × odds − 1    (profit per 1 unit staked, on average)
edge                     = p_model − implied probability
Kelly fraction           = (p × odds − 1) / (odds − 1)          (share of bankroll; we use a fraction of it)
closing line value (CLV) = bet odds × fair closing probability − 1

All functions accept floats or numpy arrays / pandas Series.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import brentq

ArrayLike = float | np.ndarray


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def implied_probability(odds: ArrayLike) -> ArrayLike:
    o = _arr(odds)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(o > 1.0, 1.0 / o, np.nan)
    return out if out.ndim else float(out)


def overround(odds_by_outcome: Sequence[ArrayLike]) -> ArrayLike:
    """Sum of implied probabilities across all outcomes of one market."""
    probs = [implied_probability(o) for o in odds_by_outcome]
    total = np.sum(np.vstack([np.atleast_1d(p) for p in probs]), axis=0)
    return total if np.ndim(odds_by_outcome[0]) else float(total[0])


def margin(odds_by_outcome: Sequence[ArrayLike]) -> ArrayLike:
    return overround(odds_by_outcome) - 1.0


def expected_value(prob: ArrayLike, odds: ArrayLike) -> ArrayLike:
    out = _arr(prob) * _arr(odds) - 1.0
    return out if np.ndim(out) else float(out)


def edge(prob: ArrayLike, odds: ArrayLike) -> ArrayLike:
    out = _arr(prob) - implied_probability(odds)
    return out if np.ndim(out) else float(out)


def fair_odds(prob: ArrayLike) -> ArrayLike:
    p = _arr(prob)
    with np.errstate(divide="ignore"):
        out = np.where(p > 0, 1.0 / p, np.inf)
    return out if out.ndim else float(out)


def kelly_fraction(prob: ArrayLike, odds: ArrayLike) -> ArrayLike:
    """Full-Kelly stake as a share of bankroll; 0 when EV ≤ 0. Use a *fraction* of this in practice."""
    p, o = _arr(prob), _arr(odds)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (p * o - 1.0) / (o - 1.0)
    f = np.where((o > 1.0) & np.isfinite(f), np.clip(f, 0.0, 1.0), 0.0)
    return f if f.ndim else float(f)


def closing_line_value(bet_odds: ArrayLike, closing_fair_prob: ArrayLike) -> ArrayLike:
    """EV of the bet measured against the (de-vigged) closing price — the standard sharpness test."""
    out = _arr(bet_odds) * _arr(closing_fair_prob) - 1.0
    return out if np.ndim(out) else float(out)


# ----------------------------------------------------------------------------- removing the margin

def devig(odds_by_outcome: Sequence[ArrayLike], method: str = "power") -> list[np.ndarray]:
    """Convert bookmaker odds for a complete market into fair probabilities that sum to 1.

    methods
    -------
    ``multiplicative`` — divide each implied probability by the overround (simple normalisation).
    ``power``          — find k with Σ (1/odds)^k = 1. Removes more margin from long shots,
                         reflecting the favourite-longshot bias. Default.
    ``shin``           — Shin (1993) insider-trading model; similar to power in practice.
    """
    probs = np.vstack([np.atleast_1d(implied_probability(o)) for o in odds_by_outcome])  # (k, n)
    n = probs.shape[1]
    out = np.full_like(probs, np.nan)
    for j in range(n):
        col = probs[:, j]
        if np.any(~np.isfinite(col)) or np.any(col <= 0):
            continue
        if method == "multiplicative":
            out[:, j] = col / col.sum()
        elif method == "power":
            out[:, j] = _power_devig(col)
        elif method == "shin":
            out[:, j] = _shin_devig(col)
        else:
            raise ValueError(f"unknown devig method {method!r}")
    return [out[i] for i in range(out.shape[0])]


def _power_devig(p: np.ndarray) -> np.ndarray:
    s = p.sum()
    if abs(s - 1.0) < 1e-12:
        return p.copy()
    f = lambda k: np.sum(p ** k) - 1.0  # noqa: E731
    lo, hi = (1.0, 10.0) if s > 1 else (0.01, 1.0)
    try:
        k = brentq(f, lo, hi, xtol=1e-12)
    except ValueError:
        return p / s
    q = p ** k
    return q / q.sum()


def _shin_devig(p: np.ndarray) -> np.ndarray:
    s = p.sum()
    if s <= 1.0:
        return p / s

    def probs(z):
        return (np.sqrt(z ** 2 + 4 * (1 - z) * p ** 2 / s) - z) / (2 * (1 - z))

    try:
        z = brentq(lambda z: probs(z).sum() - 1.0, 0.0, 0.4, xtol=1e-12)
    except ValueError:
        return p / s
    q = probs(z)
    return q / q.sum()


def devig_frame(df, cols: Sequence[str], method: str = "power", prefix: str = "fair_") -> None:
    """In-place: add fair probability columns for the given odds columns of a DataFrame."""
    fair = devig([df[c].to_numpy(dtype=float) for c in cols], method=method)
    for c, p in zip(cols, fair, strict=True):
        df[f"{prefix}{c}"] = p
