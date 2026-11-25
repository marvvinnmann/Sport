"""Probability-forecast metrics. Lower is better for log loss, Brier and RPS.

* log loss  — penalises confident wrong predictions heavily; the main metric for betting models.
* Brier     — mean squared error of the probability vector.
* RPS       — ranked probability score; respects that a draw is 'between' home and away wins.
* calibration — do events predicted at 30% happen about 30% of the time?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def _onehot(y: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), y] = 1.0
    return out


def log_loss_per_match(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:  # binary: p = P(y=1)
        p = np.column_stack([1 - p, p])
    p = p / p.sum(axis=1, keepdims=True)
    return -np.log(np.clip(p[np.arange(len(y)), y], EPS, 1))


def brier_per_match(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        p = np.column_stack([1 - p, p])
    return ((p - _onehot(y, p.shape[1])) ** 2).sum(axis=1)


def rps_per_match(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ranked probability score for ordered outcomes (home, draw, away)."""
    p = np.asarray(p, dtype=float)
    cp = np.cumsum(p, axis=1)[:, :-1]
    co = np.cumsum(_onehot(y, p.shape[1]), axis=1)[:, :-1]
    return ((cp - co) ** 2).sum(axis=1) / (p.shape[1] - 1)


def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(p)
    if p.ndim == 1:
        return float(((p > 0.5).astype(int) == y).mean())
    return float((p.argmax(axis=1) == y).mean())


def summary_metrics(p: np.ndarray, y: np.ndarray, ordered: bool = True) -> dict:
    ll = log_loss_per_match(p, y)
    out = {
        "n": int(len(y)),
        "log_loss": float(ll.mean()),
        "brier": float(brier_per_match(p, y).mean()),
        "accuracy": accuracy(p, y),
    }
    if ordered and np.asarray(p).ndim == 2 and np.asarray(p).shape[1] == 3:
        out["rps"] = float(rps_per_match(p, y).mean())
    return out


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 2000, level: float = 0.95, seed: int = 0,
                      block: np.ndarray | None = None) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI. If ``block`` (e.g. match dates) is given, resample whole
    blocks to respect correlation between bets on the same day."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    if block is None:
        idx = rng.integers(0, len(v), size=(n_boot, len(v)))
        means = v[idx].mean(axis=1)
    else:
        blocks = pd.Series(v).groupby(np.asarray(block)[: len(v)]).agg(["sum", "count"])
        s, c = blocks["sum"].to_numpy(), blocks["count"].to_numpy()
        idx = rng.integers(0, len(s), size=(n_boot, len(s)))
        means = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    a = (1 - level) / 2
    return float(v.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def paired_comparison(loss_a: np.ndarray, loss_b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Is model A better than model B? Returns mean(loss_a − loss_b) with a 95% CI (negative = A better)."""
    diff = np.asarray(loss_a) - np.asarray(loss_b)
    mean, lo, hi = bootstrap_mean_ci(diff, n_boot=n_boot, seed=seed)
    return {"mean_diff": mean, "ci_low": lo, "ci_high": hi,
            "a_better_significant": bool(hi < 0), "b_better_significant": bool(lo > 0)}


def wilson_interval(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        phat = k / n
        denom = 1 + z ** 2 / n
        centre = (phat + z ** 2 / (2 * n)) / denom
        half = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def calibration_table(prob: np.ndarray, outcome: np.ndarray, n_bins: int = 10,
                      strategy: str = "quantile") -> pd.DataFrame:
    """Reliability table for binary events (e.g. 'home win' with predicted P)."""
    prob = np.asarray(prob, float)
    outcome = np.asarray(outcome, float)
    ok = np.isfinite(prob) & np.isfinite(outcome)
    prob, outcome = prob[ok], outcome[ok]
    if len(prob) == 0:
        return pd.DataFrame(columns=["bin", "mean_pred", "observed", "count", "ci_low", "ci_high"])
    if strategy == "quantile":
        edges = np.unique(np.quantile(prob, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    if len(edges) < 2:
        edges = np.array([0.0, 1.0])
    bins = np.clip(np.digitize(prob, edges[1:-1], right=True), 0, len(edges) - 2)
    df = pd.DataFrame({"bin": bins, "p": prob, "y": outcome})
    g = df.groupby("bin").agg(mean_pred=("p", "mean"), observed=("y", "mean"), count=("y", "size"),
                              k=("y", "sum")).reset_index()
    lo, hi = wilson_interval(g["k"], g["count"])
    g["ci_low"], g["ci_high"] = lo, hi
    return g.drop(columns="k")


def expected_calibration_error(prob: np.ndarray, outcome: np.ndarray, n_bins: int = 10) -> float:
    t = calibration_table(prob, outcome, n_bins)
    if t.empty:
        return float("nan")
    return float((t["count"] * (t["mean_pred"] - t["observed"]).abs()).sum() / t["count"].sum())


def max_drawdown(equity: np.ndarray) -> tuple[float, float]:
    """(absolute drawdown, drawdown as fraction of the running peak)."""
    e = np.asarray(equity, float)
    if len(e) == 0:
        return 0.0, 0.0
    peak = np.maximum.accumulate(e)
    dd = peak - e
    i = int(np.argmax(dd))
    frac = dd[i] / peak[i] if peak[i] > 0 else 0.0
    return float(dd.max()), float(frac)


def longest_losing_streak(won: np.ndarray) -> int:
    best = cur = 0
    for w in np.asarray(won, bool):
        cur = 0 if w else cur + 1
        best = max(best, cur)
    return int(best)
