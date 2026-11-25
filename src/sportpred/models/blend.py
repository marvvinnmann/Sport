"""Combining forecasts: logarithmic opinion pooling (a.k.a. stacking on log-probabilities).

    log p(outcome) ∝ bias[outcome] + Σ_k w_k · log p_k(outcome)

Fitted by maximum likelihood on *past out-of-sample* predictions only. Weights tell us how much
each source deserves:

* weight on the market ≈ 1 and on the model ≈ 0  →  the model adds nothing beyond the odds;
* model weight > 0 with a better out-of-sample log loss → the model carries extra information.

Pooling also recalibrates over-confident models (a weight < 1 shrinks extreme probabilities).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit, softmax

EPS = 1e-6


def fit_log_pool_multiclass(P: list[np.ndarray], y: np.ndarray, l2: float = 1.0) -> dict:
    """P: list of (n, k) probability arrays. Returns weights (one per source) and k−1 biases."""
    L = [np.log(np.clip(p, EPS, 1)) for p in P]
    k = P[0].shape[1]
    n_src = len(P)

    def nll(theta):
        w = theta[:n_src]
        b = np.concatenate([[0.0], theta[n_src:]])
        z = sum(wi * li for wi, li in zip(w, L, strict=True)) + b
        p = softmax(z, axis=1)
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).sum() + l2 * np.sum((w - 1.0 / n_src) ** 2)

    x0 = np.concatenate([np.full(n_src, 1.0 / n_src), np.zeros(k - 1)])
    res = minimize(nll, x0, method="L-BFGS-B")
    return {"weights": res.x[:n_src].tolist(), "bias": [0.0, *res.x[n_src:].tolist()], "n": int(len(y))}


def apply_log_pool_multiclass(P: list[np.ndarray], params: dict) -> np.ndarray:
    L = [np.log(np.clip(p, EPS, 1)) for p in P]
    z = sum(w * li for w, li in zip(params["weights"], L, strict=True)) + np.asarray(params["bias"])
    return softmax(z, axis=1)


def fit_log_pool_binary(P: list[np.ndarray], y: np.ndarray, l2: float = 1.0) -> dict:
    X = [logit(np.clip(p, EPS, 1 - EPS)) for p in P]
    n_src = len(P)

    def nll(theta):
        w, b = theta[:n_src], theta[n_src]
        z = sum(wi * xi for wi, xi in zip(w, X, strict=True)) + b
        p = np.clip(expit(z), 1e-12, 1 - 1e-12)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum() + l2 * np.sum((w - 1.0 / n_src) ** 2)

    res = minimize(nll, np.concatenate([np.full(n_src, 1.0 / n_src), [0.0]]), method="L-BFGS-B")
    return {"weights": res.x[:n_src].tolist(), "bias": float(res.x[n_src]), "n": int(len(y))}


def apply_log_pool_binary(P: list[np.ndarray], params: dict) -> np.ndarray:
    X = [logit(np.clip(p, EPS, 1 - EPS)) for p in P]
    return expit(sum(w * x for w, x in zip(params["weights"], X, strict=True)) + params["bias"])


def walk_forward_pool(df: pd.DataFrame, sources_1x2: list[list[str]], sources_ou: list[str] | None,
                      min_train: int = 1500, by: str = "season") -> tuple[pd.DataFrame, dict]:
    """Season-by-season pooling: weights for season S are fitted on rows from earlier seasons only.

    ``sources_1x2``: list of [colH, colD, colA] per source. ``sources_ou``: list of P(over) columns.
    Rows missing any source are left with NaN predictions.
    """
    out = pd.DataFrame(index=df.index, columns=["p_H", "p_D", "p_A", "p_OVER", "p_UNDER"], dtype=float)
    params_log: dict = {"1X2": {}, "OU25": {}}
    seasons = sorted(df[by].dropna().unique())
    cols_1x2 = [c for src in sources_1x2 for c in src]
    ok1 = df[cols_1x2].notna().all(axis=1)
    okou = df[sources_ou].notna().all(axis=1) if sources_ou else pd.Series(False, index=df.index)
    for s in seasons:
        past = df[by] < s
        cur = df[by] == s
        tr = past & ok1
        if tr.sum() >= min_train:
            P = [df.loc[tr, src].to_numpy(float) for src in sources_1x2]
            prm = fit_log_pool_multiclass(P, df.loc[tr, "y_1x2"].to_numpy())
            te = cur & ok1
            if te.any():
                pred = apply_log_pool_multiclass([df.loc[te, src].to_numpy(float) for src in sources_1x2], prm)
                out.loc[te, ["p_H", "p_D", "p_A"]] = pred
            params_log["1X2"][str(s)] = prm
        if sources_ou:
            tr = past & okou
            if tr.sum() >= min_train:
                prm = fit_log_pool_binary([df.loc[tr, c].to_numpy(float) for c in sources_ou],
                                          df.loc[tr, "y_over"].to_numpy())
                te = cur & okou
                if te.any():
                    po = apply_log_pool_binary([df.loc[te, c].to_numpy(float) for c in sources_ou], prm)
                    out.loc[te, "p_OVER"] = po
                    out.loc[te, "p_UNDER"] = 1 - po
                params_log["OU25"][str(s)] = prm
    return out, params_log
