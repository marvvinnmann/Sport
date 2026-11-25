"""Elo model: pre-match Elo difference -> 1X2 probabilities through an ordered logistic link.

The link has three parameters (slope, two cut-points) fitted on training matches only.
It is deliberately simple and serves as a strong, transparent baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from sportpred.features.elo import compute_elo
from sportpred.models.base import MatchModel, finished_only, outcome_index


def ordered_logit_probs(z: np.ndarray, c1: float, c2: float) -> np.ndarray:
    """z = latent home strength. P(away)=σ(c1−z), P(away or draw)=σ(c2−z)."""
    p_a = expit(c1 - z)
    p_ad = expit(c2 - z)
    p = np.column_stack([1 - p_ad, p_ad - p_a, p_a])
    p = np.clip(p, 1e-6, 1)
    return p / p.sum(axis=1, keepdims=True)


class EloModel(MatchModel):
    name = "elo"
    markets = {"1X2"}

    def __init__(self, k: float = 20.0, home_adv: float = 60.0, window_days: int = 1500):
        self.k = k
        self.home_adv = home_adv
        self.window_days = window_days

    def fit(self, history: pd.DataFrame, as_of=None) -> EloModel:
        h = finished_only(history, as_of)
        if len(h) < 30:
            raise ValueError("Elo model needs at least 30 finished matches")
        cols = [c for c in ["league", "season", "date", "home_team", "away_team", "fthg", "ftag", "status"]
                if c in h.columns]
        self._history = h[cols].copy()
        elo = compute_elo(self._history, k=self.k, home_adv=self.home_adv)
        d = ((elo["elo_home"] + self.home_adv - elo["elo_away"]) / 400.0).to_numpy()
        dates = pd.to_datetime(h["date"])
        as_of_ts = pd.Timestamp(as_of) if as_of is not None else dates.max() + pd.Timedelta(days=1)
        use = (dates >= as_of_ts - pd.Timedelta(days=self.window_days)).to_numpy().copy()
        if "season" in h.columns:  # first season: ratings still settling
            first = h["season"].min()
            not_first = (h["season"] != first).to_numpy()
            if (use & not_first).sum() > 200:
                use &= not_first
        y = outcome_index(h)[use]
        d = d[use]

        def nll(theta):
            b, c1, gap = theta
            p = ordered_logit_probs(b * d, c1, c1 + np.exp(gap))
            return -np.log(p[np.arange(len(y)), y]).sum()

        res = minimize(nll, x0=np.array([2.0, -0.8, 0.0]), method="Nelder-Mead",
                       options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6})
        b, c1, gap = res.x
        self.slope, self.c1, self.c2 = float(b), float(c1), float(c1 + np.exp(gap))
        self.n_train = int(use.sum())
        return self

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        fx = fixtures.copy()
        fx["fthg"] = np.nan
        fx["ftag"] = np.nan
        fx["status"] = "scheduled"
        if "season" not in fx.columns and "season" in self._history.columns:
            fx["season"] = self._history["season"].iloc[-1]
        cols = list(self._history.columns)
        both = pd.concat([self._history, fx[cols]], ignore_index=True)
        elo = compute_elo(both, k=self.k, home_adv=self.home_adv).iloc[len(self._history):]
        elo.index = fixtures.index
        z = self.slope * (elo["elo_home"] + self.home_adv - elo["elo_away"]).to_numpy() / 400.0
        p = ordered_logit_probs(z, self.c1, self.c2)
        out = pd.DataFrame(p, columns=["p_H", "p_D", "p_A"], index=fixtures.index)
        out["elo_home"], out["elo_away"] = elo["elo_home"].to_numpy(), elo["elo_away"].to_numpy()
        return out
