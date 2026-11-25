"""Elo ratings for football (per league), computed strictly sequentially.

* Ratings for all matches on a given day use ratings from before that day (no same-day leakage).
* Goal-difference multiplier (World Football Elo style): bigger wins move ratings more.
* Between seasons ratings regress 25% towards the league mean.
* A team entering the league (e.g. promoted) starts at the average of the three lowest
  current ratings — roughly the strength of the teams it replaced.
* Matches without a result (fixtures) receive pre-match ratings but never update ratings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def goal_diff_multiplier(gd) -> np.ndarray | float:
    g = np.abs(np.asarray(gd, dtype=float))
    out = np.where(g <= 1, 1.0, np.where(g == 2, 1.5, (11.0 + g) / 8.0))
    return out if out.ndim else float(out)


def compute_elo(matches: pd.DataFrame, k: float = 20.0, home_adv: float = 60.0, base: float = 1500.0,
                season_regression: float = 0.25) -> pd.DataFrame:
    """Return a frame aligned with ``matches`` with pre-match ratings ``elo_home``/``elo_away``.

    Must be called per league (ratings are not comparable across leagues).
    """
    n = len(matches)
    if n == 0:
        return pd.DataFrame({"elo_home": [], "elo_away": []}, index=matches.index)
    d = pd.to_datetime(matches["date"]).to_numpy()
    order = np.argsort(d, kind="mergesort")
    days = d[order]
    homes = matches["home_team"].to_numpy()[order]
    aways = matches["away_team"].to_numpy()[order]
    hg = pd.to_numeric(matches["fthg"], errors="coerce").to_numpy(dtype=float)[order]
    ag = pd.to_numeric(matches["ftag"], errors="coerce").to_numpy(dtype=float)[order]
    if "status" in matches.columns:
        done = (matches["status"].to_numpy() == "finished")[order] & ~np.isnan(hg) & ~np.isnan(ag)
    else:
        done = ~np.isnan(hg) & ~np.isnan(ag)
    seasons = (matches["season"].astype(str).to_numpy()[order] if "season" in matches.columns
               else np.array([""] * n))

    ratings: dict[str, float] = {}
    pending: dict[str, float] = {}
    out_h = np.empty(n)
    out_a = np.empty(n)
    cur_day = None
    cur_season = None
    for i in range(n):
        if days[i] != cur_day:
            for t, delta in pending.items():
                ratings[t] += delta
            pending = {}
            cur_day = days[i]
            if cur_season is not None and seasons[i] != cur_season and ratings:
                mean = float(np.mean(list(ratings.values())))
                ratings = {t: r + season_regression * (mean - r) for t, r in ratings.items()}
            cur_season = seasons[i]
        h, a = homes[i], aways[i]
        for t in (h, a):
            if t not in ratings:
                ratings[t] = float(np.mean(sorted(ratings.values())[:3])) if len(ratings) >= 6 else base
        rh, ra = ratings[h], ratings[a]
        out_h[i], out_a[i] = rh, ra
        if done[i]:
            exp_h = 1.0 / (1.0 + 10 ** (-(rh + home_adv - ra) / 400.0))
            gd = hg[i] - ag[i]
            score = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
            delta = k * goal_diff_multiplier(gd) * (score - exp_h)
            pending[h] = pending.get(h, 0.0) + delta
            pending[a] = pending.get(a, 0.0) - delta

    res_h = np.empty(n)
    res_a = np.empty(n)
    res_h[order] = out_h
    res_a[order] = out_a
    return pd.DataFrame({"elo_home": res_h, "elo_away": res_a}, index=matches.index)


def elo_by_league(matches: pd.DataFrame, **kw) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame({"elo_home": [], "elo_away": []}, index=matches.index)
    parts = [compute_elo(g, **kw) for _, g in matches.groupby("league")]
    return pd.concat(parts).reindex(matches.index)
