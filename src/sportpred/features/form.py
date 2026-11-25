"""Rolling team form features, strictly pre-match.

Method: build each team's state *after* every finished match (exponentially weighted averages of
goals, shots, shots on target, xG, points), then attach to any target match the latest state from a
strictly earlier date (``merge_asof`` with ``allow_exact_matches=False``). This works identically for
historical matches and for upcoming fixtures, so training and live features cannot diverge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STATE_COLS = ["gf", "ga", "sf", "sa", "stf", "sta", "xgf", "xga", "pts"]


def _long(fin: pd.DataFrame) -> pd.DataFrame:
    def side(is_home: bool) -> pd.DataFrame:
        h = "home" if is_home else "away"
        o = "away" if is_home else "home"
        gf, ga = ("fthg", "ftag") if is_home else ("ftag", "fthg")
        sf, sa = ("hs", "as_") if is_home else ("as_", "hs")
        stf, sta = ("hst", "ast") if is_home else ("ast", "hst")
        xgf, xga = ("hxg", "axg") if is_home else ("axg", "hxg")
        d = pd.DataFrame({
            "league": fin["league"].to_numpy(), "date": pd.to_datetime(fin["date"]).to_numpy(),
            "team": fin[f"{h}_team"].to_numpy(), "opp": fin[f"{o}_team"].to_numpy(), "is_home": is_home,
        })
        for name, col in (("gf", gf), ("ga", ga), ("sf", sf), ("sa", sa), ("stf", stf), ("sta", sta),
                          ("xgf", xgf), ("xga", xga)):
            d[name] = pd.to_numeric(fin[col], errors="coerce").to_numpy() if col in fin.columns else np.nan
        d["pts"] = np.select([d["gf"] > d["ga"], d["gf"] == d["ga"]], [3.0, 1.0], 0.0)
        return d
    return pd.concat([side(True), side(False)], ignore_index=True)


def team_states(fin: pd.DataFrame, halflife: float = 6.0) -> pd.DataFrame:
    """State of every team after each of its finished matches."""
    lg = _long(fin).sort_values(["league", "team", "date"], kind="mergesort")
    g = lg.groupby(["league", "team"], sort=False)
    out = lg[["league", "team", "date"]].copy()
    for c in STATE_COLS:
        out[f"ew_{c}"] = g[c].transform(lambda s: s.ewm(halflife=halflife, ignore_na=True).mean())
    out["n_played"] = g.cumcount() + 1
    out["last_date"] = out["date"]
    return out.reset_index(drop=True)


def pre_match_form(targets: pd.DataFrame, fin: pd.DataFrame, halflife: float = 6.0) -> pd.DataFrame:
    """Features for ``targets`` (any matches) using only ``fin`` matches from strictly earlier dates."""
    st = team_states(fin, halflife).sort_values("date")
    t = targets[["league", "date", "home_team", "away_team"]].copy()
    t["date"] = pd.to_datetime(t["date"])
    t["_ix"] = np.arange(len(t))
    t = t.sort_values("date")
    feats = t[["_ix"]].copy()
    for side in ("home", "away"):
        right = st.rename(columns={"team": f"{side}_team"})
        merged = pd.merge_asof(t, right, on="date", by=["league", f"{side}_team"], allow_exact_matches=False,
                               direction="backward")
        pre = "h_" if side == "home" else "a_"
        for c in [f"ew_{x}" for x in STATE_COLS] + ["n_played"]:
            feats[pre + c] = merged[c].to_numpy()
        feats[pre + "rest_days"] = (t["date"].to_numpy() - merged["last_date"].to_numpy()) / np.timedelta64(1, "D")
    feats = feats.sort_values("_ix").drop(columns="_ix")
    feats.index = targets.index
    feats["h_rest_days"] = feats["h_rest_days"].clip(upper=60)
    feats["a_rest_days"] = feats["a_rest_days"].clip(upper=60)
    for c in ["gf", "ga", "stf", "sta", "pts", "xgf"]:
        feats[f"d_ew_{c}"] = feats[f"h_ew_{c}"] - feats[f"a_ew_{c}"]
    return feats
