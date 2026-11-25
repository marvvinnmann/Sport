"""Descriptive statistics for the Match Analysis page and prediction explanations.

These numbers *describe* the teams. The model itself only uses scores (Dixon-Coles) or results
(Elo); form tables and head-to-head records are shown for context and are not double-counted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def team_matches(fin: pd.DataFrame, team: str, before: pd.Timestamp | None = None, n: int | None = None) -> pd.DataFrame:
    """A team's finished matches from its own perspective (most recent first)."""
    d = fin[(fin["home_team"] == team) | (fin["away_team"] == team)]
    if before is not None:
        d = d[pd.to_datetime(d["date"]) < pd.Timestamp(before)]
    d = d.sort_values("date", ascending=False)
    if n:
        d = d.head(n)
    is_home = d["home_team"] == team
    out = pd.DataFrame({
        "date": pd.to_datetime(d["date"]).dt.date,
        "venue": np.where(is_home, "H", "A"),
        "opponent": np.where(is_home, d["away_team"], d["home_team"]),
        "gf": np.where(is_home, d["fthg"], d["ftag"]),
        "ga": np.where(is_home, d["ftag"], d["fthg"]),
        "shots_for": np.where(is_home, d["hs"], d["as_"]) if "hs" in d else np.nan,
        "shots_against": np.where(is_home, d["as_"], d["hs"]) if "hs" in d else np.nan,
        "sot_for": np.where(is_home, d["hst"], d["ast"]) if "hst" in d else np.nan,
        "sot_against": np.where(is_home, d["ast"], d["hst"]) if "hst" in d else np.nan,
        "xg_for": np.where(is_home, d["hxg"], d["axg"]) if "hxg" in d else np.nan,
        "xg_against": np.where(is_home, d["axg"], d["hxg"]) if "hxg" in d else np.nan,
    })
    out["result"] = np.where(out["gf"] > out["ga"], "W", np.where(out["gf"] == out["ga"], "D", "L"))
    out["points"] = np.select([out["result"] == "W", out["result"] == "D"], [3, 1], 0)
    return out.reset_index(drop=True)


def form_summary(fin: pd.DataFrame, team: str, before=None, n: int = 5) -> dict:
    tm = team_matches(fin, team, before, n)
    if tm.empty:
        return {"matches": 0}
    def mean(col):
        v = pd.to_numeric(tm[col], errors="coerce")
        return float(v.mean()) if v.notna().any() else None
    return {
        "matches": int(len(tm)),
        "form": "".join(tm["result"].tolist()),
        "points_per_game": float(tm["points"].mean()),
        "goals_for_avg": mean("gf"),
        "goals_against_avg": mean("ga"),
        "shots_on_target_for_avg": mean("sot_for"),
        "shots_on_target_against_avg": mean("sot_against"),
        "xg_for_avg": mean("xg_for"),
        "xg_against_avg": mean("xg_against"),
        "last_match": str(tm["date"].iloc[0]),
    }


def rest_days(fin: pd.DataFrame, team: str, match_date) -> int | None:
    tm = team_matches(fin, team, before=match_date, n=1)
    if tm.empty:
        return None
    return int((pd.Timestamp(match_date) - pd.Timestamp(tm["date"].iloc[0])).days)


def head_to_head(fin: pd.DataFrame, home: str, away: str, before=None, n: int = 6) -> pd.DataFrame:
    d = fin[((fin["home_team"] == home) & (fin["away_team"] == away))
            | ((fin["home_team"] == away) & (fin["away_team"] == home))]
    if before is not None:
        d = d[pd.to_datetime(d["date"]) < pd.Timestamp(before)]
    d = d.sort_values("date", ascending=False).head(n)
    return pd.DataFrame({"date": pd.to_datetime(d["date"]).dt.date, "home": d["home_team"],
                         "away": d["away_team"], "score": d["fthg"].astype("Int64").astype(str) + "-"
                         + d["ftag"].astype("Int64").astype(str)}).reset_index(drop=True)


def league_table(fin: pd.DataFrame, league: str, season: str, before=None) -> pd.DataFrame:
    d = fin[(fin["league"] == league) & (fin["season"] == season)]
    if before is not None:
        d = d[pd.to_datetime(d["date"]) < pd.Timestamp(before)]
    if d.empty:
        return pd.DataFrame()
    rows = []
    for side, gf, ga in (("home_team", "fthg", "ftag"), ("away_team", "ftag", "fthg")):
        t = d[[side, gf, ga]].rename(columns={side: "team", gf: "gf", ga: "ga"})
        rows.append(t)
    t = pd.concat(rows)
    t["w"] = (t["gf"] > t["ga"]).astype(int)
    t["d"] = (t["gf"] == t["ga"]).astype(int)
    t["l"] = (t["gf"] < t["ga"]).astype(int)
    tab = t.groupby("team").agg(played=("gf", "size"), won=("w", "sum"), drawn=("d", "sum"), lost=("l", "sum"),
                                gf=("gf", "sum"), ga=("ga", "sum"))
    tab["gd"] = tab["gf"] - tab["ga"]
    tab["points"] = 3 * tab["won"] + tab["drawn"]
    tab = tab.sort_values(["points", "gd", "gf"], ascending=False).reset_index()
    tab.insert(0, "pos", range(1, len(tab) + 1))
    for c in ["played", "won", "drawn", "lost", "gf", "ga", "gd", "points"]:
        tab[c] = tab[c].astype(int)
    return tab
