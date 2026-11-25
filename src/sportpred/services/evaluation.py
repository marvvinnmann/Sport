"""Walk-forward evaluation service: produce out-of-sample predictions for every candidate model,
compare them with the market on identical matches, and save everything for the dashboard.

Timeline (defaults)
-------------------
2012/13 → 2015/16  history only (models need data to start)
2015-07 →          Dixon-Coles and Elo walk-forward predictions (weekly refits)
2017-07 →          GBM walk-forward (monthly refits; needs earlier DC predictions as features)
2019-07 →          TEST PERIOD — all reported metrics use only these matches
Pooled models (blend, ensemble) fit their weights season by season on earlier out-of-sample rows.

Candidate models
----------------
market      fair probabilities from the sharp pre-closing price (Pinnacle → Betfair Exchange → average)
elo         Elo + ordered logit (1X2 only)
dixon_coles Dixon-Coles Poisson model (1X2 + O/U 2.5)
gbm         LightGBM on form/shots/xG/rest/Elo/Dixon-Coles features
blend       log-linear pool of Dixon-Coles and market
ensemble    log-linear pool of Dixon-Coles, Elo, GBM and market
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from sportpred.config import ModelSettings, get_settings, load_model_settings
from sportpred.data.repository import analysis_frame
from sportpred.db import ModelVersion, session_scope
from sportpred.evaluation.metrics import (
    expected_calibration_error,
    log_loss_per_match,
    paired_comparison,
    summary_metrics,
)
from sportpred.evaluation.walkforward import attach_outcomes, walk_forward
from sportpred.models.registry import single_league_factory

ALL_MODELS = ["market", "elo", "dixon_coles", "gbm", "blend", "ensemble"]
BASE_COLS = ["id", "league", "season", "date", "home_team", "away_team", "fthg", "ftag", "status",
             "hs", "as_", "hst", "ast", "hxg", "axg"]


def reports_dir() -> Path:
    d = get_settings().resolved_data_dir() / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def walk_forward_path(model: str) -> Path:
    return reports_dir() / f"walkforward_{model}.parquet"


def load_walk_forward(model: str, test_only: bool = True) -> pd.DataFrame | None:
    p = walk_forward_path(model)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if test_only and "is_test" in df.columns:
        df = df[df["is_test"]]
    return df


def available_walk_forward_models() -> list[str]:
    found = {p.stem.replace("walkforward_", "") for p in reports_dir().glob("walkforward_*.parquet")}
    return [m for m in ALL_MODELS if m in found]


def model_version_id(name: str, ms: ModelSettings) -> str:
    params = {"name": name, "dc_xi": ms.dc_xi, "dc_window_days": ms.dc_window_days, "dc_ridge": ms.dc_ridge,
              "elo_k": ms.elo_k, "elo_home_advantage": ms.elo_home_advantage}
    return f"{name}-" + hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]


# ----------------------------------------------------------------------------- comparison

def market_comparison(df: pd.DataFrame) -> dict:
    """Model vs bookmaker probabilities on identical matches (1X2 and O/U 2.5)."""
    out: dict = {}
    c1 = ["p_H", "p_D", "p_A"]
    if {"p_H", "ref_close_H"}.issubset(df.columns):
        d = df.dropna(subset=c1 + ["ref_close_H", "ref_close_D", "ref_close_A"])
        if len(d):
            y = d["y_1x2"].to_numpy()
            pm = d[c1].to_numpy(float)
            pc = d[["ref_close_H", "ref_close_D", "ref_close_A"]].to_numpy(float)
            res = {"model": summary_metrics(pm, y), "market_closing": summary_metrics(pc, y)}
            res["vs_closing"] = paired_comparison(log_loss_per_match(pm, y), log_loss_per_match(pc, y))
            dp = d.dropna(subset=["ref_pre_H", "ref_pre_D", "ref_pre_A"])
            if len(dp):
                yp = dp["y_1x2"].to_numpy()
                pp = dp[["ref_pre_H", "ref_pre_D", "ref_pre_A"]].to_numpy(float)
                res["market_preclosing"] = summary_metrics(pp, yp)
                res["vs_preclosing"] = paired_comparison(log_loss_per_match(dp[c1].to_numpy(float), yp),
                                                         log_loss_per_match(pp, yp))
            res["ece_home"] = expected_calibration_error(d["p_H"], (y == 0).astype(float))
            res["ece_draw"] = expected_calibration_error(d["p_D"], (y == 1).astype(float))
            res["ece_away"] = expected_calibration_error(d["p_A"], (y == 2).astype(float))
            out["1X2"] = res
    if {"p_OVER", "ref_close_O"}.issubset(df.columns):
        d = df.dropna(subset=["p_OVER", "ref_close_O"])
        if len(d):
            y = d["y_over"].to_numpy()
            pm = d["p_OVER"].to_numpy(float)
            pc = d["ref_close_O"].to_numpy(float)
            res = {"model": summary_metrics(pm, y, ordered=False),
                   "market_closing": summary_metrics(pc, y, ordered=False)}
            res["vs_closing"] = paired_comparison(log_loss_per_match(pm, y), log_loss_per_match(pc, y))
            dp = d.dropna(subset=["ref_pre_O"])
            if len(dp):
                yp = dp["y_over"].to_numpy()
                res["market_preclosing"] = summary_metrics(dp["ref_pre_O"].to_numpy(float), yp, ordered=False)
                res["vs_preclosing"] = paired_comparison(log_loss_per_match(dp["p_OVER"].to_numpy(float), yp),
                                                         log_loss_per_match(dp["ref_pre_O"].to_numpy(float), yp))
            res["ece_over"] = expected_calibration_error(d["p_OVER"], y.astype(float))
            out["OU25"] = res
    return out


# ----------------------------------------------------------------------------- orchestration

def _market_preds(frame: pd.DataFrame, wf_from: pd.Timestamp) -> pd.DataFrame:
    d = frame[pd.to_datetime(frame["date"]) >= wf_from]
    p = pd.DataFrame({"match_id": d["id"].to_numpy(), "league": d["league"].to_numpy(),
                      "p_H": d["ref_pre_H"].to_numpy(), "p_D": d["ref_pre_D"].to_numpy(),
                      "p_A": d["ref_pre_A"].to_numpy(), "p_OVER": d["ref_pre_O"].to_numpy(),
                      "p_UNDER": d["ref_pre_U"].to_numpy(), "fit_date": pd.to_datetime(d["date"]).to_numpy()})
    return p.dropna(subset=["p_H"])


def run_evaluation(models: list[str] | None = None, leagues: list[str] | None = None,
                   wf_from: str = "2015-07-01", gbm_from: str = "2017-07-01", test_from: str = "2019-07-01",
                   n_jobs: int = 4, progress: Callable[[str], None] = print,
                   ms: ModelSettings | None = None) -> dict:
    ms = ms or load_model_settings()
    models = models or ALL_MODELS
    leagues = leagues or ms.leagues
    wf_ts, gbm_ts, test_ts = pd.Timestamp(wf_from), pd.Timestamp(gbm_from), pd.Timestamp(test_from)
    need = set(models)
    if need & {"gbm", "ensemble"}:
        need |= {"dixon_coles", "gbm"}
    if need & {"blend", "ensemble"}:
        need |= {"dixon_coles", "market"}
    if "ensemble" in need:
        need |= {"elo"}

    progress(f"Loading data for {len(leagues)} league(s)…")
    frame = analysis_frame(leagues=leagues, statuses=["finished"])
    if frame.empty:
        raise RuntimeError("No finished matches in the database. Run `sportpred update` first.")
    base = frame[BASE_COLS]
    preds: dict[str, pd.DataFrame] = {}

    if "market" in need:
        preds["market"] = _market_preds(frame, wf_ts)
    for name in ("dixon_coles", "elo"):
        if name in need:
            progress(f"Walk-forward '{name}' (weekly refits from {wf_from})…")
            preds[name] = walk_forward(base, single_league_factory(name, ms), wf_ts, n_jobs=n_jobs)
    if "gbm" in need:
        from sportpred.models.gbm import build_features, walk_forward_gbm

        progress(f"Building GBM features and walk-forward (monthly refits from {gbm_from})…")
        X = build_features(base, preds.get("dixon_coles"))
        y1 = pd.Series(np.where(base["fthg"] > base["ftag"], 0, np.where(base["fthg"] == base["ftag"], 1, 2)),
                       index=base.index).where(base["fthg"].notna())
        yo = ((base["fthg"] + base["ftag"]) > 2.5).astype(float).where(base["fthg"].notna())
        g = walk_forward_gbm(X, y1, yo, start=gbm_ts, train_from=wf_ts + pd.Timedelta(days=60), progress=progress)
        g["league"] = g["match_id"].map(base.set_index("id")["league"])
        preds["gbm"] = g
    if need & {"blend", "ensemble"}:
        from sportpred.models.blend import walk_forward_pool

        joined = _join_components(preds, frame)
        if "blend" in need:
            progress("Fitting season-by-season blend (Dixon-Coles + market)…")
            pb, prm = walk_forward_pool(joined, [["dc_p_H", "dc_p_D", "dc_p_A"], ["mk_p_H", "mk_p_D", "mk_p_A"]],
                                        ["dc_p_OVER", "mk_p_OVER"])
            preds["blend"] = _pool_frame(joined, pb, prm)
        if "ensemble" in need:
            progress("Fitting season-by-season ensemble (Dixon-Coles + Elo + GBM + market)…")
            pe, prm = walk_forward_pool(joined, [["dc_p_H", "dc_p_D", "dc_p_A"], ["elo_p_H", "elo_p_D", "elo_p_A"],
                                                 ["gbm_p_H", "gbm_p_D", "gbm_p_A"], ["mk_p_H", "mk_p_D", "mk_p_A"]],
                                        ["dc_p_OVER", "gbm_p_OVER", "mk_p_OVER"])
            preds["ensemble"] = _pool_frame(joined, pe, prm)

    summary: dict = {"generated_at": dt.datetime.now().isoformat(timespec="seconds"), "wf_from": wf_from,
                     "test_from": test_from, "leagues": leagues, "models": {}}
    results: dict[str, pd.DataFrame] = {}
    for name in [m for m in ALL_MODELS if m in preds and (m in models)]:
        p = preds[name]
        if p is None or p.empty:
            progress(f"  no predictions produced for {name}")
            continue
        df = attach_outcomes(p, frame)
        df["model"] = name
        df["is_test"] = pd.to_datetime(df["date"]) >= test_ts
        df.to_parquet(walk_forward_path(name), index=False)
        results[name] = df[df["is_test"]]
        progress(f"  {name}: {int(df['is_test'].sum())} test predictions ({len(df)} incl. warm-up) saved")

    # identical-match comparison: restrict every model to matches all of them predicted
    common_1x2 = _common_ids(results, "p_H")
    common_ou = _common_ids({k: v for k, v in results.items() if "p_OVER" in v and v["p_OVER"].notna().any()},
                            "p_OVER")
    summary["common_matches"] = {"1X2": len(common_1x2), "OU25": len(common_ou)}
    for name, df in results.items():
        d1 = df[df["match_id"].isin(common_1x2)]
        dou = df[df["match_id"].isin(common_ou)]
        overall = {**{k: v for k, v in market_comparison(d1).items() if k == "1X2"},
                   **{k: v for k, v in market_comparison(dou).items() if k == "OU25"}}
        by_league = {lg: market_comparison(g) for lg, g in df.groupby("league")}
        by_season = {s: market_comparison(g) for s, g in df.groupby("season")}
        summary["models"][name] = {"overall": overall, "by_league": by_league, "by_season": by_season,
                                   "n_predictions": int(len(df))}
        if name in preds and "pool_params" in preds[name].attrs:
            summary["models"][name]["pool_params"] = preds[name].attrs["pool_params"]
        _store_model_version(name, ms, df, overall)
    summary["overall"] = {m: v["overall"] for m, v in summary["models"].items()}
    (reports_dir() / "evaluation_summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    return summary


def _join_components(preds: dict[str, pd.DataFrame], frame: pd.DataFrame) -> pd.DataFrame:
    j = frame[["id", "league", "season", "date", "fthg", "ftag"]].rename(columns={"id": "match_id"}).copy()
    j["y_1x2"] = np.where(j["fthg"] > j["ftag"], 0, np.where(j["fthg"] == j["ftag"], 1, 2))
    j["y_over"] = ((j["fthg"] + j["ftag"]) > 2.5).astype(int)
    for name, prefix in (("dixon_coles", "dc"), ("elo", "elo"), ("gbm", "gbm"), ("market", "mk")):
        p = preds.get(name)
        if p is None or p.empty:
            continue
        cols = [c for c in ["p_H", "p_D", "p_A", "p_OVER"] if c in p.columns]
        j = j.merge(p[["match_id", *cols]].rename(columns={c: f"{prefix}_{c}" for c in cols}), on="match_id", how="left")
    return j


def _pool_frame(joined: pd.DataFrame, pooled: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = pooled.copy()
    out["match_id"] = joined["match_id"].to_numpy()
    out["league"] = joined["league"].to_numpy()
    out["fit_date"] = pd.to_datetime(joined["date"]).to_numpy()
    out = out.dropna(subset=["p_H"])
    out.attrs["pool_params"] = params
    return out


def _common_ids(results: dict[str, pd.DataFrame], col: str) -> set:
    sets = [set(df.loc[df[col].notna(), "match_id"]) for df in results.values() if col in df.columns]
    return set.intersection(*sets) if sets else set()


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _store_model_version(name: str, ms: ModelSettings, df: pd.DataFrame, overall: dict) -> None:
    vid = model_version_id(name, ms)
    metrics = json.loads(json.dumps(overall, default=_json_default))
    with session_scope() as s:
        mv = s.query(ModelVersion).filter_by(version=vid).one_or_none()
        if mv is None:
            mv = ModelVersion(name=name, version=vid)
            s.add(mv)
        mv.params = {k: v for k, v in ms.model_dump().items() if k.startswith(("dc_", "elo_"))}
        mv.trained_until = pd.to_datetime(df["date"]).max().date()
        mv.n_training_matches = int(df["n_train"].max()) if "n_train" in df and df["n_train"].notna().any() else None
        mv.metrics = metrics
        mv.notes = "walk-forward evaluation (test period)"


def load_evaluation_summary() -> dict | None:
    p = reports_dir() / "evaluation_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def latest_pool_params(model: str = "blend") -> dict | None:
    """Most recent season's pooling weights (used for live 'blend' predictions)."""
    s = load_evaluation_summary()
    if not s or model not in s.get("models", {}):
        return None
    prm = s["models"][model].get("pool_params")
    if not prm:
        return None
    out = {}
    for market in ("1X2", "OU25"):
        seasons = sorted(prm.get(market, {}))
        if seasons:
            out[market] = prm[market][seasons[-1]]
    return out or None
