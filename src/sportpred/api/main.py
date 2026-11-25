"""REST API exposing predictions, data status, performance and calculators.

Read-only except for /backtest and /calculator, which compute but store nothing.
No endpoint places bets or talks to bookmaker accounts.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from sportpred import __version__
from sportpred.betting.odds import devig, expected_value, implied_probability, kelly_fraction, margin
from sportpred.config import StakingSettings, get_settings, load_model_settings
from sportpred.db import DataUpdate, Match, Odds, Prediction, init_db, session_scope
from sportpred.leagues import LEAGUES

app = FastAPI(
    title="sportpred API",
    version=__version__,
    description="Transparent football prediction & value-bet research API. Paper trading only — "
                "positive expected value is an estimate, not a guarantee.",
)


@app.on_event("startup")
def _startup() -> None:
    init_db()


def _clean(obj):
    """Make pandas/numpy values JSON-safe (NaN -> None)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(float(obj)) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (pd.Timestamp, dt.datetime, dt.date)):
        return obj.isoformat()
    return obj


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/status")
def status() -> dict:
    s = get_settings()
    with session_scope() as db:
        n_matches = db.execute(select(func.count()).select_from(Match)).scalar_one()
        n_odds = db.execute(select(func.count()).select_from(Odds)).scalar_one()
        n_upcoming = db.execute(select(func.count()).select_from(Match).where(
            Match.status == "scheduled", Match.date >= dt.date.today())).scalar_one()
        last = db.execute(select(DataUpdate).order_by(DataUpdate.id.desc()).limit(1)).scalar_one_or_none()
        last_pred = db.execute(select(func.max(Prediction.created_at))).scalar_one()
    return {
        "matches": n_matches, "odds_rows": n_odds, "upcoming_fixtures": n_upcoming,
        "last_data_update": _clean(last.finished_at) if last else None,
        "last_update_status": last.status if last else None,
        "last_prediction_run": _clean(last_pred),
        "providers": {
            "football-data.co.uk": "connected (free CSV files, no key needed)",
            "the-odds-api": "configured" if s.odds_api_key else "not configured (optional key)",
            "injuries/lineups": "not connected — predictions exclude availability information",
        },
        "mode": "prediction + paper trading only",
    }


@app.get("/leagues")
def leagues() -> list[dict]:
    ms = load_model_settings()
    return [{"code": lg.code, "country": lg.country, "name": lg.name, "tier": lg.tier,
             "enabled": lg.code in ms.leagues} for lg in LEAGUES.values()]


@app.get("/fixtures")
def fixtures(league: str | None = None, days: int = Query(7, ge=1, le=60)) -> list[dict]:
    from sportpred.data.repository import load_matches

    today = dt.date.today()
    df = load_matches(leagues=[league] if league else None, statuses=["scheduled"], from_date=today,
                      to_date=today + dt.timedelta(days=days))
    cols = ["id", "league", "date", "kickoff", "home_team", "away_team"]
    return _clean(df[cols].to_dict("records")) if not df.empty else []


@app.get("/predictions")
def predictions(
    league: str | None = None,
    market: Literal["1X2", "OU25"] | None = None,
    verdict: Literal["value", "insufficient_evidence", "no_bet"] | None = None,
    min_ev: float | None = None,
    min_odds: float | None = None,
    confidence: Literal["low", "medium", "high"] | None = None,
    recommended_only: bool = False,
) -> list[dict]:
    from sportpred.services.predictions import latest_predictions

    df = latest_predictions()
    if df.empty:
        return []
    if league:
        df = df[df["league"] == league]
    if market:
        df = df[df["market"] == market]
    if verdict:
        df = df[df["verdict"] == verdict]
    if min_ev is not None:
        df = df[df["ev"] >= min_ev]
    if min_odds is not None:
        df = df[df["best_odds"] >= min_odds]
    if confidence:
        order = {"low": 0, "medium": 1, "high": 2}
        df = df[df["confidence"].map(order) >= order[confidence]]
    if recommended_only:
        df = df[df["is_recommended"] == 1]
    cols = ["match_id", "league", "date", "kickoff", "home_team", "away_team", "market", "selection", "prob",
            "prob_low", "prob_high", "best_odds", "best_bookmaker", "implied_prob", "fair_prob", "edge", "ev",
            "ev_low", "margin", "confidence", "verdict", "is_recommended", "odds_collected_at", "created_at"]
    df = df.sort_values(["date", "league", "match_id", "market"])
    return _clean(df[cols].to_dict("records"))


@app.get("/predictions/{match_id}")
def prediction_detail(match_id: int) -> dict:
    from sportpred.services.predictions import latest_predictions

    df = latest_predictions()
    d = df[df["match_id"] == match_id] if not df.empty else df
    if d.empty:
        raise HTTPException(404, "No current prediction for this match")
    first = d.iloc[0]
    return _clean({
        "match_id": match_id, "league": first["league"], "date": first["date"], "home_team": first["home_team"],
        "away_team": first["away_team"],
        "selections": d.drop(columns=["details"]).to_dict("records"),
        "explanation": first["details"].get("explanation"), "missing_info": first["details"].get("missing_info"),
    })


@app.get("/performance")
def performance() -> dict:
    from sportpred.services.evaluation import load_evaluation_summary

    s = load_evaluation_summary()
    if not s:
        raise HTTPException(404, "No evaluation yet. Run `sportpred evaluate`.")
    return _clean({k: s[k] for k in ("generated_at", "test_from", "leagues", "common_matches", "overall") if k in s})


@app.get("/paper-bets")
def paper_bets() -> list[dict]:
    from sportpred.services.paper import paper_bets_frame

    df = paper_bets_frame()
    return _clean(df.to_dict("records")) if not df.empty else []


@app.get("/quality")
def quality() -> list[dict]:
    from sportpred.data.quality import run_quality_checks

    with session_scope() as db:
        return [i.to_dict() for i in run_quality_checks(db)]


class CalculatorIn(BaseModel):
    odds: list[float] = Field(..., description="Decimal odds for every outcome of one market, e.g. [2.1, 3.4, 3.6]")
    model_probs: list[float] | None = Field(None, description="Your probabilities for the same outcomes")
    devig_method: Literal["multiplicative", "power", "shin"] = "power"


@app.post("/calculator")
def calculator(body: CalculatorIn) -> dict:
    if any(o <= 1 for o in body.odds):
        raise HTTPException(422, "Decimal odds must be greater than 1")
    arr = [np.array([o]) for o in body.odds]
    fair = [float(f[0]) for f in devig(arr, method=body.devig_method)] if len(arr) > 1 else None
    out = {"implied_probabilities": [float(implied_probability(o)) for o in body.odds],
           "margin": float(margin(arr)[0]) if len(arr) > 1 else None, "fair_probabilities": fair}
    if body.model_probs:
        if len(body.model_probs) != len(body.odds):
            raise HTTPException(422, "model_probs must have one value per outcome")
        out["expected_value"] = [float(expected_value(p, o)) for p, o in zip(body.model_probs, body.odds, strict=True)]
        out["full_kelly_fraction"] = [float(kelly_fraction(p, o)) for p, o in zip(body.model_probs, body.odds, strict=True)]
    return out


class BacktestIn(BaseModel):
    model: str = "dixon_coles"
    market: Literal["1X2", "OU25"] = "1X2"
    price_source: str = "market_avg"
    min_ev: float = 0.04
    min_edge: float = 0.0
    min_odds: float = 1.5
    max_odds: float = 5.0
    leagues: list[str] | None = None
    date_from: str | None = None
    date_to: str | None = None
    staking: StakingSettings = StakingSettings()


@app.post("/backtest")
def backtest(body: BacktestIn) -> dict:
    from sportpred.evaluation.backtest import BacktestConfig, baseline_strategies, run_backtest
    from sportpred.services.evaluation import load_walk_forward

    df = load_walk_forward(body.model)
    if df is None:
        raise HTTPException(404, f"No walk-forward predictions for model {body.model!r}. Run `sportpred evaluate`.")
    cfg = BacktestConfig(**body.model_dump(exclude={"model", "staking"}), staking=body.staking)
    bets, summary = run_backtest(df, cfg)
    return _clean({"summary": summary, "baselines": baseline_strategies(df, cfg, n_random=50).to_dict("records")})
