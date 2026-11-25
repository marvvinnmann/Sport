"""Generate predictions for upcoming fixtures and classify each selection.

Verdicts
--------
value                  positive expected value that passes every threshold and robustness check
insufficient_evidence  EV looks positive, but the evidence is too weak (uncertainty, stale/missing odds,
                       little data, or a large unexplained disagreement with the market)
no_bet                 expected value below the threshold (the usual case — markets are efficient)

Predictions are append-only: each run gets a ``run_id`` and older predictions are kept for audit.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Callable

import numpy as np
import pandas as pd

from sportpred.betting.odds import devig, expected_value, implied_probability, margin
from sportpred.config import ModelSettings, load_model_settings
from sportpred.data.football_data import AGGREGATE_BOOKMAKERS
from sportpred.data.repository import (
    EXCHANGE_COMMISSION,
    REFERENCE_CHAIN,
    drop_implausible_odds,
    load_matches,
    load_odds_long,
)
from sportpred.db import ModelVersion, Prediction, session_scope, utcnow
from sportpred.models.dixon_coles import DixonColes
from sportpred.models.elo_model import EloModel
from sportpred.services.analysis import form_summary, head_to_head, rest_days

SELECTION_LABELS = {"H": "Home win", "D": "Draw", "A": "Away win", "OVER": "Over 2.5 goals",
                    "UNDER": "Under 2.5 goals"}
MARKET_SELECTIONS = {"1X2": ["H", "D", "A"], "OU25": ["OVER", "UNDER"]}
PRICE_FIELD = {"H": "price_home", "D": "price_draw", "A": "price_away", "OVER": "price_over",
               "UNDER": "price_under"}
PROB_FIELD = {"H": "p_H", "D": "p_D", "A": "p_A", "OVER": "p_OVER", "UNDER": "p_UNDER"}
CONF_ORDER = {"low": 0, "medium": 1, "high": 2}
MAX_MARKET_DISAGREEMENT = 0.12  # model − fair market prob above this => probably missing information
# Median absolute move of the sharp fair probability between the provider's pre-closing snapshot and
# kick-off, measured on 53,459 matches (2019-2026). Used as the minimum uncertainty band for the
# market-anchored blend: the price we anchor to is itself only an estimate of the final price.
MARKET_MOVE_BAND = {"H": 0.020, "D": 0.009, "A": 0.017, "OVER": 0.018, "UNDER": 0.018}


def _latest_preclosing_odds(odds: pd.DataFrame) -> pd.DataFrame:
    o = drop_implausible_odds(odds[~odds["is_closing"]]).copy()
    o = o[(o["market"] == "1X2") | ((o["market"] == "OU") & np.isclose(o["line"], 2.5))]
    return o.sort_values("collected_at").drop_duplicates(["match_id", "bookmaker", "market"], keep="last")


def _market_rows(odds: pd.DataFrame, match_id: int, market: str) -> pd.DataFrame:
    mk = "1X2" if market == "1X2" else "OU"
    return odds[(odds["match_id"] == match_id) & (odds["market"] == mk)]


def _fair_reference(rows: pd.DataFrame, market: str) -> tuple[dict, str | None]:
    sels = MARKET_SELECTIONS[market]
    for book in REFERENCE_CHAIN:
        r = rows[rows["bookmaker"] == book]
        if r.empty:
            continue
        prices = [r.iloc[0][PRICE_FIELD[s]] for s in sels]
        if any(pd.isna(p) for p in prices):
            continue
        fair = devig([np.array([p]) for p in prices], method="multiplicative" if book == "betfair_exchange" else "power")
        return {s: float(f[0]) for s, f in zip(sels, fair, strict=True)}, book
    return {}, None


def _confidence(width: float | None, n_min: float, unseen: bool, odds_age_h: float | None,
                ms: ModelSettings) -> tuple[str, list[str]]:
    level = 2
    notes = []
    if unseen:
        level = 0
        notes.append("a team has no matches in this league in the training window (estimate = prior)")
    if n_min < ms.min_team_matches:
        level = 0
        notes.append(f"only {int(n_min)} recent league matches for one team")
    elif n_min < 25:
        level = min(level, 1)
        notes.append(f"limited data: {int(n_min)} recent league matches for one team")
    if width is None:
        level = min(level, 1)
        notes.append("no uncertainty interval for this model")
    elif width > 0.16:
        level = 0
        notes.append(f"wide uncertainty interval ({width:.0%})")
    elif width > 0.11:
        level = min(level, 1)
        notes.append(f"moderate uncertainty interval ({width:.0%})")
    if odds_age_h is not None and odds_age_h > ms.max_odds_age_hours:
        level = max(level - 1, 0)
        notes.append(f"odds snapshot is {odds_age_h:.0f} hours old")
    return ["low", "medium", "high"][level], notes


def classify(sel: dict, ms: ModelSettings) -> tuple[str, list[str]]:
    """Apply the value-bet rules to one selection. Returns (verdict, reasons)."""
    reasons: list[str] = []
    if sel.get("best_odds") is None or not np.isfinite(sel["best_odds"]):
        return "insufficient_evidence", ["no bookmaker odds available"]
    ev, edge_ = sel["ev"], sel["edge"]
    if ev < ms.min_ev:
        return "no_bet", [f"expected value {ev:+.1%} is below the {ms.min_ev:.0%} threshold"]
    if edge_ < ms.min_edge:
        return "no_bet", [f"edge {edge_:+.1%} is below the {ms.min_edge:.0%} threshold"]
    if not (ms.min_odds <= sel["best_odds"] <= ms.max_odds):
        return "no_bet", [f"odds {sel['best_odds']:.2f} outside the allowed range {ms.min_odds}-{ms.max_odds}"]
    reasons.append(f"EV {ev:+.1%} ≥ {ms.min_ev:.0%} and edge {edge_:+.1%} ≥ {ms.min_edge:.0%}")
    weak = []
    if CONF_ORDER[sel["confidence"]] < CONF_ORDER[ms.min_confidence]:
        weak.append(f"confidence '{sel['confidence']}' below required '{ms.min_confidence}'")
    if ms.require_robust_ev:
        if sel.get("ev_low") is None:
            weak.append("no uncertainty interval to confirm the EV is robust")
        elif sel["ev_low"] <= 0:
            weak.append(f"EV turns negative at the low end of the uncertainty interval ({sel['ev_low']:+.1%})")
    if sel.get("fair_prob") is not None and sel["prob"] - sel["fair_prob"] > MAX_MARKET_DISAGREEMENT:
        weak.append(f"model is {sel['prob'] - sel['fair_prob']:.0%} above the sharp market — likely information "
                    "the model lacks (injuries, line-ups, motivation)")
    if sel.get("odds_age_h") is not None and sel["odds_age_h"] > ms.max_odds_age_hours:
        weak.append("odds may be stale")
    if weak:
        return "insufficient_evidence", reasons + weak
    return "value", reasons + ["EV stays positive across the uncertainty interval" if ms.require_robust_ev else ""]


def predict_fixtures(fin: pd.DataFrame, fixtures: pd.DataFrame, odds: pd.DataFrame, ms: ModelSettings,
                     now: dt.datetime | None = None) -> tuple[pd.DataFrame, dict]:
    """Core prediction logic (no database writes). Returns one row per fixture × selection."""
    now = now or utcnow()
    rows = []
    model_info: dict = {}
    pre = _latest_preclosing_odds(odds) if not odds.empty else odds
    for league, fx in fixtures.groupby("league"):
        hist = fin[fin["league"] == league]
        if len(hist) < 200:
            for _, f in fx.iterrows():
                for market in ms.markets:
                    for s in MARKET_SELECTIONS[market]:
                        rows.append(_empty_row(f, market, s, "insufficient_evidence",
                                               [f"only {len(hist)} historical matches for {league}"]))
            continue
        as_of = pd.Timestamp(now.date())
        dc = DixonColes(xi=ms.dc_xi, window_days=ms.dc_window_days, ridge=ms.dc_ridge)
        dcp = dc.predict_with_uncertainty(hist, fx, as_of=as_of, replicates=ms.uncertainty_replicates)
        elo = EloModel(k=ms.elo_k, home_adv=ms.elo_home_advantage).fit(hist, as_of=as_of)
        elop = elo.predict(fx)
        model_info[league] = {"dc_home_adv": dc.home_adv, "dc_rho": dc.rho, "dc_mu": dc.mu,
                              "n_train": dc.n_train, "trained_until": str(hist["date"].max().date())}
        for ix, f in fx.iterrows():
            p = dcp.loc[ix]
            ep = elop.loc[ix]
            ah, dh, _ = dc.team_params(f["home_team"])
            aa, da, _ = dc.team_params(f["away_team"])
            explanation = {
                "expected_goals": {"home": round(float(p["xg_home"]), 2), "away": round(float(p["xg_away"]), 2)},
                "attack_vs_avg_pct": {"home": round((np.exp(ah) - 1) * 100, 1), "away": round((np.exp(aa) - 1) * 100, 1)},
                "defence_vs_avg_pct": {"home": round((np.exp(dh) - 1) * 100, 1), "away": round((np.exp(da) - 1) * 100, 1)},
                "home_advantage_pct": round((np.exp(dc.home_adv) - 1) * 100, 1),
                "elo": {"home": round(float(ep["elo_home"])), "away": round(float(ep["elo_away"]))},
                "elo_probs": {"H": round(float(ep["p_H"]), 3), "D": round(float(ep["p_D"]), 3), "A": round(float(ep["p_A"]), 3)},
                "form_home": form_summary(hist, f["home_team"], f["date"]),
                "form_away": form_summary(hist, f["away_team"], f["date"]),
                "rest_days": {"home": rest_days(hist, f["home_team"], f["date"]),
                              "away": rest_days(hist, f["away_team"], f["date"])},
                "head_to_head": head_to_head(hist, f["home_team"], f["away_team"], f["date"]).astype(str).to_dict("records"),
                "matches_in_window": {"home": int(p["n_home"]), "away": int(p["n_away"])},
            }
            missing = [
                "Injuries, suspensions and confirmed line-ups are NOT included (no availability provider connected).",
                "Cup and European matches are not in the dataset, so rest days and fatigue may be understated.",
            ]
            unseen = not (bool(p["seen_home"]) and bool(p["seen_away"]))
            if unseen:
                missing.append("At least one team has no recent matches in this league (e.g. newly promoted).")
            n_min = float(min(p["n_home"], p["n_away"]))
            for market in ms.markets:
                mrows = _market_rows(pre, f["id"], market) if not pre.empty else pre
                fair, ref_book = _fair_reference(mrows, market) if len(mrows) else ({}, None)
                odds_time = mrows["collected_at"].max() if len(mrows) else None
                odds_age_h = ((now - pd.Timestamp(odds_time).to_pydatetime()).total_seconds() / 3600
                              if odds_time is not None and pd.notna(odds_time) else None)
                sels = MARKET_SELECTIONS[market]
                real = mrows[~mrows["bookmaker"].isin(AGGREGATE_BOOKMAKERS)] if len(mrows) else mrows
                margins = []
                for _, r in real.iterrows():
                    prices = [r[PRICE_FIELD[s]] for s in sels]
                    if all(pd.notna(x) for x in prices):
                        margins.append(float(margin([np.array([x]) for x in prices])[0]))
                probs, model_used, model_note = _market_probabilities(p, ep, market, fair, ms)
                if probs is None:
                    continue
                for s in sels:
                    prob, low, high = probs[s]
                    best_odds, best_book = None, None
                    book_prices = {}
                    for _, r in real.iterrows():
                        price = r[PRICE_FIELD[s]]
                        if pd.isna(price) or price <= 1:
                            continue
                        eff = 1 + (price - 1) * (1 - EXCHANGE_COMMISSION) if r["bookmaker"] == "betfair_exchange" else price
                        book_prices[r["bookmaker"]] = round(float(price), 3)
                        if best_odds is None or eff > best_odds:
                            best_odds, best_book = float(eff), r["bookmaker"]
                    width = (high - low) if low is not None and high is not None else None
                    conf, conf_notes = _confidence(width, n_min, unseen, odds_age_h, ms)
                    sel = {
                        "match_id": int(f["id"]), "league": league, "date": f["date"], "kickoff": f.get("kickoff"),
                        "home_team": f["home_team"], "away_team": f["away_team"], "market": market, "selection": s,
                        "prob": prob, "prob_low": low, "prob_high": high, "best_odds": best_odds,
                        "best_bookmaker": best_book, "confidence": conf, "odds_age_h": odds_age_h,
                        "odds_collected_at": odds_time, "fair_prob": fair.get(s), "reference_bookmaker": ref_book,
                        "margin": float(np.mean(margins)) if margins else None,
                    }
                    if best_odds:
                        sel["implied_prob"] = float(implied_probability(best_odds))
                        sel["edge"] = prob - sel["implied_prob"]
                        sel["ev"] = float(expected_value(prob, best_odds))
                        sel["ev_low"] = float(expected_value(low, best_odds)) if low is not None else None
                    else:
                        sel.update(implied_prob=None, edge=None, ev=None, ev_low=None)
                    verdict, reasons = classify(sel, ms)
                    if model_note:
                        reasons = [*reasons, model_note]
                        if verdict == "value" and model_used != ms.active_model:
                            verdict = "insufficient_evidence"
                    sel["verdict"] = verdict
                    sel["details"] = {
                        "reasons": [r for r in reasons if r], "confidence_notes": conf_notes, "missing_info": missing,
                        "explanation": explanation, "bookmaker_prices": book_prices,
                        "model": model_used, "dixon_coles_prob": float(p[PROB_FIELD[s]]),
                    }
                    rows.append(sel)
    out = pd.DataFrame(rows)
    if out.empty:
        return out, model_info
    # recommended = the value selection with highest EV per match & market
    out["is_recommended"] = False
    val = out[out["verdict"] == "value"]
    if not val.empty:
        best_idx = val.sort_values("ev", ascending=False).groupby(["match_id", "market"]).head(1).index
        out.loc[best_idx, "is_recommended"] = True
    return out, model_info


def blend_params(ms: ModelSettings) -> dict:
    """Newest pooling weights from the last walk-forward evaluation, else the shipped defaults."""
    try:
        from sportpred.services.evaluation import latest_pool_params

        fitted = latest_pool_params("blend")
    except Exception:  # pragma: no cover - evaluation files unreadable
        fitted = None
    return fitted or ms.blend_weights


def _market_probabilities(p: pd.Series, ep: pd.Series, market: str, fair: dict, ms: ModelSettings):
    """Probabilities (point, low, high) per selection for the active model.

    Returns (probs, model_used, note). ``probs`` is None when the active model does not cover the market.
    """
    from sportpred.models.blend import apply_log_pool_binary, apply_log_pool_multiclass

    sels = MARKET_SELECTIONS[market]
    dc = {s: (float(p[PROB_FIELD[s]]), float(p[f"{PROB_FIELD[s]}_low"]), float(p[f"{PROB_FIELD[s]}_high"]))
          for s in sels}
    if ms.active_model == "elo":
        if market != "1X2":
            return None, "elo", None
        return {s: (float(ep[PROB_FIELD[s]]), None, None) for s in sels}, "elo", None
    if ms.active_model == "dixon_coles":
        return dc, "dixon_coles", None
    # blend: Dixon-Coles pooled with the sharp market's fair probabilities
    if len(fair) != len(sels):
        return dc, "dixon_coles", ("no sharp reference price available, so the market blend could not be "
                                   "computed (Dixon-Coles shown; not eligible as a value bet)")
    prm = blend_params(ms).get(market)
    if not prm:
        return dc, "dixon_coles", "blend weights missing for this market"
    fair_vec = np.array([[fair[s] for s in sels]])
    out = {}
    if market == "1X2":
        def pooled(vec):
            return apply_log_pool_multiclass([np.array([vec]), fair_vec], prm)[0]
        point = pooled([dc[s][0] for s in sels])
        for i, s in enumerate(sels):
            ends = []
            for bound in (dc[s][1], dc[s][2]):  # move selection s to its bound, rescale the others
                rest = 1 - dc[s][0]
                vec = [bound if j == i else dc[t][0] * (1 - bound) / rest for j, t in enumerate(sels)]
                ends.append(pooled(vec)[i])
            band = MARKET_MOVE_BAND[s]
            out[s] = (float(point[i]), float(min(min(ends), point[i] - band)), float(max(max(ends), point[i] + band)))
    else:
        f_over = np.array([fair["OVER"]])
        def pooled_over(x):
            return float(apply_log_pool_binary([np.array([x]), f_over], prm)[0])
        po = pooled_over(dc["OVER"][0])
        lo, hi = sorted([pooled_over(dc["OVER"][1]), pooled_over(dc["OVER"][2])])
        lo, hi = min(lo, po - MARKET_MOVE_BAND["OVER"]), max(hi, po + MARKET_MOVE_BAND["OVER"])
        out["OVER"] = (po, lo, hi)
        out["UNDER"] = (1 - po, 1 - hi, 1 - lo)
    return out, "blend", None


def _empty_row(f, market, s, verdict, reasons) -> dict:
    return {"match_id": int(f["id"]), "league": f["league"], "date": f["date"], "home_team": f["home_team"],
            "away_team": f["away_team"], "market": market, "selection": s, "prob": np.nan, "verdict": verdict,
            "confidence": "low", "details": {"reasons": reasons}}


def generate_predictions(ms: ModelSettings | None = None, progress: Callable[[str], None] = print,
                         today: dt.date | None = None, log_paper: bool = True) -> dict:
    ms = ms or load_model_settings()
    today = today or dt.date.today()
    matches = load_matches(leagues=ms.leagues)
    if matches.empty:
        progress("No matches in database — run `sportpred update` first.")
        return {"n_predictions": 0, "n_matches": 0, "n_value": 0, "run_id": None}
    fin = matches[matches["status"] == "finished"]
    upcoming = matches[(matches["status"] == "scheduled") & (matches["date"] >= pd.Timestamp(today))]
    if upcoming.empty:
        progress("No upcoming fixtures for the configured leagues in the latest fixtures file.")
        return {"n_predictions": 0, "n_matches": 0, "n_value": 0, "run_id": None}
    odds = load_odds_long(match_ids=upcoming["id"].tolist())
    progress(f"Predicting {len(upcoming)} upcoming matches in {upcoming['league'].nunique()} league(s)…")
    preds, info = predict_fixtures(fin, upcoming, odds, ms)
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    with session_scope() as s:
        mv = ModelVersion(name=ms.active_model, version=f"live-{run_id}", params=ms.model_dump(),
                          trained_until=fin["date"].max().date() if not fin.empty else None,
                          n_training_matches=int(len(fin)), notes="live prediction run", metrics={"leagues": info})
        s.add(mv)
        s.flush()
        for r in preds.to_dict("records"):
            s.add(Prediction(
                run_id=run_id, match_id=r["match_id"], model_version_id=mv.id, market=r["market"],
                selection=r["selection"], prob=_f(r.get("prob")), prob_low=_f(r.get("prob_low")),
                prob_high=_f(r.get("prob_high")), best_odds=_f(r.get("best_odds")),
                best_bookmaker=r.get("best_bookmaker"), implied_prob=_f(r.get("implied_prob")),
                fair_prob=_f(r.get("fair_prob")), edge=_f(r.get("edge")), ev=_f(r.get("ev")),
                ev_low=_f(r.get("ev_low")), margin=_f(r.get("margin")), confidence=r.get("confidence"),
                verdict=r["verdict"], is_recommended=bool(r.get("is_recommended", False)),
                odds_collected_at=_dt(r.get("odds_collected_at")), details=r.get("details") or {},
            ))
    n_value = int((preds["verdict"] == "value").sum()) if not preds.empty else 0
    if log_paper and n_value:
        from sportpred.services.paper import log_paper_bets

        log_paper_bets(run_id, ms)
    return {"n_predictions": int(len(preds)), "n_matches": int(upcoming["id"].nunique()), "n_value": n_value,
            "run_id": run_id}


def _f(v):
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _dt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NaT:
        return None
    return pd.Timestamp(v).to_pydatetime()


def latest_predictions(engine=None) -> pd.DataFrame:
    """Most recent prediction per match/market/selection for matches not yet played."""
    from sportpred.db import get_engine

    engine = engine or get_engine()
    q = """
    SELECT p.*, m.league, m.season, m.date, m.kickoff, m.home_team, m.away_team, m.status
    FROM predictions p JOIN matches m ON m.id = p.match_id
    WHERE p.run_id = (SELECT run_id FROM predictions ORDER BY created_at DESC, id DESC LIMIT 1)
    """
    df = pd.read_sql(q, engine)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    import json

    df["details"] = df["details"].apply(lambda d: json.loads(d) if isinstance(d, str) else (d or {}))
    return df
