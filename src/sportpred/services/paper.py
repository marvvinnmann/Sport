"""Paper trading: record hypothetical bets for 'value' predictions and settle them after the match.

No money is involved and no bookmaker account is ever contacted. Paper results are the honest,
forward-looking test of the model — they cannot be affected by look-ahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import select

from sportpred.betting.odds import devig
from sportpred.config import ModelSettings, load_model_settings
from sportpred.data.repository import REFERENCE_CHAIN
from sportpred.db import Match, Odds, PaperBet, Prediction, session_scope, utcnow

SEL_PRICE = {"H": "price_home", "D": "price_draw", "A": "price_away", "OVER": "price_over", "UNDER": "price_under"}
MARKET_SELS = {"1X2": ["H", "D", "A"], "OU25": ["OVER", "UNDER"]}


def log_paper_bets(run_id: str, ms: ModelSettings | None = None) -> int:
    """Create a paper bet for each recommended prediction of a run (one per match & market).
    Flat 1-unit stakes, so paper results are directly comparable with the flat-stake backtest."""
    ms = ms or load_model_settings()
    n = 0
    with session_scope() as s:
        preds = s.execute(select(Prediction).where(Prediction.run_id == run_id,
                                                   Prediction.is_recommended.is_(True))).scalars().all()
        for p in preds:
            exists = s.execute(select(PaperBet.id).where(PaperBet.match_id == p.match_id,
                                                         PaperBet.market == p.market)).first()
            if exists:
                continue
            s.add(PaperBet(prediction_id=p.id, match_id=p.match_id, market=p.market, selection=p.selection,
                           odds=p.best_odds, bookmaker=p.best_bookmaker, model_prob=p.prob, stake=1.0))
            n += 1
    return n


def _closing_fair_prob(s, match_id: int, market: str, selection: str) -> tuple[float | None, float | None]:
    mk = "1X2" if market == "1X2" else "OU"
    rows = s.execute(select(Odds).where(Odds.match_id == match_id, Odds.market == mk,
                                        Odds.is_closing.is_(True))).scalars().all()
    by_book = {r.bookmaker: r for r in rows if mk == "1X2" or abs(r.line - 2.5) < 1e-9}
    sels = MARKET_SELS[market]
    for book in REFERENCE_CHAIN:
        r = by_book.get(book)
        if r is None:
            continue
        prices = [getattr(r, SEL_PRICE[x]) for x in sels]
        if any(p is None for p in prices):
            continue
        fair = devig([np.array([p]) for p in prices], method="multiplicative" if book == "betfair_exchange" else "power")
        i = sels.index(selection)
        return float(fair[i][0]), float(prices[i])
    return None, None


def settle_paper_bets() -> int:
    n = 0
    with session_scope() as s:
        open_bets = s.execute(select(PaperBet).where(PaperBet.status == "open")).scalars().all()
        for b in open_bets:
            m = s.get(Match, b.match_id)
            if m is None:
                continue
            if m.status in ("stale", "superseded"):
                b.status, b.pnl, b.settled_at = "void", 0.0, utcnow()
                n += 1
                continue
            if m.status != "finished" or m.fthg is None:
                continue
            if b.market == "1X2":
                res = "H" if m.fthg > m.ftag else ("D" if m.fthg == m.ftag else "A")
                won = res == b.selection
            else:
                over = (m.fthg + m.ftag) > 2.5
                won = over if b.selection == "OVER" else not over
            b.status = "won" if won else "lost"
            b.pnl = b.stake * (b.odds - 1) if won else -b.stake
            fair, close_price = _closing_fair_prob(s, b.match_id, b.market, b.selection)
            b.closing_odds = close_price
            b.clv = b.odds * fair - 1 if fair else None
            b.settled_at = utcnow()
            n += 1
    return n


def paper_bets_frame(engine=None) -> pd.DataFrame:
    from sportpred.db import get_engine

    engine = engine or get_engine()
    q = """SELECT b.*, m.league, m.date, m.home_team, m.away_team, m.fthg, m.ftag
           FROM paper_bets b JOIN matches m ON m.id = b.match_id ORDER BY m.date"""
    df = pd.read_sql(q, engine)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df
