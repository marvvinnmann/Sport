"""Read helpers that turn database tables into analysis-ready pandas frames."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.engine import Engine

from sportpred.betting.odds import devig
from sportpred.data.football_data import AGGREGATE_BOOKMAKERS
from sportpred.db import Match, Odds, get_engine

MATCH_COLS = ["id", "match_key", "league", "season", "date", "kickoff", "home_team", "away_team", "status",
              "fthg", "ftag", "hthg", "htag", "hs", "as", "hst", "ast", "hc", "ac", "hy", "ay", "hr", "ar",
              "hxg", "axg", "referee", "data_label", "updated_at"]

# Preference order for the CLOSING reference price (used to measure closing line value). At kick-off
# the exchange is liquid and its prices are sharp.
REFERENCE_CHAIN = ["pinnacle", "betfair_exchange", "market_avg"]
# Preference order for the PRE-CLOSING anchor used to value bets. Measured out-of-sample (see
# docs/RESEARCH.md §5.4): the provider's Tue/Fri Betfair Exchange snapshot is a poor anchor (thin
# liquidity, back-price only) while the de-vigged market average is nearly as good as Pinnacle.
ANCHOR_CHAIN = ["pinnacle", "market_avg", "betfair_exchange"]
EXCHANGE_COMMISSION = 0.02  # typical Betfair commission on net winnings (varies by account)


def load_matches(engine: Engine | None = None, leagues: list[str] | None = None,
                 statuses: list[str] | None = None, from_date: dt.date | None = None,
                 to_date: dt.date | None = None) -> pd.DataFrame:
    engine = engine or get_engine()
    t = Match.__table__
    q = select(t)
    if leagues:
        q = q.where(t.c.league.in_(leagues))
    if statuses:
        q = q.where(t.c.status.in_(statuses))
    if from_date:
        q = q.where(t.c.date >= from_date)
    if to_date:
        q = q.where(t.c.date <= to_date)
    df = pd.read_sql(q, engine)
    if df.empty:
        return pd.DataFrame(columns=MATCH_COLS + ["as_"])
    df = df.rename(columns={"as": "as_"})
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "league", "home_team"]).reset_index(drop=True)


def load_odds_long(engine: Engine | None = None, match_ids: list[int] | None = None,
                   markets: tuple[str, ...] = ("1X2", "OU")) -> pd.DataFrame:
    engine = engine or get_engine()
    t = Odds.__table__
    q = select(t).where(t.c.market.in_(markets))
    if match_ids is not None:
        if len(match_ids) == 0:
            return pd.DataFrame(columns=[c.name for c in t.columns])
        if len(match_ids) < 5000:
            q = q.where(t.c.match_id.in_([int(x) for x in match_ids]))
    df = pd.read_sql(q, engine)
    if match_ids is not None and len(match_ids) >= 5000:
        df = df[df["match_id"].isin(set(match_ids))]
    return df


def drop_implausible_odds(odds_long: pd.DataFrame) -> pd.DataFrame:
    """Remove real-bookmaker prices whose margin is impossible (data-entry errors).

    1X2 overround must lie in [0.97, 1.30]; two-way markets in [0.97, 1.20]. Aggregates
    (market max/avg) are exempt: a market maximum below 100% is normal across bookmakers."""
    if odds_long.empty:
        return odds_long
    o = odds_long
    real = ~o["bookmaker"].isin(AGGREGATE_BOOKMAKERS)
    with np.errstate(divide="ignore", invalid="ignore"):
        x12 = 1 / o["price_home"] + 1 / o["price_draw"] + 1 / o["price_away"]
        two_way = np.where(o["market"] == "OU", 1 / o["price_over"] + 1 / o["price_under"],
                           1 / o["price_home"] + 1 / o["price_away"])
    over = np.where(o["market"] == "1X2", x12, two_way)
    lo, hi = 0.97, np.where(o["market"] == "1X2", 1.30, 1.20)
    bad = real & ((over < lo) | (over > hi))
    return o[~bad]


def odds_wide(odds_long: pd.DataFrame) -> pd.DataFrame:
    """One row per match. Columns like ``pinnacle_H``, ``pinnacle_c_H`` (closing), ``market_avg_O``."""
    if odds_long.empty:
        return pd.DataFrame()
    odds_long = drop_implausible_odds(odds_long)
    parts = []
    x12 = odds_long[odds_long["market"] == "1X2"]
    if not x12.empty:
        x12 = x12.assign(tag=x12["bookmaker"] + np.where(x12["is_closing"], "_c", ""))
        w = x12.pivot_table(index="match_id", columns="tag", values=["price_home", "price_draw", "price_away"],
                            aggfunc="first")
        w.columns = [f"{tag}_{ {'price_home': 'H', 'price_draw': 'D', 'price_away': 'A'}[v] }" for v, tag in w.columns]
        parts.append(w)
    ou = odds_long[(odds_long["market"] == "OU") & (np.isclose(odds_long["line"], 2.5))]
    if not ou.empty:
        ou = ou.assign(tag=ou["bookmaker"] + np.where(ou["is_closing"], "_c", ""))
        w = ou.pivot_table(index="match_id", columns="tag", values=["price_over", "price_under"], aggfunc="first")
        w.columns = [f"{tag}_{ {'price_over': 'O', 'price_under': 'U'}[v] }" for v, tag in w.columns]
        parts.append(w)
    out = pd.concat(parts, axis=1)
    out.columns = [c.replace(" ", "") for c in out.columns]
    return out


def bookmakers_in(wide: pd.DataFrame, suffix: str = "_H", closing: bool = False) -> list[str]:
    books = set()
    for c in wide.columns:
        if not c.endswith(suffix):
            continue
        base = c[: -len(suffix)]
        if base.endswith("_c") != closing:
            continue
        books.add(base[:-2] if closing else base)
    return sorted(books)


def add_reference_probabilities(df: pd.DataFrame, method: str = "power") -> pd.DataFrame:
    """Add fair (margin-free) probabilities from the best available sharp price.

    ``ref_close_*``  closing reference (Pinnacle → Betfair Exchange → market average)
    ``ref_pre_*``    pre-closing anchor (Pinnacle → market average → Betfair Exchange)
    plus ``ref_close_source`` / ``ref_pre_source`` telling which bookmaker was used.
    """
    df = df.copy()
    for stage, suffix, chain in (("close", "_c", REFERENCE_CHAIN), ("pre", "", ANCHOR_CHAIN)):
        for market, sels in (("1X2", ["H", "D", "A"]), ("OU", ["O", "U"])):
            names = [f"ref_{stage}_{s}" for s in sels]
            for n in names:
                df[n] = np.nan
            src_col = f"ref_{stage}_source_{market}"
            df[src_col] = None
            for book in chain:
                cols = [f"{book}{suffix}_{s}" for s in sels]
                if not all(c in df.columns for c in cols):
                    continue
                need = df[names[0]].isna() & df[cols].notna().all(axis=1)
                if not need.any():
                    continue
                prices = [df.loc[need, c].to_numpy(dtype=float) for c in cols]
                if book == "betfair_exchange":
                    # exchange prices are near-fair; still normalise to remove the tiny over-round
                    fair = devig(prices, method="multiplicative")
                else:
                    fair = devig(prices, method=method)
                for n, p in zip(names, fair, strict=True):
                    df.loc[need, n] = p
                df.loc[need, src_col] = book
    return df


def best_price(df: pd.DataFrame, sel: str, closing: bool = False, exclude_aggregates: bool = True) -> tuple[pd.Series, pd.Series]:
    """Best real-bookmaker price per match for selection ``sel`` (H/D/A/O/U) and which bookmaker."""
    suffix = f"_c_{sel}" if closing else f"_{sel}"
    cols = [c for c in df.columns if c.endswith(suffix) and (closing or not c.endswith(f"_c_{sel}"))]
    if exclude_aggregates:
        cols = [c for c in cols if not any(c.startswith(a) for a in AGGREGATE_BOOKMAKERS)]
    cols = [c for c in cols if not c.startswith("ref_")]
    if not cols:
        return pd.Series(np.nan, index=df.index), pd.Series(None, index=df.index, dtype=object)
    sub = df[cols].astype(float)
    if any(c.startswith("betfair_exchange") for c in cols):
        exch = [c for c in cols if c.startswith("betfair_exchange")]
        sub[exch] = 1 + (sub[exch] - 1) * (1 - EXCHANGE_COMMISSION)
    best = sub.max(axis=1)
    has_any = sub.notna().any(axis=1)
    which = pd.Series(None, index=df.index, dtype=object)
    which[has_any] = sub[has_any].idxmax(axis=1).str.replace(suffix, "", regex=False)
    return best, which


def analysis_frame(engine: Engine | None = None, leagues: list[str] | None = None,
                   statuses: list[str] | None = None) -> pd.DataFrame:
    """Matches joined with wide odds and reference probabilities."""
    engine = engine or get_engine()
    m = load_matches(engine, leagues, statuses)
    if m.empty:
        return m
    o = odds_wide(load_odds_long(engine, m["id"].tolist()))
    df = m.merge(o, left_on="id", right_index=True, how="left") if not o.empty else m
    return add_reference_probabilities(df)
