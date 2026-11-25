"""Betting backtest on walk-forward predictions.

Realism rules
-------------
* Bets are struck at **pre-closing** prices (collected Tue/Fri by the provider) — never at the
  closing price, which is only known at kick-off.
* ``price_source`` chooses which price you could realistically get:
    pinnacle      — one sharp bookmaker (until 2025/26)
    bet365        — one mainstream bookmaker
    market_avg    — average across bookmakers (conservative)
    best_book     — best real bookmaker per selection (needs accounts everywhere; optimistic)
    market_max    — provider's market maximum (MOST optimistic: often stale/outlier prices)
    betfair_exchange — exchange price after 2% commission
* Closing line value (CLV) is measured against the sharp closing reference (Pinnacle → Betfair
  Exchange → market average), the most reliable short-sample indicator of real skill.
* One bet per match per market at most (the selection with the highest EV that passes all rules).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from sportpred.betting.odds import closing_line_value, expected_value, implied_probability
from sportpred.betting.staking import simulate_bankroll
from sportpred.config import StakingSettings
from sportpred.data.repository import EXCHANGE_COMMISSION, best_price
from sportpred.evaluation.metrics import bootstrap_mean_ci, longest_losing_streak, max_drawdown

SELECTIONS = {"1X2": ["H", "D", "A"], "OU25": ["O", "U"]}
PROB_COL = {"H": "p_H", "D": "p_D", "A": "p_A", "O": "p_OVER", "U": "p_UNDER"}
REF_COL = {"H": "ref_close_H", "D": "ref_close_D", "A": "ref_close_A", "O": "ref_close_O", "U": "ref_close_U"}
PRICE_SOURCES = ["pinnacle", "bet365", "market_avg", "best_book", "market_max", "betfair_exchange"]


@dataclass
class BacktestConfig:
    market: str = "1X2"
    price_source: str = "market_avg"
    min_ev: float = 0.04
    min_edge: float = 0.0
    min_odds: float = 1.5
    max_odds: float = 5.0
    min_prob: float = 0.0
    selections: list[str] | None = None  # restrict, e.g. ["H", "A"]
    staking: StakingSettings = field(default_factory=StakingSettings)
    leagues: list[str] | None = None
    date_from: str | None = None
    date_to: str | None = None
    label: str = "model"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["staking"] = self.staking.model_dump()
        return d


def bet_prices(df: pd.DataFrame, sel: str, source: str) -> tuple[pd.Series, pd.Series]:
    """Pre-closing price available for a selection under a price-source assumption."""
    if source == "best_book":
        return best_price(df, sel, closing=False)
    col = f"{source}_{sel}"
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index), pd.Series(None, index=df.index, dtype=object)
    price = df[col].astype(float)
    if source == "betfair_exchange":
        price = 1 + (price - 1) * (1 - EXCHANGE_COMMISSION)
    return price, pd.Series(np.where(price.notna(), source, None), index=df.index)


def candidate_bets(df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """All selections with model probability, available price, EV, edge and outcome."""
    rows = []
    sels = cfg.selections or SELECTIONS[cfg.market]
    for sel in sels:
        pcol = PROB_COL[sel]
        if pcol not in df.columns:
            continue
        price, book = bet_prices(df, sel, cfg.price_source)
        part = pd.DataFrame({
            "match_id": df["match_id"].to_numpy(), "date": df["date"].to_numpy(),
            "league": df["league"].to_numpy(), "season": df["season"].to_numpy(),
            "home_team": df["home_team"].to_numpy(), "away_team": df["away_team"].to_numpy(),
            "selection": sel, "prob": df[pcol].to_numpy(dtype=float), "odds": price.to_numpy(dtype=float),
            "bookmaker": book.to_numpy(), "ref_close_prob": df[REF_COL[sel]].to_numpy(dtype=float)
            if REF_COL[sel] in df.columns else np.nan,
        })
        if cfg.market == "1X2":
            y = df["y_1x2"].to_numpy()
            part["won"] = y == {"H": 0, "D": 1, "A": 2}[sel]
        else:
            yo = df["y_over"].to_numpy()
            part["won"] = (yo == 1) if sel == "O" else (yo == 0)
        rows.append(part)
    if not rows:
        return pd.DataFrame()
    c = pd.concat(rows, ignore_index=True)
    c["implied"] = implied_probability(c["odds"].to_numpy())
    c["ev"] = expected_value(c["prob"].to_numpy(), c["odds"].to_numpy())
    c["edge"] = c["prob"] - c["implied"]
    c["clv"] = closing_line_value(c["odds"].to_numpy(), c["ref_close_prob"].to_numpy())
    return c


def select_bets(cands: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    c = cands.dropna(subset=["prob", "odds"])
    ok = ((c["ev"] >= cfg.min_ev) & (c["edge"] >= cfg.min_edge) & (c["odds"] >= cfg.min_odds)
          & (c["odds"] <= cfg.max_odds) & (c["prob"] >= cfg.min_prob))
    c = c[ok]
    if c.empty:
        return c
    c = c.sort_values("ev", ascending=False).drop_duplicates("match_id", keep="first")
    return c.sort_values(["date", "match_id"])


def summarize(bets: pd.DataFrame, cfg: BacktestConfig, n_matches: int | None = None) -> dict:
    if bets.empty:
        return {"n_bets": 0, "n_matches": n_matches, "message": "No bets met the criteria."}
    flat_pnl = np.where(bets["won"], bets["odds"] - 1, -1.0)
    y_mean, y_lo, y_hi = bootstrap_mean_ci(flat_pnl, block=bets["date"].to_numpy())
    clv = bets["clv"].to_numpy(dtype=float)
    clv_ok = np.isfinite(clv)
    clv_mean, clv_lo, clv_hi = bootstrap_mean_ci(clv[clv_ok]) if clv_ok.any() else (np.nan,) * 3
    equity = cfg.staking.starting_bankroll + bets["pnl"].cumsum().to_numpy()
    dd_abs, dd_pct = max_drawdown(np.concatenate([[cfg.staking.starting_bankroll], equity]))
    turnover = float(bets["stake"].sum())
    profit = float(bets["pnl"].sum())
    se = flat_pnl.std(ddof=1) / np.sqrt(len(flat_pnl)) if len(flat_pnl) > 1 else np.nan
    return {
        "n_bets": int(len(bets)),
        "n_matches": n_matches,
        "hit_rate": float(bets["won"].mean()),
        "avg_odds": float(bets["odds"].mean()),
        "avg_model_prob": float(bets["prob"].mean()),
        "avg_ev_expected": float(bets["ev"].mean()),
        "flat_yield": y_mean,  # profit per 1 unit staked, flat stakes
        "flat_yield_ci_low": y_lo,
        "flat_yield_ci_high": y_hi,
        "flat_profit_units": float(flat_pnl.sum()),
        "t_stat": float(y_mean / se) if se and np.isfinite(se) and se > 0 else None,
        "staked_turnover": turnover,
        "staked_profit": profit,
        "staked_yield": profit / turnover if turnover > 0 else None,
        "roi_on_bankroll": profit / cfg.staking.starting_bankroll,
        "final_bankroll": float(equity[-1]),
        "max_drawdown": dd_abs,
        "max_drawdown_pct": dd_pct,
        "longest_losing_streak": longest_losing_streak(bets["won"].to_numpy()),
        "avg_clv": clv_mean,
        "avg_clv_ci_low": clv_lo,
        "avg_clv_ci_high": clv_hi,
        "share_positive_clv": float((clv[clv_ok] > 0).mean()) if clv_ok.any() else None,
        "clv_coverage": float(clv_ok.mean()),
    }


def run_backtest(df: pd.DataFrame, cfg: BacktestConfig) -> tuple[pd.DataFrame, dict]:
    d = _filter(df, cfg)
    cands = candidate_bets(d, cfg)
    bets = select_bets(cands, cfg) if not cands.empty else cands
    if bets.empty:
        return bets, summarize(bets, cfg, n_matches=int(d["match_id"].nunique()) if len(d) else 0)
    bets = simulate_bankroll(bets, cfg.staking)
    return bets, summarize(bets, cfg, n_matches=int(d["match_id"].nunique()))


def _filter(df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    d = df
    if cfg.leagues:
        d = d[d["league"].isin(cfg.leagues)]
    if cfg.date_from:
        d = d[pd.to_datetime(d["date"]) >= pd.Timestamp(cfg.date_from)]
    if cfg.date_to:
        d = d[pd.to_datetime(d["date"]) <= pd.Timestamp(cfg.date_to)]
    return d


# ----------------------------------------------------------------------------- baselines

def baseline_strategies(df: pd.DataFrame, cfg: BacktestConfig, n_random: int = 200, seed: int = 0) -> pd.DataFrame:
    """Simple strategies on the same matches and prices (flat 1-unit stakes) for comparison."""
    d = _filter(df, cfg)
    rows = []
    sels = SELECTIONS[cfg.market]
    prices = {s: bet_prices(d, s, cfg.price_source)[0].to_numpy(dtype=float) for s in sels}
    if cfg.market == "1X2":
        won = {s: d["y_1x2"].to_numpy() == i for i, s in enumerate(sels)}
    else:
        won = {"O": d["y_over"].to_numpy() == 1, "U": d["y_over"].to_numpy() == 0}

    def add(name, mask_sel):
        pnl = []
        for s, mask in mask_sel.items():
            p = prices[s][mask]
            w = won[s][mask]
            ok = np.isfinite(p)
            pnl.append(np.where(w[ok], p[ok] - 1, -1.0))
        pnl = np.concatenate(pnl) if pnl else np.array([])
        if len(pnl):
            rows.append({"strategy": name, "n_bets": len(pnl), "flat_yield": float(pnl.mean()),
                         "profit_units": float(pnl.sum())})

    n = len(d)
    for s in sels:
        add(f"Back every {_sel_name(s)}", {s: np.ones(n, bool)})
    P = np.column_stack([prices[s] for s in sels])
    valid = np.isfinite(P).all(axis=1)
    fav = np.where(valid, np.nanargmin(np.where(np.isfinite(P), P, np.inf), axis=1), -1)
    dog = np.where(valid, np.nanargmax(np.where(np.isfinite(P), P, -np.inf), axis=1), -1)
    add("Back the favourite", {s: fav == i for i, s in enumerate(sels)})
    add("Back the outsider", {s: dog == i for i, s in enumerate(sels)})
    # random selections inside the same odds band as the strategy
    rng = np.random.default_rng(seed)
    band = (P >= cfg.min_odds) & (P <= cfg.max_odds) & np.isfinite(P)
    W = np.column_stack([won[s] for s in sels])
    rand_yields = []
    for _ in range(n_random):
        choice = np.array([rng.choice(np.where(b)[0]) if b.any() else -1 for b in band])
        ok = choice >= 0
        if ok.sum() == 0:
            break
        p = P[ok, choice[ok]]
        w = W[ok, choice[ok]]
        rand_yields.append(np.where(w, p - 1, -1).mean())
    if rand_yields:
        rows.append({"strategy": f"Random pick in odds band {cfg.min_odds}-{cfg.max_odds} (avg of {len(rand_yields)})",
                     "n_bets": int(band.any(axis=1).sum()), "flat_yield": float(np.mean(rand_yields)),
                     "profit_units": float(np.mean(rand_yields) * band.any(axis=1).sum())})
    return pd.DataFrame(rows)


def _sel_name(s: str) -> str:
    return {"H": "home win", "D": "draw", "A": "away win", "O": "over 2.5", "U": "under 2.5"}[s]


def threshold_sensitivity(df: pd.DataFrame, cfg: BacktestConfig,
                          ev_grid=(0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15)) -> pd.DataFrame:
    """How results change with the minimum-EV threshold (flat stakes). Stable = more credible."""
    d = _filter(df, cfg)
    cands = candidate_bets(d, cfg)
    rows = []
    for t in ev_grid:
        c2 = BacktestConfig(**{**cfg.__dict__, "min_ev": t})
        b = select_bets(cands, c2)
        if b.empty:
            rows.append({"min_ev": t, "n_bets": 0})
            continue
        pnl = np.where(b["won"], b["odds"] - 1, -1.0)
        m, lo, hi = bootstrap_mean_ci(pnl, n_boot=1000, block=b["date"].to_numpy())
        rows.append({"min_ev": t, "n_bets": len(b), "flat_yield": m, "ci_low": lo, "ci_high": hi,
                     "avg_clv": float(np.nanmean(b["clv"])) if b["clv"].notna().any() else None})
    return pd.DataFrame(rows)
