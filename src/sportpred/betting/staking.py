"""Stake sizing with exposure limits, plus Monte-Carlo bankroll simulation.

The application is paper-trading only. These functions size *hypothetical* stakes so strategies
can be compared honestly, including how painful the losing streaks would be.

Implementation note: the per-day rules run in plain NumPy on pre-computed day groups, so a
Monte-Carlo run over thousands of bets takes seconds rather than minutes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sportpred.betting.odds import kelly_fraction
from sportpred.config import StakingSettings
from sportpred.evaluation.metrics import max_drawdown


@dataclass
class _Day:
    idx: np.ndarray        # row positions of the day's bets
    match_inv: np.ndarray  # local group id per bet (same match)
    league_inv: np.ndarray  # local group id per bet (same competition)


def _cap_groups(stakes: np.ndarray, inv: np.ndarray, limit: float) -> np.ndarray:
    sums = np.bincount(inv, weights=stakes)
    scale = np.where(sums > limit, limit / np.maximum(sums, 1e-12), 1.0)
    return stakes * scale[inv]


def _stakes(prob: np.ndarray, odds: np.ndarray, match_inv: np.ndarray, league_inv: np.ndarray,
            bankroll: float, cfg: StakingSettings, start_bankroll: float) -> np.ndarray:
    """Stakes (money) for one day's bets; the rules documented on `size_stakes_for_day`."""
    if len(prob) == 0 or bankroll <= 0:
        return np.zeros(len(prob))
    if cfg.method == "flat":
        stakes = np.full(len(prob), cfg.flat_stake_pct / 100 * start_bankroll)
    else:
        stakes = cfg.kelly_fraction * np.asarray(kelly_fraction(prob, odds), dtype=float) * bankroll
    stakes = np.minimum(stakes, cfg.max_stake_pct / 100 * bankroll)
    stakes = _cap_groups(stakes, match_inv, cfg.max_event_exposure_pct / 100 * bankroll)
    stakes = _cap_groups(stakes, league_inv, cfg.max_competition_exposure_pct / 100 * bankroll)
    total = stakes.sum()
    day_cap = cfg.max_daily_exposure_pct / 100 * bankroll
    if total > day_cap:
        stakes = stakes * day_cap / total
    return np.minimum(stakes, bankroll)


def size_stakes_for_day(day_bets: pd.DataFrame, bankroll: float, cfg: StakingSettings,
                        start_bankroll: float) -> np.ndarray:
    """Stakes (money) for one day's bets. Bets are settled together at the end of the day.

    Rules: flat % of *starting* bankroll, or fractional Kelly of *current* bankroll; then caps per bet,
    per event (match), per competition and per day — scaling down proportionally when exceeded.
    """
    if day_bets.empty:
        return np.zeros(0)
    match_inv = pd.factorize(day_bets["match_id"])[0]
    league_inv = pd.factorize(day_bets["league"])[0]
    return _stakes(day_bets["prob"].to_numpy(float), day_bets["odds"].to_numpy(float), match_inv, league_inv,
                   bankroll, cfg, start_bankroll)


def _day_groups(bets: pd.DataFrame) -> list[_Day]:
    """Bets must be sorted by date. Returns one group per date with local group ids."""
    days = []
    dates = bets["date"].to_numpy()
    matches = bets["match_id"].to_numpy()
    leagues = bets["league"].to_numpy()
    boundaries = np.flatnonzero(dates[1:] != dates[:-1]) + 1
    for idx in np.split(np.arange(len(bets)), boundaries):
        if len(idx) == 0:
            continue
        days.append(_Day(idx, pd.factorize(matches[idx])[0], pd.factorize(leagues[idx])[0]))
    return days


def simulate_bankroll(bets: pd.DataFrame, cfg: StakingSettings) -> pd.DataFrame:
    """Sequentially apply the staking rules to settled bets. Requires columns
    date, match_id, league, prob, odds, won. Returns bets with stake, pnl, bankroll_after."""
    bets = bets.sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)
    prob = bets["prob"].to_numpy(float)
    odds = bets["odds"].to_numpy(float)
    won = bets["won"].to_numpy(bool)
    stakes = np.zeros(len(bets))
    pnls = np.zeros(len(bets))
    after = np.zeros(len(bets))
    bankroll = cfg.starting_bankroll
    for d in _day_groups(bets):
        s = _stakes(prob[d.idx], odds[d.idx], d.match_inv, d.league_inv, bankroll, cfg, cfg.starting_bankroll)
        pnl = np.where(won[d.idx], s * (odds[d.idx] - 1), -s)
        bankroll += pnl.sum()
        stakes[d.idx], pnls[d.idx], after[d.idx] = s, pnl, bankroll
    bets["stake"] = stakes
    bets["pnl"] = pnls
    bets["bankroll_after"] = after
    return bets


def monte_carlo(bets: pd.DataFrame, cfg: StakingSettings, prob_col: str = "prob", n_sims: int = 1000,
                seed: int = 0) -> dict:
    """Re-play the same bets with outcomes drawn from ``prob_col`` (e.g. model or closing-market
    probabilities). Shows the spread of outcomes you could plausibly get with identical skill."""
    rng = np.random.default_rng(seed)
    bets = bets.sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)
    p_true = bets[prob_col].to_numpy(dtype=float)
    prob = bets["prob"].to_numpy(float)
    odds = bets["odds"].to_numpy(float)
    days = _day_groups(bets)
    start = cfg.starting_bankroll
    finals, dds, ruins = [], [], 0
    for _ in range(n_sims):
        won_all = rng.random(len(bets)) < p_true
        bankroll = start
        curve = [bankroll]
        for d in days:
            s = _stakes(prob[d.idx], odds[d.idx], d.match_inv, d.league_inv, bankroll, cfg, start)
            bankroll += np.where(won_all[d.idx], s * (odds[d.idx] - 1), -s).sum()
            curve.append(bankroll)
            if bankroll <= 0.01 * start:
                ruins += 1
                break
        finals.append(bankroll)
        dds.append(max_drawdown(np.array(curve))[1])
    finals = np.array(finals)
    return {
        "n_sims": n_sims,
        "median_final": float(np.median(finals)),
        "p5_final": float(np.quantile(finals, 0.05)),
        "p95_final": float(np.quantile(finals, 0.95)),
        "prob_profit": float((finals > start).mean()),
        "prob_lose_half": float((finals < 0.5 * start).mean()),
        "prob_ruin": ruins / n_sims,
        "median_max_drawdown_pct": float(np.median(dds) * 100),
        "finals": finals.tolist(),
    }
