"""Stake sizing with exposure limits, plus Monte-Carlo bankroll simulation.

The application is paper-trading only. These functions size *hypothetical* stakes so strategies
can be compared honestly, including how painful the losing streaks would be.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sportpred.betting.odds import kelly_fraction
from sportpred.config import StakingSettings
from sportpred.evaluation.metrics import max_drawdown


def size_stakes_for_day(day_bets: pd.DataFrame, bankroll: float, cfg: StakingSettings,
                        start_bankroll: float) -> np.ndarray:
    """Stakes (money) for one day's bets. Bets are settled together at the end of the day.

    Rules: flat % of *starting* bankroll, or fractional Kelly of *current* bankroll; then caps per bet,
    per event (match), per competition and per day — scaling down proportionally when exceeded.
    """
    if day_bets.empty or bankroll <= 0:
        return np.zeros(len(day_bets))
    if cfg.method == "flat":
        stakes = np.full(len(day_bets), cfg.flat_stake_pct / 100 * start_bankroll)
    else:
        f = kelly_fraction(day_bets["prob"].to_numpy(), day_bets["odds"].to_numpy())
        stakes = cfg.kelly_fraction * np.asarray(f) * bankroll
    stakes = np.minimum(stakes, cfg.max_stake_pct / 100 * bankroll)

    def cap_group(keys: np.ndarray, limit: float):
        nonlocal stakes
        s = pd.Series(stakes).groupby(keys).transform("sum").to_numpy()
        scale = np.where(s > limit, limit / np.maximum(s, 1e-12), 1.0)
        stakes = stakes * scale

    cap_group(day_bets["match_id"].to_numpy(), cfg.max_event_exposure_pct / 100 * bankroll)
    cap_group(day_bets["league"].to_numpy(), cfg.max_competition_exposure_pct / 100 * bankroll)
    total = stakes.sum()
    day_cap = cfg.max_daily_exposure_pct / 100 * bankroll
    if total > day_cap:
        stakes = stakes * day_cap / total
    return np.minimum(stakes, bankroll)


def simulate_bankroll(bets: pd.DataFrame, cfg: StakingSettings) -> pd.DataFrame:
    """Sequentially apply the staking rules to settled bets. Requires columns
    date, match_id, league, prob, odds, won. Returns bets with stake, pnl, bankroll_after."""
    bets = bets.sort_values(["date", "match_id"]).copy()
    bankroll = cfg.starting_bankroll
    stakes = np.zeros(len(bets))
    pnls = np.zeros(len(bets))
    after = np.zeros(len(bets))
    pos = 0
    for _, day in bets.groupby("date", sort=True):
        n = len(day)
        s = size_stakes_for_day(day, bankroll, cfg, cfg.starting_bankroll)
        won = day["won"].to_numpy(bool)
        pnl = np.where(won, s * (day["odds"].to_numpy() - 1), -s)
        bankroll += pnl.sum()
        stakes[pos:pos + n] = s
        pnls[pos:pos + n] = pnl
        after[pos:pos + n] = bankroll
        pos += n
    bets["stake"] = stakes
    bets["pnl"] = pnls
    bets["bankroll_after"] = after
    return bets


def monte_carlo(bets: pd.DataFrame, cfg: StakingSettings, prob_col: str = "prob", n_sims: int = 1000,
                seed: int = 0) -> dict:
    """Re-play the same bets with outcomes drawn from ``prob_col`` (e.g. model or closing-market
    probabilities). Shows the spread of outcomes you could plausibly get with identical skill."""
    rng = np.random.default_rng(seed)
    bets = bets.sort_values(["date", "match_id"]).reset_index(drop=True)
    p_true = bets[prob_col].to_numpy(dtype=float)
    finals, dds, ruins = [], [], 0
    day_groups = [g.index.to_numpy() for _, g in bets.groupby("date", sort=True)]
    for _ in range(n_sims):
        won_all = rng.random(len(bets)) < p_true
        bankroll = cfg.starting_bankroll
        curve = [bankroll]
        for idx in day_groups:
            day = bets.loc[idx]
            s = size_stakes_for_day(day, bankroll, cfg, cfg.starting_bankroll)
            w = won_all[idx]
            bankroll += np.where(w, s * (day["odds"].to_numpy() - 1), -s).sum()
            curve.append(bankroll)
            if bankroll <= 0.01 * cfg.starting_bankroll:
                ruins += 1
                break
        finals.append(bankroll)
        dds.append(max_drawdown(np.array(curve))[1])
    finals = np.array(finals)
    start = cfg.starting_bankroll
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
