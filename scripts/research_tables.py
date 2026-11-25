"""Reproduce the pooled backtest, CLV-by-season and anchor-comparison tables used in docs/RESEARCH.md.

Run after `sportpred evaluate`:   python scripts/research_tables.py
Writes reports/pooled_backtests.csv, reports/clv_by_season.csv, reports/anchor_comparison.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from sportpred.betting.odds import devig  # noqa: E402
from sportpred.config import StakingSettings  # noqa: E402
from sportpred.data.repository import best_price  # noqa: E402
from sportpred.evaluation.backtest import BacktestConfig, run_backtest  # noqa: E402
from sportpred.services.evaluation import available_walk_forward_models, load_walk_forward  # noqa: E402

OUT = ROOT / "reports"
FLAT = StakingSettings(method="flat")


def pooled() -> pd.DataFrame:
    rows = []
    for model in available_walk_forward_models():
        df = load_walk_forward(model)
        for market in ("1X2", "OU25"):
            if market == "OU25" and ("p_OVER" not in df or df["p_OVER"].isna().all()):
                continue
            for src in ("market_avg", "pinnacle", "bet365", "best_book", "market_max"):
                for ev in (0.02, 0.03, 0.04):
                    _, s = run_backtest(df, BacktestConfig(market=market, price_source=src, min_ev=ev, staking=FLAT))
                    rows.append({"model": model, "market": market, "price_source": src, "min_ev": ev,
                                 "bets": s.get("n_bets", 0), "flat_yield": s.get("flat_yield"),
                                 "yield_ci_low": s.get("flat_yield_ci_low"), "yield_ci_high": s.get("flat_yield_ci_high"),
                                 "avg_clv": s.get("avg_clv"), "clv_ci_low": s.get("avg_clv_ci_low"),
                                 "clv_ci_high": s.get("avg_clv_ci_high"), "share_beat_close": s.get("share_positive_clv"),
                                 "avg_odds": s.get("avg_odds")})
        print("pooled:", model)
    return pd.DataFrame(rows)


def by_season() -> pd.DataFrame:
    rows = []
    for model in [m for m in ("market", "blend", "ensemble") if m in available_walk_forward_models()]:
        df = load_walk_forward(model)
        for market in ("1X2", "OU25"):
            for season, g in df.groupby("season"):
                _, s = run_backtest(g, BacktestConfig(market=market, price_source="best_book", min_ev=0.03, staking=FLAT))
                rows.append({"model": model, "market": market, "season": season, "bets": s.get("n_bets", 0),
                             "flat_yield": s.get("flat_yield"), "yield_ci_low": s.get("flat_yield_ci_low"),
                             "yield_ci_high": s.get("flat_yield_ci_high"), "avg_clv": s.get("avg_clv"),
                             "clv_ci_low": s.get("avg_clv_ci_low"), "clv_ci_high": s.get("avg_clv_ci_high"),
                             "closing_reference": g["ref_close_source_1X2"].mode().iloc[0],
                             "anchor": g["ref_pre_source_1X2"].mode().iloc[0]})
    return pd.DataFrame(rows)


def anchors() -> pd.DataFrame:
    """Which pre-closing price is the best anchor? Bet best-book prices with EV >= 3% vs each anchor."""
    d0 = load_walk_forward("market")
    rows = []
    eras = {"2019-2025 (Pinnacle available)": ["1920", "2021", "2122", "2223", "2324", "2425"],
            "2025-2027 (Pinnacle gone)": ["2526", "2627"]}
    for era, seasons in eras.items():
        d = d0[d0["season"].isin(seasons)]
        close = d[["ref_close_H", "ref_close_D", "ref_close_A"]].to_numpy(float)
        prices = [best_price(d, s)[0].to_numpy(float) for s in "HDA"]
        for name, method in (("pinnacle", "power"), ("market_avg", "power"), ("betfair_exchange", "multiplicative")):
            cols = [f"{name}_{s}" for s in "HDA"]
            if not all(c in d for c in cols):
                continue
            ok = d[cols].notna().all(axis=1).to_numpy()
            P = np.full((len(d), 3), np.nan)
            P[ok] = np.column_stack(devig([d.loc[ok, c].to_numpy(float) for c in cols], method=method))
            clv = []
            for i in range(3):
                ev = P[:, i] * prices[i] - 1
                c = prices[i] * close[:, i] - 1
                m = (ev >= 0.03) & (prices[i] >= 1.5) & (prices[i] <= 5) & np.isfinite(c)
                clv.append(c[m])
            c = np.concatenate(clv)
            if len(c):
                rows.append({"era": era, "anchor": name, "bets": len(c), "avg_clv": c.mean(),
                             "std_error": c.std(ddof=1) / np.sqrt(len(c))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    pooled().to_csv(OUT / "pooled_backtests.csv", index=False)
    by_season().to_csv(OUT / "clv_by_season.csv", index=False)
    anchors().to_csv(OUT / "anchor_comparison.csv", index=False)
    print("wrote reports/pooled_backtests.csv, clv_by_season.csv, anchor_comparison.csv")
