"""League & market screening: which league/market/model combination shows any credible value?

Reads the walk-forward predictions produced by `sportpred evaluate` and writes
reports/league_screening.md (+ .json). Run:  python scripts/league_screening.py

For every league × market × model it reports, on the out-of-sample test period:
  * log loss vs the closing market (negative = model better), with a 95% bootstrap CI
  * flat-stake yield and average CLV when betting EV ≥ 4% at realistic (market average) and
    optimistic (market maximum) pre-closing prices, with 95% CIs
  * the bookmaker margin in that league (cost of playing)
A combination is only called promising if CLV is positive with a CI above zero — realised yield on a
few hundred bets is far too noisy on its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sportpred.config import StakingSettings  # noqa: E402
from sportpred.evaluation.backtest import BacktestConfig, run_backtest  # noqa: E402
from sportpred.evaluation.metrics import bootstrap_mean_ci, log_loss_per_match  # noqa: E402
from sportpred.leagues import league_label  # noqa: E402
from sportpred.services.evaluation import (  # noqa: E402
    available_walk_forward_models,
    load_evaluation_summary,
    load_walk_forward,
)

OUT = ROOT / "reports"
MODELS = ["dixon_coles", "elo", "gbm", "blend", "ensemble"]


def margin_1x2(df: pd.DataFrame) -> float:
    cols = ["market_avg_H", "market_avg_D", "market_avg_A"]
    if not all(c in df for c in cols):
        return float("nan")
    return float((1 / df[cols]).sum(axis=1).mean() - 1)


def margin_ou(df: pd.DataFrame) -> float:
    cols = ["market_avg_O", "market_avg_U"]
    if not all(c in df for c in cols):
        return float("nan")
    return float((1 / df[cols]).sum(axis=1).mean() - 1)


def ll_diff(df: pd.DataFrame, market: str) -> tuple[float, float, float, int]:
    if market == "1X2":
        d = df.dropna(subset=["p_H", "ref_close_H"])
        if d.empty:
            return (np.nan,) * 3 + (0,)
        y = d["y_1x2"].to_numpy()
        diff = log_loss_per_match(d[["p_H", "p_D", "p_A"]].to_numpy(float), y) - log_loss_per_match(
            d[["ref_close_H", "ref_close_D", "ref_close_A"]].to_numpy(float), y)
    else:
        d = df.dropna(subset=["p_OVER", "ref_close_O"])
        if d.empty:
            return (np.nan,) * 3 + (0,)
        y = d["y_over"].to_numpy()
        diff = log_loss_per_match(d["p_OVER"].to_numpy(float), y) - log_loss_per_match(d["ref_close_O"].to_numpy(float), y)
    m, lo, hi = bootstrap_mean_ci(diff, n_boot=1000)
    return m, lo, hi, len(d)


def main() -> None:
    models = [m for m in MODELS if m in available_walk_forward_models()]
    rows = []
    for model in models:
        df = load_walk_forward(model)
        for league, g in df.groupby("league"):
            for market in ("1X2", "OU25"):
                if market == "OU25" and ("p_OVER" not in g or g["p_OVER"].isna().all()):
                    continue
                m, lo, hi, n = ll_diff(g, market)
                row = {"model": model, "league": league, "market": market, "n": n, "ll_diff": m, "ll_lo": lo, "ll_hi": hi,
                       "margin": margin_1x2(g) if market == "1X2" else margin_ou(g)}
                for src in ("market_avg", "market_max"):
                    cfg = BacktestConfig(market=market, price_source=src, min_ev=0.04,
                                         staking=StakingSettings(method="flat"))
                    bets, s = run_backtest(g, cfg)
                    row[f"{src}_bets"] = s.get("n_bets", 0)
                    row[f"{src}_yield"] = s.get("flat_yield")
                    row[f"{src}_yield_lo"] = s.get("flat_yield_ci_low")
                    row[f"{src}_yield_hi"] = s.get("flat_yield_ci_high")
                    row[f"{src}_clv"] = s.get("avg_clv")
                    row[f"{src}_clv_lo"] = s.get("avg_clv_ci_low")
                    row[f"{src}_clv_hi"] = s.get("avg_clv_ci_high")
                rows.append(row)
        print(f"screened {model}")
    res = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    res.to_json(OUT / "league_screening.json", orient="records", indent=1)
    write_markdown(res)


def f(x, pct=True, signed=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return (f"{x:+.1%}" if signed else f"{x:.1%}") if pct else f"{x:+.4f}"


def write_markdown(res: pd.DataFrame) -> None:
    ev = load_evaluation_summary() or {}
    lines = ["# League & market screening (out-of-sample)", "",
             f"Generated from walk-forward predictions; test period from {ev.get('test_from', '2019-07-01')}. "
             "Reproduce with `sportpred evaluate` then `python scripts/league_screening.py`.", "",
             "* **Δ log loss** = model − closing market (negative = model better). 95% bootstrap CI in brackets.",
             "* **Yield / CLV** = flat 1-unit bets on every selection with model EV ≥ 4%, odds 1.5–5.0, priced at the "
             "pre-closing *market average* (realistic) or *market maximum* (optimistic, often stale prices).",
             "* **Margin** = average bookmaker over-round (the cost of betting) in that league.", ""]
    # overall model table
    lines += ["## 1. Model comparison (all leagues pooled)", "",
              "| Model | Market | Matches | Δ log loss vs closing (95% CI) |", "|---|---|---|---|"]
    for name, r in (ev.get("overall") or {}).items():
        for market, x in r.items():
            vs = x["vs_closing"]
            lines.append(f"| {name} | {market} | {x['model']['n']:,} | {vs['mean_diff']:+.4f} "
                         f"({vs['ci_low']:+.4f} … {vs['ci_high']:+.4f}) |")
    for model in res["model"].unique():
        sub = res[res["model"] == model]
        lines += ["", f"## Model: {model}", ""]
        for market in ("1X2", "OU25"):
            s = sub[sub["market"] == market].sort_values("ll_diff")
            if s.empty:
                continue
            lines += [f"### {market}", "",
                      "| League | Matches | Margin | Δ log loss (95% CI) | Bets (avg) | Yield avg (95% CI) | CLV avg (95% CI) "
                      "| Bets (max) | Yield max (95% CI) | CLV max |", "|---|---|---|---|---|---|---|---|---|---|"]
            for r in s.itertuples():
                lines.append(
                    f"| {league_label(r.league)} | {r.n:,} | {f(r.margin, signed=False)} | {r.ll_diff:+.4f} "
                    f"({r.ll_lo:+.4f} … {r.ll_hi:+.4f}) | {r.market_avg_bets} | {f(r.market_avg_yield)} "
                    f"({f(r.market_avg_yield_lo)} … {f(r.market_avg_yield_hi)}) | {f(r.market_avg_clv)} "
                    f"({f(r.market_avg_clv_lo)} … {f(r.market_avg_clv_hi)}) | {r.market_max_bets} | "
                    f"{f(r.market_max_yield)} ({f(r.market_max_yield_lo)} … {f(r.market_max_yield_hi)}) | {f(r.market_max_clv)} |")
            lines.append("")
    credible = res[(res["market_avg_clv_lo"] > 0) | (res["market_max_clv_lo"] > 0)]
    lines += ["## Combinations with CLV significantly above zero", ""]
    if credible.empty:
        lines.append("None. No league/market/model combination beat the closing line with statistical confidence.")
    else:
        for r in credible.itertuples():
            lines.append(f"- {r.model} · {league_label(r.league)} · {r.market}: CLV avg {f(r.market_avg_clv)} "
                         f"(CI {f(r.market_avg_clv_lo)}…{f(r.market_avg_clv_hi)}), CLV max {f(r.market_max_clv)} "
                         f"(CI {f(r.market_max_clv_lo)}…{f(r.market_max_clv_hi)})")
    (OUT / "league_screening.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT / 'league_screening.md'}")


if __name__ == "__main__":
    main()
