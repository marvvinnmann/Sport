"""PAGE 7 — Backtesting strategies on out-of-sample (walk-forward) predictions."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import MODEL_LABEL, SERIES, disclaimer, load_wf, pct, setup_page, style_fig, wf_mtime

from sportpred.betting.staking import monte_carlo
from sportpred.config import StakingSettings, load_model_settings
from sportpred.evaluation.backtest import (
    PRICE_SOURCES,
    BacktestConfig,
    baseline_strategies,
    run_backtest,
    threshold_sensitivity,
)
from sportpred.leagues import league_label
from sportpred.services.evaluation import available_walk_forward_models

setup_page("Backtesting", "🧪")
disclaimer()

models = available_walk_forward_models()
if not models:
    st.info("Backtests use stored walk-forward predictions. Run `sportpred evaluate` first.")
    st.stop()
ms = load_model_settings()

PRICE_HELP = {"market_avg": "Average bookmaker price (conservative, realistic)",
              "pinnacle": "Pinnacle only (sharp bookmaker; until 2025/26)",
              "bet365": "Bet365 only", "best_book": "Best named bookmaker (needs many accounts)",
              "market_max": "Market maximum (optimistic — includes stale/outlier prices)",
              "betfair_exchange": "Betfair Exchange after 2% commission (from 2024/25)"}

with st.form("bt"):
    c1, c2, c3 = st.columns(3)
    model = c1.selectbox("Model", models, format_func=lambda m: MODEL_LABEL.get(m, m),
                         index=models.index("blend") if "blend" in models else 0)
    df_all = load_wf(model, wf_mtime(model))
    market = c2.selectbox("Market", ["1X2", "OU25"], format_func=lambda m: {"1X2": "Match result", "OU25": "Over/Under 2.5"}[m])
    price = c3.selectbox("Bet at prices from", PRICE_SOURCES, format_func=lambda s: PRICE_HELP[s])
    lg_all = sorted(df_all["league"].unique())
    c1, c2 = st.columns([2, 1])
    leagues = c1.multiselect("Leagues (empty = all)", lg_all, format_func=league_label)
    dmin, dmax = pd.to_datetime(df_all["date"]).min().date(), pd.to_datetime(df_all["date"]).max().date()
    dates = c2.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
    c1, c2, c3, c4 = st.columns(4)
    min_ev = c1.number_input("Min EV (%)", -10.0, 50.0, ms.min_ev * 100, 0.5) / 100
    min_edge = c2.number_input("Min edge (pp)", 0.0, 30.0, 0.0, 0.5) / 100
    min_odds = c3.number_input("Min odds", 1.01, 20.0, ms.min_odds, 0.05)
    max_odds = c4.number_input("Max odds", 1.1, 100.0, ms.max_odds, 0.25)
    c1, c2, c3, c4 = st.columns(4)
    method = c1.selectbox("Staking", ["flat", "kelly"], format_func=lambda m: {"flat": "Flat", "kelly": "Fractional Kelly"}[m])
    kf = c2.number_input("Kelly fraction", 0.05, 1.0, ms.staking.kelly_fraction, 0.05)
    flat = c3.number_input("Flat stake (% bankroll)", 0.1, 10.0, ms.staking.flat_stake_pct, 0.1)
    bank = c4.number_input("Starting bankroll", 10.0, 1e7, ms.staking.starting_bankroll, 100.0)
    run = st.form_submit_button("▶️ Run backtest", type="primary")

if not run and "bt_result" not in st.session_state:
    st.info("Choose settings and press **Run backtest**. Only out-of-sample predictions are used: each was made "
            "by a model trained on earlier matches only.")
    st.stop()

if run:
    staking = StakingSettings(**{**ms.staking.model_dump(), "method": method, "kelly_fraction": kf,
                                 "flat_stake_pct": flat, "starting_bankroll": bank})
    cfg = BacktestConfig(market=market, price_source=price, min_ev=min_ev, min_edge=min_edge, min_odds=min_odds,
                         max_odds=max_odds, leagues=leagues or None, staking=staking,
                         date_from=str(dates[0]) if isinstance(dates, tuple) and len(dates) == 2 else None,
                         date_to=str(dates[1]) if isinstance(dates, tuple) and len(dates) == 2 else None)
    with st.spinner("Running backtest…"):
        bets, summary = run_backtest(df_all, cfg)
        base = baseline_strategies(df_all, cfg, n_random=100)
        sens = threshold_sensitivity(df_all, cfg)
    st.session_state["bt_result"] = (bets, summary, base, sens, cfg, model)

bets, summary, base, sens, cfg, model = st.session_state["bt_result"]
st.caption(f"Model: {MODEL_LABEL.get(model, model)} · market {cfg.market} · prices {cfg.price_source} · "
           f"EV ≥ {cfg.min_ev:.1%} · odds {cfg.min_odds}–{cfg.max_odds} · staking {cfg.staking.method}")
if not summary.get("n_bets"):
    st.warning(summary.get("message", "No bets."))
    st.stop()

# ---------------------------------------------------------------- KPIs
c = st.columns(6)
c[0].metric("Bets", f"{summary['n_bets']:,}", help=f"out of {summary['n_matches']:,} matches")
c[1].metric("Profit (staked)", f"{summary['staked_profit']:+,.0f}")
c[2].metric("ROI on bankroll", pct(summary["roi_on_bankroll"], digits=0, signed=True))
c[3].metric("Yield (flat)", pct(summary["flat_yield"], signed=True),
            help=f"95% CI {summary['flat_yield_ci_low']:+.1%} to {summary['flat_yield_ci_high']:+.1%}")
c[4].metric("Max drawdown", pct(summary["max_drawdown_pct"]))
c[5].metric("Avg CLV", pct(summary["avg_clv"], signed=True), help="Bet price vs fair closing price")

lo, hi = summary["flat_yield_ci_low"], summary["flat_yield_ci_high"]
if lo > 0:
    st.success(f"Flat-stake yield {summary['flat_yield']:+.1%} with a 95% CI entirely above zero ({lo:+.1%} to {hi:+.1%}). "
               "Still check CLV, the sensitivity table and paper-trading results before trusting it — many strategies "
               "were tried, so some will look good by chance.")
elif hi < 0:
    st.error(f"This strategy lost money: yield {summary['flat_yield']:+.1%}, 95% CI {lo:+.1%} to {hi:+.1%} "
             "(entirely below zero).")
else:
    st.warning(f"Result is not statistically distinguishable from zero: yield {summary['flat_yield']:+.1%}, "
               f"95% CI {lo:+.1%} to {hi:+.1%}. Treat as no demonstrated edge.")

# ---------------------------------------------------------------- equity + drawdown
eq = bets.sort_values(["date", "match_id"]).copy()
eq["date"] = pd.to_datetime(eq["date"])
fig = go.Figure()
fig.add_trace(go.Scatter(x=eq["date"], y=eq["bankroll_after"], mode="lines", name="Bankroll",
                         line=dict(width=2, color=SERIES[0]), hovertemplate="%{x|%d %b %Y}: %{y:,.0f}<extra></extra>"))
fig.add_hline(y=cfg.staking.starting_bankroll, line_color="rgba(128,128,128,0.6)", line_width=1, line_dash="dot")
st.plotly_chart(style_fig(fig, height=320, y_title="Bankroll"), width="stretch")

peak = eq["bankroll_after"].cummax()
dd = (eq["bankroll_after"] - peak) / peak
fig = go.Figure(go.Scatter(x=eq["date"], y=dd, mode="lines", fill="tozeroy", name="Drawdown",
                           line=dict(width=1.5, color=SERIES[7]),
                           hovertemplate="%{x|%d %b %Y}: %{y:.1%}<extra></extra>"))
fig.update_yaxes(tickformat=".0%")
st.plotly_chart(style_fig(fig, height=200, y_title="Drawdown from peak"), width="stretch")

# ---------------------------------------------------------------- monthly P&L + details
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Profit by month (staked)**")
    mon = eq.groupby(eq["date"].dt.to_period("M"))["pnl"].sum()
    fig = go.Figure(go.Bar(x=mon.index.to_timestamp(), y=mon.values,
                           marker_color=[SERIES[0] if v >= 0 else SERIES[7] for v in mon.values],
                           hovertemplate="%{x|%b %Y}: %{y:+,.0f}<extra></extra>"))
    st.plotly_chart(style_fig(fig, height=260), width="stretch")
with c2:
    st.markdown("**Details**")
    st.dataframe(pd.DataFrame({
        "Metric": ["Hit rate", "Average odds", "Average model probability", "Average expected EV (model's view)",
                   "Longest losing streak", "Bets beating the closing line", "Flat-stake profit (units)", "t-statistic"],
        "Value": [pct(summary["hit_rate"]), f"{summary['avg_odds']:.2f}", pct(summary["avg_model_prob"]),
                  pct(summary["avg_ev_expected"], signed=True), summary["longest_losing_streak"],
                  pct(summary["share_positive_clv"]), f"{summary['flat_profit_units']:+.1f}",
                  f"{summary['t_stat']:.2f}" if summary["t_stat"] is not None else "—"]}),
        hide_index=True, width="stretch")
    st.caption("If the model's expected EV is far above the realised yield, the model is over-confident.")

# ---------------------------------------------------------------- baselines
st.subheader("Comparison with baseline strategies (same matches, same prices, flat stakes)")
row = pd.DataFrame([{"strategy": f"This strategy ({MODEL_LABEL.get(model, model)})", "n_bets": summary["n_bets"],
                     "flat_yield": summary["flat_yield"], "profit_units": summary["flat_profit_units"]}])
comp = pd.concat([row, base], ignore_index=True)
comp["Yield"] = comp["flat_yield"].map(lambda x: pct(x, signed=True))
st.dataframe(comp.rename(columns={"strategy": "Strategy", "n_bets": "Bets", "profit_units": "Profit (units)"})[
    ["Strategy", "Bets", "Yield", "Profit (units)"]].round(1), hide_index=True, width="stretch")
st.caption("Random and 'back everything' strategies lose roughly the bookmaker margin. A real edge must clearly "
           "beat them.")

# ---------------------------------------------------------------- sensitivity
st.subheader("Sensitivity to the EV threshold")
if not sens.empty and "flat_yield" in sens:
    s2 = sens.dropna(subset=["flat_yield"])
    fig = go.Figure(go.Scatter(
        x=s2["min_ev"], y=s2["flat_yield"], mode="lines+markers", line=dict(width=2, color=SERIES[0]),
        marker=dict(size=8), name="Flat yield",
        error_y=dict(type="data", symmetric=False, array=s2["ci_high"] - s2["flat_yield"],
                     arrayminus=s2["flat_yield"] - s2["ci_low"], thickness=1.5, width=4),
        customdata=s2["n_bets"], hovertemplate="EV ≥ %{x:.0%}: yield %{y:+.1%} (%{customdata} bets)<extra></extra>"))
    fig.add_hline(y=0, line_color="rgba(128,128,128,0.7)", line_width=1)
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(style_fig(fig, height=300, x_title="Minimum EV threshold", y_title="Yield (95% CI)"), width="stretch")
    st.caption("A genuine edge should improve (or at least stay positive) as the threshold rises. Erratic jumps "
               "suggest noise.")

# ---------------------------------------------------------------- Monte Carlo
st.subheader("Staking simulation under uncertainty")
st.write("Re-play exactly these bets 500 times with random outcomes drawn from two different assumptions about "
         "the true probabilities.")
if st.button("Run simulation"):
    with st.spinner("Simulating…"):
        a = monte_carlo(bets, cfg.staking, prob_col="prob", n_sims=500)
        b_bets = bets.dropna(subset=["ref_close_prob"])
        b = monte_carlo(b_bets, cfg.staking, prob_col="ref_close_prob", n_sims=500) if len(b_bets) else None
    rows = [{"Assumption": "Model probabilities are correct", **{k: v for k, v in a.items() if k != "finals"}}]
    if b:
        rows.append({"Assumption": "Closing market is correct", **{k: v for k, v in b.items() if k != "finals"}})
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=a["finals"], name="If the model is right", marker_color=SERIES[0], opacity=0.7, nbinsx=40))
    if b:
        fig.add_trace(go.Histogram(x=b["finals"], name="If the market is right", marker_color=SERIES[1], opacity=0.7,
                                   nbinsx=40))
    fig.update_layout(barmode="overlay")
    fig.add_vline(x=cfg.staking.starting_bankroll, line_color="rgba(128,128,128,0.8)", line_width=1)
    st.plotly_chart(style_fig(fig, height=300, x_title="Final bankroll", y_title="Simulations"), width="stretch")
    st.caption("Even with a genuine edge, variance is large. Positive expected value does not guarantee profit. "
               "Aggressive (full) Kelly staking is deliberately not offered.")

st.download_button("Download bets (CSV)", bets.to_csv(index=False).encode(), "backtest_bets.csv", "text/csv")
