"""PAGE 4 — Model performance: accuracy, calibration, profitability and CLV, all out-of-sample."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (
    MODEL_LABEL,
    SERIES,
    disclaimer,
    load_evaluation,
    load_wf,
    pct,
    setup_page,
    style_fig,
    wf_mtime,
)

from sportpred.config import StakingSettings
from sportpred.evaluation.backtest import BacktestConfig, run_backtest
from sportpred.evaluation.metrics import calibration_table, expected_calibration_error
from sportpred.leagues import league_label
from sportpred.services.evaluation import available_walk_forward_models
from sportpred.services.paper import paper_bets_frame

setup_page("Model Performance")
disclaimer()

ev = load_evaluation()
models = available_walk_forward_models()
if not ev or not models:
    st.info("No evaluation results yet. Run `sportpred evaluate` from the command line "
            "(takes ~10-20 minutes for all leagues). It trains only on the past and predicts week by week.")
    st.stop()

st.caption(f"Evaluation generated {ev['generated_at']} · test period from {ev['test_from']} · "
           f"{len(ev['leagues'])} leagues · every number below is out-of-sample (walk-forward).")

# ---------------------------------------------------------------- headline table
st.subheader("Accuracy and probability quality vs the closing market")
st.write("All models are scored on **identical matches** "
         f"({ev.get('common_matches', {}).get('1X2', 0):,} for 1X2, {ev.get('common_matches', {}).get('OU25', 0):,} "
         "for over/under). The *closing market* is the bookmaker price at kick-off with the margin removed — the "
         "toughest benchmark available.")
rows = []
for model, res in ev["overall"].items():
    for market, r in res.items():
        vs = r.get("vs_closing", {})
        rows.append({"Model": MODEL_LABEL.get(model, model), "Market": market, "Matches": r["model"]["n"],
                     "Accuracy": pct(r["model"]["accuracy"]), "Brier": round(r["model"]["brier"], 4),
                     "Log loss": round(r["model"]["log_loss"], 4),
                     "RPS": f"{r['model']['rps']:.4f}" if "rps" in r["model"] else "—",
                     "Closing market log loss": round(r["market_closing"]["log_loss"], 4),
                     "Δ log loss (95% CI)": f"{vs['mean_diff']:+.4f} ({vs['ci_low']:+.4f}…{vs['ci_high']:+.4f})"})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption("Δ log loss < 0 with a CI entirely below 0 would mean the model beats the closing market. "
           "Positive Δ means the market is better.")

# ---------------------------------------------------------------- choose model for detail
model = st.selectbox("Model for detailed analysis", models, format_func=lambda m: MODEL_LABEL.get(m, m),
                     index=models.index("dixon_coles") if "dixon_coles" in models else 0)
df = load_wf(model, wf_mtime(model))
if df is None or df.empty:
    st.stop()



@st.cache_data(show_spinner="Running backtests…")
def roi_table(model: str, mtime: float) -> list[dict]:
    data = load_wf(model, mtime)
    rows = []
    for mk in ("1X2", "OU25"):
        for src in ("market_avg", "best_book", "market_max"):
            cfg = BacktestConfig(market=mk, price_source=src, min_ev=0.04, staking=StakingSettings(method="flat"))
            _, s = run_backtest(data, cfg)
            if s.get("n_bets"):
                rows.append({"Market": mk, "Price source": src, "Bets": s["n_bets"], "Hit rate": pct(s["hit_rate"]),
                             "Avg odds": round(s["avg_odds"], 2), "Yield (flat)": pct(s["flat_yield"], signed=True),
                             "Yield 95% CI": f"{s['flat_yield_ci_low']:+.1%} … {s['flat_yield_ci_high']:+.1%}",
                             "Avg CLV": pct(s["avg_clv"], signed=True), "Beat closing line": pct(s["share_positive_clv"])})
    return rows


tab_cal, tab_league, tab_time, tab_roi, tab_paper = st.tabs(
    ["Calibration", "By league", "Over time", "ROI, yield & CLV", "Paper trading (live)"])

with tab_cal:
    outcome = st.radio("Outcome", ["Home win", "Draw", "Away win", "Over 2.5"], horizontal=True)
    col, y = {"Home win": ("p_H", df["y_1x2"] == 0), "Draw": ("p_D", df["y_1x2"] == 1),
              "Away win": ("p_A", df["y_1x2"] == 2), "Over 2.5": ("p_OVER", df["y_over"] == 1)}[outcome]
    if col in df and df[col].notna().any():
        ok = df[col].notna()
        ct = calibration_table(df.loc[ok, col], y[ok].astype(float), n_bins=10)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                                 line=dict(color="rgba(128,128,128,0.6)", width=1, dash="dot"), hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=ct["mean_pred"], y=ct["observed"], mode="lines+markers", name=MODEL_LABEL.get(model, model),
            line=dict(color=SERIES[0], width=2), marker=dict(size=9),
            error_y=dict(type="data", symmetric=False, array=ct["ci_high"] - ct["observed"],
                         arrayminus=ct["observed"] - ct["ci_low"], thickness=1.5, width=4),
            customdata=ct["count"], hovertemplate="predicted %{x:.1%} → happened %{y:.1%} (n=%{customdata})<extra></extra>"))
        lo, hi = float(ct["mean_pred"].min()) - 0.05, float(ct["mean_pred"].max()) + 0.05
        fig.update_xaxes(range=[max(lo, 0), min(hi, 1)], tickformat=".0%")
        fig.update_yaxes(range=[max(lo, 0), min(hi, 1)], tickformat=".0%")
        st.plotly_chart(style_fig(fig, height=420, x_title="Predicted probability", y_title="Observed frequency"),
                        width="stretch")
        st.write(f"Expected calibration error: **{expected_calibration_error(df.loc[ok, col], y[ok].astype(float)):.3f}** "
                 "(average gap between predicted and observed; < 0.02 is good). Error bars: 95% Wilson intervals.")
    else:
        st.write("This model does not predict that market.")

with tab_league:
    rows = []
    for lg, res in ev["models"].get(model, {}).get("by_league", {}).items():
        for market, r in res.items():
            vs = r.get("vs_closing", {})
            rows.append({"league": lg, "League": league_label(lg), "Market": market, "Matches": r["model"]["n"],
                         "Model log loss": r["model"]["log_loss"], "Market log loss": r["market_closing"]["log_loss"],
                         "diff": vs.get("mean_diff"), "lo": vs.get("ci_low"), "hi": vs.get("ci_high")})
    t = pd.DataFrame(rows)
    if not t.empty:
        mk = st.radio("Market", sorted(t["Market"].unique()), horizontal=True, key="lg_mk")
        tt = t[t["Market"] == mk].sort_values("diff")
        fig = go.Figure(go.Scatter(
            x=tt["diff"], y=tt["League"], mode="markers", marker=dict(size=10, color=SERIES[0]),
            error_x=dict(type="data", symmetric=False, array=tt["hi"] - tt["diff"], arrayminus=tt["diff"] - tt["lo"],
                         thickness=2, width=5),
            hovertemplate="%{y}: Δ log loss %{x:+.4f}<extra></extra>", name="Δ log loss vs closing market"))
        fig.add_vline(x=0, line_color="rgba(128,128,128,0.7)", line_width=1)
        st.plotly_chart(style_fig(fig, height=max(300, 26 * len(tt)), x_title="Model log loss − closing market log loss "
                                  "(left of 0 = model better)"), width="stretch")
        st.dataframe(tt.drop(columns=["league", "lo", "hi"]).round(4), hide_index=True, width="stretch")

with tab_time:
    rows = []
    for season, res in ev["models"].get(model, {}).get("by_season", {}).items():
        for market, r in res.items():
            rows.append({"Season": season, "Market": market, "Model": r["model"]["log_loss"],
                         "Closing market": r["market_closing"]["log_loss"], "Matches": r["model"]["n"]})
    t = pd.DataFrame(rows)
    if not t.empty:
        mk = st.radio("Market", sorted(t["Market"].unique()), horizontal=True, key="time_mk")
        tt = t[t["Market"] == mk].sort_values("Season")
        tt["Season label"] = tt["Season"].map(lambda s: f"20{s[:2]}/{s[2:]}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tt["Season label"], y=tt["Model"], mode="lines+markers", name=MODEL_LABEL.get(model, model),
                                 line=dict(width=2, color=SERIES[0]), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=tt["Season label"], y=tt["Closing market"], mode="lines+markers",
                                 name="Closing market", line=dict(width=2, color=SERIES[1]), marker=dict(size=8)))
        st.plotly_chart(style_fig(fig, height=340, y_title="Log loss (lower is better)"), width="stretch")
        st.caption("Season 2019/20 was curtailed by COVID-19 and partly played without crowds.")

with tab_roi:
    st.write("Flat 1-unit stakes on every selection the model rates at ≥ +4% EV, odds 1.5–5.0. "
             "Prices: market average at the pre-closing snapshot (realistic) vs market maximum (optimistic).")
    rows = roi_table(model, wf_mtime(model))
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Yield = profit ÷ amount staked. CLV (closing line value) = bet odds × fair closing probability − 1; "
               "consistently positive CLV is the most reliable sign of genuine skill. Use the Backtesting page for "
               "custom strategies, staking and drawdowns.")

with tab_paper:
    pb = paper_bets_frame()
    if pb.empty:
        st.write("No paper bets yet. They are logged automatically for every positive-EV recommendation and "
                 "settled after the match — the honest forward test of the model.")
    else:
        done = pb[pb["status"].isin(["won", "lost"])]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Paper bets", len(pb))
        c2.metric("Settled", len(done))
        if len(done):
            c3.metric("Profit (units)", f"{done['pnl'].sum():+.2f}")
            c4.metric("Avg CLV", pct(done["clv"].mean(), signed=True) if done["clv"].notna().any() else "—")
        st.dataframe(pb[["date", "league", "home_team", "away_team", "market", "selection", "odds", "bookmaker",
                         "model_prob", "status", "pnl", "closing_odds", "clv"]], hide_index=True, width="stretch")
