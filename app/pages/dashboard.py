"""PAGE 1 — Main dashboard."""

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (
    MODEL_LABEL,
    SERIES,
    data_label_banner,
    disclaimer,
    last_update_text,
    load_evaluation,
    load_latest_predictions,
    load_matches_cached,
    pct,
    setup_page,
    style_fig,
)

from sportpred.config import load_model_settings
from sportpred.leagues import league_label

setup_page("Dashboard")
disclaimer()
data_label_banner()

ms = load_model_settings()
preds = load_latest_predictions()
today = pd.Timestamp(dt.date.today())
upcoming = load_matches_cached(tuple(ms.leagues), ("scheduled",))
upcoming = upcoming[upcoming["date"] >= today] if not upcoming.empty else upcoming
evaluation = load_evaluation()

# ---------------------------------------------------------------- KPI row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Upcoming fixtures", f"{len(upcoming):,}", help="Scheduled matches in the enabled leagues")
n_value = int((preds["verdict"] == "value").sum()) if not preds.empty else 0
n_insuff = int((preds["verdict"] == "insufficient_evidence").sum()) if not preds.empty else 0
c2.metric("Positive-EV selections", n_value, help="Selections passing every threshold and robustness check")
c3.metric("Insufficient evidence", n_insuff, help="EV looked positive but uncertainty/data checks failed")
c4.metric("Latest data update", last_update_text().split(" (")[-1].rstrip(")"), help=last_update_text())

if preds.empty:
    st.info("No predictions yet. Run **Data Management → Update data** and then **Model Settings → Generate "
            "predictions**, or `sportpred daily` from the command line.")

# ---------------------------------------------------------------- highest value
st.subheader("Highest estimated value opportunities")
if not preds.empty:
    rec = preds[preds["verdict"].isin(["value", "insufficient_evidence"])].sort_values("ev", ascending=False).head(12)
    if rec.empty:
        st.success("No selection currently shows positive expected value above the thresholds. "
                   "That is the normal outcome in efficient markets — the app will not force a pick.")
    else:
        show = pd.DataFrame({
            "Date": rec["date"].dt.strftime("%a %d %b"), "League": rec["league_name"], "Match": rec["match"],
            "Market": rec["market"], "Selection": rec["selection_label"],
            "Odds": rec["best_odds"].round(2), "Bookmaker": rec["best_bookmaker"],
            "Model p": rec["prob"].map(pct), "Implied p": rec["implied_prob"].map(pct),
            "EV": rec["ev"].map(lambda x: pct(x, signed=True)), "Confidence": rec["confidence"],
            "Verdict": rec["verdict_label"],
        })
        st.dataframe(show, hide_index=True, width="stretch")

# ---------------------------------------------------------------- today's fixtures
left, right = st.columns([3, 2])
with left:
    st.subheader("Today's & next fixtures")
    if upcoming.empty:
        st.write("No upcoming fixtures in the latest fixtures file for the enabled leagues. "
                 "football-data.co.uk refreshes fixtures around Tuesday and Friday.")
    else:
        nxt = upcoming.sort_values(["date", "kickoff"]).head(20)
        p1 = preds[preds["market"] == "1X2"].pivot_table(index="match_id", columns="selection", values="prob") \
            if not preds.empty else pd.DataFrame()
        rows = []
        for r in nxt.itertuples():
            pr = p1.loc[r.id] if r.id in p1.index else None
            rows.append({"Date": r.date.strftime("%a %d %b"), "Time": r.kickoff or "", "League": league_label(r.league),
                         "Match": f"{r.home_team} v {r.away_team}",
                         "Home": pct(pr["H"]) if pr is not None else "—",
                         "Draw": pct(pr["D"]) if pr is not None else "—",
                         "Away": pct(pr["A"]) if pr is not None else "—"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

with right:
    st.subheader("Predictions by league")
    if not preds.empty:
        g = preds.groupby("league_name").agg(matches=("match_id", "nunique"),
                                             value=("verdict", lambda s: (s == "value").sum())).reset_index()
        g = g.sort_values("matches")
        fig = go.Figure()
        fig.add_bar(y=g["league_name"], x=g["matches"], orientation="h", name="Matches predicted",
                    marker_color=SERIES[0], hovertemplate="%{y}: %{x} matches<extra></extra>")
        fig.add_bar(y=g["league_name"], x=g["value"], orientation="h", name="Positive-EV selections",
                    marker_color=SERIES[1], hovertemplate="%{y}: %{x} positive-EV<extra></extra>")
        fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.1)
        st.plotly_chart(style_fig(fig, height=320), width="stretch")
        st.caption("Predictions by sport: ⚽ football only in this MVP (other sports are on the roadmap).")

# ---------------------------------------------------------------- model performance overview
st.subheader("Model performance overview (out-of-sample)")
if not evaluation:
    st.write("No evaluation yet — run `sportpred evaluate` (or the Model Performance page).")
else:
    rows = []
    for model, res in evaluation.get("overall", {}).items():
        for market, r in res.items():
            vs = r.get("vs_closing", {})
            rows.append({"Model": MODEL_LABEL.get(model, model), "Market": market, "Matches": r["model"]["n"],
                         "Model log loss": round(r["model"]["log_loss"], 4),
                         "Closing-market log loss": round(r["market_closing"]["log_loss"], 4),
                         "Difference (95% CI)": f"{vs.get('mean_diff', 0):+.4f} ({vs.get('ci_low', 0):+.4f} to "
                                                f"{vs.get('ci_high', 0):+.4f})",
                         "Beats closing market?": "yes" if vs.get("a_better_significant") else "no"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Test period from {evaluation.get('test_from')} — walk-forward, no future data. Lower log loss is "
               "better. The closing market is the hardest benchmark there is.")

st.caption(f"Latest data update: {last_update_text()}")
