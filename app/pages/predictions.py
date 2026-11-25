"""PAGE 2 — Predictions with filters."""

import datetime as dt

import pandas as pd
import streamlit as st
from common import data_label_banner, disclaimer, load_latest_predictions, pct, setup_page

setup_page("Predictions")
disclaimer()
data_label_banner()

preds = load_latest_predictions()
if preds.empty:
    st.info("No predictions yet. Generate them on the **Model Settings** page or run `sportpred predict`.")
    st.stop()

st.caption(f"Run `{preds['run_id'].iloc[0]}` created {pd.Timestamp(preds['created_at'].iloc[0]):%Y-%m-%d %H:%M} UTC")

with st.container(border=True):
    f1, f2, f3, f4 = st.columns(4)
    sport = f1.selectbox("Sport", ["Football"], help="More sports are on the roadmap")
    leagues = f2.multiselect("League", sorted(preds["league_name"].unique()))
    markets = f3.multiselect("Market", ["1X2", "OU25"], format_func=lambda m: {"1X2": "Match result (1X2)",
                                                                                  "OU25": "Over/Under 2.5"}[m])
    verdicts = f4.multiselect("Verdict", ["value", "insufficient_evidence", "no_bet"],
                              default=["value", "insufficient_evidence"],
                              format_func=lambda v: {"value": "Positive EV", "insufficient_evidence": "Insufficient evidence",
                                                     "no_bet": "No bet"}[v])
    g1, g2, g3, g4, g5 = st.columns(5)
    dmin, dmax = preds["date"].min().date(), preds["date"].max().date()
    dates = g1.date_input("Dates", (dmin, dmax), min_value=dmin, max_value=dmax)
    min_odds = g2.number_input("Min odds", 1.0, 50.0, 1.0, 0.05)
    min_ev = g3.number_input("Min EV (%)", -100.0, 100.0, -100.0, 1.0) / 100
    conf = g4.selectbox("Min confidence", ["low", "medium", "high"])
    sort = g5.selectbox("Sort by", ["Expected value", "Model probability", "Date"])

df = preds.copy()
if leagues:
    df = df[df["league_name"].isin(leagues)]
if markets:
    df = df[df["market"].isin(markets)]
if verdicts:
    df = df[df["verdict"].isin(verdicts)]
if isinstance(dates, tuple) and len(dates) == 2:
    df = df[(df["date"].dt.date >= dates[0]) & (df["date"].dt.date <= dates[1])]
df = df[(df["best_odds"].fillna(0) >= min_odds) | (min_odds <= 1.0)]
df = df[(df["ev"].fillna(-9) >= min_ev)]
order = {"low": 0, "medium": 1, "high": 2}
df = df[df["confidence"].map(order).fillna(0) >= order[conf]]
key = {"Expected value": ("ev", False), "Model probability": ("prob", False), "Date": ("date", True)}[sort]
df = df.sort_values(key[0], ascending=key[1], na_position="last")

st.write(f"**{len(df)}** selections match the filters "
         f"({int((df['verdict'] == 'value').sum())} positive EV, "
         f"{int((df['verdict'] == 'insufficient_evidence').sum())} insufficient evidence).")


def interval(r):
    if pd.isna(r["prob_low"]):
        return pct(r["prob"])
    return f"{r['prob']:.1%} ({r['prob_low']:.0%}–{r['prob_high']:.0%})"


now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
table = pd.DataFrame({
    "Date": df["date"].dt.strftime("%a %d %b") + " " + df["kickoff"].fillna(""),
    "League": df["league_name"], "Match": df["match"], "Market": df["market"], "Selection": df["selection_label"],
    "Best odds": df["best_odds"].round(2), "Bookmaker": df["best_bookmaker"],
    "Model p (90% range)": df.apply(interval, axis=1),
    "Implied p": df["implied_prob"].map(pct), "Fair market p": df["fair_prob"].map(pct),
    "Edge": df["edge"].map(lambda x: pct(x, signed=True)), "EV": df["ev"].map(lambda x: pct(x, signed=True)),
    "Margin": df["margin"].map(pct), "Confidence": df["confidence"], "Verdict": df["verdict_label"],
    "Odds age (h)": df["odds_collected_at"].map(
        lambda t: round((now - pd.Timestamp(t).to_pydatetime()).total_seconds() / 3600) if pd.notna(t) else None),
    "match_id": df["match_id"],
})
event = st.dataframe(table.drop(columns="match_id"), hide_index=True, width="stretch", on_select="rerun",
                     selection_mode="single-row")
if event and event.selection and event.selection.rows:
    row = df.iloc[event.selection.rows[0]]
    st.session_state["analysis_match_id"] = int(row["match_id"])
    with st.container(border=True):
        st.markdown(f"**{row['match']} — {row['selection_label']} ({row['market']})**")
        for r in row["details"].get("reasons", []):
            st.write("• " + r)
        for r in row["details"].get("confidence_notes", []):
            st.write("• Confidence: " + r)
        st.page_link("pages/match_analysis.py", label="Open full match analysis →")

st.download_button("Download as CSV", table.to_csv(index=False).encode(), "predictions.csv", "text/csv")
with st.expander("How to read this table"):
    st.markdown("""
- **Model p** — the model's probability, with a 90% range from re-fitting it on re-weighted data.
- **Implied p** = 1 / best odds (includes the bookmaker's margin).
- **Fair market p** — the sharpest available price (Pinnacle → Betfair Exchange → market average) with the margin removed.
- **Edge** = model p − implied p. **EV** = model p × odds − 1 (average profit per 1 unit staked *if the model is right*).
- **Verdict** — *Positive EV* only if EV, edge, odds range, confidence and robustness checks all pass.
  *Insufficient evidence* means EV looks positive but the evidence is weak. *No bet* is the normal case.
""")
