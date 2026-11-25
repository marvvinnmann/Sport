"""PAGE 3 — Match analysis: everything behind one prediction."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (
    SELECTION_LABEL,
    SERIES,
    VERDICT_LABEL,
    disclaimer,
    load_latest_predictions,
    load_matches_cached,
    pct,
    setup_page,
    style_fig,
)

from sportpred.services.analysis import league_table, team_matches

setup_page("Match Analysis")
disclaimer()

preds = load_latest_predictions()
if preds.empty:
    st.info("No predictions yet — generate them first (Model Settings page or `sportpred predict`).")
    st.stop()

matches = preds.drop_duplicates("match_id").sort_values(["date", "league"])
labels = {int(r.match_id): f"{r.date:%a %d %b} · {r.league_name} · {r.match}" for r in matches.itertuples()}
ids = list(labels)
default = st.session_state.get("analysis_match_id")
idx = ids.index(default) if default in ids else 0
match_id = st.selectbox("Match", ids, index=idx, format_func=lambda i: labels[i])
st.session_state["analysis_match_id"] = match_id

m = preds[preds["match_id"] == match_id]
first = m.iloc[0]
det = first["details"]
exp = det.get("explanation", {})
home, away = first["home_team"], first["away_team"]
st.header(f"{home} v {away}")
st.caption(f"{first['league_name']} · {first['date']:%A %d %B %Y} {first['kickoff'] or ''} (UK time) · "
           f"model trained on matches before this date")

# ---------------------------------------------------------------- probabilities and EV
st.subheader("Model probability vs bookmaker odds")
for market, title in (("1X2", "Match result"), ("OU25", "Total goals over/under 2.5")):
    mm = m[m["market"] == market]
    if mm.empty:
        continue
    st.markdown(f"**{title}**")
    cols = st.columns(len(mm))
    for c, r in zip(cols, mm.itertuples(), strict=True):
        with c.container(border=True):
            st.markdown(f"**{SELECTION_LABEL[r.selection]}**")
            rng = f" ({r.prob_low:.0%}–{r.prob_high:.0%})" if pd.notna(r.prob_low) else ""
            st.write(f"Model: **{r.prob:.1%}**{rng}")
            if pd.notna(r.best_odds):
                st.write(f"Best odds: **{r.best_odds:.2f}** ({r.best_bookmaker})")
                st.write(f"Implied: {r.implied_prob:.1%} · Fair market: {pct(r.fair_prob)}")
                st.write(f"EV = {r.prob:.3f} × {r.best_odds:.2f} − 1 = **{r.ev:+.1%}**")
            else:
                st.write("No odds available")
            st.write(f"{VERDICT_LABEL[r.verdict]} · confidence {r.confidence}")

# uncertainty chart (dot = point estimate, line = 90% range, diamond = fair market probability)
plot = m.dropna(subset=["prob"]).copy()
plot["label"] = plot["selection"].map(SELECTION_LABEL)
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=plot["prob"], y=plot["label"], mode="markers", name="Model probability", marker=dict(size=11, color=SERIES[0]),
    error_x=dict(type="data", symmetric=False, array=(plot["prob_high"] - plot["prob"]).fillna(0),
                 arrayminus=(plot["prob"] - plot["prob_low"]).fillna(0), thickness=2, width=6),
    hovertemplate="%{y}: model %{x:.1%}<extra></extra>"))
fig.add_trace(go.Scatter(x=plot["fair_prob"], y=plot["label"], mode="markers", name="Fair market probability",
                         marker=dict(size=10, symbol="diamond", color=SERIES[1]),
                         hovertemplate="%{y}: market %{x:.1%}<extra></extra>"))
fig.add_trace(go.Scatter(x=plot["implied_prob"], y=plot["label"], mode="markers", name="Implied by best odds",
                         marker=dict(size=9, symbol="line-ns-open", color=SERIES[2], line=dict(width=2)),
                         hovertemplate="%{y}: implied %{x:.1%}<extra></extra>"))
fig.update_xaxes(tickformat=".0%", range=[0, 1])
st.plotly_chart(style_fig(fig, height=300, x_title="Probability"), width="stretch")
st.caption("Bars show the 90% uncertainty range of the model (Bayesian bootstrap refits). If the market's fair "
           "probability lies inside the bar, the apparent edge may just be noise.")

# ---------------------------------------------------------------- bookmaker odds table
prices = det.get("bookmaker_prices", {})
odds_rows = []
for r in m.itertuples():
    for book, price in r.details.get("bookmaker_prices", {}).items():
        odds_rows.append({"Bookmaker": book, "Selection": SELECTION_LABEL[r.selection], "Odds": price})
if odds_rows:
    with st.expander("Available bookmaker odds", expanded=False):
        ot = pd.DataFrame(odds_rows).pivot_table(index="Bookmaker", columns="Selection", values="Odds")
        st.dataframe(ot, width="stretch")
        st.caption(f"Snapshot collected {first['odds_collected_at']} (UTC). Margins: average "
                   f"{pct(m[m['market'] == '1X2']['margin'].iloc[0]) if (m['market'] == '1X2').any() else '—'} (1X2).")

# ---------------------------------------------------------------- factors
st.subheader("What drives the prediction")
xg = exp.get("expected_goals", {})
att, dfn = exp.get("attack_vs_avg_pct", {}), exp.get("defence_vs_avg_pct", {})
supporting, weakening = [], []
if xg:
    st.write(f"Expected goals (Dixon-Coles): **{home} {xg['home']:.2f} – {xg['away']:.2f} {away}** · home advantage "
             f"+{exp.get('home_advantage_pct', 0):.0f}% goals")
c1, c2 = st.columns(2)
with c1:
    comp = pd.DataFrame({
        "": ["Attack vs league average", "Goals conceded vs average", "Elo rating", "Rest days",
             "Matches in training window"],
        home: [f"{att.get('home', 0):+.0f}%", f"{dfn.get('home', 0):+.0f}%", exp.get("elo", {}).get("home"),
               exp.get("rest_days", {}).get("home"), exp.get("matches_in_window", {}).get("home")],
        away: [f"{att.get('away', 0):+.0f}%", f"{dfn.get('away', 0):+.0f}%", exp.get("elo", {}).get("away"),
               exp.get("rest_days", {}).get("away"), exp.get("matches_in_window", {}).get("away")],
    }).astype(str)
    st.dataframe(comp, hide_index=True, width="stretch")
    st.caption("Goals conceded: negative = better defence than average.")
with c2:
    ep = exp.get("elo_probs", {})
    dc1 = m[m["market"] == "1X2"].set_index("selection")["prob"]
    if ep and len(dc1):
        agree = pd.DataFrame({"Outcome": ["Home", "Draw", "Away"],
                              "Dixon-Coles": [pct(dc1.get("H")), pct(dc1.get("D")), pct(dc1.get("A"))],
                              "Elo (second opinion)": [pct(ep["H"]), pct(ep["D"]), pct(ep["A"])],
                              "Fair market": [pct(x) for x in m[m["market"] == "1X2"].set_index("selection")
                                              .reindex(["H", "D", "A"])["fair_prob"]]})
        st.dataframe(agree, hide_index=True, width="stretch")
        gap = max(abs(dc1.get(k, 0) - ep[k]) for k in ("H", "D", "A"))
        (supporting if gap < 0.06 else weakening).append(
            f"Dixon-Coles and Elo {'agree' if gap < 0.06 else 'disagree'} (largest gap {gap:.0%}).")

fh, fa = exp.get("form_home", {}), exp.get("form_away", {})
for team, f in ((home, fh), (away, fa)):
    if f.get("matches"):
        line = f"{team}: last {f['matches']} league games {f['form']} ({f['points_per_game']:.1f} pts/game)"
        (supporting if f["points_per_game"] >= 1.5 else weakening).append(line)
rd = exp.get("rest_days", {})
if rd.get("home") is not None and rd.get("away") is not None and abs(rd["home"] - rd["away"]) >= 3:
    fresher = home if rd["home"] > rd["away"] else away
    supporting.append(f"{fresher} has had {abs(rd['home'] - rd['away'])} more days of rest (league games only).")
for n in det.get("confidence_notes", []):
    weakening.append(n)
s1, s2 = st.columns(2)
with s1:
    st.markdown("**Context**")
    for x in supporting or ["—"]:
        st.write("• " + x)
with s2:
    st.markdown("**Weakening factors / caveats**")
    for x in weakening or ["—"]:
        st.write("• " + x)

st.markdown("**Important missing information**")
for x in det.get("missing_info", []):
    st.warning(x)

# ---------------------------------------------------------------- recent performance
st.subheader("Recent performance (league matches)")
fin = load_matches_cached((first["league"],), ("finished",))
c1, c2 = st.columns(2)
for col, team in ((c1, home), (c2, away)):
    with col:
        st.markdown(f"**{team}**")
        tm = team_matches(fin, team, before=first["date"], n=8)
        if tm.empty:
            st.write("No recent matches in this league.")
            continue
        show = tm[["date", "venue", "opponent", "gf", "ga", "result", "sot_for", "sot_against"]].rename(
            columns={"gf": "GF", "ga": "GA", "sot_for": "SoT for", "sot_against": "SoT against"})
        st.dataframe(show, hide_index=True, width="stretch")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**League table (this season, before the match)**")
    tab = league_table(fin, first["league"], first["season"], before=first["date"])
    if not tab.empty:
        st.dataframe(tab.style.apply(lambda r: ["font-weight: bold" if r["team"] in (home, away) else "" for _ in r],
                                     axis=1), hide_index=True, width="stretch", height=300)
with c2:
    st.markdown("**Head-to-head (context only)**")
    h2h = pd.DataFrame(exp.get("head_to_head", []))
    if h2h.empty:
        st.write("No meetings in the data.")
    else:
        st.dataframe(h2h, hide_index=True, width="stretch")
    st.caption("Head-to-head records are small samples and are deliberately NOT used by the model.")
