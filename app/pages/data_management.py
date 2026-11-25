"""PAGE 5 — Data management: sources, connection status, update history, quality, CSV import."""

import pandas as pd
import streamlit as st
from common import load_updates, odds_api_configured, setup_page

from sportpred.config import get_settings, load_model_settings
from sportpred.data.quality import coverage_table, run_quality_checks
from sportpred.db import session_scope
from sportpred.leagues import LEAGUES

setup_page("Data Management", "🗄️")
ms = load_model_settings()
settings = get_settings()

# ---------------------------------------------------------------- sources
st.subheader("Data sources")
updates = load_updates(500)
fd_ok = updates[(updates["source"] == "football-data.co.uk") & updates["status"].isin(["success", "unchanged"])] \
    if not updates.empty else updates
src = pd.DataFrame([
    {"Provider": "football-data.co.uk", "Provides": "Results, match stats, xG (from 2026/27), pre-closing & closing "
     "odds (1X2, O/U 2.5, Asian handicap), upcoming fixtures", "Key needed": "No (free)",
     "Status": "🟢 working" if len(fd_ok) else "⚪ not yet used",
     "Last success": str(fd_ok["finished_at"].max()) if len(fd_ok) else "—"},
    {"Provider": "The Odds API", "Provides": "Live odds from many bookmakers (h2h, totals)",
     "Key needed": "Yes (free tier 500 credits/month)",
     "Status": "🟢 key configured" if odds_api_configured() else "⚪ not configured (optional)", "Last success": "—"},
    {"Provider": "Injuries / line-ups", "Provides": "Player availability", "Key needed": "Paid provider",
     "Status": "🔴 not connected — predictions exclude this", "Last success": "—"},
])
st.dataframe(src, hide_index=True, width="stretch")
with st.expander("How to add an API key (optional)"):
    st.markdown("""
1. Create a free account at **the-odds-api.com** and copy your key.
2. In the project folder, copy `.env.example` to `.env`.
3. Put the key after `SPORTPRED_ODDS_API_KEY=` and save. Never paste keys into code or chat.
4. Restart the dashboard. The status above changes to *key configured*.

The key stays on your computer (`.env` is excluded from git).
""")

# ---------------------------------------------------------------- update
st.subheader("Update data")
c1, c2 = st.columns([2, 1])
with c1:
    st.write(f"Enabled leagues: {', '.join(ms.leagues)} (change on **Model Settings**). Past seasons are cached; "
             "the current season and fixtures are re-downloaded every time.")
with c2:
    full = st.checkbox("Include all seasons since first_season", value=False,
                       help="Needed once after enabling a new league")
    if st.button("⬇️ Update now", type="primary"):
        from sportpred.data.ingest import update_all
        from sportpred.leagues import current_season_start_year

        log = st.empty()
        lines = []

        def progress(r):
            lines.append(f"{r.resource}: {r.status} (+{r.inserted} new, {r.updated} updated)")
            log.code("\n".join(lines[-12:]))

        with st.spinner("Downloading…"):
            res = update_all(first_season=None if full else current_season_start_year() - 1, progress=progress)
        failed = [r for r in res if r.status == "failed"]
        (st.error if failed else st.success)(f"Finished: {len(res)} files, {len(failed)} failed.")
        st.cache_data.clear()

# ---------------------------------------------------------------- history
st.subheader("Update history")
if updates.empty:
    st.write("No updates yet.")
else:
    st.dataframe(updates.head(200), hide_index=True, width="stretch", height=260)

# ---------------------------------------------------------------- quality
st.subheader("Data quality")
with session_scope() as s:
    issues = run_quality_checks(s, ms.leagues)
icons = {"error": "🔴", "warning": "🟠", "info": "🔵"}
qi = pd.DataFrame([{"": icons[i.severity], "Severity": i.severity, "Check": i.check,
                    "Where": " ".join(x for x in [i.league, i.season] if x), "Message": i.message,
                    "Count": i.count or "", "Examples": i.examples or ""} for i in issues])
st.dataframe(qi, hide_index=True, width="stretch")

with st.expander("Coverage by league and season (missing-data report)"):
    with session_scope() as s:
        cov = coverage_table(s)
    if not cov.empty:
        cov = cov[cov["league"].isin(ms.leagues)]
        pcts = ["shots", "xg", "pinnacle_pre", "pinnacle_close", "exchange_close", "avg_close", "ou_odds"]
        st.dataframe(cov[["league_name", "season", "matches", "finished", *pcts]].style.format({c: "{:.0%}" for c in pcts}),
                     hide_index=True, width="stretch")
        st.caption("Pinnacle prices stop after 2025/26; from then the Betfair Exchange price is the sharp reference.")

# ---------------------------------------------------------------- CSV import
st.subheader("Manual CSV import")
st.write("Upload a CSV in football-data.co.uk format (columns Div, Date, HomeTeam, AwayTeam, FTHG, FTAG, odds…).")
up = st.file_uploader("CSV file", type=["csv"])
c1, c2, c3 = st.columns(3)
league = c1.selectbox("League (if the file has no Div column)", [""] + list(LEAGUES))
season = c2.text_input("Season code (e.g. 2324)", "")
label = c3.selectbox("This data is…", ["live", "sample", "simulated"],
                     help="Be honest: sample/simulated data is flagged everywhere and never counted as real performance")
if up is not None and st.button("Import file"):
    from sportpred.data.ingest import import_csv_file

    res = import_csv_file(up.getvalue(), up.name, league=league or None, season=season or None, data_label=label)
    (st.success if res.status == "success" else st.error)(
        f"{res.status}: +{res.inserted} new, {res.updated} updated, {res.odds_rows} odds rows. {res.message or ''}")
    for w in res.warnings:
        st.warning(w)
    st.cache_data.clear()

st.caption(f"Database: `{settings.resolved_database_url().split('///')[-1]}`")
