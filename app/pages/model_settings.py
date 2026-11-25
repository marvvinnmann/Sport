"""PAGE 6 — Model settings: model choice, value thresholds, leagues/markets, staking, retraining."""

import streamlit as st
from common import setup_page

from sportpred.config import ModelSettings, StakingSettings, load_model_settings, save_model_settings
from sportpred.leagues import LEAGUES
from sportpred.models.registry import MODEL_DESCRIPTIONS

setup_page("Model Settings", "🎛️")
ms = load_model_settings()

LIVE_MODELS = ["blend", "dixon_coles", "elo"]  # models available for live predictions

with st.form("settings"):
    st.subheader("Model")
    active = st.selectbox("Active model for live predictions", LIVE_MODELS, index=LIVE_MODELS.index(ms.active_model)
                          if ms.active_model in LIVE_MODELS else 0, format_func=lambda m: MODEL_DESCRIPTIONS[m])
    st.caption("Gradient boosting and the ensemble are evaluated in walk-forward tests (Model Performance page). "
               "See docs/MODEL_CARD.md for why the market blend is the live default: standalone models lost money "
               "out-of-sample.")

    st.subheader("Supported sports, leagues and markets")
    c1, c2 = st.columns([3, 1])
    leagues = c1.multiselect("Leagues", list(LEAGUES), default=ms.leagues, format_func=lambda c: f"{c} — {LEAGUES[c].label}")
    markets = c2.multiselect("Markets", ["1X2", "OU25"], default=ms.markets)
    st.caption("Sport: football (soccer). After enabling a new league run Data Management → Update with "
               "'Include all seasons'.")

    st.subheader("Value-bet thresholds")
    c1, c2, c3, c4 = st.columns(4)
    min_ev = c1.number_input("Minimum EV (%)", 0.0, 50.0, ms.min_ev * 100, 0.5) / 100
    min_edge = c2.number_input("Minimum edge (pp)", 0.0, 30.0, ms.min_edge * 100, 0.5) / 100
    min_odds = c3.number_input("Minimum odds", 1.01, 20.0, ms.min_odds, 0.05)
    max_odds = c4.number_input("Maximum odds", 1.1, 100.0, ms.max_odds, 0.25)
    c1, c2, c3 = st.columns(3)
    min_conf = c1.selectbox("Required confidence", ["low", "medium", "high"],
                            index=["low", "medium", "high"].index(ms.min_confidence))
    robust = c2.checkbox("EV must stay positive at the low end of the uncertainty range", ms.require_robust_ev)
    max_age = c3.number_input("Flag odds older than (hours)", 1.0, 336.0, float(ms.max_odds_age_hours), 1.0)

    st.subheader("Model parameters")
    c1, c2, c3, c4 = st.columns(4)
    xi = c1.number_input("Dixon-Coles time decay ξ (per day)", 0.0, 0.01, ms.dc_xi, 0.0001, format="%.4f",
                         help="0.0019 ≈ matches lose half their weight after ~1 year")
    window = c2.number_input("Training window (days)", 200, 3000, ms.dc_window_days, 50)
    ridge = c3.number_input("Shrinkage (ridge)", 0.0, 20.0, ms.dc_ridge, 0.1,
                            help="Higher = team ratings pulled more strongly towards the prior")
    reps = c4.number_input("Uncertainty refits", 0, 200, ms.uncertainty_replicates, 5)
    c1, c2, c3 = st.columns(3)
    elo_k = c1.number_input("Elo K-factor", 1.0, 80.0, ms.elo_k, 1.0)
    elo_h = c2.number_input("Elo home advantage", 0.0, 200.0, ms.elo_home_advantage, 5.0)
    retrain = c3.number_input("Retrain every (days)", 1, 60, ms.retrain_every_days, 1,
                              help="The daily job re-fits the model on every run; this is used by walk-forward tests")

    st.subheader("Staking (paper trading & backtests)")
    s = ms.staking
    c1, c2, c3, c4 = st.columns(4)
    method = c1.selectbox("Method", ["flat", "kelly"], index=["flat", "kelly"].index(s.method),
                          format_func=lambda m: {"flat": "Flat stakes", "kelly": "Fractional Kelly"}[m])
    flat = c2.number_input("Flat stake (% bankroll)", 0.1, 10.0, s.flat_stake_pct, 0.1)
    kf = c3.number_input("Kelly fraction", 0.05, 1.0, s.kelly_fraction, 0.05, help="0.25 = quarter Kelly")
    cap = c4.number_input("Max stake per bet (% bankroll)", 0.1, 20.0, s.max_stake_pct, 0.1)
    c1, c2, c3, c4 = st.columns(4)
    ev_cap = c1.number_input("Max per event (%)", 0.1, 50.0, s.max_event_exposure_pct, 0.5)
    day_cap = c2.number_input("Max per day (%)", 0.5, 100.0, s.max_daily_exposure_pct, 0.5)
    comp_cap = c3.number_input("Max per competition per day (%)", 0.5, 100.0, s.max_competition_exposure_pct, 0.5)
    bank = c4.number_input("Starting bankroll", 10.0, 1e7, s.starting_bankroll, 100.0)

    submitted = st.form_submit_button("💾 Save settings", type="primary")

if submitted:
    if min_odds >= max_odds:
        st.error("Minimum odds must be below maximum odds.")
    elif not leagues or not markets:
        st.error("Choose at least one league and one market.")
    else:
        new = ModelSettings(
            leagues=leagues, first_season=ms.first_season, markets=markets, active_model=active, min_ev=min_ev,
            min_edge=min_edge, min_odds=min_odds, max_odds=max_odds, min_confidence=min_conf, require_robust_ev=robust,
            max_odds_age_hours=max_age, dc_xi=xi, dc_window_days=int(window), dc_ridge=ridge,
            elo_k=elo_k, elo_home_advantage=elo_h, uncertainty_replicates=int(reps),
            min_team_matches=ms.min_team_matches, retrain_every_days=int(retrain),
            staking=StakingSettings(method=method, flat_stake_pct=flat, kelly_fraction=kf, max_stake_pct=cap,
                                    max_event_exposure_pct=ev_cap, max_daily_exposure_pct=day_cap,
                                    max_competition_exposure_pct=comp_cap, starting_bankroll=bank),
        )
        path = save_model_settings(new)
        st.success(f"Saved to {path}. New predictions will use these settings.")
        st.cache_data.clear()

st.divider()
st.subheader("Retrain & predict now")
st.write("Re-fits the active model on all finished matches and predicts the upcoming fixtures (about 1–3 s per league).")
if st.button("🔄 Generate predictions"):
    from sportpred.services.predictions import generate_predictions

    with st.spinner("Fitting models and pricing fixtures…"):
        res = generate_predictions(progress=lambda m: None)
    st.success(f"{res['n_predictions']} predictions for {res['n_matches']} matches; {res['n_value']} positive-EV "
               f"selections. Run id {res['run_id']}.")
    st.cache_data.clear()
