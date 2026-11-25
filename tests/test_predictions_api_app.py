"""Prediction rules, end-to-end prediction service, REST API and dashboard pages (SIMULATED data)."""

import pytest

from sportpred.config import ModelSettings


def sel(**kw):
    base = dict(best_odds=2.2, prob=0.52, ev=0.52 * 2.2 - 1, edge=0.52 - 1 / 2.2, ev_low=0.03, confidence="high",
                fair_prob=0.47, odds_age_h=10)
    base.update(kw)
    return base


def test_classify_rules():
    from sportpred.services.predictions import classify

    ms = ModelSettings()
    assert classify(sel(), ms)[0] == "value"
    assert classify(sel(prob=0.40, ev=0.40 * 2.2 - 1, edge=0.40 - 1 / 2.2), ms)[0] == "no_bet"
    assert classify(sel(best_odds=None), ms)[0] == "insufficient_evidence"
    assert classify(sel(ev_low=-0.05), ms)[0] == "insufficient_evidence"  # not robust to uncertainty
    assert classify(sel(confidence="low"), ms)[0] == "insufficient_evidence"
    assert classify(sel(fair_prob=0.30), ms)[0] == "insufficient_evidence"  # suspicious disagreement
    v, reasons = classify(sel(best_odds=7.0, prob=0.2, ev=0.4, edge=0.06), ms)
    assert v == "no_bet" and "outside" in reasons[0]


def test_generate_predictions_end_to_end(simulated_db):
    from sportpred.config import save_model_settings
    from sportpred.services.paper import paper_bets_frame, settle_paper_bets
    from sportpred.services.predictions import generate_predictions, latest_predictions

    ms = ModelSettings(leagues=["E0"], uncertainty_replicates=5, min_confidence="low", require_robust_ev=False,
                       min_ev=0.0, min_edge=0.0)
    save_model_settings(ms)
    res = generate_predictions(ms=ms, progress=lambda m: None)
    assert res["n_matches"] == 3
    assert res["n_predictions"] == 3 * 5  # 3 matches x (3 + 2) selections
    df = latest_predictions()
    assert set(df["verdict"]) <= {"value", "insufficient_evidence", "no_bet"}
    one = df[df["market"] == "1X2"].groupby("match_id")["prob"].sum()
    assert one.round(6).eq(1.0).all()
    assert "explanation" in df.iloc[0]["details"]
    assert settle_paper_bets() == 0  # nothing finished yet
    pb = paper_bets_frame()
    assert len(pb) == int(df["is_recommended"].sum())


def test_api_endpoints(simulated_db):
    from fastapi.testclient import TestClient

    from sportpred.api.main import app

    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        st = c.get("/status").json()
        assert st["matches"] > 0 and st["providers"]["the-odds-api"].startswith("not configured")
        r = c.post("/calculator", json={"odds": [2.0, 3.5, 4.0], "model_probs": [0.55, 0.25, 0.20]}).json()
        assert r["expected_value"][0] == pytest.approx(0.10)
        assert sum(r["fair_probabilities"]) == pytest.approx(1.0)
        assert c.post("/calculator", json={"odds": [1.0, 2.0]}).status_code == 422
        assert c.get("/predictions").json() == []
        assert c.get("/performance").status_code == 404
        assert c.get("/fixtures").status_code == 200
        assert isinstance(c.get("/quality").json(), list)


@pytest.mark.parametrize("page", ["dashboard", "predictions", "match_analysis", "model_performance",
                                  "data_management", "model_settings", "backtesting"])
def test_dashboard_pages_render(simulated_db, page):
    """Every dashboard page renders without exceptions (with and without predictions)."""
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    from sportpred.services.predictions import generate_predictions

    generate_predictions(ms=ModelSettings(leagues=["E0"], uncertainty_replicates=3), progress=lambda m: None)
    path = Path(__file__).resolve().parents[1] / "app" / "pages" / f"{page}.py"
    at = AppTest.from_file(str(path), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if page == "dashboard":
        # non-live data must be flagged loudly
        assert any("NON-LIVE" in e.value for e in at.error)


def test_blend_probabilities_are_market_anchored():
    import pandas as pd

    from sportpred.services.predictions import _market_probabilities

    p = pd.Series({"p_H": 0.55, "p_H_low": 0.45, "p_H_high": 0.65, "p_D": 0.25, "p_D_low": 0.2, "p_D_high": 0.3,
                   "p_A": 0.20, "p_A_low": 0.15, "p_A_high": 0.27, "p_OVER": 0.6, "p_OVER_low": 0.5,
                   "p_OVER_high": 0.7, "p_UNDER": 0.4, "p_UNDER_low": 0.3, "p_UNDER_high": 0.5})
    ep = pd.Series({"p_H": 0.5, "p_D": 0.27, "p_A": 0.23})
    fair = {"H": 0.45, "D": 0.28, "A": 0.27}
    probs, model, note = _market_probabilities(p, ep, "1X2", fair, ModelSettings())
    assert model == "blend" and note is None
    assert sum(v[0] for v in probs.values()) == pytest.approx(1.0)
    for s, (pt, lo, hi) in probs.items():
        assert lo <= pt <= hi
        assert abs(pt - fair[s]) < 0.03  # anchored to the market, not to the 55% model view
    ou, _, _ = _market_probabilities(p, ep, "OU25", {"OVER": 0.52, "UNDER": 0.48}, ModelSettings())
    assert ou["OVER"][0] + ou["UNDER"][0] == pytest.approx(1.0)
    # without a sharp reference the blend falls back to Dixon-Coles and says so
    probs, model, note = _market_probabilities(p, ep, "1X2", {}, ModelSettings())
    assert model == "dixon_coles" and "no sharp reference" in note
    # Elo covers 1X2 only
    assert _market_probabilities(p, ep, "OU25", fair, ModelSettings(active_model="elo"))[0] is None
