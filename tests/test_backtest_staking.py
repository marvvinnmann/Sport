import numpy as np
import pandas as pd
import pytest

from sportpred.betting.staking import simulate_bankroll, size_stakes_for_day
from sportpred.config import StakingSettings
from sportpred.evaluation.backtest import (
    BacktestConfig,
    baseline_strategies,
    run_backtest,
    threshold_sensitivity,
)
from sportpred.evaluation.metrics import (
    bootstrap_mean_ci,
    brier_per_match,
    calibration_table,
    log_loss_per_match,
    longest_losing_streak,
    max_drawdown,
    rps_per_match,
)


def frame():
    """Three hand-made matches with known outcomes and prices."""
    return pd.DataFrame({
        "match_id": [1, 2, 3], "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-08"]),
        "league": "E0", "season": "2324", "home_team": ["A", "C", "E"], "away_team": ["B", "D", "F"],
        "p_H": [0.60, 0.30, 0.50], "p_D": [0.25, 0.30, 0.25], "p_A": [0.15, 0.40, 0.25],
        "market_avg_H": [2.0, 3.0, 1.8], "market_avg_D": [3.4, 3.2, 3.6], "market_avg_A": [5.0, 2.4, 4.5],
        "ref_close_H": [0.52, 0.32, 0.55], "ref_close_D": [0.27, 0.30, 0.25], "ref_close_A": [0.21, 0.38, 0.20],
        "y_1x2": [0, 2, 1], "y_over": [1, 0, 0],
    })


def test_backtest_selects_and_settles_correctly():
    cfg = BacktestConfig(market="1X2", price_source="market_avg", min_ev=0.05, min_odds=1.5, max_odds=5.0,
                         staking=StakingSettings(method="flat", flat_stake_pct=1.0, starting_bankroll=100))
    bets, s = run_backtest(frame(), cfg)
    # match 1: H EV = 0.6*2-1 = +0.20 -> bet, won at 2.0
    # match 2: A EV = 0.4*2.4-1 = -0.04, H 0.3*3-1=-0.1 -> none... D 0.3*3.2-1=-0.04 -> no bet
    # match 3: H EV = 0.5*1.8-1=-0.10, A EV=0.25*4.5-1=+0.125 -> bet, lost (draw)
    assert list(bets["match_id"]) == [1, 3]
    assert list(bets["selection"]) == ["H", "A"]
    assert s["n_bets"] == 2
    assert s["flat_profit_units"] == pytest.approx(1.0 - 1.0)
    assert s["hit_rate"] == pytest.approx(0.5)
    # CLV for bet 1: 2.0 * 0.52 - 1 = +0.04
    assert bets.iloc[0]["clv"] == pytest.approx(0.04)


def test_no_bet_when_nothing_qualifies():
    cfg = BacktestConfig(min_ev=0.5)
    bets, s = run_backtest(frame(), cfg)
    assert bets.empty and s["n_bets"] == 0


def test_baselines_and_sensitivity_run():
    cfg = BacktestConfig(price_source="market_avg", min_ev=0.0)
    b = baseline_strategies(frame(), cfg, n_random=5)
    home = b[b["strategy"] == "Back every home win"].iloc[0]
    assert home["profit_units"] == pytest.approx((2.0 - 1) - 1 - 1)
    sens = threshold_sensitivity(frame(), cfg, ev_grid=(0.0, 0.1, 0.3))
    assert list(sens["n_bets"]) == [2, 2, 0]


def test_stake_caps():
    cfg = StakingSettings(method="kelly", kelly_fraction=1.0, max_stake_pct=5, max_event_exposure_pct=50,
                          max_daily_exposure_pct=6, max_competition_exposure_pct=100, starting_bankroll=1000)
    day = pd.DataFrame({"match_id": [1, 2], "league": ["E0", "E0"], "prob": [0.7, 0.7], "odds": [2.0, 2.0]})
    s = size_stakes_for_day(day, 1000, cfg, 1000)
    # full Kelly would be 40% each; capped at 5% (50) each, then daily cap 6% total -> 30 each
    np.testing.assert_allclose(s, [30, 30])


def test_simulate_bankroll_flat():
    cfg = StakingSettings(method="flat", flat_stake_pct=10, max_stake_pct=20, max_daily_exposure_pct=100,
                          max_event_exposure_pct=50, max_competition_exposure_pct=100, starting_bankroll=100)
    bets = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02"]), "match_id": [1, 2],
                         "league": "E0", "prob": [0.6, 0.6], "odds": [2.5, 2.0], "won": [True, False]})
    out = simulate_bankroll(bets, cfg)
    assert out["pnl"].tolist() == pytest.approx([15.0, -10.0])
    assert out["bankroll_after"].iloc[-1] == pytest.approx(105.0)


def test_metrics():
    p = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    y = np.array([0, 2])
    np.testing.assert_allclose(log_loss_per_match(p, y), [-np.log(0.5)] * 2)
    np.testing.assert_allclose(brier_per_match(p, y), [0.25 + 0.09 + 0.04] * 2)
    # RPS for a perfect forecast is 0
    assert rps_per_match(np.array([[1.0, 0, 0]]), np.array([0]))[0] == pytest.approx(0)
    assert max_drawdown(np.array([100, 120, 90, 130]))[0] == pytest.approx(30)
    assert max_drawdown(np.array([100, 120, 90, 130]))[1] == pytest.approx(0.25)
    assert longest_losing_streak(np.array([1, 0, 0, 1, 0, 0, 0])) == 3
    m, lo, hi = bootstrap_mean_ci(np.ones(50))
    assert m == lo == hi == 1.0


def test_calibration_table_perfect():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.1, 0.9, 20000)
    y = rng.random(20000) < p
    t = calibration_table(p, y, n_bins=5)
    assert np.all(np.abs(t["mean_pred"] - t["observed"]) < 0.03)
