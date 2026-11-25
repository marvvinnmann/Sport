import numpy as np
import pytest

from sportpred.betting.odds import (
    closing_line_value,
    devig,
    edge,
    expected_value,
    fair_odds,
    implied_probability,
    kelly_fraction,
    margin,
    overround,
)


def test_implied_probability():
    assert implied_probability(2.0) == pytest.approx(0.5)
    assert implied_probability(4.0) == pytest.approx(0.25)
    assert np.isnan(implied_probability(1.0))  # invalid decimal odds
    np.testing.assert_allclose(implied_probability(np.array([2.0, 5.0])), [0.5, 0.2])


def test_overround_and_margin():
    odds = [1.9, 3.6, 4.2]
    expected = 1 / 1.9 + 1 / 3.6 + 1 / 4.2
    assert overround(odds) == pytest.approx(expected)
    assert margin(odds) == pytest.approx(expected - 1)
    assert margin([2.0, 2.0]) == pytest.approx(0.0)


def test_expected_value_and_edge():
    assert expected_value(0.55, 2.0) == pytest.approx(0.10)
    assert expected_value(0.5, 2.0) == pytest.approx(0.0)
    assert expected_value(0.4, 2.0) == pytest.approx(-0.2)
    assert edge(0.55, 2.0) == pytest.approx(0.05)


def test_likely_winner_is_not_value():
    # 80% likely but priced at 1.20 (83.3% implied) -> negative EV
    assert expected_value(0.80, 1.20) < 0


def test_kelly():
    # p=0.55 at evens: f* = (0.55*2-1)/(2-1) = 0.10
    assert kelly_fraction(0.55, 2.0) == pytest.approx(0.10)
    assert kelly_fraction(0.40, 2.0) == 0.0  # never stake on negative EV
    np.testing.assert_allclose(kelly_fraction(np.array([0.6, 0.3]), np.array([2.0, 3.0])), [0.2, 0.0])


def test_fair_odds_and_clv():
    assert fair_odds(0.25) == pytest.approx(4.0)
    # bet at 2.2 when the fair closing probability is 50% -> +10% CLV
    assert closing_line_value(2.2, 0.5) == pytest.approx(0.10)


@pytest.mark.parametrize("method", ["multiplicative", "power", "shin"])
def test_devig_sums_to_one_and_preserves_order(method):
    odds = [np.array([1.5, 2.5]), np.array([4.2, 3.3]), np.array([6.5, 2.9])]
    fair = devig(odds, method=method)
    total = fair[0] + fair[1] + fair[2]
    np.testing.assert_allclose(total, 1.0, atol=1e-9)
    assert fair[0][0] > fair[1][0] > fair[2][0]  # favourite stays favourite
    for f, o in zip(fair, odds, strict=True):
        assert np.all(f < implied_probability(o))  # removing the margin lowers each probability


def test_power_devig_takes_more_margin_from_longshots():
    odds = [np.array([1.30]), np.array([5.5]), np.array([11.0])]
    mult = devig(odds, "multiplicative")
    power = devig(odds, "power")
    assert power[0][0] > mult[0][0]  # favourite gets relatively more probability
    assert power[2][0] < mult[2][0]  # longshot gets less


def test_devig_handles_missing():
    fair = devig([np.array([2.0, np.nan]), np.array([2.0, 2.0])])
    assert fair[0][0] == pytest.approx(0.5)
    assert np.isnan(fair[0][1])
