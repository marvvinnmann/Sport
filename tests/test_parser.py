"""Parser tests on small hand-written CSV snippets in football-data.co.uk format."""

import datetime as dt

import pytest

from sportpred.data.football_data import approx_preclosing_collected_at, parse_football_data_csv

CSV_2025 = b"""Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,B365H,B365D,B365A,BFDH,BFDD,BFDA,PSH,PSD,PSA,MaxH,MaxD,MaxA,AvgH,AvgD,AvgA,B365>2.5,B365<2.5,AHh,B365AHH,B365AHA,B365CH,B365CD,B365CA,PSCH,PSCD,PSCA,AHCh,B365CAHH,B365CAHA
E0,16/08/2025,15:00,Arsenal,Chelsea,2,1,H,15,9,6,3,2.0,3.5,4.0,1.95,3.4,4.1,2.05,3.6,4.1,2.1,3.7,4.3,2.0,3.5,4.0,1.8,2.0,-0.5,1.95,1.95,1.9,3.6,4.3,1.95,3.7,4.4,-0.5,1.9,2.0
E0,16/08/2025,17:30,Leeds,Everton,0,0,D,8,10,2,4,2.6,3.2,2.9,2.6,3.1,2.9,2.65,3.25,2.95,2.7,3.3,3.0,2.6,3.2,2.9,2.1,1.75,0,1.9,2.0,,,,,,,,,
"""


def test_parse_results_file_matches_and_odds():
    p = parse_football_data_csv(CSV_2025, league="E0", season="2526")
    assert len(p.matches) == 2
    m = p.matches.set_index("home_team")
    assert m.loc["Arsenal", "ftr"] == "H"
    assert m.loc["Leeds", "ftr"] == "D"
    assert m.loc["Arsenal", "status"] == "finished"
    assert m.loc["Arsenal", "hst"] == 6
    assert m.loc["Arsenal", "date"] == dt.date(2025, 8, 16)
    o = p.odds
    books = set(o.loc[(o.market == "1X2") & ~o.is_closing, "bookmaker"])
    assert {"bet365", "betfred", "pinnacle", "market_max", "market_avg"} <= books
    # BFDH (Betfred) must not be confused with BFD (Betfair draw)
    assert "betfair_sportsbook" not in books
    closing = o[(o.market == "1X2") & o.is_closing]
    assert set(closing["bookmaker"]) == {"bet365", "pinnacle"}
    assert len(closing) == 2  # second match has empty closing prices -> skipped
    ah = o[(o.market == "AH") & ~o.is_closing]
    assert ah["line"].tolist() == [-0.5, 0.0]
    ou = o[o.market == "OU"]
    assert ou["line"].eq(2.5).all()


def test_old_format_two_digit_years_and_bb_columns():
    csv = b"""Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A,BbMxH,BbMxD,BbMxA,BbAvH,BbAvD,BbAvA,BbAv>2.5,BbAv<2.5,PH,PD,PA
E0,18/08/12,Arsenal,Sunderland,0,0,D,1.36,4.75,9.5,1.4,5.0,10.0,1.36,4.6,8.7,1.8,2.0,1.38,4.9,10.2
"""
    p = parse_football_data_csv(csv, league="E0", season="1213")
    assert p.matches.iloc[0]["date"] == dt.date(2012, 8, 18)
    books = set(p.odds["bookmaker"])
    assert {"market_max", "market_avg", "pinnacle", "bet365"} <= books


def test_malformed_and_duplicate_rows_are_reported():
    csv = CSV_2025 + b"E0,16/08/2025,15:00,Arsenal,Chelsea,2,1,H,15,9,6,3,2.0,3.5,4.0,,,,,,,,,,,,,,,,,,,,,,,,,,\n"
    csv += b"E0,notadate,15:00,X,Y,1,1,D\n"
    csv += b"E0,17/08/2025,15:00,A,B,1,0,H,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37\n"
    p = parse_football_data_csv(csv, league="E0", season="2526")
    assert len(p.matches) == 2  # duplicate collapsed, bad date dropped, too-long line skipped
    text = " ".join(p.warnings)
    assert "duplicate" in text and "without a valid date" in text and "malformed" in text


def test_fixture_file_is_scheduled_with_snapshot_time():
    csv = b"""Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A,BFEH,BFED,BFEA
E1,24/11/2026,19:45,Bristol City,Wrexham,2.38,3.2,3.0,2.56,3.3,3.15
"""
    snap = dt.datetime(2026, 11, 24, 9, 30)
    p = parse_football_data_csv(csv, is_fixture_file=True, snapshot_time=snap)
    assert p.matches.iloc[0]["status"] == "scheduled"
    assert p.matches.iloc[0]["season"] == "2627"
    assert set(p.odds["bookmaker"]) == {"bet365", "betfair_exchange"}
    assert (p.odds["collected_at"] == snap).all()


def test_invalid_prices_are_dropped():
    csv = b"""Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A
E0,01/09/2024,A,B,1,0,1.0,3.0,4.0
"""
    p = parse_football_data_csv(csv, league="E0", season="2425")
    assert p.odds.empty  # 1.0 is not a valid decimal price -> incomplete market


def test_missing_required_columns():
    with pytest.raises(ValueError):
        parse_football_data_csv(b"Div,Date,Foo\nE0,01/01/2024,x\n")


@pytest.mark.parametrize("match_day,expected", [
    (dt.date(2025, 8, 16), dt.date(2025, 8, 15)),  # Saturday -> Friday
    (dt.date(2025, 8, 18), dt.date(2025, 8, 15)),  # Monday -> previous Friday
    (dt.date(2025, 8, 20), dt.date(2025, 8, 19)),  # Wednesday -> Tuesday
    (dt.date(2025, 8, 15), dt.date(2025, 8, 15)),  # Friday -> same day
])
def test_preclosing_collection_time(match_day, expected):
    assert approx_preclosing_collected_at(match_day).date() == expected
