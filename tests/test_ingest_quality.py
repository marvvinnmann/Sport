"""Database ingestion and data-quality checks (temporary database, simulated data)."""

import datetime as dt

from sqlalchemy import func, select

from sportpred.data.football_data import parse_football_data_csv
from sportpred.data.ingest import import_csv_file, mark_stale_fixtures, upsert_matches_and_odds
from sportpred.data.quality import run_quality_checks
from sportpred.db import DataUpdate, Match, Odds, session_scope

RESULTS = b"""Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A,PSH,PSD,PSA
E0,16/08/2025,15:00,Arsenal,Chelsea,2,1,2.0,3.5,4.0,2.05,3.6,4.1
E0,16/08/2025,15:00,Leeds,Everton,0,0,2.6,3.2,2.9,2.65,3.25,2.95
"""


def counts():
    with session_scope() as s:
        return (s.execute(select(func.count()).select_from(Match)).scalar_one(),
                s.execute(select(func.count()).select_from(Odds)).scalar_one())


def test_import_is_idempotent(temp_env):
    r1 = import_csv_file(RESULTS, "a.csv", data_label="sample")
    assert r1.status == "success" and r1.inserted == 2
    c1 = counts()
    r2 = import_csv_file(RESULTS, "a.csv", data_label="sample")
    assert r2.inserted == 0 and r2.updated == 2
    assert counts() == c1 == (2, 4)
    with session_scope() as s:
        assert s.execute(select(func.count()).select_from(DataUpdate)).scalar_one() == 2
        assert {m.data_label for m in s.execute(select(Match)).scalars()} == {"sample"}


def test_fixture_never_downgrades_finished_match(temp_env):
    import_csv_file(RESULTS, "a.csv")
    fixture = b"Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A\nE0,16/08/2025,15:00,Arsenal,Chelsea,2.0,3.5,4.0\n"
    parsed = parse_football_data_csv(fixture, is_fixture_file=True, snapshot_time=dt.datetime(2025, 8, 15))
    with session_scope() as s:
        upsert_matches_and_odds(s, parsed.matches, parsed.odds)
    with session_scope() as s:
        m = s.execute(select(Match).where(Match.home_team == "Arsenal")).scalar_one()
        assert m.status == "finished" and m.fthg == 2


def test_stale_fixtures_are_marked(temp_env):
    fixture = b"Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A\nE0,01/01/2020,15:00,A,B,2.0,3.5,4.0\n"
    parsed = parse_football_data_csv(fixture, is_fixture_file=True, snapshot_time=dt.datetime(2019, 12, 30))
    with session_scope() as s:
        upsert_matches_and_odds(s, parsed.matches, parsed.odds)
        assert mark_stale_fixtures(s) == 1


def test_quality_checks_detect_problems(temp_env):
    bad = b"""Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,HS,AS,HST,AST,B365H,B365D,B365A
E0,16/08/2025,15:00,Arsenal,Chelsea,2,1,5,5,9,3,2.0,3.5,4.0
E0,16/08/2025,17:30,Arsenal,Leeds,20,0,5,5,1,1,1.5,1.6,1.7
"""
    import_csv_file(bad, "bad.csv", data_label="simulated")
    with session_scope() as s:
        checks = {i.check for i in run_quality_checks(s)}
    assert "team_twice_same_day" in checks
    assert "implausible_goals" in checks
    assert "shots_inconsistent" in checks
    assert "non_live_data" in checks
    assert "implausible_margin" in checks


def test_quality_empty_database(temp_env):
    with session_scope() as s:
        issues = run_quality_checks(s)
    assert issues[0].check == "no_data"
