"""Test fixtures: every test that touches the database gets its own temporary SQLite file."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "app"))


@pytest.fixture()
def temp_env(tmp_path, monkeypatch):
    """Isolated database + data folder. Returns the data directory."""
    import sportpred.config as config
    import sportpred.db as db

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("SPORTPRED_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("SPORTPRED_DATA_DIR", str(data_dir))
    monkeypatch.delenv("SPORTPRED_ODDS_API_KEY", raising=False)
    config.get_settings.cache_clear()
    db._engines.clear()
    db.init_db()
    yield data_dir
    for e in db._engines.values():
        e.dispose()
    db._engines.clear()
    config.get_settings.cache_clear()


@pytest.fixture()
def simulated_db(temp_env):
    """Database filled with SIMULATED matches + odds (labelled 'simulated') and some fixtures."""
    import datetime as dt

    import numpy as np
    from synthetic import simulate_league, to_football_data_csv

    from sportpred.data.football_data import parse_football_data_csv
    from sportpred.data.ingest import import_csv_file, upsert_matches_and_odds
    from sportpred.db import session_scope

    df, truth = simulate_league(n_teams=12, seasons=4, start_year=2021, seed=1)
    res = import_csv_file(to_football_data_csv(df), "simulated_E0.csv", data_label="simulated")
    assert res.status == "success", res.message
    # upcoming fixtures (no result) with odds, dated in the future
    today = dt.date.today()
    teams = truth["teams"]
    rows = ["Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A,AvgH,AvgD,AvgA,PSH,PSD,PSA,B365>2.5,B365<2.5,Avg>2.5,Avg<2.5"]
    for i in range(0, 6, 2):
        d = (today + dt.timedelta(days=2)).strftime("%d/%m/%Y")
        rows.append(f"E0,{d},15:00,{teams[i]},{teams[i + 1]},2.10,3.40,3.60,2.05,3.35,3.55,2.12,3.45,3.70,1.90,1.95,1.88,1.93")
    parsed = parse_football_data_csv("\n".join(rows).encode(), is_fixture_file=True,
                                     snapshot_time=dt.datetime.now())
    with session_scope() as s:
        upsert_matches_and_odds(s, parsed.matches, parsed.odds, data_label="simulated")
    np.random.seed(0)
    return {"truth": truth, "matches": df, "data_dir": temp_env}
