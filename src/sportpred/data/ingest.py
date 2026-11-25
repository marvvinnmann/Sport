"""Write parsed provider data into the database, with an audit trail for every update."""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from sportpred.config import ModelSettings, get_settings, load_model_settings
from sportpred.data.football_data import SOURCE, FootballDataClient, parse_football_data_csv
from sportpred.db import DataUpdate, Match, Odds, session_scope, utcnow
from sportpred.leagues import current_season_start_year, season_code

log = logging.getLogger(__name__)

MATCH_FIELDS = [
    "match_key", "league", "season", "date", "kickoff", "home_team", "away_team", "status",
    "fthg", "ftag", "ftr", "hthg", "htag", "hs", "as_", "hst", "ast", "hc", "ac", "hf", "af",
    "hy", "ay", "hr", "ar", "hxg", "axg", "referee", "source", "source_file",
]
ODDS_FIELDS = ["bookmaker", "market", "line", "is_closing", "price_home", "price_draw", "price_away",
               "price_over", "price_under", "source", "collected_at"]


@dataclass
class IngestResult:
    resource: str
    status: str
    inserted: int = 0
    updated: int = 0
    odds_rows: int = 0
    warnings: list[str] = field(default_factory=list)
    message: str | None = None


def _insert_fn(session: Session):
    return pg_insert if session.bind.dialect.name == "postgresql" else sqlite_insert


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    if v is pd.NA or v is pd.NaT:
        return None
    return v


def _records(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    out = []
    for row in df[cols].itertuples(index=False, name=None):
        out.append({c: _clean(v) for c, v in zip(cols, row, strict=True)})
    return out


def upsert_matches_and_odds(
    session: Session, matches: pd.DataFrame, odds: pd.DataFrame, data_label: str = "live",
    replace_odds_source: str | None = SOURCE,
) -> tuple[int, int, int]:
    """Insert or update matches (by ``match_key``) and replace their odds from this source.

    A finished result is never downgraded back to 'scheduled' by a later fixtures file.
    Returns (inserted, updated, odds_rows).
    """
    if matches.empty:
        return 0, 0, 0
    ins = _insert_fn(session)
    matches = matches.copy()
    for c in MATCH_FIELDS:
        if c not in matches.columns:
            matches[c] = None
    keys = matches["match_key"].tolist()

    existing: dict[str, tuple[int, str]] = {}
    for chunk_start in range(0, len(keys), 500):
        chunk = keys[chunk_start:chunk_start + 500]
        for mid, key, status in session.execute(
            select(Match.id, Match.match_key, Match.status).where(Match.match_key.in_(chunk))
        ):
            existing[key] = (mid, status)

    # never overwrite a finished match with an unfinished row
    finished_keys = {k for k, (_, st) in existing.items() if st == "finished"}
    incoming_unfinished = matches["status"] != "finished"
    protect = matches["match_key"].isin(finished_keys) & incoming_unfinished
    writable = matches.loc[~protect]

    rows = _records(writable, MATCH_FIELDS)
    now = utcnow()
    for r in rows:
        r["as"] = r.pop("as_")
        r["data_label"] = data_label
        r["updated_at"] = now
        r["sport"] = "football"
    inserted = sum(1 for r in rows if r["match_key"] not in existing)
    updated = len(rows) - inserted
    for start in range(0, len(rows), 500):
        batch = rows[start:start + 500]
        stmt = ins(Match.__table__).values(batch)
        update_cols = {c: stmt.excluded[c] for c in batch[0] if c not in ("match_key",)}
        stmt = stmt.on_conflict_do_update(index_elements=["match_key"], set_=update_cols)
        session.execute(stmt)

    # map keys -> ids (fresh)
    ids: dict[str, int] = {}
    for chunk_start in range(0, len(keys), 500):
        chunk = keys[chunk_start:chunk_start + 500]
        for mid, key in session.execute(select(Match.id, Match.match_key).where(Match.match_key.in_(chunk))):
            ids[key] = mid

    odds_rows = 0
    if odds is not None and not odds.empty:
        odds = odds[odds["match_key"].isin(writable["match_key"])].copy()
        odds["match_id"] = odds["match_key"].map(ids)
        odds = odds.dropna(subset=["match_id"])
        match_ids = sorted(set(int(x) for x in odds["match_id"]))
        if replace_odds_source and match_ids:
            for start in range(0, len(match_ids), 500):
                chunk = match_ids[start:start + 500]
                session.execute(delete(Odds).where(Odds.match_id.in_(chunk), Odds.source == replace_odds_source))
        if "source" not in odds.columns:
            odds["source"] = replace_odds_source or "manual"
        recs = _records(odds, ["match_id", *ODDS_FIELDS])
        for r in recs:
            r["match_id"] = int(r["match_id"])
            r["is_closing"] = bool(r["is_closing"])
            r["line"] = float(r["line"]) if r["line"] is not None else 0.0
            r["updated_at"] = now
        for start in range(0, len(recs), 1000):
            batch = recs[start:start + 1000]
            stmt = ins(Odds.__table__).values(batch)
            stmt = stmt.on_conflict_do_update(
                index_elements=["match_id", "bookmaker", "market", "line", "is_closing"],
                set_={c: stmt.excluded[c] for c in ("price_home", "price_draw", "price_away", "price_over",
                                                     "price_under", "source", "collected_at", "updated_at")},
            )
            session.execute(stmt)
        odds_rows = len(recs)
    return inserted, updated, odds_rows


def _last_success_hash(session: Session, source: str, resource: str) -> str | None:
    row = session.execute(
        select(DataUpdate.content_hash)
        .where(DataUpdate.source == source, DataUpdate.resource == resource,
               DataUpdate.status.in_(["success", "unchanged"]))
        .order_by(DataUpdate.id.desc())
        .limit(1)
    ).scalar_one_or_none()
    return row


def _log_update(session: Session, source: str, resource: str, started: dt.datetime, result: IngestResult,
                content_hash: str | None) -> None:
    msg_parts = [result.message] if result.message else []
    msg_parts += result.warnings
    session.add(DataUpdate(
        source=source, resource=resource, started_at=started, finished_at=utcnow(), status=result.status,
        rows_inserted=result.inserted, rows_updated=result.updated, content_hash=content_hash,
        message="; ".join(m for m in msg_parts if m) or None,
    ))


def ingest_season(session: Session, client: FootballDataClient, league: str, season: str,
                  force: bool = False) -> IngestResult:
    resource = f"{season}/{league}.csv"
    started = utcnow()
    dl = client.download_season(league, season, force=force)
    if dl.error:
        status = "skipped" if "404" in dl.error else "failed"
        res = IngestResult(resource, status, message=dl.error)
        _log_update(session, SOURCE, resource, started, res, None)
        return res
    if not force and dl.content_hash == _last_success_hash(session, SOURCE, resource):
        res = IngestResult(resource, "unchanged", message="file unchanged since last import")
        _log_update(session, SOURCE, resource, started, res, dl.content_hash)
        return res
    try:
        parsed = parse_football_data_csv(dl.content, league=league, season=season, source_file=resource)
        ins_, upd, n_odds = upsert_matches_and_odds(session, parsed.matches, parsed.odds)
        res = IngestResult(resource, "success", ins_, upd, n_odds, parsed.warnings,
                           message="from cache" if dl.from_cache else None)
    except Exception as exc:  # keep going with other files, but record the failure
        log.exception("Failed to ingest %s", resource)
        session.rollback()
        res = IngestResult(resource, "failed", message=f"{type(exc).__name__}: {exc}")
    _log_update(session, SOURCE, resource, started, res, dl.content_hash)
    return res


def ingest_fixtures(session: Session, client: FootballDataClient, leagues: list[str] | None = None) -> IngestResult:
    resource = "fixtures.csv"
    started = utcnow()
    dl = client.download_fixtures()
    if dl.error:
        res = IngestResult(resource, "failed", message=dl.error)
        _log_update(session, SOURCE, resource, started, res, None)
        return res
    snapshot = dl.last_modified or utcnow()
    try:
        parsed = parse_football_data_csv(dl.content, source_file=resource, is_fixture_file=True,
                                         snapshot_time=snapshot)
        m, o = parsed.matches, parsed.odds
        if leagues:
            m = m[m["league"].isin(leagues)]
            o = o[o["match_key"].isin(m["match_key"])]
        n_superseded = _supersede_rescheduled(session, m)
        ins_, upd, n_odds = upsert_matches_and_odds(session, m, o)
        msg = f"odds snapshot time {snapshot:%Y-%m-%d %H:%M} UTC"
        if n_superseded:
            msg += f"; {n_superseded} rescheduled fixture(s) superseded"
        res = IngestResult(resource, "success", ins_, upd, n_odds, parsed.warnings, message=msg)
    except Exception as exc:
        log.exception("Failed to ingest fixtures")
        session.rollback()
        res = IngestResult(resource, "failed", message=f"{type(exc).__name__}: {exc}")
    _log_update(session, SOURCE, resource, started, res, dl.content_hash)
    return res


def _supersede_rescheduled(session: Session, fixtures: pd.DataFrame) -> int:
    """If a fixture now appears on a new date, mark the old scheduled row as 'superseded'."""
    n = 0
    for row in fixtures.itertuples(index=False):
        res = session.execute(
            update(Match)
            .where(Match.league == row.league, Match.home_team == row.home_team,
                   Match.away_team == row.away_team, Match.status == "scheduled",
                   Match.match_key != row.match_key)
            .values(status="superseded")
        )
        n += res.rowcount or 0
    return n


def mark_stale_fixtures(session: Session, today: dt.date | None = None, grace_days: int = 3) -> int:
    """Scheduled matches whose date passed without a result get status 'stale'."""
    today = today or dt.date.today()
    res = session.execute(
        update(Match)
        .where(Match.status == "scheduled", Match.date < today - dt.timedelta(days=grace_days))
        .values(status="stale")
    )
    return res.rowcount or 0


def update_all(
    leagues: list[str] | None = None,
    first_season: int | None = None,
    last_season: int | None = None,
    force: bool = False,
    include_fixtures: bool = True,
    settings: ModelSettings | None = None,
    progress=None,
) -> list[IngestResult]:
    """Download (or reuse cached) season files for the configured leagues, then fixtures."""
    ms = settings or load_model_settings()
    leagues = leagues or ms.leagues
    first = first_season or ms.first_season
    last = last_season or current_season_start_year()
    client = FootballDataClient(get_settings())
    results: list[IngestResult] = []
    with session_scope() as session:
        for league in leagues:
            for year in range(first, last + 1):
                res = ingest_season(session, client, league, season_code(year), force=force)
                session.commit()
                results.append(res)
                if progress:
                    progress(res)
        if include_fixtures:
            res = ingest_fixtures(session, client, leagues=leagues)
            results.append(res)
            if progress:
                progress(res)
        mark_stale_fixtures(session)
    return results


def import_csv_file(content: bytes, filename: str, league: str | None = None, season: str | None = None,
                    data_label: str = "live") -> IngestResult:
    """Manual CSV import (football-data.co.uk format). ``data_label`` must state honestly
    whether the file is live/real data, a sample, or simulated."""
    started = utcnow()
    resource = f"manual:{filename}"
    digest = hashlib.sha256(content).hexdigest()
    with session_scope() as session:
        try:
            parsed = parse_football_data_csv(content, league=league, season=season, source_file=resource)
            if parsed.matches.empty:
                res = IngestResult(resource, "failed", message="no valid rows found")
            else:
                if parsed.matches["league"].isna().any():
                    raise ValueError("League code missing: add a 'Div' column or choose a league")
                m = parsed.matches.assign(source=f"manual:{filename}")
                o = parsed.odds.assign(source=f"manual:{filename}")
                ins_, upd, n_odds = upsert_matches_and_odds(session, m, o, data_label=data_label,
                                                            replace_odds_source=f"manual:{filename}")
                res = IngestResult(resource, "success", ins_, upd, n_odds, parsed.warnings)
        except Exception as exc:
            session.rollback()
            res = IngestResult(resource, "failed", message=f"{type(exc).__name__}: {exc}")
        _log_update(session, "manual-csv", resource, started, res, digest)
    return res
