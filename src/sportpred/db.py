"""Database schema and session helpers (SQLAlchemy 2.0).

SQLite is the default (a single file, nothing to install). PostgreSQL works by changing
``SPORTPRED_DATABASE_URL`` — the schema is identical.

Tables
------
matches          one row per match (finished or scheduled), with results and match statistics
odds             bookmaker prices per match / bookmaker / market / line, pre-closing and closing
data_updates     audit log of every download/import (source, time, rows, status, file hash)
model_versions   every trained model configuration that produced stored predictions
predictions      append-only store of predictions (never overwritten -> full history)
paper_bets       paper-trading ledger (no real money, no bookmaker connection)
backtest_runs    saved backtest configurations and summary results
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from sportpred.config import get_settings


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_key: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    sport: Mapped[str] = mapped_column(String(30), default="football")
    league: Mapped[str] = mapped_column(String(10), index=True)
    season: Mapped[str | None] = mapped_column(String(10), index=True)
    date: Mapped[dt.date] = mapped_column(Date, index=True)
    kickoff: Mapped[str | None] = mapped_column(String(10))  # "HH:MM", UK time per provider
    home_team: Mapped[str] = mapped_column(String(80), index=True)
    away_team: Mapped[str] = mapped_column(String(80), index=True)
    status: Mapped[str] = mapped_column(String(20), default="scheduled", index=True)
    # results
    fthg: Mapped[int | None] = mapped_column(Integer)
    ftag: Mapped[int | None] = mapped_column(Integer)
    ftr: Mapped[str | None] = mapped_column(String(1))
    hthg: Mapped[int | None] = mapped_column(Integer)
    htag: Mapped[int | None] = mapped_column(Integer)
    # match statistics (where available)
    hs: Mapped[int | None] = mapped_column(Integer)
    as_: Mapped[int | None] = mapped_column("as", Integer)
    hst: Mapped[int | None] = mapped_column(Integer)
    ast: Mapped[int | None] = mapped_column(Integer)
    hc: Mapped[int | None] = mapped_column(Integer)
    ac: Mapped[int | None] = mapped_column(Integer)
    hf: Mapped[int | None] = mapped_column(Integer)
    af: Mapped[int | None] = mapped_column(Integer)
    hy: Mapped[int | None] = mapped_column(Integer)
    ay: Mapped[int | None] = mapped_column(Integer)
    hr: Mapped[int | None] = mapped_column(Integer)
    ar: Mapped[int | None] = mapped_column(Integer)
    hxg: Mapped[float | None] = mapped_column(Float)
    axg: Mapped[float | None] = mapped_column(Float)
    referee: Mapped[str | None] = mapped_column(String(80))
    # provenance
    source: Mapped[str] = mapped_column(String(60))
    source_file: Mapped[str | None] = mapped_column(String(200))
    data_label: Mapped[str] = mapped_column(String(20), default="live")  # live | sample | simulated
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class Odds(Base):
    __tablename__ = "odds"
    __table_args__ = (
        UniqueConstraint("match_id", "bookmaker", "market", "line", "is_closing", name="uq_odds"),
        Index("ix_odds_match_market", "match_id", "market"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"))
    bookmaker: Mapped[str] = mapped_column(String(40))
    market: Mapped[str] = mapped_column(String(10))  # 1X2 | OU | AH
    line: Mapped[float] = mapped_column(Float, default=0.0)  # 0 for 1X2, 2.5 for OU, handicap for AH
    is_closing: Mapped[bool] = mapped_column(Boolean, default=False)
    price_home: Mapped[float | None] = mapped_column(Float)  # 1X2 home / AH home
    price_draw: Mapped[float | None] = mapped_column(Float)
    price_away: Mapped[float | None] = mapped_column(Float)  # 1X2 away / AH away
    price_over: Mapped[float | None] = mapped_column(Float)
    price_under: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(60))
    collected_at: Mapped[dt.datetime | None] = mapped_column(DateTime)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class DataUpdate(Base):
    __tablename__ = "data_updates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(60), index=True)
    resource: Mapped[str] = mapped_column(String(200))
    started_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(20))  # success | unchanged | failed | skipped
    rows_inserted: Mapped[int] = mapped_column(Integer, default=0)
    rows_updated: Mapped[int] = mapped_column(Integer, default=0)
    content_hash: Mapped[str | None] = mapped_column(String(64))
    message: Mapped[str | None] = mapped_column(Text)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(40))
    version: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    trained_until: Mapped[dt.date | None] = mapped_column(Date)
    n_training_matches: Mapped[int | None] = mapped_column(Integer)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    notes: Mapped[str | None] = mapped_column(Text)


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (Index("ix_pred_match_market", "match_id", "market"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(40), index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, index=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"))
    model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"))
    market: Mapped[str] = mapped_column(String(10))  # 1X2 | OU25
    selection: Mapped[str] = mapped_column(String(10))  # H D A | OVER UNDER
    prob: Mapped[float] = mapped_column(Float)
    prob_low: Mapped[float | None] = mapped_column(Float)
    prob_high: Mapped[float | None] = mapped_column(Float)
    best_odds: Mapped[float | None] = mapped_column(Float)
    best_bookmaker: Mapped[str | None] = mapped_column(String(40))
    implied_prob: Mapped[float | None] = mapped_column(Float)
    fair_prob: Mapped[float | None] = mapped_column(Float)
    edge: Mapped[float | None] = mapped_column(Float)
    ev: Mapped[float | None] = mapped_column(Float)
    ev_low: Mapped[float | None] = mapped_column(Float)
    margin: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[str | None] = mapped_column(String(10))
    verdict: Mapped[str] = mapped_column(String(30))  # value | insufficient_evidence | no_bet
    is_recommended: Mapped[bool] = mapped_column(Boolean, default=False)
    odds_collected_at: Mapped[dt.datetime | None] = mapped_column(DateTime)
    details: Mapped[dict] = mapped_column(JSON, default=dict)


class PaperBet(Base):
    __tablename__ = "paper_bets"
    __table_args__ = (UniqueConstraint("match_id", "market", "selection", name="uq_paper_bet"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[int | None] = mapped_column(ForeignKey("predictions.id"))
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"))
    placed_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    market: Mapped[str] = mapped_column(String(10))
    selection: Mapped[str] = mapped_column(String(10))
    odds: Mapped[float] = mapped_column(Float)
    bookmaker: Mapped[str | None] = mapped_column(String(40))
    model_prob: Mapped[float] = mapped_column(Float)
    stake: Mapped[float] = mapped_column(Float, default=1.0)
    status: Mapped[str] = mapped_column(String(10), default="open")  # open | won | lost | void
    pnl: Mapped[float | None] = mapped_column(Float)
    closing_odds: Mapped[float | None] = mapped_column(Float)
    clv: Mapped[float | None] = mapped_column(Float)
    settled_at: Mapped[dt.datetime | None] = mapped_column(DateTime)


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    name: Mapped[str | None] = mapped_column(String(120))
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    summary: Mapped[dict] = mapped_column(JSON, default=dict)


# --------------------------------------------------------------------------- engine/session

_engines: dict[str, Engine] = {}


def get_engine(url: str | None = None) -> Engine:
    url = url or get_settings().resolved_database_url()
    if url not in _engines:
        kwargs = {}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        engine = create_engine(url, json_serializer=lambda o: json.dumps(o, default=str), **kwargs)
        if url.startswith("sqlite"):

            @event.listens_for(engine, "connect")
            def _sqlite_pragmas(dbapi_conn, _record):
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA foreign_keys=ON")
                cur.execute("PRAGMA journal_mode=WAL")
                cur.close()

        _engines[url] = engine
    return _engines[url]


def init_db(url: str | None = None) -> Engine:
    engine = get_engine(url)
    Base.metadata.create_all(engine)
    return engine


@contextmanager
def session_scope(url: str | None = None) -> Iterator[Session]:
    engine = init_db(url)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
