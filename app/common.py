"""Shared helpers for the Streamlit dashboard: cached data loaders, formatting, chart styling."""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import select

from sportpred.config import get_settings, load_model_settings
from sportpred.db import DataUpdate, get_engine, init_db, session_scope
from sportpred.leagues import LEAGUES, league_label

# Validated categorical palette (fixed order, never cycled) and reserved status colours.
SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
STATUS = {"good": "#0aac23", "warning": "#fbb315", "serious": "#f18a59", "critical": "#dc3c3d"}
NEUTRAL = "#8a8a8a"

VERDICT_LABEL = {
    "value": "✅ Positive EV",
    "insufficient_evidence": "⚠️ Insufficient evidence",
    "no_bet": "— No bet",
}
SELECTION_LABEL = {"H": "Home win", "D": "Draw", "A": "Away win", "OVER": "Over 2.5", "UNDER": "Under 2.5"}
MODEL_LABEL = {"market": "Market (pre-closing)", "elo": "Elo", "dixon_coles": "Dixon-Coles", "gbm": "Gradient boosting",
               "blend": "Blend: Dixon-Coles + market", "ensemble": "Ensemble (all + market)"}


def setup_page(title: str, icon: str = "⚽") -> None:
    init_db()
    st.title(f"{icon} {title}")


def disclaimer() -> None:
    st.caption("Research and paper-trading tool. Probabilities are estimates; positive expected value is not a "
               "guarantee of profit and most selections are correctly classified as *no bet*. No bets are placed.")


def data_label_banner() -> None:
    """Tell the viewer whether they are looking at live, sample or simulated data."""
    labels = load_data_labels()
    if not labels:
        st.warning("No data loaded yet. Go to **Data Management** and run an update.")
        return
    non_live = {k: v for k, v in labels.items() if k != "live"}
    if non_live:
        st.error(f"⚠️ Database contains NON-LIVE data: {non_live}. Do not judge real performance on it.")
    else:
        st.caption(f"🟢 LIVE data from football-data.co.uk — {sum(labels.values()):,} matches. "
                   "Not simulated, not sample data.")


@st.cache_data(ttl=60)
def load_data_labels() -> dict:
    df = pd.read_sql("SELECT data_label, COUNT(*) AS n FROM matches GROUP BY data_label", get_engine())
    return dict(zip(df["data_label"], df["n"], strict=True)) if not df.empty else {}


@st.cache_data(ttl=60)
def load_latest_predictions() -> pd.DataFrame:
    from sportpred.services.predictions import latest_predictions

    df = latest_predictions()
    if df.empty:
        return df
    df["league_name"] = df["league"].map(league_label)
    df["match"] = df["home_team"] + " v " + df["away_team"]
    df["selection_label"] = df["selection"].map(SELECTION_LABEL)
    df["verdict_label"] = df["verdict"].map(VERDICT_LABEL)
    return df


@st.cache_data(ttl=60)
def load_matches_cached(leagues: tuple[str, ...] | None = None, statuses: tuple[str, ...] | None = None) -> pd.DataFrame:
    from sportpred.data.repository import load_matches

    return load_matches(leagues=list(leagues) if leagues else None, statuses=list(statuses) if statuses else None)


@st.cache_data(ttl=60)
def load_updates(limit: int = 300) -> pd.DataFrame:
    with session_scope() as s:
        rows = s.execute(select(DataUpdate).order_by(DataUpdate.id.desc()).limit(limit)).scalars().all()
        return pd.DataFrame([{
            "finished_at": r.finished_at, "source": r.source, "resource": r.resource, "status": r.status,
            "new": r.rows_inserted, "updated": r.rows_updated, "message": r.message} for r in rows])


@st.cache_data(ttl=300)
def load_evaluation() -> dict | None:
    from sportpred.services.evaluation import load_evaluation_summary

    return load_evaluation_summary()


@st.cache_data(ttl=300)
def load_wf(model: str, mtime: float) -> pd.DataFrame | None:  # mtime busts the cache when the file changes
    from sportpred.services.evaluation import load_walk_forward

    return load_walk_forward(model)


def wf_mtime(model: str) -> float:
    from sportpred.services.evaluation import walk_forward_path

    p = walk_forward_path(model)
    return p.stat().st_mtime if p.exists() else 0.0


def last_update_text() -> str:
    df = load_updates(50)
    ok = df[df["status"].isin(["success", "unchanged"])] if not df.empty else df
    if ok.empty:
        return "never"
    t = pd.Timestamp(ok["finished_at"].max())
    age = dt.datetime.now(dt.UTC).replace(tzinfo=None) - t.to_pydatetime()
    hours = age.total_seconds() / 3600
    when = f"{hours:.0f} h ago" if hours < 48 else f"{hours / 24:.0f} days ago"
    return f"{t:%Y-%m-%d %H:%M} UTC ({when})"


def pct(x, digits: int = 1, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{x:+.{digits}%}" if signed else f"{x:.{digits}%}"


def style_fig(fig: go.Figure, height: int = 340, y_title: str | None = None, x_title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10), colorway=SERIES, hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(size=13),
    )
    fig.update_xaxes(showgrid=False, title=x_title)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False, title=y_title)
    return fig


def league_options() -> dict[str, str]:
    ms = load_model_settings()
    return {c: league_label(c) for c in ms.leagues if c in LEAGUES}


def odds_api_configured() -> bool:
    return get_settings().odds_api_key is not None


def details_of(row) -> dict:
    d = row.get("details") if isinstance(row, dict) else row["details"]
    if isinstance(d, str):
        return json.loads(d)
    return d or {}
