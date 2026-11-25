"""Common model interface and helpers.

A model is fitted on *finished* matches strictly before the prediction date and returns a
DataFrame aligned with the fixtures it is asked about. Probability column names:

    p_H, p_D, p_A          1X2 market (home win, draw, away win)
    p_OVER, p_UNDER        total goals over/under 2.5
    xg_home, xg_away       expected goals (models that produce them)
"""

from __future__ import annotations

import abc
from typing import ClassVar

import numpy as np
import pandas as pd

PROB_COLS_1X2 = ["p_H", "p_D", "p_A"]
PROB_COLS_OU = ["p_OVER", "p_UNDER"]


class MatchModel(abc.ABC):
    name: ClassVar[str] = "base"
    markets: ClassVar[set[str]] = {"1X2"}

    @abc.abstractmethod
    def fit(self, history: pd.DataFrame, as_of: pd.Timestamp | None = None) -> MatchModel:
        """Fit on finished matches. ``as_of`` = the moment predictions are made (no data on/after it)."""

    @abc.abstractmethod
    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """Return probabilities indexed like ``fixtures``."""

    def get_params(self) -> dict:
        return {k: v for k, v in vars(self).items() if not k.startswith("_") and _jsonable(v)}

    def explain(self, fixture: pd.Series) -> dict:  # pragma: no cover - optional
        return {}


def _jsonable(v) -> bool:
    return isinstance(v, (int, float, str, bool, type(None), list, tuple))


def finished_only(history: pd.DataFrame, as_of: pd.Timestamp | None) -> pd.DataFrame:
    h = history
    if "status" in h.columns:
        h = h[h["status"] == "finished"]
    h = h.dropna(subset=["fthg", "ftag"])
    if as_of is not None:
        h = h[pd.to_datetime(h["date"]) < pd.Timestamp(as_of)]
    return h


def outcome_index(df: pd.DataFrame) -> np.ndarray:
    """0 = home win, 1 = draw, 2 = away win."""
    hg, ag = df["fthg"].to_numpy(), df["ftag"].to_numpy()
    return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))


def over_index(df: pd.DataFrame, line: float = 2.5) -> np.ndarray:
    return ((df["fthg"] + df["ftag"]).to_numpy() > line).astype(int)


class PerLeague(MatchModel):
    """Fit one independent model per league and route predictions by league code."""

    def __init__(self, factory, name: str | None = None):
        self._factory = factory
        self._models: dict[str, MatchModel] = {}
        proto = factory()
        self.name = name or proto.name  # type: ignore[misc]
        self.markets = proto.markets  # type: ignore[misc]
        self._proto = proto

    def fit(self, history: pd.DataFrame, as_of=None) -> PerLeague:
        self._models = {}
        for league, h in history.groupby("league"):
            self._models[league] = self._factory().fit(h, as_of=as_of)
        return self

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for league, f in fixtures.groupby("league"):
            model = self._models.get(league)
            if model is None:
                continue
            parts.append(model.predict(f))
        if not parts:
            return pd.DataFrame(index=fixtures.index)
        return pd.concat(parts).reindex(fixtures.index)

    def model_for(self, league: str) -> MatchModel | None:
        return self._models.get(league)

    def get_params(self) -> dict:
        return self._proto.get_params()
