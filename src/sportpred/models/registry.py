"""Model registry: build models by name from ModelSettings."""

from __future__ import annotations

from sportpred.config import ModelSettings
from sportpred.models.base import MatchModel, PerLeague
from sportpred.models.dixon_coles import DixonColes
from sportpred.models.elo_model import EloModel

MODEL_DESCRIPTIONS = {
    "dixon_coles": "Dixon-Coles Poisson model with time decay (1X2 + Over/Under). Default.",
    "elo": "Elo ratings + ordered logistic link (1X2 only). Transparent baseline.",
    "gbm": "LightGBM on pre-match features incl. Elo, Dixon-Coles, form, shots, rest days (1X2 + O/U).",
    "ensemble": "Average of Dixon-Coles, Elo and GBM with calibration (1X2 + O/U).",
}


def dixon_coles_factory(ms: ModelSettings | None = None):
    ms = ms or ModelSettings()
    return lambda: DixonColes(xi=ms.dc_xi, window_days=ms.dc_window_days, ridge=ms.dc_ridge)


def elo_factory(ms: ModelSettings | None = None):
    ms = ms or ModelSettings()
    return lambda: EloModel(k=ms.elo_k, home_adv=ms.elo_home_advantage)


def build_model(name: str, ms: ModelSettings | None = None) -> MatchModel:
    ms = ms or ModelSettings()
    if name == "dixon_coles":
        return PerLeague(dixon_coles_factory(ms))
    if name == "elo":
        return PerLeague(elo_factory(ms))
    if name in ("gbm", "ensemble"):
        from sportpred.models.gbm import build_stacked_model

        return build_stacked_model(name, ms)
    raise ValueError(f"unknown model {name!r}")


def single_league_factory(name: str, ms: ModelSettings | None = None):
    if name == "dixon_coles":
        return dixon_coles_factory(ms)
    if name == "elo":
        return elo_factory(ms)
    raise ValueError(f"{name!r} has no single-league factory")
