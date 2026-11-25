"""Configuration.

Two kinds of settings exist:

1. ``Settings`` — environment settings (database location, API keys). Read from environment
   variables or a ``.env`` file. Secrets live here and are never written to disk by the app.
2. ``ModelSettings`` — user-tunable modelling/betting settings (leagues, thresholds, staking).
   Defaults live in ``config/default_settings.yaml`` (committed); changes made in the dashboard
   are saved to ``<data_dir>/user_settings.yaml`` (not committed).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_FILE = PROJECT_ROOT / "config" / "default_settings.yaml"


class Settings(BaseSettings):
    """Environment-level settings. Prefix every variable with ``SPORTPRED_``."""

    model_config = SettingsConfigDict(
        env_prefix="SPORTPRED_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str = "sqlite:///data/sportpred.db"
    data_dir: Path = Path("data")
    football_data_base_url: str = "https://football-data.co.uk"
    http_user_agent: str = "sportpred/0.1 (research; non-commercial)"
    http_timeout_seconds: float = 30.0
    request_pause_seconds: float = 1.0
    odds_api_key: SecretStr | None = None
    odds_api_regions: str = "uk,eu"
    odds_api_base_url: str = "https://api.the-odds-api.com"

    @field_validator("odds_api_key", mode="before")
    @classmethod
    def _empty_key_is_none(cls, v):
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return v

    def resolved_data_dir(self) -> Path:
        d = self.data_dir if self.data_dir.is_absolute() else PROJECT_ROOT / self.data_dir
        d.mkdir(parents=True, exist_ok=True)
        return d

    def resolved_database_url(self) -> str:
        """Make relative SQLite paths relative to the project root, so every entry point
        (CLI, API, dashboard, tests) uses the same file regardless of the current folder."""
        url = self.database_url
        prefix = "sqlite:///"
        if url.startswith(prefix) and not url.startswith("sqlite:////") and ":memory:" not in url:
            rel = url[len(prefix):]
            path = PROJECT_ROOT / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            return f"{prefix}{path}"
        return url


class StakingSettings(BaseModel):
    method: Literal["flat", "kelly"] = "flat"
    flat_stake_pct: float = Field(1.0, ge=0, le=10, description="% of starting bankroll per bet")
    kelly_fraction: float = Field(0.25, ge=0, le=1)
    max_stake_pct: float = Field(2.0, ge=0, le=20, description="cap per bet, % of bankroll")
    max_event_exposure_pct: float = Field(3.0, ge=0, le=50)
    max_daily_exposure_pct: float = Field(10.0, ge=0, le=100)
    max_competition_exposure_pct: float = Field(6.0, ge=0, le=100)
    starting_bankroll: float = Field(1000.0, gt=0)


class ModelSettings(BaseModel):
    """User-tunable settings. All money/odds values use decimal odds."""

    leagues: list[str] = ["E0"]
    first_season: int = Field(2012, description="first season start year to download")
    markets: list[Literal["1X2", "OU25"]] = ["1X2", "OU25"]
    active_model: Literal["blend", "dixon_coles", "elo"] = "blend"

    # Value-bet thresholds
    min_ev: float = Field(0.03, description="minimum expected value per unit stake")
    min_edge: float = Field(0.01, description="minimum model prob minus implied prob")
    min_odds: float = 1.5
    max_odds: float = 5.0
    min_confidence: Literal["low", "medium", "high"] = "medium"
    require_robust_ev: bool = Field(
        True, description="EV must stay positive at the lower end of the uncertainty interval"
    )
    max_odds_age_hours: float = Field(96, description="older odds snapshots are flagged stale")

    # Model hyper-parameters
    dc_xi: float = Field(0.0019, description="Dixon-Coles time-decay per day (half-life ~1 year)")
    dc_window_days: int = 1100
    dc_ridge: float = 0.5
    elo_k: float = 20.0
    elo_home_advantage: float = 60.0
    uncertainty_replicates: int = 30
    min_team_matches: int = 10

    # Retraining
    retrain_every_days: int = 7

    # Log-linear pooling weights for the 'blend' model: sources = [Dixon-Coles, sharp market].
    # Fallback values estimated out-of-sample (2015-2026, 20 leagues); `sportpred evaluate`
    # re-estimates them and the newest estimate is used automatically.
    blend_weights: dict = Field(default_factory=lambda: {
        "1X2": {"weights": [-0.069, 1.099], "bias": [0.0, 0.038, 0.009]},
        "OU25": {"weights": [-0.043, 1.060], "bias": 0.010},
    })

    staking: StakingSettings = StakingSettings()


def load_model_settings(data_dir: Path | None = None) -> ModelSettings:
    """Defaults from config/default_settings.yaml, overridden by <data_dir>/user_settings.yaml."""
    merged: dict = {}
    if DEFAULT_SETTINGS_FILE.exists():
        merged.update(yaml.safe_load(DEFAULT_SETTINGS_FILE.read_text()) or {})
    data_dir = data_dir or get_settings().resolved_data_dir()
    user_file = data_dir / "user_settings.yaml"
    if user_file.exists():
        user = yaml.safe_load(user_file.read_text()) or {}
        staking = {**merged.get("staking", {}), **user.pop("staking", {})}
        merged.update(user)
        merged["staking"] = staking
    return ModelSettings(**merged)


def save_model_settings(ms: ModelSettings, data_dir: Path | None = None) -> Path:
    data_dir = data_dir or get_settings().resolved_data_dir()
    path = data_dir / "user_settings.yaml"
    path.write_text(yaml.safe_dump(ms.model_dump(), sort_keys=False))
    return path


@lru_cache
def get_settings() -> Settings:
    return Settings()
