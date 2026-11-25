"""football-data.co.uk provider: download and parse results + bookmaker odds CSV files.

Provider facts (from https://www.football-data.co.uk/notes.txt, checked 2026-11-25):

* One CSV per league per season at ``/mmz4281/{season}/{league}.csv`` (e.g. ``2324/E0.csv``).
* Upcoming fixtures with odds at ``/fixtures.csv`` (updated around Tuesday and Friday).
* Pre-closing odds are collected on Friday afternoons for weekend games and Tuesday afternoons
  for midweek games. Closing odds use the same columns with an extra ``C`` (``B365CH``...)
  and exist from 2019/20 (Pinnacle closing 1X2 from 2012/13).
* Pinnacle columns disappear from 2026/27 files; Betfair Exchange (``BFE``) is present from
  2024/25 and becomes the sharpest reference price.

The parser is defensive: column sets change between seasons, older files use other
bookmakers and some rows are malformed. Everything we skip is counted and reported.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import logging
import time
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from sportpred.config import Settings, get_settings
from sportpred.leagues import LEAGUES, current_season_start_year, season_code

log = logging.getLogger(__name__)

SOURCE = "football-data.co.uk"

# 1X2 bookmaker column prefixes -> our bookmaker key. Closing columns insert "C" after prefix.
BOOKMAKERS_1X2: dict[str, str] = {
    "B365": "bet365",
    "BW": "bwin",
    "IW": "interwetten",
    "PS": "pinnacle",
    "P": "pinnacle",  # older alias (PH/PD/PA)
    "WH": "williamhill",
    "VC": "betvictor",
    "BV": "betvictor",
    "1XB": "1xbet",
    "BF": "betfair_sportsbook",
    "BFD": "betfred",
    "BMGM": "betmgm",
    "CL": "coral",
    "LB": "ladbrokes",
    "PP": "paddypower",
    "SK": "skybet",
    "SKB": "skybet",
    "BFE": "betfair_exchange",
    "Max": "market_max",
    "Avg": "market_avg",
    "BbMx": "market_max",
    "BbAv": "market_avg",
}

# Over/Under 2.5 goals prefixes ("{p}>2.5", "{p}<2.5"); closing: "{p}C>2.5".
BOOKMAKERS_OU: dict[str, str] = {
    "B365": "bet365",
    "P": "pinnacle",
    "Max": "market_max",
    "Avg": "market_avg",
    "BFE": "betfair_exchange",
    "BbMx": "market_max",
    "BbAv": "market_avg",
}

# Asian handicap prefixes ("{p}AHH", "{p}AHA"); closing: "{p}CAHH".
BOOKMAKERS_AH: dict[str, str] = {
    "B365": "bet365",
    "P": "pinnacle",
    "Max": "market_max",
    "Avg": "market_avg",
    "BFE": "betfair_exchange",
    "BbMx": "market_max",
    "BbAv": "market_avg",
}

# Aggregates are not real bookmakers: you cannot bet "market_max" at one place.
AGGREGATE_BOOKMAKERS = {"market_max", "market_avg"}

STAT_COLUMNS = {
    "FTHG": "fthg", "FTAG": "ftag", "FTR": "ftr", "HTHG": "hthg", "HTAG": "htag",
    "HS": "hs", "AS": "as_", "HST": "hst", "AST": "ast", "HC": "hc", "AC": "ac",
    "HF": "hf", "AF": "af", "HY": "hy", "AY": "ay", "HR": "hr", "AR": "ar",
    "HxG": "hxg", "AxG": "axg", "Referee": "referee", "Time": "kickoff",
}
INT_STATS = ["fthg", "ftag", "hthg", "htag", "hs", "as_", "hst", "ast", "hc", "ac",
             "hf", "af", "hy", "ay", "hr", "ar"]


@dataclass
class ParsedFile:
    matches: pd.DataFrame
    odds: pd.DataFrame
    rows_in_file: int
    rows_skipped: int
    warnings: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------- parsing helpers

def _read_csv_bytes(content: bytes) -> tuple[pd.DataFrame, int, list[str]]:
    warnings: list[str] = []
    text = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = content.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:  # pragma: no cover - latin-1 always decodes
        raise ValueError("Could not decode CSV")
    text = text.lstrip("﻿")
    raw_lines = [ln for ln in text.splitlines() if ln.strip(" ,\t")]
    n_data_lines = max(len(raw_lines) - 1, 0)
    df = pd.read_csv(io.StringIO(text), on_bad_lines="skip", low_memory=False)
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]
    df = df.dropna(how="all")
    skipped = n_data_lines - len(df)
    if skipped > 0:
        warnings.append(f"{skipped} malformed or empty line(s) skipped")
    return df, max(skipped, 0), warnings


def _parse_dates(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    out = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    short = out.isna()
    if short.any():
        out[short] = pd.to_datetime(s[short], format="%d/%m/%y", errors="coerce")
    return out.dt.date


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _valid_price(s: pd.Series) -> pd.Series:
    """Decimal odds must be > 1. Anything else is treated as missing."""
    return s.where(s > 1.0)


def approx_preclosing_collected_at(match_date: dt.date) -> dt.datetime:
    """Provider collects pre-closing odds Friday afternoon (weekend) / Tuesday afternoon (midweek)."""
    wd = match_date.weekday()  # Mon=0
    if wd in (1, 2, 3):  # Tue-Thu -> Tuesday of that week
        back = wd - 1
    else:  # Fri-Mon -> preceding Friday
        back = (wd - 4) % 7
    d = match_date - dt.timedelta(days=back)
    return dt.datetime.combine(d, dt.time(15, 0))


def parse_football_data_csv(
    content: bytes,
    league: str | None = None,
    season: str | None = None,
    source_file: str | None = None,
    is_fixture_file: bool = False,
    snapshot_time: dt.datetime | None = None,
) -> ParsedFile:
    """Parse a football-data.co.uk CSV (results file or fixtures file) into normalised frames.

    Returns ``matches`` (one row per match, ``match_key`` column) and ``odds`` (long format:
    one row per match/bookmaker/market/line/closing flag).
    """
    df, skipped, warnings = _read_csv_bytes(content)
    rows_in_file = len(df) + skipped
    if df.empty:
        return ParsedFile(pd.DataFrame(), pd.DataFrame(), rows_in_file, skipped, warnings)

    if "HomeTeam" not in df.columns and "Home" in df.columns:  # 'extra leagues' format
        df = df.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam", "HG": "FTHG", "AG": "FTAG",
                                "Res": "FTR"})
    required = {"Date", "HomeTeam", "AwayTeam"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    m = pd.DataFrame(index=df.index)
    m["league"] = df["Div"].astype(str).str.strip() if "Div" in df.columns else league
    if league is not None:
        m["league"] = m["league"].fillna(league)
    m["date"] = _parse_dates(df["Date"])
    m["home_team"] = df["HomeTeam"].astype(str).str.strip()
    m["away_team"] = df["AwayTeam"].astype(str).str.strip()
    for src, dst in STAT_COLUMNS.items():
        if src in df.columns:
            m[dst] = df[src]
        else:
            m[dst] = np.nan
    for c in INT_STATS:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m["hxg"] = pd.to_numeric(m["hxg"], errors="coerce")
    m["axg"] = pd.to_numeric(m["axg"], errors="coerce")
    m["kickoff"] = m["kickoff"].where(m["kickoff"].notna(), None)
    m["referee"] = m["referee"].where(m["referee"].notna(), None)

    bad_rows = m["date"].isna() | m["home_team"].isin(["", "nan"]) | m["away_team"].isin(["", "nan"])
    if bad_rows.any():
        warnings.append(f"{int(bad_rows.sum())} row(s) without a valid date/team dropped")
        m = m.loc[~bad_rows]
        df = df.loc[m.index]

    if season is None:
        season_series = m["date"].map(lambda d: season_code(current_season_start_year(d)))
    else:
        season_series = pd.Series(season, index=m.index)
    m["season"] = season_series

    has_result = m["fthg"].notna() & m["ftag"].notna()
    m["status"] = np.where(has_result & (not is_fixture_file), "finished", "scheduled")
    # derive FTR when missing; validate when present
    derived = np.select([m["fthg"] > m["ftag"], m["fthg"] < m["ftag"]], ["H", "A"], default="D")
    provided = m["ftr"].astype("string").str.strip()
    mismatch = has_result & provided.notna() & (provided != pd.Series(derived, index=m.index))
    if mismatch.any():
        warnings.append(f"{int(mismatch.sum())} row(s) where FTR disagrees with goals; goals used")
    m["ftr"] = np.where(has_result, derived, None)

    m["match_key"] = (
        m["league"].astype(str) + "|" + m["date"].astype(str) + "|" + m["home_team"] + "|" + m["away_team"]
    )
    dup = m["match_key"].duplicated(keep="last")
    if dup.any():
        warnings.append(f"{int(dup.sum())} duplicate match row(s) in file; kept the last one")
        m = m.loc[~dup]
        df = df.loc[m.index]
    m["source"] = SOURCE
    m["source_file"] = source_file

    odds = _extract_odds(df, m, snapshot_time=snapshot_time, is_fixture_file=is_fixture_file)
    return ParsedFile(m.reset_index(drop=True), odds, rows_in_file, skipped, warnings)


def _extract_odds(
    df: pd.DataFrame, m: pd.DataFrame, snapshot_time: dt.datetime | None, is_fixture_file: bool
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    keys = m["match_key"]
    pre_collected = (
        pd.Series(snapshot_time, index=m.index)
        if snapshot_time is not None
        else m["date"].map(approx_preclosing_collected_at)
    )

    def closing_time(row_dates: pd.Series, kickoff: pd.Series) -> pd.Series:
        def f(d, k):
            try:
                hh, mm = str(k).split(":")[:2]
                return dt.datetime.combine(d, dt.time(int(hh), int(mm)))
            except Exception:
                return dt.datetime.combine(d, dt.time(15, 0))
        return pd.Series([f(d, k) for d, k in zip(row_dates, kickoff, strict=False)], index=row_dates.index)

    close_collected = closing_time(m["date"], m["kickoff"])

    def add(book: str, market: str, closing: bool, line: pd.Series | float, **prices: pd.Series):
        part = pd.DataFrame({"match_key": keys, "bookmaker": book, "market": market,
                             "is_closing": closing})
        part["line"] = line if isinstance(line, pd.Series) else float(line)
        for name in ("price_home", "price_draw", "price_away", "price_over", "price_under"):
            part[name] = _valid_price(prices[name]) if name in prices else np.nan
        part["collected_at"] = close_collected if closing else pre_collected
        price_cols = [f"price_{p}" for p in ("home", "draw", "away", "over", "under")]
        complete = part[price_cols].notna().sum(axis=1) >= (3 if market == "1X2" else 2)
        if market != "1X2":
            complete &= part["line"].notna()
        part = part.loc[complete]
        if not part.empty:
            frames.append(part)

    seen: set[tuple[str, str, bool]] = set()
    for prefix, book in BOOKMAKERS_1X2.items():
        for closing in (False, True):
            p = f"{prefix}C" if closing else prefix
            cols = [f"{p}H", f"{p}D", f"{p}A"]
            if all(c in df.columns for c in cols) and (book, "1X2", closing) not in seen:
                add(book, "1X2", closing, 0.0, price_home=_num(df, cols[0]), price_draw=_num(df, cols[1]),
                    price_away=_num(df, cols[2]))
                seen.add((book, "1X2", closing))

    for prefix, book in BOOKMAKERS_OU.items():
        for closing in (False, True):
            p = f"{prefix}C" if closing else prefix
            o, u = f"{p}>2.5", f"{p}<2.5"
            if o in df.columns and u in df.columns and (book, "OU", closing) not in seen:
                add(book, "OU", closing, 2.5, price_over=_num(df, o), price_under=_num(df, u))
                seen.add((book, "OU", closing))

    for prefix, book in BOOKMAKERS_AH.items():
        for closing in (False, True):
            p = f"{prefix}C" if closing else prefix
            h, a = f"{p}AHH", f"{p}AHA"
            if not (h in df.columns and a in df.columns) or (book, "AH", closing) in seen:
                continue
            line_candidates = (
                [f"{prefix}CAH", "AHCh"] if closing else [f"{prefix}AH", "AHh", "BbAHh"]
            )
            line = next((_num(df, c) for c in line_candidates if c in df.columns), None)
            if line is None:
                continue
            add(book, "AH", closing, line, price_home=_num(df, h), price_away=_num(df, a))
            seen.add((book, "AH", closing))

    if not frames:
        return pd.DataFrame(columns=["match_key", "bookmaker", "market", "line", "is_closing",
                                     "price_home", "price_draw", "price_away", "price_over",
                                     "price_under", "collected_at"])
    out = pd.concat(frames, ignore_index=True)
    # The same bookmaker can appear under two prefixes in one file (e.g. VC and BV); keep first.
    out = out.drop_duplicates(["match_key", "bookmaker", "market", "line", "is_closing"], keep="first")
    out["source"] = SOURCE
    return out


# ----------------------------------------------------------------------------- downloading

@dataclass
class DownloadResult:
    resource: str
    content: bytes | None
    content_hash: str | None
    last_modified: dt.datetime | None
    from_cache: bool
    error: str | None = None


class FootballDataClient:
    """Polite HTTP client with a local file cache in ``<data_dir>/raw/football-data``."""

    def __init__(self, settings: Settings | None = None, client: httpx.Client | None = None):
        self.settings = settings or get_settings()
        self.cache_dir = self.settings.resolved_data_dir() / "raw" / "football-data"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = client
        self._last_request = 0.0

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.settings.http_timeout_seconds,
                follow_redirects=True,
                headers={"User-Agent": self.settings.http_user_agent},
            )
        return self._client

    def _cache_path(self, resource: str) -> Path:
        return self.cache_dir / resource

    def season_url(self, league: str, season: str) -> str:
        return f"{self.settings.football_data_base_url}/mmz4281/{season}/{league}.csv"

    def fixtures_url(self) -> str:
        return f"{self.settings.football_data_base_url}/fixtures.csv"

    def fetch(self, url: str, resource: str, use_cache: bool) -> DownloadResult:
        path = self._cache_path(resource)
        if use_cache and path.exists():
            content = path.read_bytes()
            return DownloadResult(resource, content, hashlib.sha256(content).hexdigest(), None, True)
        wait = self.settings.request_pause_seconds - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = self._http().get(url)
            self._last_request = time.monotonic()
        except httpx.HTTPError as exc:
            return DownloadResult(resource, None, None, None, False, error=f"network error: {exc}")
        if resp.status_code == 404:
            return DownloadResult(resource, None, None, None, False, error="not found (404)")
        if resp.status_code != 200:
            return DownloadResult(resource, None, None, None, False, error=f"HTTP {resp.status_code}")
        content = resp.content
        if b"," not in content[:200]:
            return DownloadResult(resource, None, None, None, False, error="response is not a CSV file")
        last_modified = None
        if "last-modified" in resp.headers:
            try:
                last_modified = parsedate_to_datetime(resp.headers["last-modified"]).astimezone(
                    dt.UTC).replace(tzinfo=None)
            except (TypeError, ValueError):
                last_modified = None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return DownloadResult(resource, content, hashlib.sha256(content).hexdigest(), last_modified, False)

    def download_season(self, league: str, season: str, force: bool = False) -> DownloadResult:
        """Past seasons are cached forever (they no longer change); the current season is
        always re-downloaded unless cached and not forced... we re-download it every time."""
        is_current = season == season_code(current_season_start_year())
        use_cache = (not force) and (not is_current)
        return self.fetch(self.season_url(league, season), f"{season}/{league}.csv", use_cache=use_cache)

    def download_fixtures(self) -> DownloadResult:
        return self.fetch(self.fixtures_url(), "fixtures.csv", use_cache=False)


def known_league(code: str) -> bool:
    return code in LEAGUES
