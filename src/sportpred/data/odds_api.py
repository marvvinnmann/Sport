"""Optional live odds from The Odds API (https://the-odds-api.com, v4).

Status: implemented and unit-tested with recorded-format responses; it is only *connected* when you
set ``SPORTPRED_ODDS_API_KEY``. Without a key the app says "not configured" and uses the free
football-data.co.uk odds snapshot instead.

Cost model (check the provider's pricing page): each request costs 1 credit per market per region.
We request ``h2h,totals`` for the configured regions, i.e. 2 × regions credits per league per refresh.
The free plan (500 credits/month at the time of writing) is enough for roughly one refresh per
match day for a few leagues.
"""

from __future__ import annotations

import datetime as dt
import difflib
import re
from dataclasses import dataclass, field

import httpx
import pandas as pd

from sportpred.config import Settings, get_settings

SOURCE = "the-odds-api"

SPORT_KEYS = {
    "E0": "soccer_epl", "E1": "soccer_efl_champ", "E2": "soccer_england_league1",
    "E3": "soccer_england_league2", "SC0": "soccer_spl", "D1": "soccer_germany_bundesliga",
    "D2": "soccer_germany_bundesliga2", "I1": "soccer_italy_serie_a", "I2": "soccer_italy_serie_b",
    "SP1": "soccer_spain_la_liga", "SP2": "soccer_spain_segunda_division", "F1": "soccer_france_ligue_one",
    "F2": "soccer_france_ligue_two", "N1": "soccer_netherlands_eredivisie", "B1": "soccer_belgium_first_div",
    "P1": "soccer_portugal_primeira_liga", "T1": "soccer_turkey_super_league", "G1": "soccer_greece_super_league",
}

BOOKMAKER_MAP = {"betfair_ex_uk": "betfair_exchange", "betfair_ex_eu": "betfair_exchange",
                 "williamhill": "williamhill", "pinnacle": "pinnacle", "bet365": "bet365"}

# provider name -> football-data.co.uk name (extend as needed; unmatched names are reported)
TEAM_ALIASES = {
    "manchester city": "Man City", "manchester united": "Man United", "tottenham hotspur": "Tottenham",
    "wolverhampton wanderers": "Wolves", "brighton and hove albion": "Brighton", "nottingham forest": "Nott'm Forest",
    "newcastle united": "Newcastle", "west ham united": "West Ham", "leicester city": "Leicester",
    "leeds united": "Leeds", "sheffield united": "Sheffield United", "sheffield wednesday": "Sheffield Weds",
    "afc bournemouth": "Bournemouth", "ipswich town": "Ipswich", "luton town": "Luton",
    "queens park rangers": "QPR", "west bromwich albion": "West Brom", "bayern munich": "Bayern Munich",
    "borussia dortmund": "Dortmund", "borussia monchengladbach": "M'gladbach", "bayer leverkusen": "Leverkusen",
    "eintracht frankfurt": "Ein Frankfurt", "inter milan": "Inter", "ac milan": "Milan", "as roma": "Roma",
    "atletico madrid": "Ath Madrid", "athletic bilbao": "Ath Bilbao", "real sociedad": "Sociedad",
    "paris saint germain": "Paris SG", "olympique marseille": "Marseille", "olympique lyonnais": "Lyon",
}


def _norm(name: str) -> str:
    s = name.lower().replace("&", "and")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\b(fc|afc|cf|sc|ac)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def match_team(provider_name: str, candidates: list[str]) -> str | None:
    low = provider_name.lower().strip()
    if low in TEAM_ALIASES and TEAM_ALIASES[low] in candidates:
        return TEAM_ALIASES[low]
    norm_map = {_norm(c): c for c in candidates}
    n = _norm(provider_name)
    if n in norm_map:
        return norm_map[n]
    for k, v in norm_map.items():
        if k and (k in n or n in k):
            return v
    best = difflib.get_close_matches(n, list(norm_map), n=1, cutoff=0.75)
    return norm_map[best[0]] if best else None


@dataclass
class OddsApiResult:
    league: str
    events: int = 0
    matched: int = 0
    unmatched: list[str] = field(default_factory=list)
    odds: pd.DataFrame | None = None
    credits_remaining: str | None = None
    error: str | None = None


class OddsApiClient:
    def __init__(self, settings: Settings | None = None, client: httpx.Client | None = None):
        self.settings = settings or get_settings()
        self._client = client

    @property
    def configured(self) -> bool:
        return self.settings.odds_api_key is not None

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.settings.http_timeout_seconds,
                                        headers={"User-Agent": self.settings.http_user_agent})
        return self._client

    def fetch_league(self, league: str, fixtures: pd.DataFrame) -> OddsApiResult:
        """Fetch current odds for one league and match events to our scheduled fixtures."""
        res = OddsApiResult(league)
        if not self.configured:
            res.error = "not configured (no SPORTPRED_ODDS_API_KEY)"
            return res
        sport = SPORT_KEYS.get(league)
        if not sport:
            res.error = f"league {league} not covered by The Odds API mapping"
            return res
        url = f"{self.settings.odds_api_base_url}/v4/sports/{sport}/odds"
        params = {"apiKey": self.settings.odds_api_key.get_secret_value(), "regions": self.settings.odds_api_regions,
                  "markets": "h2h,totals", "oddsFormat": "decimal", "dateFormat": "iso"}
        try:
            r = self._http().get(url, params=params)
        except httpx.HTTPError as exc:
            res.error = f"network error: {exc}"
            return res
        res.credits_remaining = r.headers.get("x-requests-remaining")
        if r.status_code == 401:
            res.error = "invalid API key (HTTP 401)"
            return res
        if r.status_code == 429:
            res.error = "rate limit or credits exhausted (HTTP 429)"
            return res
        if r.status_code != 200:
            res.error = f"HTTP {r.status_code}"
            return res
        return self.parse_events(league, r.json(), fixtures, res)

    @staticmethod
    def parse_events(league: str, events: list[dict], fixtures: pd.DataFrame, res: OddsApiResult | None = None) -> OddsApiResult:
        res = res or OddsApiResult(league)
        res.events = len(events)
        fx = fixtures[fixtures["league"] == league]
        teams = sorted(set(fx["home_team"]) | set(fx["away_team"]))
        rows = []
        for ev in events:
            home = match_team(ev.get("home_team", ""), teams)
            away = match_team(ev.get("away_team", ""), teams)
            when = pd.Timestamp(ev.get("commence_time")).tz_convert(None) if ev.get("commence_time") else None
            cand = fx[(fx["home_team"] == home) & (fx["away_team"] == away)]
            if when is not None and len(cand):
                cand = cand[(pd.to_datetime(cand["date"]) - when.normalize()).abs() <= pd.Timedelta(days=1)]
            if home is None or away is None or cand.empty:
                res.unmatched.append(f"{ev.get('home_team')} v {ev.get('away_team')}")
                continue
            res.matched += 1
            match_key = cand.iloc[0]["match_key"]
            for bm in ev.get("bookmakers", []):
                book = BOOKMAKER_MAP.get(bm["key"], bm["key"])
                collected = pd.Timestamp(bm.get("last_update")).tz_convert(None).to_pydatetime() if bm.get(
                    "last_update") else dt.datetime.utcnow()
                for mk in bm.get("markets", []):
                    if mk["key"] == "h2h":
                        prices = {o["name"]: o["price"] for o in mk["outcomes"]}
                        ph, pa, pd_ = prices.get(ev["home_team"]), prices.get(ev["away_team"]), prices.get("Draw")
                        if ph and pa and pd_:
                            rows.append({"match_key": match_key, "bookmaker": book, "market": "1X2", "line": 0.0,
                                         "is_closing": False, "price_home": ph, "price_draw": pd_, "price_away": pa,
                                         "collected_at": collected})
                    elif mk["key"] == "totals":
                        over = [o for o in mk["outcomes"] if o["name"] == "Over" and abs(o.get("point", 0) - 2.5) < 1e-9]
                        under = [o for o in mk["outcomes"] if o["name"] == "Under" and abs(o.get("point", 0) - 2.5) < 1e-9]
                        if over and under:
                            rows.append({"match_key": match_key, "bookmaker": book, "market": "OU", "line": 2.5,
                                         "is_closing": False, "price_over": over[0]["price"],
                                         "price_under": under[0]["price"], "collected_at": collected})
        res.odds = pd.DataFrame(rows)
        if not res.odds.empty:
            res.odds["source"] = SOURCE
        return res


def refresh_live_odds(leagues: list[str], days_ahead: int = 4) -> list[OddsApiResult]:
    """Fetch The Odds API prices for leagues with fixtures in the next ``days_ahead`` days."""
    from sportpred.data.ingest import upsert_matches_and_odds
    from sportpred.data.repository import load_matches
    from sportpred.db import DataUpdate, session_scope, utcnow

    client = OddsApiClient()
    today = dt.date.today()
    fx = load_matches(leagues=leagues, statuses=["scheduled"], from_date=today,
                      to_date=today + dt.timedelta(days=days_ahead))
    out = []
    for league in sorted(fx["league"].unique()) if not fx.empty else []:
        started = utcnow()
        res = client.fetch_league(league, fx)
        out.append(res)
        with session_scope() as s:
            if res.odds is not None and not res.odds.empty:
                m = fx[fx["match_key"].isin(res.odds["match_key"])].copy()
                m["date"] = m["date"].dt.date
                m["as_"] = m.get("as_")
                upsert_matches_and_odds(s, m, res.odds, replace_odds_source=SOURCE)
            s.add(DataUpdate(source=SOURCE, resource=f"odds/{league}", started_at=started, finished_at=utcnow(),
                             status="failed" if res.error else "success", rows_inserted=0,
                             rows_updated=res.matched,
                             message=res.error or f"{res.events} events, {res.matched} matched, "
                             f"unmatched: {', '.join(res.unmatched) or 'none'}; credits left {res.credits_remaining}"))
    return out
