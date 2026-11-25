"""Data quality checks. Each check returns issues with a severity, so the dashboard and the
prediction service can warn (or refuse) instead of silently using bad data."""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from sportpred.betting.odds import overround
from sportpred.db import DataUpdate, Match, Odds
from sportpred.leagues import LEAGUES, current_season_start_year, season_code

# Matches each team plays per season when the format is not a plain double round-robin.
MATCHES_PER_TEAM = {"SC0": 38, "SC1": 36, "SC2": 36, "SC3": 36}


def expected_season_matches(league: str, n_teams: int) -> int:
    per_team = MATCHES_PER_TEAM.get(league, 2 * (n_teams - 1))
    return n_teams * per_team // 2


@dataclass
class QualityIssue:
    severity: str  # info | warning | error
    check: str
    message: str
    league: str | None = None
    season: str | None = None
    count: int = 0
    examples: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def load_matches_frame(session: Session, leagues: list[str] | None = None) -> pd.DataFrame:
    q = select(Match)
    if leagues:
        q = q.where(Match.league.in_(leagues))
    rows = session.execute(q).scalars().all()
    if not rows:
        return pd.DataFrame()
    cols = ["id", "league", "season", "date", "home_team", "away_team", "status", "fthg", "ftag",
            "hs", "as_", "hst", "ast", "hxg", "axg", "data_label", "source"]
    return pd.DataFrame([{c: getattr(r, c) for c in cols} for r in rows])


def run_quality_checks(session: Session, leagues: list[str] | None = None,
                       today: dt.date | None = None) -> list[QualityIssue]:
    today = today or dt.date.today()
    issues: list[QualityIssue] = []
    df = load_matches_frame(session, leagues)
    if df.empty:
        return [QualityIssue("error", "no_data", "No matches in the database yet. Run a data update.")]

    fin = df[df["status"] == "finished"]

    # 1. matches whose date passed but no result
    stale = df[(df["status"].isin(["scheduled", "stale"])) & (pd.to_datetime(df["date"]) < pd.Timestamp(today))]
    if len(stale):
        issues.append(QualityIssue(
            "warning", "missing_results", "Past matches without a result (postponed, abandoned or not yet "
            "updated). They are excluded from training and predictions.", count=len(stale),
            examples=_examples(stale)))

    # 2. implausible values
    bad_goals = fin[(fin["fthg"] < 0) | (fin["ftag"] < 0) | (fin["fthg"] > 15) | (fin["ftag"] > 15)]
    if len(bad_goals):
        issues.append(QualityIssue("error", "implausible_goals", "Goal counts outside 0-15.",
                                   count=len(bad_goals), examples=_examples(bad_goals)))
    extreme = fin[((fin["fthg"] >= 9) | (fin["ftag"] >= 9)) & (fin["fthg"] <= 15) & (fin["ftag"] <= 15)]
    if len(extreme):
        issues.append(QualityIssue("info", "extreme_scores", "Very high scores (9+ goals for one team). Usually "
                                   "real (e.g. VVV 0-13 Ajax, 2020) but worth a glance.", count=len(extreme),
                                   examples=_examples(extreme)))
    sot = fin.dropna(subset=["hs", "hst"])
    bad_sot = sot[(sot["hst"] > sot["hs"]) | (sot["ast"] > sot["as_"])]
    if len(bad_sot):
        issues.append(QualityIssue("warning", "shots_inconsistent", "Shots on target exceed total shots.",
                                   count=len(bad_sot), examples=_examples(bad_sot)))

    # 3. a team playing twice on one day / possible duplicates within 2 days
    long = pd.concat([
        df[["league", "date", "home_team", "status"]].rename(columns={"home_team": "team"}),
        df[["league", "date", "away_team", "status"]].rename(columns={"away_team": "team"}),
    ])
    long = long[long["status"].isin(["finished", "scheduled"])]
    dupe = long[long.duplicated(["league", "date", "team"], keep=False)]
    if len(dupe):
        issues.append(QualityIssue("error", "team_twice_same_day", "A team appears in two matches on the "
                                   "same date (possible duplicate rows).", count=len(dupe) // 2,
                                   examples=", ".join(sorted(set(dupe["team"] + " " + dupe["date"].astype(str))))[:300]))
    pairs = df[df["status"].isin(["finished", "scheduled"])].sort_values("date")
    pairs = pairs.assign(d=pd.to_datetime(pairs["date"]))
    pairs["gap"] = pairs.groupby(["league", "home_team", "away_team"])["d"].diff().dt.days
    near = pairs[pairs["gap"].notna() & (pairs["gap"] <= 2)]
    if len(near):
        issues.append(QualityIssue("warning", "possible_duplicates", "Same fixture listed twice within 2 days.",
                                   count=len(near), examples=_examples(near)))

    # 4. season completeness
    cur = season_code(current_season_start_year(today))
    for (lg, season), g in fin.groupby(["league", "season"]):
        n = len(g)
        n_teams = len(set(g["home_team"]) | set(g["away_team"]))
        exp = expected_season_matches(lg, n_teams)
        if season != cur and n < 0.97 * exp:
            issues.append(QualityIssue("warning", "incomplete_season",
                                       f"Only {n} of ~{exp} matches present (season curtailed or file incomplete).",
                                       league=lg, season=season, count=int(exp - n)))

    # 5. teams that appear only a handful of times in a full season (possible name typos)
    for (lg, season), g in fin.groupby(["league", "season"]):
        if season == cur:
            continue
        team_counts = pd.concat([g["home_team"], g["away_team"]]).value_counts()
        rare = team_counts[team_counts <= 3]
        if len(rare):
            issues.append(QualityIssue("warning", "rare_team_name", "Team name appears ≤3 times in a season "
                                       "(possible spelling variant).", league=lg, season=season,
                                       count=len(rare), examples=", ".join(rare.index[:5])))

    # 6. odds coverage and plausibility
    issues += _odds_checks(session, df)

    # 7. freshness
    issues += freshness_issues(session, today)

    # 8. labels
    labels = df["data_label"].value_counts().to_dict()
    non_live = {k: v for k, v in labels.items() if k != "live"}
    if non_live:
        issues.append(QualityIssue("warning", "non_live_data", f"Database contains non-live data: {non_live}. "
                                   "Sample/simulated rows must never be used to judge real performance.",
                                   count=sum(non_live.values())))
    if not issues:
        issues.append(QualityIssue("info", "all_clear", "No data quality problems found."))
    return issues


def _odds_checks(session: Session, matches: pd.DataFrame) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    odds = pd.read_sql(
        select(Odds.match_id, Odds.bookmaker, Odds.market, Odds.is_closing, Odds.price_home, Odds.price_draw,
               Odds.price_away, Odds.price_over, Odds.price_under)
        .where(Odds.market.in_(["1X2", "OU"])),
        session.bind,
    )
    if odds.empty:
        issues.append(QualityIssue("warning", "no_odds", "No bookmaker odds stored: value bets cannot be assessed."))
        return issues
    x12 = odds[(odds["market"] == "1X2") & ~odds["bookmaker"].isin(["market_max", "market_avg"])].copy()
    if not x12.empty:
        x12["overround"] = overround([x12["price_home"], x12["price_draw"], x12["price_away"]])
        weird = x12[(x12["overround"] < 0.97) | (x12["overround"] > 1.30)]
        if len(weird):
            issues.append(QualityIssue(
                "warning", "implausible_margin", "1X2 prices with bookmaker margin below -3% or above 30% "
                "(likely data entry errors). These prices are ignored for value calculations.", count=len(weird)))
    fin = matches[matches["status"] == "finished"]
    with_odds = set(odds.loc[odds["market"] == "1X2", "match_id"])
    by_season = fin.assign(has=fin["id"].isin(with_odds)).groupby(["league", "season"])["has"].mean()
    low = by_season[by_season < 0.95]
    for (lg, season), share in low.items():
        issues.append(QualityIssue("warning", "odds_coverage", f"Only {share:.0%} of matches have 1X2 odds.",
                                   league=lg, season=season))
    return issues


def freshness_issues(session: Session, today: dt.date | None = None) -> list[QualityIssue]:
    today = today or dt.date.today()
    issues: list[QualityIssue] = []
    last = session.execute(
        select(DataUpdate.resource, func.max(DataUpdate.finished_at))
        .where(DataUpdate.status.in_(["success", "unchanged"]))
        .group_by(DataUpdate.resource)
    ).all()
    if not last:
        issues.append(QualityIssue("warning", "never_updated", "No successful data update recorded."))
        return issues
    latest = max(t for _, t in last if t is not None)
    age_days = (dt.datetime.combine(today, dt.time()) - latest).days
    if age_days > 4:
        issues.append(QualityIssue("warning", "stale_data", f"Last successful data update was {age_days} days ago."))
    fixtures = [t for r, t in last if r == "fixtures.csv"]
    if not fixtures:
        issues.append(QualityIssue("info", "no_fixtures", "Upcoming fixtures have not been downloaded yet."))
    failed = session.execute(
        select(func.count()).select_from(DataUpdate).where(DataUpdate.status == "failed")
    ).scalar_one()
    if failed:
        issues.append(QualityIssue("warning", "failed_updates", f"{failed} failed download/import attempt(s) "
                                   "in the update history.", count=failed))
    return issues


def _examples(df: pd.DataFrame, n: int = 3) -> str:
    rows = df.head(n)
    return "; ".join(f"{r.league} {r.date} {r.home_team} v {r.away_team}" for r in rows.itertuples())


def coverage_table(session: Session) -> pd.DataFrame:
    """Matches and odds coverage per league/season for the Data Management page."""
    m = pd.read_sql(
        select(Match.id, Match.league, Match.season, Match.status, Match.hst, Match.hxg), session.bind
    )
    if m.empty:
        return pd.DataFrame()
    o = pd.read_sql(select(Odds.match_id, Odds.bookmaker, Odds.market, Odds.is_closing), session.bind)
    has = lambda book, market, closing: set(  # noqa: E731
        o.loc[(o.bookmaker == book) & (o.market == market) & (o.is_closing == closing), "match_id"])
    sets = {
        "pinnacle_pre": has("pinnacle", "1X2", False),
        "pinnacle_close": has("pinnacle", "1X2", True),
        "exchange_close": has("betfair_exchange", "1X2", True),
        "avg_close": has("market_avg", "1X2", True),
        "ou_odds": set(o.loc[o.market == "OU", "match_id"]),
    }
    g = m.groupby(["league", "season"])
    out = g.agg(matches=("id", "size"), finished=("status", lambda s: (s == "finished").sum()),
                shots=("hst", lambda s: s.notna().mean()), xg=("hxg", lambda s: s.notna().mean()))
    for name, ids in sets.items():
        out[name] = g["id"].apply(lambda s, ids=ids: s.isin(ids).mean())
    out = out.reset_index()
    out["league_name"] = out["league"].map(lambda c: LEAGUES[c].label if c in LEAGUES else c)
    return out.sort_values(["league", "season"])
