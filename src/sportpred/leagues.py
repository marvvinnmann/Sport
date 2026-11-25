"""League catalogue for football-data.co.uk 'main' leagues (code = file name on the site)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class League:
    code: str
    country: str
    name: str
    tier: int
    sport: str = "football"

    @property
    def label(self) -> str:
        return f"{self.country} — {self.name}"


LEAGUES: dict[str, League] = {
    lg.code: lg
    for lg in [
        League("E0", "England", "Premier League", 1),
        League("E1", "England", "Championship", 2),
        League("E2", "England", "League One", 3),
        League("E3", "England", "League Two", 4),
        League("EC", "England", "National League", 5),
        League("SC0", "Scotland", "Premiership", 1),
        League("SC1", "Scotland", "Championship", 2),
        League("SC2", "Scotland", "League One", 3),
        League("SC3", "Scotland", "League Two", 4),
        League("D1", "Germany", "Bundesliga", 1),
        League("D2", "Germany", "2. Bundesliga", 2),
        League("I1", "Italy", "Serie A", 1),
        League("I2", "Italy", "Serie B", 2),
        League("SP1", "Spain", "La Liga", 1),
        League("SP2", "Spain", "Segunda Division", 2),
        League("F1", "France", "Ligue 1", 1),
        League("F2", "France", "Ligue 2", 2),
        League("N1", "Netherlands", "Eredivisie", 1),
        League("B1", "Belgium", "Pro League", 1),
        League("P1", "Portugal", "Primeira Liga", 1),
        League("T1", "Turkey", "Super Lig", 1),
        League("G1", "Greece", "Super League", 1),
    ]
}


def league_label(code: str) -> str:
    lg = LEAGUES.get(code)
    return lg.label if lg else code


def season_code(start_year: int) -> str:
    """2023 -> '2324' (football-data.co.uk folder name)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(code: str) -> str:
    """'2324' -> '2023/24'."""
    start = int(code[:2])
    century = 1900 if start >= 90 else 2000
    return f"{century + start}/{code[2:]}"


def season_start_year(code: str) -> int:
    start = int(code[:2])
    return (1900 if start >= 90 else 2000) + start


def current_season_start_year(today=None) -> int:
    """European football seasons start in July/August."""
    import datetime as _dt

    today = today or _dt.date.today()
    return today.year if today.month >= 7 else today.year - 1
