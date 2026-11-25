"""SIMULATED data for automated tests only. Never shown as real data; rows are labelled 'simulated'.

Generates leagues where the true team strengths are known, so tests can check that models recover them.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd


def simulate_league(n_teams: int = 12, seasons: int = 3, start_year: int = 2020, league: str = "E0",
                    home_adv: float = 0.25, mu: float = 0.15, seed: int = 0, margin: float = 0.05) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    teams = [f"Team {chr(65 + i)}" for i in range(n_teams)]
    att = rng.normal(0, 0.3, n_teams)
    att -= att.mean()
    dfn = rng.normal(0, 0.25, n_teams)
    dfn -= dfn.mean()
    rows = []
    for s in range(seasons):
        year = start_year + s
        season = f"{year % 100:02d}{(year + 1) % 100:02d}"
        fixtures = [(h, a) for h in range(n_teams) for a in range(n_teams) if h != a]
        rng.shuffle(fixtures)
        day = dt.date(year, 8, 10)
        per_round = n_teams // 2
        for k, (h, a) in enumerate(fixtures):
            if k and k % per_round == 0:
                day += dt.timedelta(days=7)
            lam = np.exp(mu + home_adv + att[h] + dfn[a])
            nu = np.exp(mu + att[a] + dfn[h])
            hg, ag = rng.poisson(lam), rng.poisson(nu)
            rows.append({"league": league, "season": season, "date": day, "home_team": teams[h],
                         "away_team": teams[a], "fthg": hg, "ftag": ag, "lam": lam, "nu": nu})
    df = pd.DataFrame(rows)
    truth = {"teams": teams, "att": dict(zip(teams, att, strict=True)), "def": dict(zip(teams, dfn, strict=True)),
             "home_adv": home_adv, "mu": mu}
    return df, truth


def to_football_data_csv(df: pd.DataFrame, margin: float = 0.05, seed: int = 0) -> bytes:
    """Render simulated matches as a football-data.co.uk style CSV with odds (SIMULATED)."""
    from scipy.stats import poisson

    rng = np.random.default_rng(seed)
    out = []
    for r in df.itertuples():
        g = np.arange(11)
        m = np.outer(poisson.pmf(g, r.lam), poisson.pmf(g, r.nu))
        ph, pd_, pa = np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()
        po = sum(m[i, j] for i in range(11) for j in range(11) if i + j > 2)
        noise = rng.normal(0, 0.02, 3)
        p = np.clip(np.array([ph, pd_, pa]) + noise, 0.02, None)
        p /= p.sum()
        odds = np.round(1 / (p * (1 + margin)), 2)
        oo = np.round(1 / (np.array([po, 1 - po]) * (1 + margin)), 2)
        out.append({
            "Div": r.league, "Date": r.date.strftime("%d/%m/%Y"), "Time": "15:00", "HomeTeam": r.home_team,
            "AwayTeam": r.away_team, "FTHG": r.fthg, "FTAG": r.ftag,
            "FTR": "H" if r.fthg > r.ftag else ("D" if r.fthg == r.ftag else "A"),
            "HS": r.fthg * 3 + 5, "AS": r.ftag * 3 + 4, "HST": r.fthg + 2, "AST": r.ftag + 1,
            "B365H": odds[0], "B365D": odds[1], "B365A": odds[2],
            "PSH": round(odds[0] * 1.02, 2), "PSD": round(odds[1] * 1.02, 2), "PSA": round(odds[2] * 1.02, 2),
            "AvgH": odds[0], "AvgD": odds[1], "AvgA": odds[2], "MaxH": round(odds[0] * 1.05, 2),
            "MaxD": round(odds[1] * 1.05, 2), "MaxA": round(odds[2] * 1.05, 2),
            "Avg>2.5": oo[0], "Avg<2.5": oo[1], "B365>2.5": oo[0], "B365<2.5": oo[1],
            "PSCH": round(odds[0] * 1.02, 2), "PSCD": round(odds[1] * 1.02, 2), "PSCA": round(odds[2] * 1.02, 2),
            "AvgC>2.5": oo[0], "AvgC<2.5": oo[1],
        })
    return pd.DataFrame(out).to_csv(index=False).encode()
