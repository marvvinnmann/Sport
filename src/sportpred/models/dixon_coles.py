"""Dixon-Coles (1997) bivariate-Poisson-style model with time decay.

Each team has an attack and a defence strength. Expected goals:

    home goals ~ Poisson(λ),  log λ = μ + home_adv + attack[home] + defence[away]
    away goals ~ Poisson(ν),  log ν = μ +            attack[away] + defence[home]

A correction factor τ(ρ) adjusts the four low scores (0-0, 1-0, 0-1, 1-1), which plain
Poisson models get wrong. Older matches count less: weight = exp(−ξ · days_ago).

Why this model: it is the standard, well-studied baseline for football; it needs only scores,
gives a full score distribution (so 1X2, over/under, BTTS and handicaps come from one model)
and its parameters are easy to explain ("Arsenal attack +0.35 = 42% more goals than average").

Additions for robustness:
* ridge penalty shrinks teams with few matches towards a prior (avoids small-sample overreaction);
* newly promoted teams are shrunk towards the strength of the weakest established teams
  instead of towards "league average" (promoted teams are usually below average);
* optional Bayesian-bootstrap refits give an uncertainty interval for every probability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from sportpred.models.base import MatchModel, finished_only

MAX_GOALS = 10


def score_matrix(lam: np.ndarray, nu: np.ndarray, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """Probability of every scoreline: array (n, max_goals+1, max_goals+1) [home goals, away goals]."""
    lam, nu = np.atleast_1d(lam), np.atleast_1d(nu)
    g = np.arange(max_goals + 1)
    ph = poisson.pmf(g[None, :], lam[:, None])
    pa = poisson.pmf(g[None, :], nu[:, None])
    m = ph[:, :, None] * pa[:, None, :]
    m[:, 0, 0] *= np.maximum(1 - lam * nu * rho, 1e-10)
    m[:, 0, 1] *= np.maximum(1 + lam * rho, 1e-10)
    m[:, 1, 0] *= np.maximum(1 + nu * rho, 1e-10)
    m[:, 1, 1] *= np.maximum(1 - rho, 1e-10)
    m /= m.sum(axis=(1, 2), keepdims=True)
    return m


def market_probs_from_matrix(m: np.ndarray, ou_line: float = 2.5) -> dict[str, np.ndarray]:
    n, k, _ = m.shape
    hg, ag = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    home = (hg > ag)[None]
    draw = (hg == ag)[None]
    over = ((hg + ag) > ou_line)[None]
    btts = ((hg > 0) & (ag > 0))[None]
    return {
        "p_H": (m * home).sum(axis=(1, 2)),
        "p_D": (m * draw).sum(axis=(1, 2)),
        "p_A": (m * ~(home | draw)).sum(axis=(1, 2)),
        "p_OVER": (m * over).sum(axis=(1, 2)),
        "p_UNDER": (m * ~over).sum(axis=(1, 2)),
        "p_BTTS": (m * btts).sum(axis=(1, 2)),
    }


class DixonColes(MatchModel):
    name = "dixon_coles"
    markets = {"1X2", "OU25"}

    def __init__(self, xi: float = 0.0019, window_days: int = 1100, ridge: float = 0.5,
                 newcomer_prior: bool = True, rho_bounds: tuple[float, float] = (-0.2, 0.2)):
        self.xi = xi
        self.window_days = window_days
        self.ridge = ridge
        self.newcomer_prior = newcomer_prior
        self.rho_bounds = rho_bounds
        self._fitted = False

    # ------------------------------------------------------------------ fitting
    def fit(self, history: pd.DataFrame, as_of=None, sample_weights: np.ndarray | None = None) -> DixonColes:
        h = finished_only(history, as_of)
        dates = pd.to_datetime(h["date"])
        as_of = pd.Timestamp(as_of) if as_of is not None else dates.max() + pd.Timedelta(days=1)
        age = (as_of - dates).dt.days.to_numpy()
        keep = age <= self.window_days
        h, age = h.loc[keep], age[keep]
        if len(h) < 30:
            raise ValueError(f"Dixon-Coles needs at least 30 finished matches, got {len(h)}")
        self._as_of = as_of
        teams = sorted(set(h["home_team"]) | set(h["away_team"]))
        self._teams = teams
        idx = {t: i for i, t in enumerate(teams)}
        hi = h["home_team"].map(idx).to_numpy()
        ai = h["away_team"].map(idx).to_numpy()
        x = h["fthg"].to_numpy(dtype=float)
        y = h["ftag"].to_numpy(dtype=float)
        w = np.exp(-self.xi * age)
        if sample_weights is not None:
            w = w * sample_weights[keep] if len(sample_weights) == len(keep) else w * sample_weights
        n = len(teams)

        # match counts per team (weighted, and in earlier seasons) for diagnostics / newcomer detection
        self._n_matches = np.bincount(hi, minlength=n) + np.bincount(ai, minlength=n)
        self._w_matches = np.bincount(hi, weights=w, minlength=n) + np.bincount(ai, weights=w, minlength=n)
        latest_season = h["season"].iloc[np.argmax(dates.loc[h.index].to_numpy())] if "season" in h else None
        if latest_season is not None:
            prev = h["season"] != latest_season
            prev_counts = (np.bincount(hi[prev.to_numpy()], minlength=n)
                           + np.bincount(ai[prev.to_numpy()], minlength=n))
            has_prev = prev.any()
            self._newcomer = (prev_counts == 0) & has_prev
        else:
            self._newcomer = np.zeros(n, dtype=bool)

        data = (hi, ai, x, y, w, n)
        center_att = np.zeros(n)
        center_def = np.zeros(n)
        theta0 = getattr(self, "_warm", None)
        if theta0 is None or len(theta0) != 3 + 2 * n:
            theta0 = np.concatenate([[np.log(max(x.mean(), 0.1)) / 2 + np.log(max(y.mean(), 0.1)) / 2, 0.25, -0.05],
                                     np.zeros(2 * n)])
        theta = self._optimise(theta0, data, center_att, center_def)
        if self.newcomer_prior and self._newcomer.any() and (~self._newcomer).sum() >= 6:
            att, dfn = theta[3:3 + n], theta[3 + n:]
            strength = att - dfn
            est = np.where(~self._newcomer)[0]
            weakest = est[np.argsort(strength[est])[:3]]
            center_att[self._newcomer] = att[weakest].mean()
            center_def[self._newcomer] = dfn[weakest].mean()
            self._prior_att, self._prior_def = float(att[weakest].mean()), float(dfn[weakest].mean())
            theta = self._optimise(theta, data, center_att, center_def)
        else:
            self._prior_att, self._prior_def = 0.0, 0.0
        self._theta = theta
        self._warm = theta
        self.mu, self.home_adv, self.rho = float(theta[0]), float(theta[1]), float(theta[2])
        self._att = theta[3:3 + n]
        self._def = theta[3 + n:]
        self._fitted = True
        # fallback strength for teams never seen: weakest-three average
        strength = self._att - self._def
        weakest = np.argsort(strength)[:3]
        self._unseen_att = float(self._att[weakest].mean())
        self._unseen_def = float(self._def[weakest].mean())
        self.n_train = int(len(h))
        return self

    def _optimise(self, theta0, data, center_att, center_def):
        hi, ai, x, y, w, n = data
        ridge = self.ridge
        x00 = (x == 0) & (y == 0)
        x01 = (x == 0) & (y == 1)
        x10 = (x == 1) & (y == 0)
        x11 = (x == 1) & (y == 1)

        def f(theta):
            mu, home, rho = theta[0], theta[1], theta[2]
            att, dfn = theta[3:3 + n], theta[3 + n:]
            log_lam = mu + home + att[hi] + dfn[ai]
            log_nu = mu + att[ai] + dfn[hi]
            lam, nu = np.exp(log_lam), np.exp(log_nu)
            tau = np.ones_like(lam)
            tau[x00] = 1 - lam[x00] * nu[x00] * rho
            tau[x01] = 1 + lam[x01] * rho
            tau[x10] = 1 + nu[x10] * rho
            tau[x11] = 1 - rho
            tau = np.maximum(tau, 1e-10)
            ll = w * (np.log(tau) + x * log_lam - lam + y * log_nu - nu)
            da, dd = att - center_att, dfn - center_def
            obj = -ll.sum() + ridge * (da @ da + dd @ dd)

            # gradient wrt log_lam, log_nu, rho
            g_lam = x - lam
            g_nu = y - nu
            g_rho = np.zeros_like(lam)
            t00 = tau[x00]
            g_lam[x00] += -lam[x00] * nu[x00] * rho / t00
            g_nu[x00] += -lam[x00] * nu[x00] * rho / t00
            g_rho[x00] = -lam[x00] * nu[x00] / t00
            g_lam[x01] += lam[x01] * rho / tau[x01]
            g_rho[x01] = lam[x01] / tau[x01]
            g_nu[x10] += nu[x10] * rho / tau[x10]
            g_rho[x10] = nu[x10] / tau[x10]
            g_rho[x11] = -1.0 / tau[x11]
            g_lam *= w
            g_nu *= w
            g_rho *= w
            grad = np.empty_like(theta)
            grad[0] = g_lam.sum() + g_nu.sum()
            grad[1] = g_lam.sum()
            grad[2] = g_rho.sum()
            grad[3:3 + n] = np.bincount(hi, weights=g_lam, minlength=n) + np.bincount(ai, weights=g_nu, minlength=n)
            grad[3 + n:] = np.bincount(ai, weights=g_lam, minlength=n) + np.bincount(hi, weights=g_nu, minlength=n)
            grad = -grad
            grad[3:3 + n] += 2 * ridge * da
            grad[3 + n:] += 2 * ridge * dd
            return obj, grad

        bounds = [(None, None), (-1, 1), self.rho_bounds] + [(-3, 3)] * (2 * n)
        res = minimize(f, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500, "gtol": 1e-6})
        return res.x

    # ------------------------------------------------------------------ prediction
    def team_params(self, team: str) -> tuple[float, float, bool]:
        """(attack, defence, seen_in_training)"""
        try:
            i = self._teams.index(team)
            return float(self._att[i]), float(self._def[i]), True
        except ValueError:
            return self._unseen_att, self._unseen_def, False

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        ah, dh, _ = self.team_params(home)
        aa, da, _ = self.team_params(away)
        return (float(np.exp(self.mu + self.home_adv + ah + da)), float(np.exp(self.mu + aa + dh)))

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("model not fitted")
        lam = np.empty(len(fixtures))
        nu = np.empty(len(fixtures))
        seen_h = np.empty(len(fixtures), dtype=bool)
        seen_a = np.empty(len(fixtures), dtype=bool)
        n_h = np.empty(len(fixtures))
        n_a = np.empty(len(fixtures))
        idx = {t: i for i, t in enumerate(self._teams)}
        for k, (home, away) in enumerate(zip(fixtures["home_team"], fixtures["away_team"], strict=True)):
            ah, dh, sh = self.team_params(home)
            aa, da, sa = self.team_params(away)
            lam[k] = np.exp(self.mu + self.home_adv + ah + da)
            nu[k] = np.exp(self.mu + aa + dh)
            seen_h[k], seen_a[k] = sh, sa
            n_h[k] = self._n_matches[idx[home]] if sh else 0
            n_a[k] = self._n_matches[idx[away]] if sa else 0
        m = score_matrix(lam, nu, self.rho)
        out = pd.DataFrame(market_probs_from_matrix(m), index=fixtures.index)
        out["xg_home"], out["xg_away"] = lam, nu
        out["n_home"], out["n_away"] = n_h, n_a
        out["seen_home"], out["seen_away"] = seen_h, seen_a
        return out

    def ratings_table(self) -> pd.DataFrame:
        if not self._fitted:
            return pd.DataFrame()
        df = pd.DataFrame({
            "team": self._teams, "attack": self._att, "defence": self._def,
            "matches_in_window": self._n_matches, "newcomer": self._newcomer,
        })
        # attack: + means scores more; defence: + means concedes more
        df["attack_pct"] = (np.exp(df["attack"]) - 1) * 100
        df["defence_pct"] = (np.exp(df["defence"]) - 1) * 100
        df["strength"] = df["attack"] - df["defence"]
        return df.sort_values("strength", ascending=False).reset_index(drop=True)

    def predict_with_uncertainty(self, history: pd.DataFrame, fixtures: pd.DataFrame, as_of=None,
                                 replicates: int = 30, level: float = 0.9, seed: int = 0) -> pd.DataFrame:
        """Point prediction plus a Bayesian-bootstrap interval for each probability.

        Each replicate re-weights the training matches with random Exponential(1) weights and refits.
        The spread of the replicate probabilities reflects how sensitive the estimate is to the
        particular sample of matches (it does not capture model mis-specification)."""
        self.fit(history, as_of=as_of)
        base = self.predict(fixtures)
        h = finished_only(history, as_of)
        rng = np.random.default_rng(seed)
        cols = ["p_H", "p_D", "p_A", "p_OVER", "p_UNDER", "p_BTTS"]
        reps = []
        warm = self._theta.copy()
        boot = DixonColes(self.xi, self.window_days, self.ridge, self.newcomer_prior, self.rho_bounds)
        for _ in range(replicates):
            boot._warm = warm
            wts = rng.exponential(1.0, size=len(h))
            boot.fit(h, as_of=as_of, sample_weights=wts)
            reps.append(boot.predict(fixtures)[cols].to_numpy())
        arr = np.stack(reps)  # (r, n, k)
        lo = np.quantile(arr, (1 - level) / 2, axis=0)
        hi = np.quantile(arr, 1 - (1 - level) / 2, axis=0)
        for j, c in enumerate(cols):
            base[f"{c}_low"] = lo[:, j]
            base[f"{c}_high"] = hi[:, j]
        return base
