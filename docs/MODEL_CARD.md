# Model card

## Intended use

* **Use:** research and **paper trading** of football 1X2 and over/under 2.5 selections. Comparing
  model and market probabilities; learning how betting markets price matches.
* **Not for:** automated betting, guaranteed-profit claims, in-play betting, or markets/leagues not listed.
  Outputs are probability estimates with uncertainty, not tips.

## Models

| Name | Type | Markets | Live? | Role |
|---|---|---|---|---|
| `market` | de-vigged sharp pre-closing price (Pinnacle → market average (power method) → Betfair Exchange) | 1X2, O/U 2.5 | ✅ **default** | probability estimate; value = a bookmaker paying more |
| `blend` | log-linear pool of Dixon-Coles and `market`, weights fitted season by season out-of-sample | 1X2, O/U | ✅ | best probability forecast (log loss) |
| `dixon_coles` | Dixon-Coles Poisson (time decay ξ=0.0019/day, 1,100-day window, ridge 0.5, promoted-team prior, ρ low-score correction) | 1X2, O/U (+BTTS, AH-ready) | ✅ | explanation, second opinion |
| `elo` | Elo (K=20, home +60, goal-difference multiplier, 25% season regression) + ordered logit | 1X2 | ✅ | transparent baseline |
| `gbm` | LightGBM (250 trees, 12 leaves, min 150/leaf, L2=10), pooled across leagues | 1X2, O/U | evaluation only | complex challenger |
| `ensemble` | log-linear pool of DC, Elo, GBM and market | 1X2, O/U | evaluation only | research |

## Training data

* football-data.co.uk, 20 European leagues (12 enabled by default), seasons 2012/13–2026/27,
  105,547 finished matches, 2.49 M odds rows. Live data, not simulated.
* Each model is fitted **per league** (DC, Elo) or pooled with a league feature (GBM).
* Rows excluded: matches without results; bookmaker prices with impossible margins
  (1X2 over-round outside 0.97–1.30, two-way outside 0.97–1.20).

## Features

| Model | Inputs (all known before kick-off) |
|---|---|
| Dixon-Coles | goals scored/conceded, home/away, match date (for time decay) |
| Elo | results and goal difference, sequential by day |
| GBM | exponentially weighted (half-life 6 matches) goals, shots, shots on target, xG, points for/against; rest days (league games only); matches played; pre-match Elo; Dixon-Coles walk-forward probabilities and expected goals; league |
| market / blend | pre-closing odds of the anchor bookmaker (+ Dixon-Coles for the blend) |

Not used: injuries, suspensions, line-ups, cup/European fixtures, weather, referee, motivation,
head-to-head (shown for context only; small samples).

## Uncertainty

* Dixon-Coles: 30 Bayesian-bootstrap refits (random Exponential(1) match weights) → 90% range.
* Market and blend: at least the **median absolute move of the sharp fair probability between the
  Tue/Fri snapshot and kick-off**, measured on 53,459 matches. Home ±2.0pp, draw ±0.9pp, away ±1.7pp,
  over/under ±1.8pp. The blend also propagates the Dixon-Coles range.
* These ranges reflect sampling and price noise, not model mis-specification.

## Value rules (defaults in `config/default_settings.yaml`)

A selection is **positive EV** only if all hold: EV ≥ 3%; edge ≥ 1pp; odds 1.5–5.0; confidence ≥ medium;
EV > 0 at the low end of the uncertainty range; model not more than 12pp above the sharp market;
odds snapshot ≤ 96 h old. Otherwise **insufficient evidence** (EV positive but checks fail) or **no bet**.

## Evaluation (walk-forward, test period 1 Jul 2019 – 23 Nov 2026, identical 53,459 matches)

| Model | 1X2 log loss | vs closing market (95% CI) | Accuracy | RPS | ECE home/draw/away | O/U log loss | O/U ECE |
|---|---|---|---|---|---|---|---|
| closing market | 0.9966 | — | 50.8% | — | — | 0.6737 | — |
| ensemble | 1.0000 | +0.0034 (+0.0026…+0.0041) | 50.5% | 0.2035 | 0.4/0.3/0.7% | 0.6761 | 0.7% |
| blend | 1.0001 | +0.0035 (+0.0027…+0.0042) | 50.5% | 0.2036 | 0.4/0.5/0.7% | 0.6761 | 0.7% |
| market (pre-closing) | 1.0003 | +0.0037 (+0.0030…+0.0044) | 50.4% | 0.2036 | 0.7/0.7/0.8% | 0.6762 | 0.7% |
| gbm | 1.0128 | +0.0162 (+0.0148…+0.0175) | 49.3% | 0.2074 | 0.4/1.2/0.4% | 0.6810 | 1.0% |
| elo | 1.0191 | +0.0224 (+0.0208…+0.0241) | 49.0% | 0.2096 | 0.7/0.6/0.6% | — | — |
| dixon_coles | 1.0233 | +0.0267 (+0.0249…+0.0286) | 48.7% | 0.2106 | 1.7/1.2/1.5% | 0.6916 | 4.0% |

Betting results (EV ≥ 3%, flat stakes, best named bookmaker price): market consensus 1X2
2,382 bets, yield +6.6% (95% CI +0.5…+12.9%), CLV +3.3% (+2.9…+3.8%). Blend +3.5% (CI −0.7…+7.9%),
CLV +1.8%. Standalone Dixon-Coles −5.9%, Elo −5.7%, GBM −4.4% (all CIs below zero, CLV negative).
Full detail: [`RESEARCH.md`](RESEARCH.md) §5 and `reports/`.

Blend weights (latest season fit, 1X2): Dixon-Coles −0.080, market +1.109; draw bias +0.040.
Ensemble: Dixon-Coles −0.087, Elo −0.114, GBM +0.122, market +1.098.

## Known limitations

* The fundamental models are significantly less accurate than the market. Their disagreements with
  the market are, on average, wrong.
* Dixon-Coles over/under probabilities are poorly calibrated (ECE 4%).
* Odds are one Tue/Fri snapshot per match; live opportunities may vanish before you can act.
* Since Pinnacle data stopped (2025/26), the anchor is the market average, and evidence for it in the
  newest seasons is thin (~340 bets).
* Positive CLV and backtest profits do not account for bookmaker account limits, voided bets or taxes.
* Promoted teams get a prior (the weakest established teams' strength) and low confidence for their
  first matches.

## Versioning and reproducibility

* Every evaluation run stores a `model_versions` row (parameters hash, trained-until date, metrics).
  Every live prediction run stores its own `model_versions` row, and predictions are append-only with
  a `run_id`.
* Walk-forward predictions: `data/reports/walkforward_<model>.parquet`; summary: `reports/evaluation_summary.json`.
* Deterministic seeds for bootstraps and simulations; LightGBM uses fixed parameters.

## Update history

| Date | Change | Evidence |
|---|---|---|
| 2026-11-25 | v0.1: Dixon-Coles, Elo, GBM, blend, ensemble; walk-forward on 20 leagues | `reports/evaluation_summary.json` |
| 2026-11-25 | Pre-closing anchor changed from Pinnacle → Exchange → average to Pinnacle → **average** → Exchange | `reports/anchor_comparison.csv`: the Exchange snapshot was the weakest anchor in both eras |
| 2026-11-25 | Live default changed from Dixon-Coles to **market consensus** | standalone models lost 4–11%; market consensus had positive CLV in every season |
