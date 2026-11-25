# sportpred — transparent football prediction & value-bet research

A working application that estimates probabilities for football matches, compares them with bookmaker
odds, and flags a selection as **positive expected value only when the evidence survives strict checks**.
Everything is validated walk-forward (no look-ahead) against the sharpest benchmark there is: the
bookmakers' closing prices.

> **Paper trading only.** The app never places bets and never connects to bookmaker accounts.
> Positive expected value is an estimate, not a guarantee. Read the honest results below before
> drawing any conclusion.

## What it does

* Downloads 15 seasons of results, match statistics and odds for 12 (up to 22) European leagues from
  football-data.co.uk — free, no key needed — plus upcoming fixtures with current odds.
* Fits **Dixon-Coles**, **Elo**, **LightGBM**, a **market blend** and a **market-consensus** model; predicts
  match result (1X2) and over/under 2.5 goals with uncertainty ranges and plain-language explanations.
* Classifies every selection as **Positive EV**, **Insufficient evidence** or **No bet** — it never forces a pick.
* Backtests strategies on out-of-sample predictions with realistic prices, staking rules, drawdowns,
  closing line value (CLV), baselines and Monte Carlo simulation.
* Logs paper bets automatically and settles them, so the model is also tested going forward.
* 7-page dashboard (Streamlit), REST API (FastAPI), command line, Docker deployment, 56 automated tests.

## Honest results (out-of-sample, Jul 2019 → Nov 2026, 20 leagues, 53,459 matches)

Full numbers: [`docs/RESEARCH.md`](docs/RESEARCH.md) and [`reports/`](reports/).

1. **No model beats the closing market.** Every model's log loss is worse than the de-vigged closing
   price with 95% confidence. Gradient boosting is the best standalone model; Elo and Dixon-Coles trail.
2. **Betting on a standalone model's "value" loses money:** −4% to −11% yield with negative closing line
   value (CLV). The models disagree with the market far too often, and the market is usually right.
3. **The only signal that holds up is market consensus.** It means taking a bookmaker's price when it is
   higher than the sharp, margin-free consensus (Pinnacle, or the de-vigged market average). This had
   **positive CLV in every season and every league** (+3.3% on average, 2,382 bets) and a realised yield
   of +6.6% (95% CI +0.5% to +12.9%). That yield is borderline once you account for how many strategies
   were tried. This is price dispersion between bookmakers, not better football prediction, and such
   strategies tend to get accounts limited.
4. So the live default is the **market-consensus** model, with Dixon-Coles and Elo shown as context.
   Paper-trade it for at least 3 months before trusting it.

## Quick start (your computer)

You need Python 3.11+ ([python.org](https://www.python.org/downloads/)). In a terminal:

```bash
git clone <this repository> sport
cd sport
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
sportpred init
sportpred update          # first time: downloads ~180 files, 10–15 minutes
sportpred evaluate        # optional (~25 min): walk-forward tests that fill Model Performance & Backtesting
sportpred predict
streamlit run app/main.py # opens the dashboard at http://localhost:8501
```

Every day afterwards: `sportpred daily` (or let the Docker scheduler do it). Docker, cron, backups
and hosting: [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

### Optional API key (live odds)
1. Get a free key at [the-odds-api.com](https://the-odds-api.com/) (500 credits/month).
2. `cp .env.example .env` and put it after `SPORTPRED_ODDS_API_KEY=`.
3. Restart. The Data Management page shows *key configured*. Keys stay in `.env`, which git ignores.

## Command reference

| Command | What it does |
|---|---|
| `sportpred update [--leagues E0,E1] [--force]` | download/refresh results, odds and fixtures |
| `sportpred quality` | data quality report |
| `sportpred import-csv FILE --label live/sample/simulated` | manual CSV import (football-data format) |
| `sportpred evaluate` | walk-forward evaluation of all models vs the market |
| `sportpred backtest --model market --price-source best_book --min-ev 0.03` | quick backtest in the terminal |
| `sportpred predict` | predictions for upcoming fixtures |
| `sportpred settle` | settle paper bets |
| `sportpred daily` | update → settle → predict → quality → report |
| `sportpred serve-api` | REST API at http://127.0.0.1:8000/docs |
| `python scripts/league_screening.py` | league × market × model screening report |
| `python scripts/research_tables.py` | pooled backtests, CLV by season, anchor comparison |

## Dashboard pages

1. **Dashboard** — fixtures, highest-value opportunities, predictions by league, model overview, data freshness
2. **Predictions** — filter by league, date, market, odds, EV, confidence; sort; CSV export
3. **Match Analysis** — probabilities vs odds with uncertainty, EV workings, team comparison, form, table, H2H, caveats
4. **Model Performance** — log loss/Brier/RPS vs closing market, calibration, by league/season, ROI/yield/CLV, paper trading
5. **Data Management** — sources & status, update button, history, quality warnings, coverage, CSV import
6. **Model Settings** — model, thresholds, leagues/markets, parameters, staking, generate predictions
7. **Backtesting** — any model/market/price source/date range/thresholds/staking → P&L, drawdown, baselines, sensitivity, Monte Carlo

## Documentation

| File | Contents |
|---|---|
| [`docs/RESEARCH.md`](docs/RESEARCH.md) | Phase 1: AI coding tool, stack, sport/league/market choice, providers & costs, critical review |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | system diagram, components, database, leakage protections |
| [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) | every provider: coverage, depth, cost, limits, licensing, weaknesses |
| [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | training data, features, algorithms, results, limitations, update history |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | install, Docker, scheduling, backups, monitoring, security |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | phases and research backlog |
| [`DEVELOPMENT_STATUS.md`](DEVELOPMENT_STATUS.md) | done / next / known bugs — read first when resuming |
| [`CLAUDE.md`](CLAUDE.md) / [`AGENTS.md`](AGENTS.md) | rules for AI coding agents working on this repo |

## Data credits

Results, statistics and odds: [football-data.co.uk](https://www.football-data.co.uk/) (free for research;
raw files are not redistributed here). Optional live odds: The Odds API.

## Responsible use

Betting involves risk of loss. If gambling stops being fun, get help — e.g. BeGambleAware (UK),
Gamblers Anonymous, or your national support line. This project is for research and education.
