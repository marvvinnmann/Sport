# Development status

_Last updated: 2026-11-25 (session 1). Read this first when resuming work._

## Where we are

Phases 1–3 of the brief are complete in a first iteration; Phases 4–5 are partly done
(see `docs/ROADMAP.md`). The application works end to end on real data:

```
sportpred update → sportpred evaluate → sportpred predict → streamlit run app/main.py
```

## Completed (implemented AND tested)

| Area | What | Verified by |
|---|---|---|
| Data | football-data.co.uk downloader + parser for every column layout 2012/13–2026/27 (Bb*, Pinnacle P/PS, Betfair, Betfred vs Betfair-draw clash, closing odds, xG, AH lines) | `tests/test_parser.py`; real ingest of 301 files, 0 failures |
| Data | upsert ingest with audit log + file hashes, unchanged-file skip, finished-never-downgraded, rescheduled → superseded, stale fixtures | `tests/test_ingest_quality.py` |
| Data | quality checks: missing results, duplicates, team twice/day, implausible goals/margins, shots, season completeness, odds coverage, freshness, non-live labels | tests + run on real DB |
| Data | manual CSV import with honest live/sample/simulated label | tests + dashboard |
| Data | The Odds API client (optional) | parser unit-tested with recorded-format data; **not tested against the live API (no key)** |
| Maths | implied prob, overround, margin, de-vig (multiplicative/power/Shin), EV, edge, Kelly, CLV | `tests/test_odds.py` |
| Models | Dixon-Coles (analytic gradient, time decay, ridge, promoted-team prior, bootstrap intervals) | gradient check; recovers simulated strengths (r > 0.9) |
| Models | Elo (no same-day leakage, season regression, newcomer prior) + ordered logit | tests |
| Models | LightGBM challenger; log-linear pooling (blend, ensemble) | walk-forward on 20 leagues |
| Validation | walk-forward engine with leakage guard; future-poisoning test | `test_walk_forward_has_no_lookahead` |
| Validation | metrics (log loss, Brier, RPS, ECE, calibration table, bootstrap CIs, paired comparison) | tests |
| Betting | backtest (price sources, CLV, baselines, EV sensitivity), staking (flat/fractional Kelly + caps, NumPy fast path), Monte Carlo (500 runs ≈ 8 s) | tests + dashboard (Playwright run) |
| Live | predictions with verdicts (value / insufficient evidence / no bet), reasons, missing info, explanations; append-only storage; paper bets + settlement | end-to-end test + real run on 12 fixtures |
| Live | **market consensus** (default) and **blend** models, market-move uncertainty band, anchor Pinnacle → market average → Exchange | tests + real run |
| Apps | 7-page Streamlit dashboard | AppTest renders every page; Playwright screenshots with real data, 0 exceptions |
| Apps | FastAPI (health, status, leagues, fixtures, predictions, detail, performance, paper bets, quality, calculator, backtest) | `test_api_endpoints` |
| Ops | CLI, daily job + report, scheduler loop, backup script, CI workflow | CLI runs on real data |
| Ops | Dockerfile + docker compose (PostgreSQL) | compose file validated; regular (non-editable) pip install verified incl. config discovery; **image build NOT completed** — the sandbox's build containers had no internet access |
| Research | 20-league × 6-model screening, pooled backtests, CLV by season, anchor comparison (`scripts/research_tables.py`) | `reports/` |

Test suite: **56 tests passing** (`pytest`), lint clean (`ruff`).

## Next steps (in priority order)

1. **Paper trade** for at least 3 months with the default market-consensus settings; review Model Performance →
   Paper trading weekly. Compare its CLV with the backtest (positive CLV is the go/no-go signal).
2. Build and run the Docker image on a normal machine (`docker compose build && docker compose up`) and
   fix anything environment-specific.
3. With an Odds API key: wire `refresh_live_odds` into `daily_job` and measure how many best-price
   opportunities still exist at the time you could act on them.
4. Walk-forward hyper-parameter tuning of Dixon-Coles (ξ, ridge) and GBM with nested validation.
5. Asian handicap / BTTS markets (score matrix and stored AH odds are ready).
6. Tennis module (tennis-data.co.uk).

## Known bugs / limitations

* **Odds snapshots are Tue/Fri.** football-data.co.uk collects pre-closing odds once; live "best
  prices" may have moved by the time you see them. Odds age is shown and stale odds are flagged.
* **No injuries/line-ups/cup matches.** Every prediction lists this as missing information.
* **Pinnacle data ended in 2025/26.** The live anchor is now the de-vigged market average (Exchange
  as last resort). Backtest evidence for this anchor in the newest seasons is thin (≈300 bets).
* **Uncertainty for the blend** combines Dixon-Coles bootstrap spread with the typical pre-close price
  move; it does not capture model mis-specification.
* **Database size:** the full 20-league research database is ~0.5 GB in SQLite (odds table). The
  default 12-league install is smaller. Compaction is on the roadmap.
* `backtest_runs` table exists but backtests are not yet saved to it (results are shown, downloadable as CSV).
* The Docker image has not been built successfully yet (the development sandbox blocks internet access
  inside build containers). Run `docker compose build` on your machine and report any error.
* Changes are committed locally on branch `claude/sports-betting-prediction-app-loi862`; pushing to
  GitHub failed in session 1 (the Claude GitHub App has no access to the repository yet).
* The dashboard caches data for 60 s; after an update use the ⋮ menu → *Clear cache* or wait a minute.

## How the cloud development environment is set up

Claude Code cloud sessions for this repo should use the personal environment **"sport"** (created in
session 1). It allows `football-data.co.uk` and `www.football-data.co.uk` plus package registries.
The Default environment blocks football-data.co.uk.

## Session log

| Session | Date | Summary |
|---|---|---|
| 1 | 2026-11-25 | Research, full MVP + Phase 3 models, 20-league walk-forward screening, dashboard, API, tests, docs |
