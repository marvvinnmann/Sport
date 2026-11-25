# Roadmap

Phases follow the original project brief. ✅ done · 🟡 partly done · ⬜ not started.

## Phase 1 — Research & architecture ✅
- ✅ AI coding tool recommendation, stack choice, architecture (`docs/RESEARCH.md`, `docs/ARCHITECTURE.md`)
- ✅ Data-provider comparison with costs and licensing (`docs/DATA_SOURCES.md`)
- ✅ Sport / league / market selection backed by a 20-league out-of-sample screen (`reports/league_screening.md`)

## Phase 2 — Working MVP ✅
- ✅ Football, 12 leagues enabled by default (20 screened), markets 1X2 + over/under 2.5
- ✅ Historical import 2012/13 → today, fixtures, odds (pre-closing + closing)
- ✅ Baseline models (Elo, Dixon-Coles), probability and EV calculations, value classification
- ✅ Dashboard (7 pages), REST API, backtesting

## Phase 3 — Advanced prediction engine ✅ (first iteration)
- ✅ LightGBM challenger on engineered features (form, shots, xG, rest, Elo, Dixon-Coles)
- ✅ Calibration via log-linear pooling; market blend and ensemble
- ✅ Model comparison on identical matches with bootstrap CIs; model versions table
- ✅ Prediction explanations (expected goals, attack/defence, Elo, form, rest, H2H context, missing info)
- ⬜ Hyper-parameter search for Dixon-Coles ξ / ridge (walk-forward, nested — avoid overfitting)
- ⬜ Asian-handicap and BTTS markets (score matrix already supports them; AH odds are stored)
- ⬜ xG-based Dixon-Coles variant once 2+ seasons of xG exist in the free data (xG starts 2026/27)

## Phase 4 — Automation 🟡
- ✅ `sportpred daily`: update → settle paper bets → predict → quality checks → markdown report
- ✅ Scheduler service in docker compose; cron instructions
- ✅ Data-quality alerts in the report and dashboard
- ✅ Optional live odds (The Odds API client) — needs a key; not yet called by the daily job
- ⬜ Call The Odds API in the daily job when a key exists (fresher odds closer to kick-off)
- ⬜ Email/Telegram notification of the daily report (optional)

## Phase 5 — Production readiness 🟡
- ✅ Docker image, PostgreSQL option, backups script, health endpoints, secrets in `.env`
- ⬜ Authentication (only needed if hosted publicly — use a reverse proxy, see DEPLOYMENT.md)
- ✅ CI pipeline (GitHub Actions: ruff + pytest on every push, `.github/workflows/ci.yml`)
- ⬜ Odds table compaction (the full 20-league research database is ~0.5 GB in SQLite)

## Research backlog (ordered by expected value of information)
1. **Paper-trade the market-consensus model for ≥ 3 months** and compare its CLV with the backtest. This is the only
   test that cannot be affected by look-ahead or selection bias.
2. **Sharper anchor price.** The edge in backtests depends on the anchor (Pinnacle ≫ market average ≫
   Betfair snapshot). Evaluate a paid sharp feed or exchange mid-prices before spending real money.
3. **Odds closer to kick-off.** The provider's odds are Tuesday/Friday snapshots. With The Odds API key,
   measure whether best-price opportunities survive until you could actually bet.
4. **Bookmaker limits.** Price-dispersion strategies get accounts restricted quickly (documented in the
   literature). Track which bookmakers the recommended prices come from.
5. **Tennis** (tennis-data.co.uk: results + odds since 2001): player Elo with surface adjustments.
6. **Availability data** (API-Football injuries/line-ups, ~$19/month) — only after 1–4.
