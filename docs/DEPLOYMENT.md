# Installation, deployment and maintenance

## Option A — run on your computer (simplest)

Requirements: Python 3.11–3.13. No database server needed (SQLite file).

```bash
git clone <your repo url> sport && cd sport
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env                 # optional: add API keys later
sportpred init
sportpred update                     # first run downloads ~15 seasons × 12 leagues (10–15 min)
sportpred evaluate                   # optional but recommended: walk-forward tests (~25 min, 4 CPU cores)
sportpred predict
streamlit run app/main.py            # dashboard at http://localhost:8501
```

`uv` users can replace the venv/pip lines with `uv venv && uv pip install -e ".[dev]"`.

The REST API: `sportpred serve-api` → http://127.0.0.1:8000/docs

## Option B — Docker compose (PostgreSQL + dashboard + API + scheduler)

```bash
cp .env.example .env                 # optional keys
docker compose build
docker compose run --rm scheduler sportpred init
docker compose run --rm scheduler sportpred update
docker compose run --rm scheduler sportpred evaluate
docker compose up -d                 # dashboard :8501, API :8000, scheduler every 12 h
docker compose logs -f scheduler     # watch the daily job
```

Ports are bound to `127.0.0.1` only. Do **not** expose them to the internet without adding
authentication (see below).

## Scheduling (automation)

* Docker: the `scheduler` service runs `sportpred schedule --every-hours 12`
  (update data → settle paper bets → predict → quality checks → report in `data/reports/daily_YYYY-MM-DD.md`).
* Without Docker (Linux/macOS cron), e.g. at 08:10 and 17:10 every day:
  ```cron
  10 8,17 * * * cd /path/to/sport && .venv/bin/sportpred daily >> data/daily.log 2>&1
  ```
  Windows: Task Scheduler → run `.venv\Scripts\sportpred.exe daily` in the project folder.
* Retraining: the live model is re-fitted on every prediction run (it takes seconds). Re-run
  `sportpred evaluate` monthly to refresh walk-forward evidence and the blend weights.

## Backups

```bash
scripts/backup_db.sh     # SQLite online backup or pg_dump; keeps the 14 newest in data/backups/
```
Add it to cron (e.g. daily at 03:00). Raw downloads can always be re-downloaded; the database
holds your prediction history and paper-trading ledger, which cannot.

## Monitoring and logging

* Every download/import is logged in the `data_updates` table (Data Management page → *Update history*).
* Data quality alerts appear on the Data Management page and in each daily report.
* `sportpred quality` prints the same checks in a terminal.
* Container health checks: dashboard `/_stcore/health`; API `/health` and `/status`.
* Command errors are written to stderr (captured by Docker logs or your cron log file).

## Security

* API keys only in `.env` (git-ignored) or your hosting platform's secret store — never in code.
* The app never places bets or logs into bookmaker accounts.
* Authentication: not included because the default deployment is local-only. If you host it,
  put it behind an authenticating reverse proxy (e.g. Caddy/nginx with basic auth, Cloudflare
  Access, or Tailscale), or use Streamlit Community Cloud's viewer authentication.

## Hosting options (if you want it online)

| Option | Effort | Notes |
|---|---|---|
| Small VPS (e.g. 2 vCPU / 4 GB) + docker compose | medium | full control; add a reverse proxy with authentication |
| Streamlit Community Cloud | low | free for the dashboard; needs an external database (e.g. hosted PostgreSQL) and keys in its secret manager |
| Keep it local | none | recommended while the strategy is unproven |

## Maintenance checklist

| When | What |
|---|---|
| Every day (automatic) | `sportpred daily` |
| Weekly | glance at Data Management → quality alerts; Model Performance → paper trading |
| Monthly | `sportpred evaluate`, then `python scripts/league_screening.py`; review CLV trend |
| Each new season (July/August) | nothing required — new season files are picked up automatically; check that promoted teams appear |
| When football-data.co.uk changes columns | run `pytest tests/test_parser.py`; add new bookmaker prefixes in `data/football_data.py` |
| Upgrading libraries | `pip install -U -e ".[dev]"` then `pytest` |
