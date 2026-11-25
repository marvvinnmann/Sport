"""Command-line interface. Run ``sportpred --help`` to see all commands.

Typical daily use:
    sportpred update      # download new results, fixtures and odds
    sportpred predict     # create predictions for upcoming matches
    sportpred daily       # both of the above + settle paper bets + write a report
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from sportpred.config import get_settings, load_model_settings

app = typer.Typer(add_completion=False, help="Sports prediction & value-bet research platform (paper trading only).")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)


@app.command()
def init() -> None:
    """Create the database tables (safe to run repeatedly)."""
    from sportpred.db import init_db

    init_db()
    typer.echo(f"Database ready at {get_settings().resolved_database_url()}")


@app.command()
def update(
    leagues: str = typer.Option(None, help="Comma separated league codes, e.g. E0,E1 (default: settings)"),
    first_season: int = typer.Option(None, help="First season start year (default: settings)"),
    force: bool = typer.Option(False, help="Re-download files even if cached"),
    no_fixtures: bool = typer.Option(False, help="Skip upcoming fixtures"),
    verbose: bool = False,
) -> None:
    """Download historical results + odds and upcoming fixtures from football-data.co.uk."""
    _setup_logging(verbose)
    from sportpred.data.ingest import update_all

    lg = [x.strip() for x in leagues.split(",")] if leagues else None

    def show(r):
        extra = f" — {r.message}" if r.message else ""
        warn = f" [warnings: {'; '.join(r.warnings)}]" if r.warnings else ""
        typer.echo(f"{r.resource:<22} {r.status:<10} +{r.inserted} new, {r.updated} updated, "
                   f"{r.odds_rows} odds rows{extra}{warn}")

    results = update_all(leagues=lg, first_season=first_season, force=force,
                         include_fixtures=not no_fixtures, progress=show)
    failed = [r for r in results if r.status == "failed"]
    typer.echo(f"Done: {len(results)} resources, {len(failed)} failed.")


@app.command()
def quality(leagues: str = typer.Option(None)) -> None:
    """Run data quality checks and print the issues found."""
    from sportpred.data.quality import run_quality_checks
    from sportpred.db import session_scope

    lg = [x.strip() for x in leagues.split(",")] if leagues else None
    with session_scope() as s:
        for i in run_quality_checks(s, lg):
            where = " ".join(x for x in [i.league, i.season] if x)
            typer.echo(f"[{i.severity.upper():7}] {i.check:<22} {where:<10} {i.message}"
                       + (f" (e.g. {i.examples})" if i.examples else ""))


@app.command("import-csv")
def import_csv(path: Path, league: str = typer.Option(None), season: str = typer.Option(None),
               label: str = typer.Option("live", help="live | sample | simulated")) -> None:
    """Import a CSV in football-data.co.uk format (e.g. a file you downloaded manually)."""
    from sportpred.data.ingest import import_csv_file

    res = import_csv_file(path.read_bytes(), path.name, league=league, season=season, data_label=label)
    typer.echo(f"{res.status}: +{res.inserted} new, {res.updated} updated, {res.odds_rows} odds rows. "
               f"{res.message or ''} {'; '.join(res.warnings)}")


@app.command()
def evaluate(
    models: str = typer.Option("market,elo,dixon_coles,gbm,blend,ensemble", help="Comma separated model names"),
    leagues: str = typer.Option(None),
    wf_from: str = typer.Option("2015-07-01", help="First date with walk-forward predictions (warm-up)"),
    test_from: str = typer.Option("2019-07-01", help="Start of the out-of-sample test period"),
    n_jobs: int = typer.Option(4),
    verbose: bool = False,
) -> None:
    """Walk-forward evaluation: train only on the past, predict week by week, compare to the market."""
    _setup_logging(verbose)
    from sportpred.services.evaluation import run_evaluation

    lg = [x.strip() for x in leagues.split(",")] if leagues else None
    summary = run_evaluation([m.strip() for m in models.split(",")], leagues=lg, wf_from=wf_from,
                             test_from=test_from, n_jobs=n_jobs, progress=typer.echo)
    for name, res in summary.get("overall", {}).items():
        for market, r in res.items():
            vs = r.get("vs_closing", {})
            typer.echo(f"{name:<12} {market:<5} n={r['model']['n']:>6} log loss {r['model']['log_loss']:.4f} "
                       f"vs closing market {r['market_closing']['log_loss']:.4f} "
                       f"(diff {vs.get('mean_diff', float('nan')):+.4f}, 95% CI {vs.get('ci_low', float('nan')):+.4f}"
                       f" to {vs.get('ci_high', float('nan')):+.4f})")


@app.command()
def backtest(
    model: str = typer.Option("dixon_coles"),
    market: str = typer.Option("1X2", help="1X2 or OU25"),
    price_source: str = typer.Option("market_avg"),
    min_ev: float = typer.Option(0.04),
    min_odds: float = typer.Option(1.5),
    max_odds: float = typer.Option(5.0),
    leagues: str = typer.Option(None),
    staking: str = typer.Option("flat", help="flat or kelly"),
) -> None:
    """Backtest a value-betting strategy on stored walk-forward predictions (run `evaluate` first)."""
    from sportpred.config import StakingSettings
    from sportpred.evaluation.backtest import BacktestConfig, baseline_strategies, run_backtest
    from sportpred.services.evaluation import load_walk_forward

    df = load_walk_forward(model)
    if df is None:
        raise typer.Exit("No walk-forward predictions stored. Run `sportpred evaluate` first.")
    ms = load_model_settings()
    cfg = BacktestConfig(market=market, price_source=price_source, min_ev=min_ev, min_odds=min_odds,
                         max_odds=max_odds, leagues=[x.strip() for x in leagues.split(",")] if leagues else None,
                         staking=StakingSettings(**{**ms.staking.model_dump(), "method": staking}))
    bets, summary = run_backtest(df, cfg)
    typer.echo(json.dumps(summary, indent=2, default=str))
    typer.echo(baseline_strategies(df, cfg).to_string(index=False))


@app.command()
def predict(verbose: bool = False) -> None:
    """Fit the active model on all finished matches and predict upcoming fixtures."""
    _setup_logging(verbose)
    from sportpred.services.predictions import generate_predictions

    res = generate_predictions(progress=typer.echo)
    typer.echo(f"Stored {res['n_predictions']} predictions for {res['n_matches']} matches "
               f"({res['n_value']} flagged as positive EV). Run id: {res['run_id']}")


@app.command()
def settle() -> None:
    """Settle paper bets whose matches have finished."""
    from sportpred.services.paper import settle_paper_bets

    n = settle_paper_bets()
    typer.echo(f"Settled {n} paper bet(s).")


@app.command()
def daily(verbose: bool = False) -> None:
    """Full daily job: update data → predict → log paper bets → settle → write report."""
    _setup_logging(verbose)
    from sportpred.services.jobs import daily_job

    report_path = daily_job(progress=typer.echo)
    typer.echo(f"Daily report written to {report_path}")


@app.command()
def schedule(every_hours: float = typer.Option(12.0, help="Run the daily job every N hours")) -> None:
    """Run the daily job forever on a fixed interval (used by docker compose 'scheduler')."""
    import time

    from sportpred.services.jobs import daily_job

    _setup_logging(False)
    while True:
        try:
            daily_job(progress=typer.echo)
        except Exception as exc:  # keep the scheduler alive; the error is logged
            logging.getLogger("scheduler").exception("daily job failed: %s", exc)
        time.sleep(every_hours * 3600)


@app.command("serve-api")
def serve_api(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the REST API (docs at http://127.0.0.1:8000/docs)."""
    import uvicorn

    uvicorn.run("sportpred.api.main:app", host=host, port=port)


if __name__ == "__main__":
    app()
