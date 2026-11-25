"""Scheduled jobs: the daily pipeline and its markdown report."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from pathlib import Path

from sportpred.config import load_model_settings
from sportpred.data.ingest import update_all
from sportpred.data.quality import run_quality_checks
from sportpred.db import session_scope
from sportpred.leagues import current_season_start_year, league_label
from sportpred.services.evaluation import reports_dir
from sportpred.services.paper import paper_bets_frame, settle_paper_bets
from sportpred.services.predictions import SELECTION_LABELS, generate_predictions, latest_predictions


def daily_job(progress: Callable[[str], None] = print) -> Path:
    ms = load_model_settings()
    progress("1/4 Updating data (current and previous season + fixtures)…")
    results = update_all(leagues=ms.leagues, first_season=current_season_start_year() - 1, settings=ms)
    failed = [r for r in results if r.status == "failed"]
    progress("2/4 Settling paper bets…")
    settled = settle_paper_bets()
    progress("3/4 Generating predictions…")
    pred = generate_predictions(ms=ms, progress=progress)
    progress("4/4 Running data quality checks and writing report…")
    with session_scope() as s:
        issues = run_quality_checks(s, ms.leagues)
    return write_daily_report(results, failed, settled, pred, issues)


def write_daily_report(results, failed, settled, pred, issues) -> Path:
    today = dt.date.today()
    lines = [f"# Daily report — {today}", "",
             "_Paper-trading research output. Not betting advice. Positive expected value is an estimate, "
             "not a guarantee of profit._", "",
             "## Data update", f"- Resources processed: {len(results)}; failed: {len(failed)}"]
    for r in failed:
        lines.append(f"  - FAILED {r.resource}: {r.message}")
    lines += ["", "## Predictions", f"- Matches predicted: {pred.get('n_matches', 0)}",
              f"- Selections flagged as positive EV: {pred.get('n_value', 0)}", f"- Run id: {pred.get('run_id')}"]
    df = latest_predictions()
    if not df.empty:
        rec = df[df["is_recommended"] == 1].sort_values("ev", ascending=False)
        if len(rec):
            lines += ["", "| Date | League | Match | Selection | Odds | Model p | EV | Confidence |",
                      "|---|---|---|---|---|---|---|---|"]
            for r in rec.itertuples():
                lines.append(f"| {r.date:%Y-%m-%d} | {league_label(r.league)} | {r.home_team} v {r.away_team} | "
                             f"{SELECTION_LABELS[r.selection]} | {r.best_odds:.2f} ({r.best_bookmaker}) | "
                             f"{r.prob:.1%} | {r.ev:+.1%} | {r.confidence} |")
        else:
            lines.append("- No selection met all value criteria today (this is normal).")
    lines += ["", "## Paper trading", f"- Bets settled this run: {settled}"]
    pb = paper_bets_frame()
    if not pb.empty:
        done = pb[pb["status"].isin(["won", "lost"])]
        if len(done):
            lines.append(f"- Settled bets: {len(done)}, profit {done['pnl'].sum():+.2f} units, "
                         f"yield {done['pnl'].sum() / done['stake'].sum():+.1%}, "
                         f"avg CLV {done['clv'].mean():+.1%}" if done["clv"].notna().any() else
                         f"- Settled bets: {len(done)}, profit {done['pnl'].sum():+.2f} units")
    lines += ["", "## Data quality alerts"]
    for i in issues:
        if i.severity in ("warning", "error"):
            where = " ".join(x for x in [i.league, i.season] if x)
            lines.append(f"- **{i.severity}** {i.check} {where}: {i.message}")
    if not any(i.severity in ("warning", "error") for i in issues):
        lines.append("- None")
    path = reports_dir() / f"daily_{today}.md"
    path.write_text("\n".join(lines) + "\n")
    return path
