"""Walk-forward (rolling-origin) validation.

For each league, the test period is cut into blocks (default: 7 days). For each block the model is
re-fitted on matches played strictly *before* the block starts and then predicts the block.
This mirrors real use: on Friday you only know results up to Thursday.

Random train/test splits are never used — they leak future information into the past.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from sportpred.models.base import MatchModel

log = logging.getLogger(__name__)

PRED_COLS = ["p_H", "p_D", "p_A", "p_OVER", "p_UNDER", "p_BTTS", "xg_home", "xg_away", "n_home", "n_away",
             "seen_home", "seen_away", "elo_home", "elo_away"]


class LeakageError(AssertionError):
    pass


def walk_forward_league(
    matches: pd.DataFrame,
    factory: Callable[[], MatchModel],
    test_start: pd.Timestamp,
    test_end: pd.Timestamp | None = None,
    refit_days: int = 7,
    min_train: int = 300,
) -> pd.DataFrame:
    """Walk-forward predictions for one league. ``matches`` must contain finished matches."""
    m = matches.copy()
    m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values("date")
    fin = m[(m["status"] == "finished") & m["fthg"].notna()]
    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end) if test_end is not None else fin["date"].max()
    test = fin[(fin["date"] >= test_start) & (fin["date"] <= test_end)]
    if test.empty:
        return pd.DataFrame()
    model = factory()
    out = []
    block_start = test_start
    while block_start <= test_end:
        block_end = block_start + pd.Timedelta(days=refit_days)
        block = test[(test["date"] >= block_start) & (test["date"] < block_end)]
        if not block.empty:
            train = fin[fin["date"] < block_start]
            if len(train) >= min_train:
                if train["date"].max() >= block["date"].min():
                    raise LeakageError("training data overlaps the prediction block")
                try:
                    model.fit(train, as_of=block_start)
                    pred = model.predict(block)
                    pred = pred[[c for c in PRED_COLS if c in pred.columns]].copy()
                    pred["match_id"] = block["id"].to_numpy() if "id" in block else block.index
                    pred["fit_date"] = block_start
                    pred["n_train"] = len(train)
                    out.append(pred)
                except ValueError as exc:  # not enough data etc.
                    log.warning("skip block %s: %s", block_start.date(), exc)
        block_start = block_end
    if not out:
        return pd.DataFrame()
    res = pd.concat(out)
    return res


def _run_one(args):
    league, df, factory, test_start, test_end, refit_days, min_train = args
    res = walk_forward_league(df, factory, test_start, test_end, refit_days, min_train)
    if not res.empty:
        res["league"] = league
    return res


def walk_forward(
    matches: pd.DataFrame,
    factory: Callable[[], MatchModel],
    test_start,
    test_end=None,
    refit_days: int = 7,
    min_train: int = 300,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Walk-forward predictions for all leagues in ``matches`` (each league modelled separately).

    Returns one row per test match: match_id, league, model probabilities, fit_date.
    """
    jobs = [(lg, g, factory, test_start, test_end, refit_days, min_train) for lg, g in matches.groupby("league")]
    if n_jobs != 1 and len(jobs) > 1:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs)(delayed(_run_one)(j) for j in jobs)
    else:
        results = [_run_one(j) for j in jobs]
    results = [r for r in results if r is not None and not r.empty]
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def attach_outcomes(preds: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Join walk-forward predictions with match info, results, odds and reference probabilities."""
    df = preds.merge(frame.drop(columns=[c for c in ("league",) if c in frame.columns and c in preds.columns]),
                     left_on="match_id", right_on="id", how="left", suffixes=("", "_m"))
    df["y_1x2"] = np.where(df["fthg"] > df["ftag"], 0, np.where(df["fthg"] == df["ftag"], 1, 2))
    df["y_over"] = ((df["fthg"] + df["ftag"]) > 2.5).astype(int)
    return df
