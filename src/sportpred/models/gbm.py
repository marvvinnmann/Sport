"""Gradient boosting (LightGBM) on engineered pre-match features — the 'complex' challenger.

Features (all known before kick-off):
* exponentially weighted form: goals, shots, shots on target, xG (when available), points
* rest days since the previous league match, matches played
* pre-match Elo ratings
* Dixon-Coles walk-forward probabilities and expected goals (stacking)

The model is pooled across leagues (league is a categorical feature) and re-fitted every
``refit_days`` on all earlier rows. Hyper-parameters are deliberately conservative (shallow trees,
large leaves, strong L2) because football outcomes are noisy and overfitting is the main risk.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from sportpred.features.elo import elo_by_league
from sportpred.features.form import pre_match_form

log = logging.getLogger(__name__)

LGB_PARAMS = dict(n_estimators=250, learning_rate=0.03, num_leaves=12, min_child_samples=150, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=10.0, verbose=-1, n_jobs=4)
DC_FEATURES = ["dc_p_H", "dc_p_D", "dc_p_A", "dc_p_OVER", "dc_xg_home", "dc_xg_away"]


def build_features(matches: pd.DataFrame, dc_preds: pd.DataFrame | None) -> pd.DataFrame:
    """One row per match in ``matches`` (finished or not) with all model features."""
    fin = matches[(matches["status"] == "finished") & matches["fthg"].notna()]
    parts = []
    for _, g in matches.groupby("league"):
        parts.append(pre_match_form(g, fin[fin["league"] == g["league"].iloc[0]]))
    form = pd.concat(parts).reindex(matches.index)
    elo = elo_by_league(matches)
    X = form.copy()
    X["elo_home"] = elo["elo_home"]
    X["elo_away"] = elo["elo_away"]
    X["elo_diff"] = X["elo_home"] - X["elo_away"]
    X["league_cat"] = matches["league"].astype("category")
    X["match_id"] = matches["id"].to_numpy()
    X["date"] = pd.to_datetime(matches["date"]).to_numpy()
    if dc_preds is not None and not dc_preds.empty:
        d = dc_preds.set_index("match_id")[["p_H", "p_D", "p_A", "p_OVER", "xg_home", "xg_away"]]
        d.columns = DC_FEATURES
        X = X.join(d, on="match_id")
    return X


def feature_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in ("match_id", "date")]


def _fit_predict(Xtr: pd.DataFrame, ytr: np.ndarray, Xte: pd.DataFrame, objective: str) -> np.ndarray:
    import lightgbm as lgb

    if objective == "multiclass":
        m = lgb.LGBMClassifier(objective="multiclass", **LGB_PARAMS)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xte)
    m = lgb.LGBMClassifier(objective="binary", **LGB_PARAMS)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def walk_forward_gbm(X: pd.DataFrame, y_1x2: pd.Series, y_over: pd.Series, start, end=None,
                     train_from=None, refit_days: int = 30,
                     progress: Callable[[str], None] | None = None) -> pd.DataFrame:
    """Walk-forward GBM predictions for rows with date in [start, end]."""
    X = X.copy()
    dates = pd.to_datetime(X["date"])
    start = pd.Timestamp(start)
    end = pd.Timestamp(end) if end is not None else dates.max()
    cols = feature_columns(X)
    has_y = y_1x2.notna()
    usable = has_y & (dates >= pd.Timestamp(train_from)) if train_from is not None else has_y
    if any(c in X.columns for c in DC_FEATURES):
        usable &= X["dc_p_H"].notna()
    out = []
    block = start
    n_blocks = 0
    while block <= end:
        nxt = block + pd.Timedelta(days=refit_days)
        te = (dates >= block) & (dates < nxt) & has_y
        if any(c in X.columns for c in DC_FEATURES):
            te &= X["dc_p_H"].notna()
        tr = usable & (dates < block)
        if te.any() and tr.sum() >= 2000:
            p = _fit_predict(X.loc[tr, cols], y_1x2[tr].astype(int).to_numpy(), X.loc[te, cols], "multiclass")
            po = _fit_predict(X.loc[tr, cols], y_over[tr].astype(int).to_numpy(), X.loc[te, cols], "binary")
            part = pd.DataFrame({"match_id": X.loc[te, "match_id"].to_numpy(), "p_H": p[:, 0], "p_D": p[:, 1],
                                 "p_A": p[:, 2], "p_OVER": po, "p_UNDER": 1 - po, "fit_date": block,
                                 "n_train": int(tr.sum())})
            out.append(part)
            n_blocks += 1
            if progress and n_blocks % 12 == 0:
                progress(f"    GBM refit {n_blocks} (up to {block.date()})")
        block = nxt
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def gbm_feature_importance(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    import lightgbm as lgb

    cols = feature_columns(X)
    ok = y.notna()
    m = lgb.LGBMClassifier(objective="multiclass", **{**LGB_PARAMS, "importance_type": "gain"})
    m.fit(X.loc[ok, cols], y[ok].astype(int))
    imp = pd.DataFrame({"feature": cols, "gain": m.feature_importances_})
    imp["share"] = imp["gain"] / imp["gain"].sum()
    return imp.sort_values("gain", ascending=False).reset_index(drop=True)


def build_stacked_model(name, ms):  # pragma: no cover - live use goes through services.predictions
    raise NotImplementedError(
        "GBM/ensemble are evaluated with `sportpred evaluate`; live predictions use Dixon-Coles or the "
        "market blend (see docs/MODEL_CARD.md)."
    )
