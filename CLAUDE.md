# Instructions for AI coding agents (Claude Code, Codex, Cursor)

This file is read automatically by Claude Code. `AGENTS.md` (same content) is read by OpenAI Codex
and Cursor. Read `DEVELOPMENT_STATUS.md` first — it says what is done, what is next and known bugs.

## Project in one paragraph
`sportpred` is a football (soccer) prediction and value-bet **research** platform. It downloads free
results/odds from football-data.co.uk, fits probability models (Dixon-Coles, Elo, LightGBM, and a
market blend), validates them walk-forward against bookmaker closing prices, and shows predictions,
backtests and paper-trading results in a Streamlit dashboard and a FastAPI API. It never places bets.

## Commands
```bash
source .venv/bin/activate        # or prefix with .venv/bin/
pytest                           # full test suite (~2-3 min); must pass before every commit
ruff check src app tests scripts # lint
sportpred update                 # download data
sportpred evaluate               # walk-forward evaluation (slow: ~25 min for 20 leagues)
sportpred predict                # live predictions
streamlit run app/main.py        # dashboard
sportpred serve-api              # API at :8000/docs
```

## Non-negotiable rules
1. **No look-ahead.** Anything used to predict a match must be known before kick-off. Use walk-forward
   splits only, never random train/test splits. Bets are struck at pre-closing prices; closing prices
   only measure CLV. Keep `tests/test_models.py::test_walk_forward_has_no_lookahead` passing.
2. **No fabricated data.** Simulated data lives only in `tests/` and must be labelled `simulated`.
   Never present sample/simulated numbers as real results. Never claim an API is connected if it is not.
3. **Report honestly.** If a change makes results worse, say so. A strategy is not "profitable" unless the
   confidence interval excludes zero and CLV is positive. Update `docs/MODEL_CARD.md` with new results.
4. **Never silently remove functionality.** Explain architectural changes before making them.
5. **Secrets** only in `.env` (git-ignored). Never commit keys or the `data/` folder.
6. **Paper trading only.** Do not add code that places bets or logs into bookmaker accounts.
7. After finishing a piece of work, update `DEVELOPMENT_STATUS.md` (done / next / known bugs).

## Conventions
* Library code in `src/sportpred/`; the dashboard (`app/`) and API (`src/sportpred/api/`) call
  services in `src/sportpred/services/` — do not duplicate calculations in UI code.
* Odds are decimal everywhere. Probability columns: `p_H, p_D, p_A, p_OVER, p_UNDER`.
* Market codes: `1X2`, `OU25` (predictions) / `1X2`, `OU`, `AH` (stored odds).
* Pandas ≥ 3: `.to_numpy()` may return read-only arrays — copy before in-place edits.
* Charts: palette and helpers in `app/common.py` (`SERIES`, `style_fig`); one y-axis per chart.
* New data providers go in `src/sportpred/data/` and must write a `DataUpdate` audit row.
