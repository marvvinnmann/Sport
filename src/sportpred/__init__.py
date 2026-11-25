"""sportpred — transparent sports prediction & value-bet research platform.

The package is organised in layers:

* ``sportpred.data``        — downloading, parsing, storing and checking data
* ``sportpred.features``    — pre-match features (ratings, form, rest days)
* ``sportpred.models``      — probability models (market, Elo, Dixon-Coles, gradient boosting)
* ``sportpred.betting``     — odds maths: implied probability, margin, EV, Kelly
* ``sportpred.evaluation``  — metrics, walk-forward validation and backtesting
* ``sportpred.services``    — high level jobs used by the CLI, API and dashboard

The application runs in prediction / paper-trading mode only. It never places bets.
"""

__version__ = "0.1.0"
