# Research report — Phase 1 decisions and the evidence behind them

_Session 1, 25 Nov 2026. All numbers are reproducible: `sportpred update` → `sportpred evaluate` →
`python scripts/league_screening.py` → `python scripts/research_tables.py` (outputs in `reports/`)._

## Executive summary

* **Sport:** football (soccer). It is the only sport with **free, long, clearly-sourced historical
  odds including closing prices** across many leagues — the precondition for honest testing.
* **Markets:** match result (1X2) first, over/under 2.5 goals second.
* **Leagues:** 20 European leagues screened; second tiers and smaller leagues show the most
  opportunities (Serie B, National League, League One, Belgian Pro League, Championship, Bundesliga).
  12 leagues are enabled by default.
* **Models:** Dixon-Coles, Elo, LightGBM, a market blend and an ensemble were compared walk-forward
  on **53,459 out-of-sample matches (Jul 2019 – Nov 2026)**. No model beat the closing market;
  standalone models lost 4–11% when betting on their own "value". The only consistently positive
  signal was **market-consensus value**: a bookmaker offering a price above the sharp de-vigged consensus.
* **Answer to the research question:** we can build well-calibrated models, but they do **not** find
  value by out-predicting the market. Positive expected value in this data comes from **price
  differences between bookmakers**. It showed positive closing line value in every season and in
  all 20 leagues. The realised profit is borderline significant, and that profit is
  practically limited (prices move, bookmakers limit winning accounts).
* **Recommendation:** run the app in **paper-trading mode** with the market-consensus model for at
  least three months before risking any money. Expected data cost to start: **$0**.

---

## 1. AI coding tool

| Tool | Strengths for this project | Weaknesses |
|---|---|---|
| **Claude Code** (terminal, IDE, desktop, web/cloud) | Works across the whole repository; runs commands and tests itself and fixes what fails; reads `CLAUDE.md` every session, so context survives between sessions; cloud sessions need no local setup | Long autonomous runs cost usage; you review results after the fact rather than line by line |
| **OpenAI Codex** (CLI + cloud) | Comparable agentic workflow; reads `AGENTS.md` | Similar trade-offs; switching mid-project costs re-learning |
| **Cursor** (AI code editor) | Excellent when you want to watch and approve every edit inside an editor; good for small targeted changes | Encourages fragment-by-fragment editing; less natural for "run the 20-minute evaluation and analyse it" |

**Recommendation: Claude Code**, because this project needs long multi-step work — downloading data,
running walk-forward evaluations, fixing failures and writing reports. In session 1 it built and
tested the whole application (55 tests) from an empty repository. Use Cursor or VS Code optionally just
to browse files. The repository contains both `CLAUDE.md` and `AGENTS.md`, so you can switch tools
without losing project rules or context.

**Exact way of working:** start each session with *"Read DEVELOPMENT_STATUS.md and continue with the
next step"*. End each session with *"update DEVELOPMENT_STATUS.md, run the tests, commit and push"*.

## 2. Technology stack

Python everywhere: pandas/NumPy/SciPy, scikit-learn + LightGBM, SQLAlchemy (SQLite by default,
PostgreSQL via Docker), FastAPI, Streamlit, Typer, Docker compose. Rationale and the alternatives
considered are in [`ARCHITECTURE.md`](ARCHITECTURE.md). The key trade-off: **Streamlit instead of
React/Next.js**. A single language is far easier to maintain for a non-programmer, and the REST API
already exists if a React front end is wanted later.

## 3. Which sport? (data availability decides)

| Sport | Historical odds incl. closing, free & clearly sourced | Depth | Verdict |
|---|---|---|---|
| **Football** | football-data.co.uk: ~10 bookmakers + market max/avg, pre-closing **and closing**, 1X2/OU/AH, 22 leagues | 1993→ (odds 2000→, closing 2012→/2019→) | **Chosen** |
| Tennis | tennis-data.co.uk: results + odds (ATP ~2001→, WTA 2007→), one snapshot | 20+ years | Best second candidate (roadmap) |
| NBA/NFL/MLB/NHL | closing lines in scattered archives with unclear licences; reliable feeds are paid | varies | Not without a data budget; markets are very efficient |
| Others (handball, esports…) | patchy | — | Not considered reliable |

Without closing prices you cannot measure closing line value, which is the most reliable short-sample
test of a betting edge. Football is the only sport where that is free at scale.

## 4. Method

* **Data:** 20 leagues × 2012/13–2026/27 from football-data.co.uk (105,547 finished matches; 2.49 M odds rows).
* **Walk-forward:** Dixon-Coles and Elo refitted **weekly** from Jul 2015; GBM refitted **monthly** from
  Jul 2017; pooled models refit **per season** on earlier out-of-sample rows only. Test period
  **1 Jul 2019 – 23 Nov 2026**; all models scored on identical matches.
* **Benchmarks:** the pre-closing sharp price (what you can see when you bet) and the **closing**
  price (the market's final, most informed estimate), both de-vigged.
* **Betting rules for backtests:** bet at **pre-closing** prices only; flat 1-unit stakes; odds 1.5–5.0;
  one bet per match and market; EV threshold 3% unless stated. CLV is measured against the closing
  reference (Pinnacle → Betfair Exchange → market average).
* **Uncertainty:** 95% bootstrap CIs (match-day blocks for yields).

## 5. Results

### 5.1 Probability quality (log loss; lower is better)

| Model | 1X2 log loss | Δ vs closing (95% CI) | 1X2 accuracy | O/U log loss | Δ vs closing |
|---|---|---|---|---|---|
| Closing market (benchmark) | 0.9966 | — | 50.8% | 0.6737 | — |
| **Ensemble** (DC+Elo+GBM+market) | 1.0000 | +0.0034 (+0.0026…+0.0041) | 50.5% | 0.6761 | +0.0024 |
| **Blend** (DC+market) | 1.0001 | +0.0035 (+0.0027…+0.0042) | 50.5% | 0.6761 | +0.0024 |
| Market, pre-closing | 1.0003 | +0.0037 (+0.0030…+0.0044) | 50.4% | 0.6762 | +0.0024 |
| Gradient boosting (LightGBM) | 1.0128 | +0.0162 (+0.0148…+0.0175) | 49.3% | 0.6810 | +0.0073 |
| Elo | 1.0191 | +0.0224 (+0.0208…+0.0241) | 49.0% | — | — |
| Dixon-Coles | 1.0233 | +0.0267 (+0.0249…+0.0286) | 48.7% | 0.6916 | +0.0178 |

* **Nothing beats the closing market.** That is expected: the closing price aggregates the most information.
* **Complex beats simple among standalone models:** GBM (form, shots, xG, rest, Elo, DC features)
  improves on Dixon-Coles by 0.0105 log loss — but it is still far from the market.
* **Pooling adds a sliver over the pre-closing market:** blend −0.00026 (CI −0.00041…−0.00011),
  ensemble −0.00035 (CI −0.00056…−0.00013) for 1X2; nothing for over/under.
* **Revealing weights:** the blend gives Dixon-Coles a **negative** weight (−0.08) and the market
  1.11. When the model disagrees with the market, the market tends to be even more right (the model
  lacks injury, transfer and line-up news). In the ensemble only GBM gets a positive weight (+0.12).
* **Calibration** (expected calibration error): market 0.7%, GBM 0.4–1.2%, Elo 0.6–0.7%, Dixon-Coles
  1.2–1.7% for 1X2 but **4.0% for over/under** (poorly calibrated). Accuracy alone would have hidden this.

### 5.2 Betting on each model's "value" (EV ≥ 3%, flat stakes, 2019–2026)

| Strategy | Price source | Bets | Yield (95% CI) | Avg CLV (95% CI) | Beat close |
|---|---|---|---|---|---|
| Dixon-Coles 1X2 | market average | 31,241 | −11.0% (−12.6…−9.4) | −7.3% | 19% |
| Dixon-Coles 1X2 | best named bookmaker | 39,145 | −5.9% (−7.4…−4.4) | −2.9% | 36% |
| Elo 1X2 | best named bookmaker | 38,475 | −5.7% (−7.1…−4.3) | −3.1% | 34% |
| GBM 1X2 | best named bookmaker | 37,938 | −4.4% (−5.8…−2.9) | −2.0% | 39% |
| Dixon-Coles O/U | best named bookmaker | 33,214 | −4.6% (−5.7…−3.5) | −3.1% | 29% |
| **Market consensus 1X2** | **best named bookmaker** | **2,382** | **+6.6% (+0.5…+12.9)** | **+3.3% (+2.9…+3.8)** | **67%** |
| Market consensus 1X2, robust rule¹ | best named bookmaker | 1,468 | +5.4% (−2.2…+12.9) | +4.7% (+4.2…+5.2) | — |
| Market consensus 1X2 | Bet365 only | 904 | +5.9% (−4.7…+16.5) | +1.7% (+1.1…+2.4) | 61% |
| Market consensus 1X2 | market maximum | 5,184 | +3.6% (−0.5…+8.0) | +3.3% (+3.1…+3.5) | 71% |
| Blend 1X2 | best named bookmaker | 4,353 | +3.5% (−0.7…+7.9) | +1.8% (+1.6…+2.1) | 61% |
| Ensemble 1X2 | best named bookmaker | 6,906 | +2.7% (−0.7…+6.1) | +1.3% (+1.1…+1.5) | 57% |
| Market consensus O/U | best named bookmaker | 142 | +14.3% (−4.8…+34.0) | +5.2% (+3.6…+6.9) | 72% |
| Baselines (all matches) | best named bookmaker | ~53,000 | home −3.5%, draw −2.2%, away −4.5%, favourite −1.4%, random −2.8% | — | — |

¹ The live rule: EV must stay positive even if the sharp price moves by its typical pre-close amount.
Full table: `reports/pooled_backtests.csv`.

**Interpretation.** A model that "finds value" in a third of all matches is not finding value; it is
disagreeing with a better forecaster. The signal that survives is simpler. The de-vigged sharp price
is the best available probability, so look for a bookmaker paying more than that. Kaunitz, Zhong &
Kreiner (2017) report the same mechanism. They also document that bookmakers limit accounts that
exploit it.

### 5.3 Stability over time and across leagues (market consensus, best named bookmaker, 1X2)

| Season | 19/20 | 20/21 | 21/22 | 22/23 | 23/24 | 24/25 | 25/26 | 26/27 (part) |
|---|---|---|---|---|---|---|---|---|
| Bets | 366 | 349 | 413 | 226 | 197 | 488 | 296 | 47 |
| Avg CLV | +3.5% | +2.3% | +4.7% | +4.1% | +3.8% | +3.3% | +1.9% | +1.8% |
| CLV 95% CI low | +2.5% | +1.1% | +3.5% | +2.8% | +2.3% | +2.4% | +1.0% | −1.1% |
| Flat yield | +4.7% | +1.0% | +15.9% | +5.4% | +1.9% | +4.7% | +11.2% | −4.3% |

* CLV is positive in every season, and its CI excludes zero in 7 of 8. Realised yield is far noisier,
  which is why CLV is the main yardstick.
* **The edge shrank after Pinnacle disappeared** (2025/26→). The anchor became the de-vigged market
  average, and the CLV reference became the Betfair Exchange closing price. With only ~340 bets since,
  the recent figure is uncertain.
* **By league** (`reports/market_consensus_by_league.csv`): average CLV was positive in all 20 leagues,
  with the CI above zero in 15. It was highest in the Scottish Premiership (+5.2%, only 30 bets), Serie B
  (+5.0%), National League (+4.7%), Belgian Pro League (+4.4%), League One (+4.2%), Bundesliga
  (+4.1%) and Championship (+4.0%). It was lowest in the Eredivisie (+0.4%), Ligue 1 (+1.4%) and
  Premier League (+1.9%; only 29 bets in 7 seasons). Lower and second tiers offer more opportunities: more
  price dispersion, less bookmaker attention. These per-league figures are not corrected for multiple
  comparisons; the pooled result is the primary evidence.

### 5.4 Which price should anchor "fair"? (`reports/anchor_comparison.csv`)

| Era | Anchor | Bets | Avg CLV |
|---|---|---|---|
| 2019–2025 (Pinnacle available) | Pinnacle | 2,058 | +3.6% (s.e. 0.2%) |
| | de-vigged market average | 1,027 | +4.8% (s.e. 0.4%) |
| | Betfair Exchange snapshot | 648 | +1.4% (s.e. 0.4%) |
| 2025–2027 (Pinnacle gone) | Pinnacle (partial coverage) | 191 | +2.1% (s.e. 0.5%) |
| | de-vigged market average | 292 | +0.9% (s.e. 0.6%) |
| | Betfair Exchange snapshot | 388 | −0.3% (s.e. 0.4%) |

The provider's Tuesday/Friday exchange snapshot is a poor anchor in both eras, probably because of thin
early liquidity and back prices only. The app therefore anchors on Pinnacle → market average → Exchange.
This choice was made **after** seeing the 2025–27 data, a researcher degree of freedom. It is
supported by the earlier era too, but paper trading must confirm it.

### 5.5 League and market selection

* **Default leagues (12):** Premier League, Championship, League One, League Two, National League,
  Bundesliga, 2. Bundesliga, Serie A, Serie B, La Liga, Segunda División, Belgian Pro League. That is the
  strongest evidence plus enough volume. The Premier League is kept for interest, but expect very few
  opportunities there. Greece, Turkey, Ligue 2 and the Scottish Premiership also showed positive CLV and
  can be enabled in Model Settings.
* **Markets:** 1X2 primary. Over/under 2.5 has a tighter market and very few opportunities
  (142 in 7 years), so it stays secondary. Asian handicap odds are stored but not yet modelled: it is
  the sharpest football market and the least likely to offer dispersion value.

### 5.6 Staking and risk (market consensus, best named bookmaker, 2,382 bets)

Monte Carlo simulations (500 runs) re-play the same bets, drawing outcomes as if the **closing market's
probabilities were the truth**. This is a conservative stand-in for reality. Starting bankroll: 1,000.

| Staking | Backtest max drawdown | P(profit) | P(lose half the bankroll) | Median worst drawdown | Median final bankroll |
|---|---|---|---|---|---|
| Flat 1% of starting bankroll | 21% | 86% | 6% | 37% | 1,757 |
| ¼ Kelly, capped (2%/bet, 3%/match, 10%/day) | 20% | 93% | 0% | 30% | 2,096 |
| ½ Kelly, uncapped | 34% | 90% | 3% | **54%** | 5,276 |

* Even with a genuine edge, drawdowns of 30–50% are normal over a few thousand bets. Positive
  expected value does not protect you from long losing runs: the longest backtest streak was 16 losses.
* The compounding backtests look spectacular on paper (¼ Kelly +402% over 7 years) only because they
  assume unlimited stakes at the best price, which no bookmaker allows. Treat them as illustration, not
  forecast.
* The app offers flat stakes and **fractional** Kelly with exposure caps only. Full Kelly is
  deliberately not offered.

## 6. Data providers and costs

| Need | Provider | Cost | Status |
|---|---|---|---|
| History, results, stats, odds (incl. closing), fixtures with odds | football-data.co.uk | **free** | connected |
| Fresher odds closer to kick-off | The Odds API | free 500 credits/month; reported ~$30/month for 20k | optional (key) |
| Injuries / line-ups | API-Football | free 100 req/day; reported ~$19/month Pro | not connected (low priority) |
| A sharp live anchor (Pinnacle-like) | third-party re-sellers (Pinnacle closed its public API on 23 Jul 2025) | reported from ~$99/month | not recommended yet |

**Start: $0/month.** Step up to The Odds API (~$30/month) only if paper trading confirms positive CLV
and you need prices at the moment you would bet. Details: [`DATA_SOURCES.md`](DATA_SOURCES.md).

## 7. Independent critical review (threats to validity)

1. **Can the prices really be had?** Best prices are one snapshot per match (Tue/Fri). They may have
   lasted minutes, been limited to small stakes, or been palpable errors that get voided.
   *Mitigation:* CLV against the closing price, and an implausible-margin filter. *Remaining risk:* high.
2. **Account limits.** A strategy that systematically takes the best price at soft bookmakers gets
   accounts restricted. Kaunitz et al. report this within months. Real-world capacity is small.
3. **Multiple testing.** We tried 6 models × 5 price sources × 3 thresholds × 2 markets. The best realised
   yield (+6.6%) has a CI barely above zero and **would not survive** a multiple-comparison correction.
   The CLV evidence is far stronger: t ≈ 15, positive in every season and every league. It is the more
   credible signal, but CLV is not profit.
4. **Researcher degrees of freedom.** The anchor (§5.4) and robustness band were chosen with sight of the
   data. Only forward paper trading is free of this.
5. **Changing market.** Pinnacle's data disappeared; the most recent seasons show a smaller edge.
6. **Reference consistency.** CLV is measured against Pinnacle closing until 2025/26 and against
   Betfair Exchange closing afterwards, so the eras are not perfectly comparable.
7. **Not included:** exchange commission is modelled at 2%; taxes on winnings (country-specific) and
   bookmaker promotions/limits are not.
8. **Model blind spots.** No injuries, line-ups, cup fixtures or motivation data. This is the most likely
   reason the fundamental models lose to the market.
9. **Overfitting checks done.** Walk-forward only; a future-poisoning test; simple-vs-complex comparison
   (complex GBM beats simple DC on log loss, but neither beats the market); EV-threshold sensitivity
   (Backtesting page); calibration curves; random and naive baselines.

**Conclusion:** there is **no evidence that the football models can beat the betting market**. There
is consistent evidence of **positive closing line value from price shopping against a sharp
consensus**. Its realised profitability is statistically fragile and practically constrained.
Positive EV does not guarantee profit.

## 8. Phased implementation plan

See [`ROADMAP.md`](ROADMAP.md). Phases 1–3 are delivered; Phases 4–5 are partly delivered.

## 9. Your exact first steps

1. Get the code (see `README.md` → Quick start), or continue in a Claude Code cloud session using the
   **"sport"** environment, which allows football-data.co.uk.
2. `sportpred update`, then `sportpred evaluate` (≈25 min), then `streamlit run app/main.py`.
3. Read **Model Performance** and **Backtesting** yourself and try the price sources and thresholds.
4. Let `sportpred daily` run (cron or Docker scheduler) for **3 months of paper trading**. Each week,
   check Model Performance → *Paper trading*. Go/no-go: is average CLV positive after ≥ 300 settled bets?
5. Only then consider The Odds API key (fresher odds) and small real stakes, with flat or ≤ ¼-Kelly
   stakes and the exposure caps in Model Settings.

## Sources

Web (checked Nov 2026; vendor pages are marketing — verify prices yourself):
[football-data.co.uk notes](https://www.football-data.co.uk/notes.txt) ·
[The Odds API](https://the-odds-api.com/) ·
[OddsPapi: The Odds API free-tier limits](https://oddspapi.io/blog/the-odds-api-free-tier-limits) ·
[sportsapis.dev: The Odds API review](https://sportsapis.dev/apis/the-odds-api) ·
[API-Football pricing](https://www.api-football.com/pricing) ·
[football-data.org API policies](https://docs.football-data.org/general/v4/policies.html) ·
[TheStatsAPI: football-data.org free-tier limits](https://www.thestatsapi.com/blog/football-data-org-free-tier-limits-2026) ·
[Pinnacle API documentation (closure notice)](https://github.com/pinnacleapi/pinnacleapi-documentation) ·
[odds-api.io: Pinnacle API shutdown alternatives](https://odds-api.io/blog/pinnacle-api-shutdown-alternatives) ·
[Betfair Data Scientists: historical data sources](https://betfair-datascientists.github.io/modelling/dataSources) ·
[betcode FAQ: using data](https://betcode-org.github.io/faq/using-data) ·
[Tennis Abstract: free ATP/WTA databases](https://www.tennisabstract.com/blog/2015/03/24/free-atp-and-wta-results-and-stats-databases/) ·
[SportsOddsHistory: about](https://www.sportsoddshistory.com/about-us) ·
[Princeton DSS: historical sports odds database](https://dss.princeton.edu/catalog/resource6741)

Academic (Tier 2):
* Dixon, M. & Coles, S. (1997). Modelling association football scores and inefficiencies in the football betting market. *Applied Statistics* 46(2).
* Kaunitz, L., Zhong, S. & Kreiner, J. (2017). Beating the bookies with their own numbers — and how the online sports betting market is rigged. arXiv:1710.02824.
* Constantinou, A. & Fenton, N. (2012). Solving the problem of inadequate scoring rules for assessing probabilistic football forecast models. *JQAS* 8(1).
* Štrumbelj, E. (2014). On determining probability forecasts from betting odds. *International Journal of Forecasting* 30(4).
* Hubáček, O., Šourek, G. & Železný, F. (2019). Exploiting sports-betting market using machine learning. *International Journal of Forecasting* 35(2).
* Walsh, C. & Joshi, A. (2024). Machine learning for sports betting: should model selection be based on accuracy or calibration? *Machine Learning with Applications* 16.
