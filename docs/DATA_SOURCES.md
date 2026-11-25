# Data sources

Status legend: **connected** = implemented and used; **optional** = implemented, needs your key;
**not connected** = documented only. Nothing in this project pretends a source is connected when it is not.

## 1. football-data.co.uk — connected (primary source)

| | |
|---|---|
| Provider | football-data.co.uk (independent site, free CSV downloads) |
| URL pattern | `https://football-data.co.uk/mmz4281/{season}/{league}.csv` (e.g. `2324/E0.csv`); upcoming fixtures `https://football-data.co.uk/fixtures.csv` |
| Coverage | 22 "main" European leagues (England 5 tiers, Scotland 4, Germany 2, Italy 2, Spain 2, France 2, Netherlands, Belgium, Portugal, Turkey, Greece). 20 of them are used in the research screen; 12 are enabled by default |
| Content | full-time/half-time results; shots, shots on target, corners, fouls, cards (most leagues); **expected goals (xG) from 2026/27**; 1X2, over/under 2.5 and Asian handicap odds from ~10 bookmakers + market maximum and average; **pre-closing and closing** odds |
| Historical depth | results since 1993/94 for top leagues; odds since ~2000; Pinnacle closing 1X2 since 2012/13; full closing odds since 2019/20. We load 2012/13 onwards |
| Update frequency | current-season files a few times per week; `fixtures.csv` around Tuesday and Friday |
| Odds timing | pre-closing odds collected **Friday afternoon** (weekend games) and **Tuesday afternoon** (midweek). Closing = just before kick-off |
| Cost | free |
| API limits | none published; we pause 1 s between requests and cache past seasons forever |
| Licensing | free to download for personal/research use; the site acknowledges its own sources (BBC, Flashscore, Oddsportal, bookmakers). **We do not redistribute raw files** (the `data/` folder is git-ignored). Check the site's terms before any commercial use |
| Known weaknesses | **Pinnacle columns stop after 2025/26** (Pinnacle closed its public API on 23 July 2025); some seasons curtailed (COVID 2019/20); occasional shot-count inconsistencies; one odds snapshot per match (no full price history); fixtures file lists only the next few days; cup/European matches are not included |
| Bookmakers by era | 2012–19: Bet365, Bwin, Interwetten, Ladbrokes, Pinnacle, William Hill, VC Bet (+BetBrain max/avg). 2019–25: + closing prices, 1xBet, Betfair sportsbook & Exchange. 2025/26: + Betfred, BetMGM, Coral, Ladbrokes. 2026/27: Bet365, Betfred, BetVictor, Bwin, Paddy Power, Sky Bet, Betfair Exchange (no Pinnacle) |

Provenance in the app: every file import writes a `data_updates` row with the file's SHA-256 hash,
row counts and warnings; every match row stores `source`, `source_file` and `data_label`.

## 2. The Odds API — optional (implemented, needs key)

| | |
|---|---|
| Provider | the-odds-api.com, REST API v4 |
| Coverage | 70+ sports; for football the major European leagues (mapping in `src/sportpred/data/odds_api.py`); h2h (1X2), totals, spreads from UK/EU/US/AU bookmakers |
| Historical depth | live and upcoming odds; historical snapshots are a paid feature (10× credit cost per call) |
| Update frequency | near real-time |
| Cost (third-party summaries, Nov 2026 — verify on the official pricing page) | free: 500 credits/month; paid tiers reported at about $30/month (20k credits), $59 (100k), up to $249 |
| Credit usage here | 1 credit per market per region per league per refresh → `h2h,totals` × 2 regions = 4 credits per league |
| Licensing | per provider terms; personal use |
| Known weaknesses | Pinnacle prices, where offered, are reported to come from Pinnacle's public website and may lag; team names need mapping (unmatched names are reported in the update log, never silently dropped) |
| How to enable | put `SPORTPRED_ODDS_API_KEY=...` in `.env` (see README). Status shows on the Data Management page |

## 3. Player availability (injuries, suspensions, line-ups) — not connected

No free, reliable, licensable source was found. Candidates if you decide to pay:

| Provider | What | Price (third-party summaries, Nov 2026 — verify) | Notes |
|---|---|---|---|
| API-Football (api-sports.io) | fixtures, injuries, line-ups, odds, stats | free 100 requests/day; Pro ~$19/mo (7,500/day); Ultra ~$29/mo | best value; injuries endpoint exists; data quality varies by league |
| Sportmonks | fixtures, line-ups, injuries, xG, odds | from ~€29–€99/mo depending on leagues | broad coverage, more expensive |
| football-data.org | fixtures/results for 12 competitions | free tier 10 calls/min; paid from ~€12–€49/mo | no injuries on free tier; no odds |

Every prediction lists *"Injuries, suspensions and line-ups are NOT included"* under missing information.

## 4. Other sports investigated (not built)

| Sport | Best free/cheap data with odds | Assessment |
|---|---|---|
| Tennis (ATP/WTA) | tennis-data.co.uk — results + bookmaker odds (ATP since ~2001, WTA since 2007), free CSV/XLS; Jeff Sackmann's ATP/WTA match databases (stats, no odds; non-commercial licence) | **Best second candidate.** Elo-style player ratings are well studied; one odds snapshot per match. Planned as Phase 6 |
| NBA / NFL / MLB / NHL | closing lines exist in scattered archives (SportsOddsHistory, university databases, Kaggle) with unclear licences; reliable feeds are paid (SportsDataIO, The Odds API historical) | Markets are very efficient; free historical **closing** odds with clear licensing are hard to find. Not recommended without a paid data budget |
| Betfair Exchange historical | Basic tier free (1-minute last-traded prices, no volume); Advanced/Pro paid | Useful later for exchange-based backtests; Betfair starting prices don't exist for football |

## 5. Sources deliberately avoided

* Tipster sites, "sure bet" services, unverified predictions.
* Scraping sites whose terms forbid it (e.g. odds-comparison sites); paid "Pinnacle replacement"
  feeds that scrape Pinnacle's website (unclear legality and reliability).
* Any dataset without clear provenance, and any generated/simulated data presented as real.
  Simulated data exists **only** inside `tests/` and is labelled `simulated` in the database.
