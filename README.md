# NFL Injury Analytics

Where key NFL players and top fantasy picks get hurt, how often, and how long
they're out: six completed seasons (2020–2025) plus the season in progress.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/fetch_data.py      # nflverse + ADP -> data/raw/
.venv/bin/python src/fetch_awards.py    # Pro Bowl / All-Pro rosters (Wikipedia) -> data/raw/awards.csv
.venv/bin/python src/build_dataset.py   # player-seasons, absences, fantasy picks -> data/processed/
.venv/bin/python src/injury_model.py    # train/evaluate injury-risk models, score the live week
.venv/bin/python src/build_dashboard.py # summaries + dashboard/index.html
.venv/bin/python src/audit_teams.py     # optional: team-by-team check vs. raw rosters and injury reports
```

## Outputs

| File | What it is |
|---|---|
| `data/processed/player_seasons.csv` | Every player-team-season with a snap, tagged with each key-player criterion (snap share, 1st-round rookie, salary percentile, prior-season Pro Bowl / All-Pro) |
| `data/processed/injury_episodes.csv` | One row per injury absence for any player: body region, start week, games missed, IR, season-ending |
| `data/processed/fantasy_top48.csv` / `fantasy_episodes.csv` | Top-48 PPR ADP picks per season and their fantasy-week absences |
| `data/processed/audit_team_injuries.csv` | Every injured player-week from raw rosters/reports, and whether the pipeline captured it |
| `data/processed/summary_*.csv` | Definition comparison, % injured by side, same-region recurrence, fantasy by season / position / round / team |
| `data/processed/injury_risk_live.csv` | This week's risk scores for the top-125 PPR players, with backup and hedge advice |
| `reports/injury_model_report.md` | Model evaluation: metrics, leave-one-season-out results, feature importance, hedge assumptions |
| `models/*.joblib` | Trained scikit-learn pipelines (next-game risk, baseline risk, availability) |
| `dashboard/index.html` | Interactive dashboard with Starters and Fantasy tabs |

Definitions are documented at the top of `src/build_dataset.py`.

## Keeping the current season up to date

`fetch_data.py` always re-downloads the current season (`CURRENT_SEASON` in
`src/fetch_data.py` and `src/build_dataset.py`), so re-running the four commands above
picks up the latest week. nflverse usually posts new injury reports and snap counts a
day or two after each game week. When a new season starts, bump `CURRENT_SEASON` in
`fetch_data.py`, `build_dataset.py` and `build_dashboard.py`.

## Injury models and lineup risk

`src/injury_model.py` trains three models on 2013–2022, tunes on 2023–2024, tests on 2025,
and scores the current week. The population is every QB/RB/WR/TE startable in a
12-team PPR league (top 12 QB, 30 RB, 36 WR, 12 TE by PPR per game); live scores also
cover deeper rosterable players. Its docstring explains the design; results are in
`reports/injury_model_report.md`.

- **Next week:** will a player who played last week miss the next game?
  Leave-one-season-out ROC-AUC ~0.77. At 40%+ predicted, about half of players missed.
- **Injury report:** will a Questionable/Doubtful player sit? ROC-AUC ~0.93.
- **Time out:** once out, how many games? C-index ~0.85; within one game ~82% of the time.
- Beyond next week, healthy players carry their position's league rate (no model tested
  could separate them further out).
- **My Lineup & Risk tab:** enter a QB/RB/RB/WR/WR/TE/FLEX lineup to see the chance a
  starter misses time, expected starter-games and PPR points at risk over four weeks, a
  grade vs. 2,000 random lineups, and the weakest links with time-out forecasts and backups.
