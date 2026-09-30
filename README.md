# NFL Injury Analytics

Where key NFL players and top fantasy picks get hurt, how often, and how long
they're out: six completed seasons (2020–2025) plus the season in progress.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/fetch_data.py      # nflverse + ADP -> data/raw/
.venv/bin/python src/fetch_awards.py    # Pro Bowl / All-Pro rosters (Wikipedia) -> data/raw/awards.csv
.venv/bin/python src/build_dataset.py   # player-seasons, absences, fantasy picks -> data/processed/
.venv/bin/python src/build_dashboard.py # summaries + dashboard/index.html
```

## Outputs

| File | What it is |
|---|---|
| `data/processed/player_seasons.csv` | Every player-team-season with a snap, tagged with each key-player criterion (snap share, 1st-round rookie, salary percentile, prior-season Pro Bowl / All-Pro) |
| `data/processed/injury_episodes.csv` | One row per injury absence for any player: body region, start week, games missed, IR, season-ending |
| `data/processed/fantasy_top48.csv` / `fantasy_episodes.csv` | Top-48 PPR ADP picks per season and their fantasy-week absences |
| `data/processed/summary_*.csv` | Definition comparison, % injured by side, same-region recurrence, fantasy by season / position / round / team |
| `dashboard/index.html` | Interactive dashboard with Starters and Fantasy tabs |

Definitions are documented at the top of `src/build_dataset.py`.

## Keeping the current season up to date

`fetch_data.py` always re-downloads the current season (`CURRENT_SEASON` in
`src/fetch_data.py` and `src/build_dataset.py`), so re-running the four commands above
picks up the latest week. nflverse usually posts new injury reports and snap counts a
day or two after each game week. When a new season starts, bump `CURRENT_SEASON` in
`fetch_data.py`, `build_dataset.py` and `build_dashboard.py`.
