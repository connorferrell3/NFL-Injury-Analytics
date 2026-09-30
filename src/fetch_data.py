"""Download raw data: nflverse (injuries, snaps, rosters, players, draft, contracts)
and fantasy ADP.

Sources:
  https://github.com/nflverse/nflverse-data (open data, updated weekly in-season)
  https://fantasyfootballcalculator.com/api/v1/adp (PPR, 12-team, preseason drafts; 2020-2025)
  https://api.sleeper.app/projections (PPR ADP; current season, since FFC only serves
  in-season drafts once the season starts)
Files are cached to data/raw/; the current season is re-downloaded on every run so the
dashboard picks up new weeks. Use --force to refresh everything.
Pro Bowl / All-Pro rosters come from src/fetch_awards.py.
"""
import argparse
import io
from pathlib import Path

import pandas as pd
import requests

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
ADP_URL = "https://fantasyfootballcalculator.com/api/v1/adp/ppr?teams=12&year={season}&position=all"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
CURRENT_SEASON = 2026  # in progress; always re-downloaded
SEASONS = range(2020, CURRENT_SEASON + 1)  # 6 completed seasons + the current one
SNAP_SEASONS = range(2019, CURRENT_SEASON + 1)  # +1 prior season to identify returning starters
MODEL_SEASONS = range(2012, CURRENT_SEASON + 1)  # longer history for the injury model (2012 = first snap counts)
GAMES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"  # schedule: rest days, surface
SLEEPER_URL = ("https://api.sleeper.app/projections/nfl/{season}?season_type=regular"
               "&position[]=QB&position[]=RB&position[]=WR&position[]=TE&order_by=adp_ppr")

DATASETS = {
    "injuries": "injuries/injuries_{season}.parquet",
    "snap_counts": "snap_counts/snap_counts_{season}.parquet",
    "weekly_rosters": "weekly_rosters/roster_weekly_{season}.parquet",
    "stats_player": "stats_player/stats_player_week_{season}.parquet",  # workload + PPR points
}
SINGLE_FILES = {
    "players.parquet": "players/players.parquet",
    "draft_picks.parquet": "draft_picks/draft_picks.parquet",
    "contracts.parquet": "contracts/historical_contracts.parquet",
}


def download(url: str, dest: Path, force: bool) -> None:
    if dest.exists() and not force:
        return
    # requests bundles certifi, avoiding macOS python.org SSL cert issues with urllib
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    pd.read_parquet(io.BytesIO(resp.content))  # validate before writing
    dest.write_bytes(resp.content)
    print(f"  saved {dest.relative_to(RAW.parent.parent)}")


def download_adp(force: bool) -> None:
    dest = RAW / "adp.csv"
    if dest.exists() and not force:
        return
    frames = []
    for season in SEASONS:
        if season == CURRENT_SEASON:
            frames.append(sleeper_adp(season))
            continue
        resp = requests.get(ADP_URL.format(season=season), timeout=60)
        resp.raise_for_status()
        body = resp.json()
        df = pd.DataFrame(body["players"]).assign(season=season, source="Fantasy Football Calculator")
        frames.append(df[["season", "name", "position", "team", "adp", "adp_formatted", "source"]])
    pd.concat(frames).to_csv(dest, index=False)
    print(f"  saved {dest.relative_to(RAW.parent.parent)}")


def sleeper_adp(season: int) -> pd.DataFrame:
    resp = requests.get(SLEEPER_URL.format(season=season), timeout=60)
    resp.raise_for_status()
    rows = []
    for x in resp.json():
        p, adp = x.get("player") or {}, (x.get("stats") or {}).get("adp_ppr")
        if adp and p.get("position") in {"QB", "RB", "WR", "TE"}:
            rows.append({"season": season, "name": f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
                         "position": p["position"], "team": x.get("team"), "adp": adp})
    df = pd.DataFrame(rows).sort_values("adp")
    rnd = (df["adp"].round().clip(lower=1) - 1).astype(int)
    df["adp_formatted"] = [f"{r // 12 + 1}.{r % 12 + 1:02d}" for r in rnd]
    return df.assign(source="Sleeper")


def main(force: bool = False) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, pattern in DATASETS.items():
        for season in MODEL_SEASONS:
            rel = pattern.format(season=season)
            download(f"{BASE}/{rel}", RAW / Path(rel).name, force or season == CURRENT_SEASON)
    # season-end rosters before 2020 mark IR players as RES (weekly rosters don't), used to
    # recover IR placements that never appeared on an injury report
    for season in range(MODEL_SEASONS[0], 2020):
        download(f"{BASE}/rosters/roster_{season}.parquet", RAW / f"roster_season_{season}.parquet", force)
    for fname, rel in SINGLE_FILES.items():
        download(f"{BASE}/{rel}", RAW / fname, force)
    download_adp(True)  # small; refreshed each run so the current season stays current
    resp = requests.get(GAMES_URL, timeout=120)
    resp.raise_for_status()
    (RAW / "games.csv").write_bytes(resp.content)  # refreshed each run (scores fill in weekly)
    print("Raw data ready in data/raw/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    main(ap.parse_args().force)
