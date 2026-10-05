"""Team-by-team audit: compare the pipeline's injury absences against an independent
reconstruction from raw weekly rosters and injury reports.

Ground truth (built from raw files only): for each team-season, players the team would
care about = last season's snap-share starters (any team) or this season's top-30% cap
hits on the team's roster. A player-week counts as "out injured" when he was on an
injury reserve list (IR, IR-return, PUP, NFI), or listed on that week's injury report
with an injury (illness and non-injury reasons excluded, as in the pipeline) and did
not take a snap. Every such week is then looked up in data/processed/injury_episodes.csv.

Run: .venv/bin/python src/audit_teams.py [season ...]   (default: 2025 2026)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import IR_CODES, OUT, RAW, body_region, build_crosswalk, load  # noqa: E402

FANTASY_OR_STARTER = {"QB", "RB", "FB", "WR", "TE", "T", "G", "C", "OT", "OG", "OL", "DE", "DT", "NT", "DL",
                      "LB", "ILB", "OLB", "MLB", "CB", "S", "FS", "SS", "DB"}


def ground_truth(season: int, xw: pd.DataFrame, ps: pd.DataFrame) -> pd.DataFrame:
    ros = load("roster_weekly", [season]); ros = ros[ros["game_type"] == "REG"].dropna(subset=["pfr_id"])
    inj = load("injuries", [season]); inj = inj[inj["game_type"] == "REG"].merge(xw, on="gsis_id", suffixes=("", "_x"))
    snaps = load("snap_counts", [season]); snaps = snaps[snaps["game_type"] == "REG"]
    played = set(zip(snaps["team"], snaps["pfr_player_id"], snaps["week"]))
    team_weeks = snaps.groupby("team")["week"].apply(set).to_dict()

    # who the team cares about, computed from raw sources
    prior_key = set(ps.loc[(ps.season == season - 1) & ps.is_snap_key, "pfr_player_id"])
    contracts = pd.read_parquet(RAW / "contracts.parquet")
    contracts["pfr"] = contracts["gsis_id"].map(xw.set_index("gsis_id")["pfr_id"])
    cap = {}
    for pid, hist in zip(contracts["pfr"], contracts["season_history"]):
        if pd.isna(pid) or hist is None:
            continue
        for h in hist:
            if str(h.get("year")) == str(season):
                cap[pid] = max(cap.get(pid, 0), h.get("cap_number") or 0)
    roster_players = ros[ros["position"].isin(FANTASY_OR_STARTER)][["team", "pfr_id", "full_name", "position"]].drop_duplicates(["team", "pfr_id"])
    roster_players["cap"] = roster_players["pfr_id"].map(cap).fillna(0)
    roster_players["cap_rank"] = roster_players.groupby("team")["cap"].rank(pct=True, method="first")
    cares = roster_players[(roster_players["cap_rank"] > 0.7) | roster_players["pfr_id"].isin(prior_key)]
    cares = cares.assign(reason_key=np.where(cares["pfr_id"].isin(prior_key), "prior-season starter", "top-30% cap"))

    rows = []
    reserve = ros[ros["status_description_abbr"].isin(IR_CODES)]
    res_keys = set(zip(reserve["team"], reserve["pfr_id"], reserve["week"]))
    # same injury rule as the pipeline: illness and non-injury listings don't count
    rep = inj[inj["report_primary_injury"].map(body_region).notna() | inj["practice_primary_injury"].map(body_region).notna()]
    rep_keys = set(zip(rep["team"], rep["pfr_id"], rep["week"]))
    for c in cares.itertuples(index=False):
        for w in sorted(team_weeks.get(c.team, [])):
            if (c.team, c.pfr_id, w) in played:
                continue
            on_res, on_rep = (c.team, c.pfr_id, w) in res_keys, (c.team, c.pfr_id, w) in rep_keys
            if on_res or on_rep:
                rows.append({"season": season, "team": c.team, "pfr_player_id": c.pfr_id, "player": c.full_name,
                             "position": c.position, "week": w, "evidence": "reserve list" if on_res else "injury report",
                             "why_key": c.reason_key, "cap": round(c.cap, 1)})
    return pd.DataFrame(rows)


def main(seasons: list[int]) -> None:
    xw = build_crosswalk()
    ps = pd.read_csv(OUT / "player_seasons.csv")
    ep = pd.read_csv(OUT / "injury_episodes.csv")
    covered = {(r.season, r.team, r.pfr_player_id, w) for r in ep.itertuples(index=False)
               for w in range(r.start_week, r.end_week + 1)}
    ps_keys = set(zip(ps.season, ps.team, ps.pfr_player_id))
    snaps_all = load("snap_counts", seasons)
    first_play = snaps_all.groupby(["season", "team", "pfr_player_id"])["week"].min().to_dict()
    gt = pd.concat([ground_truth(s, xw, ps) for s in seasons], ignore_index=True)
    gt["captured"] = [(r.season, r.team, r.pfr_player_id, r.week) in covered for r in gt.itertuples(index=False)]

    def why(r):
        if r.captured:
            return "captured"
        k = (r.season, r.team, r.pfr_player_id)
        if k not in ps_keys:
            return "no player-season row (never played, not on reserve at Week 1)"
        fp = first_play.get(k)
        if fp is not None and r.week < fp:
            return "before his first game (not on reserve at Week 1)"
        return "missed week not linked to an absence"
    gt["status"] = gt.apply(why, axis=1)
    gt.to_csv(OUT / "audit_team_injuries.csv", index=False)

    print(f"Player-weeks out injured (ground truth): {len(gt):,} | captured {gt.captured.mean():.1%}")
    print(gt["status"].value_counts().to_string())
    by_team = (gt.groupby(["season", "team"]).agg(weeks=("captured", "size"), captured=("captured", "mean"))
                 .reset_index().sort_values("captured"))
    print("\nLowest-coverage teams:"); print(by_team.head(8).round(3).to_string(index=False))
    miss = gt[~gt.captured].groupby(["season", "team", "player", "position", "status"]).agg(weeks=("week", "size"), first=("week", "min"), cap=("cap", "max")).reset_index()
    print("\nLargest uncaptured cases:"); print(miss.sort_values(["weeks", "cap"], ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [2025, 2026])
