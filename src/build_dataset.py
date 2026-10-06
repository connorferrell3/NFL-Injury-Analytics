"""Build the analysis dataset: player-seasons, injury absences, and fantasy top-48 picks.

Absences are computed for EVERY player who took a regular-season snap, and each
player-team-season is tagged with the criteria of each key-player definition, so
definitions can be compared on identical absence logic.

Definitions
-----------
Key player: snap-share definition (`is_snap_key`)
    An offensive or defensive player (K/P/LS excluded) who, for a given team-season,
    averaged >= 65% of his unit's snaps (50% for RBs, who rotate more) in the
    regular-season games he played (min. 2 games) -- OR was a key player the previous
    season and averaged >= 50% (35% RB) this season or played < 4 games. The second
    clause keeps starters hurt in the first weeks.

Key player: pedigree definition (`is_pedigree_key`) -- any of:
    - first-round pick still on his rookie contract (draft year through year 5, ending
      early if he signed an extension or a new deal before the season)
    - cap hit in the top 30% of his position group that season (`salary_top30`)
    Also tagged: `team_cap_top30`, the top 30% of each team-season's players by cap hit
    (an equal-share group per team, used for team comparisons).
    - Pro Bowl or All-Pro selection the PREVIOUS season (same-season honors would be
      biased toward players who stayed healthy)
    - named to the NFL Top 100 released before the season (`top100_prev`; player-voted
      on the previous season, so it has the same timing as the honors above)

Fantasy top 48
    Top 48 by PPR ADP (12-team drafts just before the season). Absences are counted
    from Week 1 through the fantasy season (Week 16 in 2020, Week 17 after).

Injury episode
    A run of consecutive team regular-season games a key player missed while on the
    team's roster, where at least one missed week has injury evidence:
    an injury-related listing on the official injury report, or an injury reserve
    designation on the weekly roster (IR R01 / R48, PUP R04, non-football injury R05).
    Runs ending in release/trade are cut at the last week the player was on the
    team's roster. Absences are counted from the week he joined the team's roster, not his
    first game, so a camp injury (on IR from Week 1, or simply listed Out) counts even if
    he never takes a snap.

Duration
    Games missed in the episode. Body part comes from injury-report entries during the
    absence, else the week before it, else the first 3 weeks after return (players sent
    straight to IR often appear on the report only once they resume practice).
    Episodes still open at the team's final regular-season game are flagged
    `out_at_season_end` (duration is right-censored); those that also involved IR are
    flagged `season_ending`. In the current season, "end" is the latest week played, so
    those absences are flagged `ongoing` (currently out) instead of season-ending. Week 18 one-game absences often reflect playoff teams
    resting starters under minor injury tags -- treat with care.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
CURRENT_SEASON = 2026  # in progress
SEASONS = list(range(2020, CURRENT_SEASON + 1))
SNAP_SEASONS = list(range(2019, CURRENT_SEASON + 1))
SALARY_TOP = 0.30  # top share of each position group's cap hits counted as key
FANTASY_TOP_N = 48
NAME_ALIASES = {"camjordan": "cameronjordan",  # award / ADP name -> roster name
                "hollywoodbrown": "marquisebrown", "robbiechosen": "robbyanderson", "robbieanderson": "robbyanderson",
                "kennygainwell": "kennethgainwell"}  # applied to both sides, so either spelling matches

# roster team -> words that identify it in award "team" text ("Buffalo", "LA Rams", "Buffalo Bills"),
# used to tell apart players who share a name (Josh Allen BUF QB / JAX edge)
TEAM_WORDS = {"ARI": "Arizona|Cardinals", "ATL": "Atlanta|Falcons", "BAL": "Baltimore|Ravens", "BUF": "Buffalo|Bills",
              "CAR": "Carolina|Panthers", "CHI": "Chicago|Bears", "CIN": "Cincinnati|Bengals", "CLE": "Cleveland|Browns",
              "DAL": "Dallas|Cowboys", "DEN": "Denver|Broncos", "DET": "Detroit|Lions", "GB": "Green Bay|Packers",
              "HOU": "Houston|Texans", "IND": "Indianapolis|Colts", "JAX": "Jacksonville|Jaguars", "KC": "Kansas City|Chiefs",
              "LA": "Rams", "LAC": "Chargers", "LV": "Las Vegas|Oakland|Raiders", "OAK": "Las Vegas|Oakland|Raiders",
              "MIA": "Miami|Dolphins", "MIN": "Minnesota|Vikings", "NE": "New England|Patriots", "NO": "New Orleans|Saints",
              "NYG": "Giants", "NYJ": "Jets", "PHI": "Philadelphia|Eagles", "PIT": "Pittsburgh|Steelers",
              "SEA": "Seattle|Seahawks", "SF": "San Francisco|49ers", "TB": "Tampa Bay|Buccaneers", "TEN": "Tennessee|Titans",
              "WAS": "Washington|Commanders|Football Team"}

KEY_MIN_GAMES = 2
KEY_PCT = {"RB": 0.50}  # default 0.65
RETURNING_KEY_PCT = {"RB": 0.35}  # default 0.50
RETURN_LOOKAHEAD = 3
# injury reserve lists: Reserve/Injured, IR - designated for return, PUP, non-football injury
IR_CODES = {"R01", "R48", "R04", "R05"}

POS_GROUP = {
    "QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
    "T": "OL", "OT": "OL", "G": "OL", "OG": "OL", "C": "OL", "OL": "OL",
    "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL",
    "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB",
    "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB", "DB": "DB",
}
OFFENSE = {"QB", "RB", "WR", "TE", "OL"}

# keyword -> body region. The earliest keyword in the label wins, so
# "toe, pec, knee, hip" maps to Foot / Toe.
REGION_KEYWORDS = {
    "concussion": "Head / Concussion", "head": "Head / Concussion", "eye": "Head / Concussion",
    "face": "Head / Concussion", "jaw": "Head / Concussion", "tooth": "Head / Concussion",
    "neck": "Neck", "stinger": "Neck", "throat": "Neck",
    "shoulder": "Shoulder", "collarbone": "Shoulder", "clavicle": "Shoulder",
    "chest": "Chest / Ribs", "pec": "Chest / Ribs", "rib": "Chest / Ribs", "sternum": "Chest / Ribs",
    "elbow": "Arm / Elbow", "bicep": "Arm / Elbow", "tricep": "Arm / Elbow",
    "forearm": "Arm / Elbow", "arm": "Arm / Elbow",
    "hand": "Hand / Wrist", "wrist": "Hand / Wrist", "finger": "Hand / Wrist", "thumb": "Hand / Wrist",
    "back": "Back",
    "abdom": "Core / Abdomen", "oblique": "Core / Abdomen", "core": "Core / Abdomen",
    "hernia": "Core / Abdomen", "kidney": "Core / Abdomen", "lung": "Core / Abdomen",
    "appendi": "Core / Abdomen", "spleen": "Core / Abdomen",
    "hip": "Hip / Groin", "groin": "Hip / Groin", "pelvis": "Hip / Groin",
    "glute": "Hip / Groin", "adductor": "Hip / Groin",
    "hamstring": "Hamstring",
    "quad": "Quad / Thigh", "thigh": "Quad / Thigh",
    "knee": "Knee", "acl": "Knee", "mcl": "Knee",
    "calf": "Lower Leg", "shin": "Lower Leg", "fibula": "Lower Leg",
    "tibia": "Lower Leg", "lower leg": "Lower Leg",
    "achilles": "Achilles",
    "ankle": "Ankle",
    "foot": "Foot / Toe", "toe": "Foot / Toe", "heel": "Foot / Toe",
}


def body_region(label) -> str | None:
    """Map a free-text injury label to a body region; None for illness/non-injury."""
    if not isinstance(label, str):
        return None
    s = label.strip().lower()
    if s.startswith("not injury"):
        return None
    hits = [(s.find(k), -len(k), r) for k, r in REGION_KEYWORDS.items() if k in s]
    return min(hits)[2] if hits else None


def load(prefix: str, seasons) -> pd.DataFrame:
    df = pd.concat(pd.read_parquet(RAW / f"{prefix}_{y}.parquet") for y in seasons)
    if prefix == "roster_weekly":
        # 25-50% of weekly-roster rows lack pfr_id; backfill it through gsis_id
        players = pd.read_parquet(RAW / "players.parquet", columns=["gsis_id", "pfr_id"]).dropna()
        df["pfr_id"] = df["pfr_id"].fillna(df["gsis_id"].map(players.drop_duplicates("gsis_id").set_index("gsis_id")["pfr_id"]))
    return df


def norm_name(name) -> str:
    """'A. J. Brown Jr.' -> 'ajbrown' for cross-source name matching."""
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", str(name).lower())
    return re.sub(r"[^a-z]", "", s)


def build_crosswalk() -> pd.DataFrame:
    """gsis_id <-> pfr_id from the players table, backfilled from weekly rosters."""
    players = pd.read_parquet(RAW / "players.parquet", columns=["gsis_id", "pfr_id"])
    rosters = load("roster_weekly", SEASONS)[["gsis_id", "pfr_id"]]
    xw = pd.concat([players, rosters]).dropna().drop_duplicates("pfr_id")
    return xw.drop_duplicates("gsis_id")


def player_seasons(snaps: pd.DataFrame) -> pd.DataFrame:
    """One row per player-team-season for everyone who took an off/def snap."""
    s = snaps.copy()
    s["pos_group"] = s["position"].map(POS_GROUP)
    s = s.dropna(subset=["pos_group"])
    s["side"] = np.where(s["pos_group"].isin(OFFENSE), "Offense", "Defense")
    s["unit_pct"] = np.where(s["side"] == "Offense", s["offense_pct"], s["defense_pct"])
    agg = (s.groupby(["season", "team", "pfr_player_id"])
             .agg(player=("player", "first"),
                  pos_group=("pos_group", lambda x: x.mode().iat[0]),
                  games=("week", "nunique"),
                  first_week=("week", "min"),
                  avg_unit_pct=("unit_pct", "mean"))
             .reset_index())
    agg["side"] = np.where(agg["pos_group"].isin(OFFENSE), "Offense", "Defense")
    return agg


def add_preplay_absences(ps: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    """Count absences from the week a player joined the team's roster, not from his first
    game, so injuries before he first plays (camp injuries, "Out" in Week 1, mid-season
    signings who arrive hurt) are captured. Injury evidence is still required for every
    absence, so healthy scratches and inactive rookies don't count. Players with injury
    evidence who never took a snap get a games = 0 row so they can qualify as key players
    (prior-season role, salary, honors)."""
    ros = load("roster_weekly", SEASONS)
    ros = ros[ros["game_type"] == "REG"].dropna(subset=["pfr_id"])
    joined = ros.groupby(["season", "team", "pfr_id"])["week"].min()
    keys = list(zip(ps["season"], ps["team"], ps["pfr_player_id"]))
    jw = pd.Series([joined.get(k, np.nan) for k in keys], index=ps.index)
    start = (jw - 1).where(jw.notna(), ps["first_week"])
    moved = int((start < ps["first_week"]).sum())
    ps["first_week"] = np.minimum(ps["first_week"], start).astype(int)

    # never played but had injury evidence while on the team (reserve list or injury report)
    inj = load("injuries", SEASONS)
    inj = inj[(inj["game_type"] == "REG") & inj["report_primary_injury"].map(body_region).notna()]
    xw = build_crosswalk().set_index("gsis_id")["pfr_id"]
    hurt = set(zip(inj["season"], inj["team"], inj["gsis_id"].map(xw)))
    hurt |= set(map(tuple, ros.loc[ros["status_description_abbr"].isin(IR_CODES), ["season", "team", "pfr_id"]].to_numpy()))
    have = set(keys)
    info = ros.sort_values("week").drop_duplicates(["season", "team", "pfr_id"])
    info = info[[k in hurt and k not in have for k in zip(info["season"], info["team"], info["pfr_id"])]]
    info = info.assign(pos_group=info["position"].map(POS_GROUP)).dropna(subset=["pos_group"])
    add = pd.DataFrame({"season": info["season"], "team": info["team"], "pfr_player_id": info["pfr_id"],
                        "player": info["full_name"], "pos_group": info["pos_group"], "games": 0,
                        "first_week": (info["week"] - 1).astype(int), "avg_unit_pct": np.nan})
    add["side"] = np.where(add["pos_group"].isin(OFFENSE), "Offense", "Defense")
    print(f"Absences now counted from roster arrival: {moved:,} player-seasons start earlier; "
          f"{len(add):,} injured players who never took a snap added")
    return pd.concat([ps, add], ignore_index=True)


def tag_snap_key(ps: pd.DataFrame) -> pd.DataFrame:
    key_pct = ps["pos_group"].map(KEY_PCT).fillna(0.65)
    returning_pct = ps["pos_group"].map(RETURNING_KEY_PCT).fillna(0.50)
    strict = (ps["avg_unit_pct"] >= key_pct) & (ps["games"] >= KEY_MIN_GAMES)
    prior = set(zip(ps.loc[strict, "season"] + 1, ps.loc[strict, "pfr_player_id"]))
    returning = pd.Series([(y, p) in prior for y, p in zip(ps["season"], ps["pfr_player_id"])],
                          index=ps.index)
    returning &= (ps["avg_unit_pct"] >= returning_pct) | (ps["games"] < 4)
    ps["is_snap_key"] = strict | returning
    ps["key_via"] = np.select([strict, returning], ["snap share", "returning starter"], "")
    return ps


def tag_pedigree(ps: pd.DataFrame, xw: pd.DataFrame) -> pd.DataFrame:
    to_pfr = xw.set_index("gsis_id")["pfr_id"]

    # first-round pick on rookie contract
    draft = pd.read_parquet(RAW / "draft_picks.parquet")
    r1 = draft[draft["round"] == 1].dropna(subset=["pfr_player_id"])
    r1_year = r1.set_index("pfr_player_id")["season"]
    contracts = pd.read_parquet(RAW / "contracts.parquet")
    contracts["pfr_player_id"] = contracts["gsis_id"].map(to_pfr)
    new_deal = {}  # pfr -> earliest year a non-rookie contract was signed
    for pid, hist in zip(contracts["pfr_player_id"], contracts["contract_history"]):
        if pd.isna(pid) or hist is None:
            continue
        for c in hist:
            if c.get("contract_type") != "Drafted" and c.get("year_signed"):
                new_deal[pid] = min(new_deal.get(pid, 9999), int(c["year_signed"]))
    dy = ps["pfr_player_id"].map(r1_year)
    nd = ps["pfr_player_id"].map(new_deal).fillna(9999)
    ps["first_round_rookie"] = dy.notna() & (ps["season"] >= dy) & (ps["season"] <= dy + 4) & (nd > ps["season"] - 1)

    # cap hit percentile within season x position group (players who took snaps)
    caps = [(pid, int(h["year"]), h.get("cap_number") or 0)
            for pid, hist in zip(contracts["pfr_player_id"], contracts["season_history"])
            if pd.notna(pid) and hist is not None for h in hist if str(h.get("year")).isdigit()]
    cap = (pd.DataFrame(caps, columns=["pfr_player_id", "season", "cap_hit"])
             .groupby(["pfr_player_id", "season"])["cap_hit"].max())
    ps["cap_hit"] = [cap.get((p, y), 0.0) for p, y in zip(ps["pfr_player_id"], ps["season"])]
    best = ps.sort_values("games").drop_duplicates(["season", "pfr_player_id"], keep="last")
    best["cap_pct_rank"] = best.groupby(["season", "pos_group"])["cap_hit"].rank(pct=True)
    ps = ps.merge(best[["season", "pfr_player_id", "cap_pct_rank"]], on=["season", "pfr_player_id"])
    ps["salary_top30"] = ps["cap_pct_rank"] > 1 - SALARY_TOP
    team_rank = ps.groupby(["season", "team"])["cap_hit"].rank(pct=True, method="first")
    ps["team_cap_top30"] = team_rank > 1 - SALARY_TOP

    # previous-season Pro Bowl / All-Pro / Top 100, matched by name on that season's rosters
    awards = pd.read_csv(RAW / "awards.csv")
    names = load("roster_weekly", SNAP_SEASONS)[["season", "full_name", "pfr_id", "team"]].dropna()
    names["n"] = names["full_name"].map(norm_name).replace(NAME_ALIASES)
    names = names.drop_duplicates(["season", "n", "pfr_id", "team"])
    by_name = dict(tuple(names.groupby("n")))

    def resolve(season: int, n: str, team_text) -> str | None:
        """Player id for an honoree: that season's roster, then any season; team text breaks name ties."""
        c = by_name.get(n)
        if c is None:
            return None
        if (c["season"] == season).any():
            c = c[c["season"] == season]
        if c["pfr_id"].nunique() > 1 and isinstance(team_text, str):
            on_team = c[[bool(re.search(TEAM_WORDS.get(t, "^$"), team_text)) for t in c["team"]]]
            if on_team["pfr_id"].nunique() == 1:
                return on_team["pfr_id"].iat[0]
        return c["pfr_id"].iat[0]

    awards["n"] = awards["name"].map(norm_name).replace(NAME_ALIASES)
    awards["pfr_player_id"] = [resolve(y, n, t) for y, n, t in zip(awards["season"], awards["n"], awards["team_text"])]
    matched = awards.dropna(subset=["pfr_player_id"])
    print(f"Awards: matched {matched['n'].nunique()} names; "
          f"unmatched non-position strings: {awards.loc[awards['pfr_player_id'].isna(), 'name'].nunique()}")
    for award in ("pro_bowl", "all_pro", "top100"):
        got = set(zip(matched.loc[matched["award"] == award, "season"] + 1,
                      matched.loc[matched["award"] == award, "pfr_player_id"]))
        ps[f"{award}_prev"] = [(y, p) in got for y, p in zip(ps["season"], ps["pfr_player_id"])]

    top = matched[matched["award"] == "top100"]
    print(f"Top 100: matched {len(top)} of {(awards['award'] == 'top100').sum()} list entries")
    ps["is_pedigree_key"] = ps[["first_round_rookie", "salary_top30", "pro_bowl_prev", "all_pro_prev", "top100_prev"]].any(axis=1)
    return ps


def fantasy_top48(xw: pd.DataFrame, top_n: int = FANTASY_TOP_N) -> pd.DataFrame:
    """Top-N (default 48) ADP players matched to pfr ids and their Week 1 team."""
    adp = pd.read_csv(RAW / "adp.csv").sort_values(["season", "adp"])
    adp = adp[adp["position"].isin(["QB", "RB", "WR", "TE"])]
    adp["adp_rank"] = adp.groupby("season").cumcount() + 1
    adp = adp[adp["adp_rank"] <= top_n].copy()
    adp["n"] = adp["name"].map(norm_name).replace(NAME_ALIASES)

    ros = load("roster_weekly", SEASONS)
    ros = ros[ros["game_type"] == "REG"].dropna(subset=["pfr_id"])
    ros["n"] = ros["full_name"].map(norm_name).replace(NAME_ALIASES)
    first = ros.sort_values("week").drop_duplicates(["season", "pfr_id"])  # Week 1 (or first) team
    cand = first[["season", "n", "position", "pfr_id", "team"]].rename(columns={"team": "wk1_team"})
    m = adp.merge(cand, on=["season", "n"], how="left", suffixes=("", "_ros"))
    # break name collisions on position, then team
    m["score"] = (m["position"] == m["position_ros"]).astype(int) * 2 + (m["team"] == m["wk1_team"]).astype(int)
    m = m.sort_values("score", ascending=False).drop_duplicates(["season", "adp_rank"])
    missing = m[m["pfr_id"].isna()]
    if len(missing):
        print("Fantasy: unmatched ADP names:", missing[["season", "name"]].values.tolist())
    m = m.dropna(subset=["pfr_id"]).rename(columns={"pfr_id": "pfr_player_id", "name": "player"})
    m["pos_group"] = m["position"]
    m["side"] = "Offense"
    m["first_week"] = 0  # count absences from Week 1, even if he never played
    m["max_week"] = np.where(m["season"] == 2020, 16, 17)
    m["team"] = m["wk1_team"]
    return m[["season", "adp_rank", "adp", "adp_formatted", "source", "player", "pos_group", "side", "team",
              "pfr_player_id", "first_week", "max_week"]].sort_values(["season", "adp_rank"])


def injury_evidence(xw: pd.DataFrame, seasons=SEASONS) -> pd.DataFrame:
    """Per (season, week, pfr_player_id): injury-report region and/or IR flag."""
    inj = load("injuries", seasons)
    inj = inj[inj["game_type"] == "REG"].copy()
    inj["region"] = inj["report_primary_injury"].map(body_region)
    # fall back to the practice report when the game-status report has no body part
    inj["region"] = inj["region"].fillna(inj["practice_primary_injury"].map(body_region))
    inj["raw_label"] = inj["report_primary_injury"].fillna(inj["practice_primary_injury"])
    inj = inj.dropna(subset=["region"])
    inj = inj.merge(xw, on="gsis_id")[["season", "week", "pfr_id", "region", "raw_label",
                                       "report_status"]]

    ros = load("roster_weekly", seasons)
    ros = ros[(ros["game_type"] == "REG") & ros["status_description_abbr"].isin(IR_CODES)]
    ir = ros[["season", "week", "pfr_id"]].dropna().drop_duplicates().assign(on_ir=True)

    ev = inj.drop_duplicates(["season", "week", "pfr_id"]).merge(
        ir, on=["season", "week", "pfr_id"], how="outer")
    ev["on_ir"] = ev["on_ir"].fillna(False).astype(bool)
    return ev.rename(columns={"pfr_id": "pfr_player_id"})


def roster_weeks(seasons=SEASONS) -> pd.DataFrame:
    """Last REG week each player was on each team's roster (any status)."""
    ros = load("roster_weekly", seasons)
    ros = ros[ros["game_type"] == "REG"].dropna(subset=["pfr_id"])
    return (ros.groupby(["season", "team", "pfr_id"])["week"].max()
               .rename("last_roster_week").reset_index()
               .rename(columns={"pfr_id": "pfr_player_id"}))


def find_episodes(key: pd.DataFrame, snaps: pd.DataFrame, ev: pd.DataFrame,
                  last_roster: pd.DataFrame) -> pd.DataFrame:
    """Absence spells for each row of `key` (season, team, pfr_player_id, first_week,
    optional max_week). Weeks after first_week and up to max_week are scanned."""
    team_weeks = (snaps.groupby(["season", "team"])["week"]
                       .apply(lambda w: sorted(w.unique())).to_dict())
    played = snaps.groupby(["season", "team", "pfr_player_id"])["week"].apply(set).to_dict()
    ev_idx = ev.set_index(["season", "pfr_player_id", "week"]).sort_index()
    ev_keys = set(ev_idx.index)
    last_roster = last_roster.set_index(["season", "team", "pfr_player_id"])["last_roster_week"]
    has_max = "max_week" in key.columns

    rows = []
    for r in key.itertuples(index=False):
        weeks = team_weeks.get((r.season, r.team))
        if not weeks:
            continue
        season_end = min(max(weeks), r.max_week) if has_max else max(weeks)
        cutoff = min(last_roster.get((r.season, r.team, r.pfr_player_id), max(weeks)), season_end)
        timeline = [w for w in weeks if r.first_week < w <= cutoff]
        played_w = played.get((r.season, r.team, r.pfr_player_id), set())

        # group consecutive missed team games into spells
        spells, cur = [], []
        for w in timeline:
            if w in played_w:
                if cur:
                    spells.append(cur)
                cur = []
            else:
                cur.append(w)
        if cur:
            spells.append(cur)

        for spell in spells:
            key_ = (r.season, r.pfr_player_id)
            hits = [(*key_, w) for w in spell if (*key_, w) in ev_keys]
            if not hits:
                continue  # no injury evidence: benching, suspension, personal, etc.
            spell_ev = ev_idx.loc[hits]

            labeled = spell_ev.dropna(subset=["region"])
            # IR only: look back one week, then ahead to the return weeks
            fallback = [spell[0] - 1] + list(range(spell[-1] + 1, spell[-1] + 1 + RETURN_LOOKAHEAD))
            for w in fallback:
                if not labeled.empty:
                    break
                if (*key_, w) in ev_keys and pd.notna(ev_idx.loc[(*key_, w), "region"]):
                    labeled = ev_idx.loc[[(*key_, w)]]
            region = labeled["region"].iat[0] if not labeled.empty else "Unspecified (IR)"
            raw = labeled["raw_label"].iat[0] if not labeled.empty else None
            on_ir = bool(spell_ev["on_ir"].any())

            rows.append({
                "season": r.season, "team": r.team, "pfr_player_id": r.pfr_player_id,
                "player": r.player, "pos_group": r.pos_group, "side": r.side,
                "start_week": spell[0], "end_week": spell[-1],
                "games_missed": len(spell),
                "games_remaining": sum(1 for w in weeks if spell[0] <= w <= season_end),
                "body_region": region, "injury_label": raw,
                "went_on_ir": on_ir,
                "out_at_season_end": spell[-1] == season_end,
                "season_ending": spell[-1] == season_end and on_ir and r.season != CURRENT_SEASON,
                "ongoing": spell[-1] == season_end and r.season == CURRENT_SEASON,
            })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    snaps = load("snap_counts", SNAP_SEASONS)
    snaps = snaps[snaps["game_type"] == "REG"]
    xw = build_crosswalk()

    ps = tag_snap_key(add_preplay_absences(player_seasons(snaps), snaps))
    ps = ps[ps["season"].isin(SEASONS)].copy()
    ps = tag_pedigree(ps, xw)

    ev = injury_evidence(xw)
    snaps_cur = snaps[snaps["season"].isin(SEASONS)]
    last_roster = roster_weeks()
    episodes = find_episodes(ps, snaps_cur, ev, last_roster)

    fantasy = fantasy_top48(xw)
    fantasy_eps = find_episodes(fantasy, snaps_cur, ev, last_roster)

    ps.to_csv(OUT / "player_seasons.csv", index=False)
    (snaps_cur[["season", "team", "week"]].drop_duplicates().sort_values(["season", "team", "week"])
        .to_csv(OUT / "team_weeks.csv", index=False))
    episodes.to_csv(OUT / "injury_episodes.csv", index=False)
    fantasy.to_csv(OUT / "fantasy_top48.csv", index=False)
    fantasy_eps.to_csv(OUT / "fantasy_episodes.csv", index=False)
    for stale in ("key_players.csv",):
        (OUT / stale).unlink(missing_ok=True)

    print(f"Player-seasons: {len(ps):,}  snap-key {ps['is_snap_key'].sum():,}  "
          f"pedigree-key {ps['is_pedigree_key'].sum():,}")
    print(f"Injury episodes (all players): {len(episodes):,}")
    print(f"Fantasy top-48 matched: {len(fantasy)} / {FANTASY_TOP_N * len(SEASONS)}; "
          f"episodes {len(fantasy_eps)}")


if __name__ == "__main__":
    main()
