"""Fantasy injury models for startable 12-team PPR players, with backup/hedge advice.

Who is covered
--------------
Each week, any QB/RB/WR/TE who is startable in a 12-team PPR league: top 12 QB, 30 RB,
36 WR, 12 TE by PPR points per game (flex included). Early in a season, points per game
blend this season with last season. Defined from production, not ADP, so it follows
in-season breakouts and works for seasons with no ADP data (2013-2019).

Three models
------------
A. Next week      Did a player who played in week t miss week t+1 with a new injury?
                  Scored the Tuesday after week t. Warning signs (left the game early,
                  played through an injury) plus schedule (rest days, short week, turf).
B. Injury report  Will a Questionable/Doubtful player sit this week? Uses game status and
                  practice participation, published before kickoff. Players ruled Out
                  sat essentially always, so they're handled by rule.
C. Time out       Once a player is out, will he return for the next game? A discrete-time
                  survival model with one row per game missed. It handles absences still
                  open at season end (censoring) and conditions on games already missed, so
                  a player out 3 weeks gets a different forecast than one just hurt.
                  Rolled forward, it gives P(misses 0 / 1 / 2-3 / 4+ of the coming games).
Beyond next week, a healthy player's risk is his position's league rate: the earlier
long-run model couldn't beat it (ROC-AUC ~0.56), so it was removed.

Design decisions
----------------
* Seasons 2013-2026 (2012 only feeds prior-season features). Split by season: train
  2013-2022, validate 2023-2024, test 2025, score 2026 live; plus leave-one-season-out.
  Older seasons record fewer injuries (IR rules and reporting changed), so every model
  gets an era feature. Tested on 2020-2025: 2013+ with era beat 2020+ only slightly
  (AUC +0.003 next week, +0.006 time out); 2013+ without era under-predicted.
* Before 2020, weekly rosters drop players placed on IR instead of listing them. Season-
  end rosters mark them RES, so a player who left the roster mid-season and ends it on
  reserve is treated as placed on IR (an absence to season end). Without this, the most
  severe injuries would silently fall out of the older data.
* No resampling or class weights: outputs are probabilities and must stay calibrated.
* Baseline = L2 logistic regression; challenger = HistGradientBoosting (LightGBM-style).
  The challenger is used only if it beats the baseline on validation log loss by >0.5%.
* Explanations: permutation importance (global) and exact logistic contributions per
  player (local).

Hedge logic (RB/WR/TE, 12-team PPR)
-----------------------------------
backup = highest-snap teammate at the position who isn't himself startable or on IR
value  = E[starter games missed, next 4 weeks] x (backup PPR when starter sits - waiver PPR)

Run: .venv/bin/python src/injury_model.py   (after fetch_data.py and build_dataset.py)
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import (CURRENT_SEASON, OUT, RAW, body_region, build_crosswalk,  # noqa: E402
                           find_episodes, injury_evidence, load, roster_weeks)

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parents[1]
REPORTS, MODELS = ROOT / "reports", ROOT / "models"

# ----------------------------------------------------------------------------- config
LIVE = CURRENT_SEASON
LIVE_LIMIT = {pos: 999 for pos in ("QB", "RB", "WR", "TE")}  # live pool: every fantasy-position player with a snap
HISTORY_START, MODEL_START = 2012, 2013
TRAIN = list(range(MODEL_START, 2023))
VAL, TEST = [2023, 2024], [2025]
RANDOM_STATE = 42
STARTER_RANK = {"QB": 12, "RB": 30, "WR": 36, "TE": 12}  # 12-team PPR incl. flex
FANTASY_POS = set(STARTER_RANK)
LAST_WEEK = lambda s: 17 if s <= 2020 else 18
ERA = lambda s: "pre2020" if s < 2020 else "2020_21" if s <= 2021 else "2022plus"
SOFT_TISSUE = {"Hamstring", "Hip / Groin", "Quad / Thigh", "Lower Leg"}
REGION_GROUP = {**{r: "soft_tissue" for r in SOFT_TISSUE}, "Head / Concussion": "concussion",
                "Knee": "knee", "Ankle": "ankle_foot", "Foot / Toe": "ankle_foot", "Achilles": "achilles",
                "Unspecified (IR)": "unspecified"}
HEDGE_POS = {"RB", "WR", "TE"}
ROSTERED = {"RB": 60, "WR": 72, "TE": 24}  # 12-team PPR: rostered players per position
ROSTER_LIMIT = {"QB": 24, **ROSTERED}  # live scoring pool: anyone a manager might start
SHRINK_M = 3
HEDGE_STRONG, HEDGE_CONSIDER = 4.0, 2.0  # next-4-week marginal PPR points
TIER_HIGH, TIER_ELEVATED = 0.40, 0.15

NEXT_NUM = ["age", "experience", "bmi", "ppg", "starter_rank_pct", "week_next",
            "opp_last", "opp_avg3", "opp_season_avg", "prior_opp_pg", "opp_ratio",
            "snap_drop", "snap_avg3", "prior1_games_missed", "prior1_absences", "prior2_games_missed",
            "ytd_absences", "ytd_games_missed", "weeks_since_return", "last_injury_soft_tissue",
            "played_through_injury", "played_through_soft_tissue", "limited_practice_last", "questionable_last",
            "rest_next", "short_week_next", "after_bye", "turf_next", "turf_last"]
# injury-history features (added after the first review found the model leaned almost
# entirely on "left last game early"): how often he gets hurt, what his last injury was
# and how often that type recurs, whether he has re-injured it, and whether he came
# back faster than that injury usually takes
HISTORY_NUM = ["abs_3y", "gm_3y", "frequent_injury", "reinjured_same_region", "last_injury_recur_rate",
               "return_vs_typical", "rushed_return"]
NEXT_NUM = NEXT_NUM + HISTORY_NUM
NEXT_CAT = ["pos", "era", "last_injury_group"]  # era: IR rules and reporting changed in 2020 and 2022
# warning signs visible by Tuesday; players without any are "quiet". A separate quiet-player
# model was tested and lost (see feature_experiment), so one model serves both groups.
WARNING = ["snap_drop", "played_through_injury", "played_through_soft_tissue", "limited_practice_last", "questionable_last"]
QUIET_NUM = [f for f in NEXT_NUM if f not in WARNING]
RETURN_LABELS = [(0.65, "Likely to play"), (0.35, "Game-time decision"), (0.10, "Unlikely to play"), (0.0, "Expected out")]
AVAIL_NUM = ["age", "ppg", "week", "missed_prev", "weeks_on_report", "opp_avg3", "snap_drop_prev", "rest"]
AVAIL_CAT = ["pos", "report_status", "practice", "region_group", "era"]
DUR_NUM = ["age", "k", "k_log", "week", "games_left", "on_ir", "left_early", "injury_game_snap",
           "prior1_games_missed", "same_region_before", "ppg"]
DUR_CAT = ["pos", "region_group", "era", "report_status", "practice"]

FEATURE_NOUNS = {
    "age": "age", "experience": "years in league", "bmi": "body mass index", "ppg": "PPR points per game",
    "week_next": "week of season", "opp_last": "workload last game", "opp_avg3": "recent workload",
    "opp_season_avg": "season workload", "prior_opp_pg": "workload last season", "opp_ratio": "workload vs. last season",
    "snap_avg3": "recent snap share", "prior1_games_missed": "games missed last season", "prior1_absences": "injuries last season",
    "prior2_games_missed": "games missed two seasons ago", "ytd_absences": "injuries this season",
    "ytd_games_missed": "games missed this season", "weeks_since_return": "weeks since returning from injury",
    "rest_next": "days of rest before next game",
}
FEATURE_NOUNS.update({"abs_3y": "injuries in the last 3 seasons", "gm_3y": "games missed in the last 3 seasons",
                      "last_injury_recur_rate": "recurrence rate of his last injury type",
                      "return_vs_typical": "recovery time vs. typical for that injury"})
FEATURE_FLAGS = {
    "last_injury_soft_tissue": "last injury was soft-tissue", "played_through_injury": "played through an injury last game",
    "played_through_soft_tissue": "played through a soft-tissue injury", "limited_practice_last": "was limited in practice last week",
    "questionable_last": "was listed Questionable/Doubtful last week", "short_week_next": "short week next (Thursday game)",
    "after_bye": "coming off a bye", "turf_next": "next game on artificial turf", "turf_last": "played on artificial turf",
    "frequent_injury": "frequently injured (3+ injuries in 3 seasons)", "reinjured_same_region": "has already re-injured the same body part",
    "rushed_return": "came back faster than that injury usually takes",
}
FEATURE_LABELS = {**FEATURE_NOUNS, **FEATURE_FLAGS, "pos": "position", "snap_drop": "left last game early (snap drop)",
                  "starter_rank_pct": "fantasy rank", "report_status": "game status", "practice": "practice participation",
                  "region_group": "type of injury", "missed_prev": "missed the previous game", "weeks_on_report": "weeks on report",
                  "snap_drop_prev": "left last game early", "rest": "days of rest", "week": "week of season",
                  "k": "games missed so far", "k_log": "games missed so far (log)", "games_left": "games left in season",
                  "on_ir": "on Injured Reserve", "left_early": "left the injury game early", "injury_game_snap": "snaps in injury game",
                  "same_region_before": "same body part hurt before", "era": "IR rules era", "last_injury_group": "type of last injury"}


# ============================================================ 1. data
def load_inputs() -> dict:
    seasons = range(HISTORY_START, LIVE + 1)
    xw = build_crosswalk()
    to_pfr = xw.set_index("gsis_id")["pfr_id"]
    snaps = load("snap_counts", seasons)
    snaps = snaps[snaps["game_type"] == "REG"]
    stats = load("stats_player_week", seasons)
    stats = stats[stats["season_type"] == "REG"].copy()
    stats["pfr_player_id"] = stats["player_id"].map(to_pfr)
    players = pd.read_parquet(RAW / "players.parquet").dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id")
    inj = load("injuries", range(MODEL_START, LIVE + 1))
    inj = inj[inj["game_type"] == "REG"].merge(xw, on="gsis_id").rename(columns={"pfr_id": "pfr_player_id"})
    inj["region"] = inj["report_primary_injury"].map(body_region).fillna(inj["practice_primary_injury"].map(body_region))
    games = pd.read_csv(RAW / "games.csv")
    games = games[(games["game_type"] == "REG") & (games["season"] >= HISTORY_START)]

    old = range(HISTORY_START, 2020)
    ev = pd.concat([injury_evidence(xw, old), injury_evidence(xw, range(2020, LIVE + 1))])
    last_roster = pd.concat([roster_weeks(old), roster_weeks(range(2020, LIVE + 1))])
    ev, last_roster, recovered = recover_old_ir(ev, last_roster, snaps, old)
    print(f"Recovered {recovered} pre-2020 IR placements from season-end rosters")
    return {"xw": xw, "snaps": snaps, "stats": stats, "players": players, "inj": inj, "games": games,
            "ev": ev, "last_roster": last_roster, "recovered_ir": recovered}


def recover_old_ir(ev, last_roster, snaps, seasons):
    """Pre-2020: a player who left the weekly roster mid-season and is RES on the season-end
    roster was placed on IR. Add IR evidence from then to season end and extend his roster."""
    team_last = snaps.groupby(["season", "team"])["week"].max()
    lr = last_roster.set_index(["season", "team", "pfr_player_id"])["last_roster_week"].copy()
    add, n = [], 0
    for s in seasons:
        sr = pd.read_parquet(RAW / f"roster_season_{s}.parquet")
        for r in sr[sr["status"] == "RES"].dropna(subset=["pfr_id"]).itertuples(index=False):
            key = (s, r.team, r.pfr_id)
            if key not in lr.index or (s, r.team) not in team_last.index:
                continue
            end = int(team_last[(s, r.team)])
            if lr[key] < end:
                add += [{"season": s, "week": w, "pfr_player_id": r.pfr_id, "on_ir": True} for w in range(int(lr[key]) + 1, end + 1)]
                lr[key] = end
                n += 1
    ev = pd.concat([ev, pd.DataFrame(add)]).drop_duplicates(["season", "week", "pfr_player_id"], keep="first")
    ev["on_ir"] = ev["on_ir"].fillna(False).astype(bool)
    return ev, lr.rename("last_roster_week").reset_index(), n


def schedule_features(games: pd.DataFrame) -> dict:
    """(season, week, team) -> rest days and artificial-turf flag."""
    turf = lambda x: int(isinstance(x, str) and "grass" not in x.lower())
    rows = {}
    for g in games.itertuples(index=False):
        for team, rest in ((g.home_team, g.home_rest), (g.away_team, g.away_rest)):
            rows[(g.season, g.week, team)] = {"rest": rest, "turf": turf(g.surface)}
    return rows


def caliber_table(stats: pd.DataFrame, limits: dict = STARTER_RANK, seasons=None) -> dict:
    """(season, week t) -> {pfr: (rank_pct, ppg, pos)} for players ranked within `limits`
    as of after week t. rank_pct is always relative to the startable line (>1 = deeper).
    Week 0 = preseason (last season's PPR per game)."""
    st = stats[stats["position"].isin(FANTASY_POS)].dropna(subset=["pfr_player_id"])
    by_season = {s: g for s, g in st.groupby("season")}
    out = {}
    for s in (seasons or range(MODEL_START, LIVE + 1)):
        g = by_season.get(s)
        pri = by_season.get(s - 1)
        pri = pri.groupby(["pfr_player_id", "position"])["fantasy_points_ppr"].agg(["mean", "size"]) if pri is not None else None
        pri = pri[pri["size"] >= 6]["mean"] if pri is not None else pd.Series(dtype=float)
        pri_val = {p: v for (p, _), v in pri.items()}
        pri_pos = {p: ps for (p, ps) in pri.index}
        for t in range(0, LAST_WEEK(s) + 1):
            vals, pos = {}, dict(pri_pos)
            if g is not None and t > 0:
                cur = g[g["week"] <= t].groupby(["pfr_player_id", "position"])["fantasy_points_ppr"].agg(["sum", "size"])
                for (p, ps), row in cur.iterrows():
                    n = row["size"]; this = row["sum"] / n; w = min(n, 4) / 4
                    vals[p] = w * this + (1 - w) * pri_val[p] if p in pri_val else this
                    pos[p] = ps
            for p, v in pri_val.items():
                vals.setdefault(p, v)
            df = pd.DataFrame({"p": list(vals), "ppg": list(vals.values())})
            df["pos"] = df["p"].map(pos)
            df["rank"] = df.groupby("pos")["ppg"].rank(ascending=False, method="first")
            df = df[df["rank"] <= df["pos"].map(limits)]
            df["lim"] = df["pos"].map(STARTER_RANK)
            out[(s, t)] = {r.p: (r.rank / r.lim, r.ppg, r.pos) for r in df.itertuples(index=False)}
    return out


def snap_ratio(last: float, prior: list) -> float:
    """Last game's snap share relative to his previous (up to 3) games; NaN if no usable base."""
    base = np.nanmean(prior) if len(prior) and not np.all(np.isnan(prior)) else np.nan
    return float(np.clip(last / base, 0, 3)) if base == base and base > 0.05 and last == last else np.nan


def next_scheduled(games: pd.DataFrame, s: int, team: str, t: int):
    g = games[(games["season"] == s) & ((games["home_team"] == team) | (games["away_team"] == team)) & (games["week"] > t)]
    return int(g["week"].min()) if len(g) else None


def games_after(games: pd.DataFrame, s: int, team: str, t: int) -> int:
    return int(((games["season"] == s) & ((games["home_team"] == team) | (games["away_team"] == team)) & (games["week"] > t)).sum())


# ============================================================ panels
def build_panels(d: dict):
    snaps, stats, players, inj = d["snaps"], d["stats"], d["players"], d["inj"]
    fs = snaps[snaps["position"].isin(FANTASY_POS)]
    team_weeks = snaps.groupby(["season", "team"])["week"].apply(lambda w: sorted(set(w))).to_dict()
    played = fs.groupby(["season", "team", "pfr_player_id"])["week"].apply(set).to_dict()
    snap_pct = fs.set_index(["season", "team", "pfr_player_id", "week"])["offense_pct"].to_dict()
    last_roster = d["last_roster"].set_index(["season", "team", "pfr_player_id"])["last_roster_week"].to_dict()
    sched = schedule_features(d["games"])
    cal = caliber_table(stats)
    cal_ext = caliber_table(stats, LIVE_LIMIT, [LIVE])  # live scoring pool only (never trained on)
    live_week = int(snaps.loc[snaps["season"] == LIVE, "week"].max())

    st = stats.dropna(subset=["pfr_player_id"]).copy()
    qb = st["position"] == "QB"
    st["opp"] = np.where(qb, st["attempts"].fillna(0) + st["carries"].fillna(0), st["carries"].fillna(0) + st["targets"].fillna(0))
    opp_week = st.groupby(["season", "pfr_player_id", "week"])["opp"].sum().to_dict()
    opp_pg = st.groupby(["season", "pfr_player_id"])["opp"].mean().to_dict()

    # absences for every fantasy-position player-team-season (same logic as the dashboard)
    pts = (fs[fs["season"] >= MODEL_START].groupby(["season", "team", "pfr_player_id"])
             .agg(player=("player", "first"), pos_group=("position", lambda x: x.mode().iat[0]), first_week=("week", "min")).reset_index())
    pts["side"] = "Offense"
    pts["max_week"] = pts["season"].map(LAST_WEEK)
    eps = find_episodes(pts, snaps[snaps["season"] >= MODEL_START], d["ev"], d["last_roster"])
    ep_by = {k: g.sort_values("start_week") for k, g in eps.groupby(["season", "pfr_player_id"])}
    grp_of = lambda r: REGION_GROUP.get(r, "other")
    tr_eps = eps[eps.season.isin(TRAIN) & (eps.body_region != "Unspecified (IR)")].sort_values(["pfr_player_id", "season", "start_week"]).copy()
    typical = tr_eps.groupby(["pos_group", "body_region"])["games_missed"].agg(["median", "size"])
    typical_region = tr_eps.groupby("body_region")["games_missed"].median()
    typical_games = lambda pos, reg: (typical.loc[(pos, reg), "median"] if (pos, reg) in typical.index and typical.loc[(pos, reg), "size"] >= 15
                                      else typical_region.get(reg, np.nan))
    tr_eps["grp"] = tr_eps["body_region"].map(grp_of)
    nxt_ep = tr_eps.groupby(["pfr_player_id", "season"]).shift(-1)
    tr_eps["recurred"] = (nxt_ep["body_region"].map(grp_of) == tr_eps["grp"]) & (nxt_ep["start_week"] - tr_eps["end_week"] <= 8)
    recur_rate = tr_eps.groupby("grp")["recurred"].mean().to_dict()  # same type again within 8 weeks
    rep = inj.drop_duplicates(["season", "pfr_player_id", "week"]).set_index(["season", "pfr_player_id", "week"])
    rep_keys = set(rep.index)
    # a team's report counts as final once it carries game statuses (Friday); Wednesday and
    # Thursday reports list practice only
    team_final = set(map(tuple, inj.loc[inj["report_status"].isin(["Out", "Doubtful", "Questionable"]), ["season", "team", "week"]].to_numpy()))
    ir_weeks = set(map(tuple, d["ev"].loc[d["ev"]["on_ir"], ["season", "pfr_player_id", "week"]].to_numpy()))

    def practice_of(x):
        x = str(x)
        return "DNP" if x.startswith("Did Not") else "Limited" if x.startswith("Limited") else "Full" if x.startswith("Full") else "Unknown"

    def status_of(x):
        return x if x in ("Questionable", "Doubtful", "Probable", "Out") else "None"

    def last_region_before(s, p, t):
        for season in (s, s - 1, s - 2):
            g = ep_by.get((season, p))
            if g is None:
                continue
            g = g[g["start_week"] <= t] if season == s else g
            g = g[g["body_region"] != "Unspecified (IR)"]
            if len(g):
                return g["body_region"].iat[-1]
        return None

    next_rows, avail_rows, dur_rows, live_out = [], [], [], []
    for r in pts.itertuples(index=False):
        s, team, p, pos = r.season, r.team, r.pfr_player_id, r.pos_group
        if pos not in FANTASY_POS:
            continue
        weeks = [w for w in team_weeks.get((s, team), []) if w <= min(last_roster.get((s, team, p), 99), LAST_WEEK(s))]
        if not weeks:
            continue
        pl = played.get((s, team, p), set())
        prof = players.loc[p] if p in players.index else None
        age = ((pd.Timestamp(f"{s}-09-01") - pd.Timestamp(prof["birth_date"])).days / 365.25
               if prof is not None and pd.notna(prof["birth_date"]) else np.nan)
        exp = s - prof["rookie_season"] if prof is not None and pd.notna(prof["rookie_season"]) else np.nan
        bmi = 703 * prof["weight"] / prof["height"] ** 2 if prof is not None and prof["height"] else np.nan
        prior_pg = opp_pg.get((s - 1, p), np.nan)
        prior1, prior2 = ep_by.get((s - 1, p)), ep_by.get((s - 2, p))
        p1_gm = prior1["games_missed"].sum() if prior1 is not None else 0
        p1_n = len(prior1) if prior1 is not None else 0
        p2_gm = prior2["games_missed"].sum() if prior2 is not None else 0
        cur_eps = ep_by.get((s, p))
        my_eps = cur_eps[cur_eps["team"] == team] if cur_eps is not None else None
        starts = set(my_eps["start_week"]) if my_eps is not None else set()
        base = {"season": s, "team": team, "pfr_player_id": p, "player": r.player, "pos": pos, "age": age, "era": ERA(s)}

        for i, t in enumerate(weeks):
            nxt = weeks[i + 1] if i + 1 < len(weeks) else None
            played_ws = sorted(w for w in pl if w <= t)
            c_now = cal.get((s, t), {}).get(p)
            if c_now is None and s == LIVE and t == live_week:
                c_now = cal_ext.get((s, t), {}).get(p)
            # ---- A. next week (he played week t and is startable)
            if t in pl and c_now and (nxt is not None or (s == LIVE and t == live_week)):
                nxt_w = nxt if nxt is not None else next_scheduled(d["games"], s, team, t)
                last3 = played_ws[-3:]
                opps = [opp_week.get((s, p, w), 0) for w in played_ws]
                opp3 = [opp_week.get((s, p, w), 0) for w in last3]
                ytd = cur_eps[cur_eps["start_week"] <= t] if cur_eps is not None else None
                on_rep = (s, p, t) in rep_keys
                rr = rep.loc[(s, p, t)] if on_rep else None
                lr = last_region_before(s, p, t)
                sn = sched.get((s, nxt_w, team), {}) if nxt_w else {}
                hist = [g for g in (prior2, prior1, ytd) if g is not None and len(g)]
                hist = pd.concat(hist).sort_values(["season", "start_week"]) if hist else None
                known = hist[hist.body_region != "Unspecified (IR)"] if hist is not None else None
                last_ep = known.iloc[-1] if known is not None and len(known) else None
                recent = ytd.iloc[-1] if ytd is not None and len(ytd) else None
                wsr = float(t - recent["end_week"]) if recent is not None else np.nan
                ratio = (recent["games_missed"] / typical_games(pos, recent["body_region"])
                         if recent is not None and wsr <= 6 and recent["body_region"] != "Unspecified (IR)" else np.nan)
                full_window = s - 2 >= MODEL_START
                next_rows.append({**base, "week": t, "week_next": nxt_w or t + 1, "experience": exp, "bmi": bmi,
                    "ppg": c_now[1], "starter_rank_pct": c_now[0],
                    "opp_last": opp_week.get((s, p, t), 0), "opp_avg3": np.mean(opp3), "opp_season_avg": np.mean(opps),
                    "prior_opp_pg": prior_pg, "opp_ratio": np.mean(opp3) / prior_pg if prior_pg and prior_pg > 0 else np.nan,
                    "snap_drop": snap_ratio(snap_pct.get((s, team, p, t), np.nan), [snap_pct.get((s, team, p, w), np.nan) for w in played_ws[:-1][-3:]]),
                    "snap_avg3": np.nanmean([snap_pct.get((s, team, p, w), np.nan) for w in last3]),
                    "prior1_games_missed": p1_gm, "prior1_absences": p1_n, "prior2_games_missed": p2_gm,
                    "ytd_absences": len(ytd) if ytd is not None else 0,
                    "ytd_games_missed": float(ytd["games_missed"].sum()) if ytd is not None and len(ytd) else 0.0,
                    "weeks_since_return": float(t - ytd["end_week"].max()) if ytd is not None and len(ytd) else 20.0,
                    "last_injury_soft_tissue": int(lr in SOFT_TISSUE),
                    "played_through_injury": int(on_rep and pd.notna(rr["region"])),
                    "played_through_soft_tissue": int(on_rep and rr["region"] in SOFT_TISSUE),
                    "limited_practice_last": int(on_rep and practice_of(rr["practice_status"]) in ("DNP", "Limited")),
                    "questionable_last": int(on_rep and status_of(rr["report_status"]) in ("Questionable", "Doubtful")),
                    "rest_next": sn.get("rest", np.nan), "short_week_next": int(sn.get("rest", 7) <= 5),
                    "after_bye": int(nxt_w is not None and nxt_w - t > 1), "turf_next": sn.get("turf", np.nan),
                    "turf_last": sched.get((s, t, team), {}).get("turf", np.nan),
                    "injury_region_now": rr["region"] if on_rep else None,
                    "abs_3y": (len(hist) if hist is not None else 0) if full_window else np.nan,
                    "gm_3y": (float(hist["games_missed"].sum()) if hist is not None else 0.0) if full_window else np.nan,
                    "frequent_injury": int(hist is not None and len(hist) >= 3),
                    "reinjured_same_region": int(last_ep is not None and (known["body_region"] == last_ep["body_region"]).sum() >= 2),
                    "last_injury_group": grp_of(last_ep["body_region"]) if last_ep is not None else "none",
                    "last_injury_recur_rate": recur_rate.get(grp_of(last_ep["body_region"]), np.nan) if last_ep is not None else np.nan,
                    "return_vs_typical": ratio, "rushed_return": int(ratio == ratio and ratio < 1),
                    "warning": int((snap_ratio(snap_pct.get((s, team, p, t), np.nan), [snap_pct.get((s, team, p, w), np.nan) for w in played_ws[:-1][-3:]]) or 1) < 0.6
                                   or (on_rep and pd.notna(rr["region"]))),
                    "y": (int(nxt in starts) if nxt is not None else np.nan)})
            # ---- B. injury report for week t (startable going into the week)
            c_prev = cal.get((s, weeks[i - 1] if i > 0 else 0), {}).get(p)
            if c_prev is None and s == LIVE:
                c_prev = cal_ext.get((s, weeks[i - 1] if i > 0 else 0), {}).get(p)
            if (s, p, t) in rep_keys and c_prev and pd.notna(rep.loc[(s, p, t), "region"]):  # includes next week's live report
                rr = rep.loc[(s, p, t)]
                prev = weeks[i - 1] if i > 0 else None
                streak = 0
                for w in reversed(weeks[:i + 1]):
                    if (s, p, w) in rep_keys:
                        streak += 1
                    else:
                        break
                pw = [w for w in played_ws if w < t]
                recent = [opp_week.get((s, p, w), 0) for w in pw][-3:]
                avail_rows.append({**base, "week": t, "ppg": c_prev[1], "report_status": status_of(rr["report_status"]),
                    "practice": practice_of(rr["practice_status"]), "region": rr["region"],
                    "region_group": REGION_GROUP.get(rr["region"], "other"),
                    "missed_prev": int(prev is not None and prev not in pl), "weeks_on_report": streak,
                    "opp_avg3": np.mean(recent) if recent else np.nan,
                    "snap_drop_prev": snap_ratio(snap_pct.get((s, team, p, pw[-1]), np.nan), [snap_pct.get((s, team, p, w), np.nan) for w in pw[:-1][-3:]]) if pw else np.nan,
                    "rest": sched.get((s, t, team), {}).get("rest", np.nan),
                    "y": int(t not in pl) if (s < LIVE or t <= live_week) else np.nan})

        # ---- C. time out: one row per game missed
        if my_eps is None:
            continue
        for e in my_eps.itertuples(index=False):
            wk0 = max([w for w in weeks if w < e.start_week], default=0)
            c_start = cal.get((s, wk0), {}).get(p)
            if c_start is None and s == LIVE and e.ongoing:
                c_start = cal_ext.get((s, wk0), {}).get(p)
            if not c_start:
                continue
            missed = [w for w in weeks if e.start_week <= w <= e.end_week]
            after = [w for w in weeks if w > e.end_week]
            returned = bool(after) and after[0] in pl
            inj_game = max([w for w in weeks if w < e.start_week and w in pl], default=None)
            pw = sorted(w for w in pl if inj_game and w < inj_game)
            ig_snap = snap_pct.get((s, team, p, inj_game), np.nan) if inj_game else np.nan
            ld = snap_ratio(ig_snap, [snap_pct.get((s, team, p, w), np.nan) for w in pw[-3:]])
            same = any(g is not None and (g["body_region"] == e.body_region).any() for g in (prior1, prior2)) or \
                   (cur_eps is not None and ((cur_eps["start_week"] < e.start_week) & (cur_eps["body_region"] == e.body_region)).any())
            for k, w in enumerate(missed, start=1):
                rr = rep.loc[(s, p, w)] if (s, p, w) in rep_keys else None
                live_last = s == LIVE and k == len(missed) and e.ongoing
                row = {**base, "week": w, "k": k, "k_log": np.log1p(k),
                       "games_left": games_after(d["games"], s, team, w) if live_last else len([x for x in weeks if x > w]),
                       "on_ir": int((s, p, w) in ir_weeks), "left_early": int(ld == ld and ld < 0.6),
                       "injury_game_snap": ig_snap, "prior1_games_missed": p1_gm, "same_region_before": int(same),
                       "ppg": c_start[1], "region": e.body_region, "region_group": REGION_GROUP.get(e.body_region, "other"),
                       "report_status": status_of(rr["report_status"]) if rr is not None else "Not listed",
                       "practice": practice_of(rr["practice_status"]) if rr is not None else "Not listed",
                       "episode": f"{s}_{p}_{e.start_week}", "games_missed_total": e.games_missed, "censored": int(not returned)}
                nxt_games = [x for x in weeks if x > w]
                w2 = nxt_games[0] if nxt_games else None
                row["next_report"] = ("No report" if w2 is None or (s, team, w2) not in team_final else
                                      status_of(rep.loc[(s, p, w2), "report_status"]) if (s, p, w2) in rep_keys else "Not listed")
                if live_last:
                    live_out.append({**row, "y": np.nan})
                elif s < LIVE or not e.ongoing:
                    dur_rows.append({**row, "y": int(k == len(missed) and returned)})
    return (pd.DataFrame(next_rows), pd.DataFrame(avail_rows), pd.DataFrame(dur_rows), pd.DataFrame(live_out),
            {"episodes": eps, "cal": cal, "cal_ext": cal_ext, "live_week": live_week, "team_final": team_final})


# ============================================================ 2-4. preprocessing, splits, models
def preprocessor(num: list, cat: list) -> ColumnTransformer:
    """Median-impute (+ missing indicators) and scale numerics; one-hot categoricals.
    Fit inside the Pipeline on training rows only, so nothing leaks from val/test."""
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)])


def logistic(num, cat, C=1.0) -> Pipeline:
    return Pipeline([("prep", preprocessor(num, cat)), ("model", LogisticRegression(C=C, max_iter=5000))])


def boosted(num, cat, depth=3, lr=0.05, iters=200) -> Pipeline:
    return Pipeline([("prep", preprocessor(num, cat)),
                     ("model", HistGradientBoostingClassifier(max_depth=depth, learning_rate=lr, max_iter=iters, min_samples_leaf=60,
                                                              l2_regularization=1.0, early_stopping=False, random_state=RANDOM_STATE))])


def validate(df: pd.DataFrame, num: list, cat: list, name: str) -> dict:
    """Data checks: label balance per split, missingness, finite values, plausible ages."""
    lab = df.dropna(subset=["y"])
    splits = {"train": lab[lab.season.isin(TRAIN)], "val": lab[lab.season.isin(VAL)], "test": lab[lab.season.isin(TEST)]}
    info = {"rows": len(df), "labeled_rows": len(lab),
            "positive_rate": {k: round(float(v["y"].mean()), 4) for k, v in splits.items()},
            "positives": {k: int(v["y"].sum()) for k, v in splits.items()},
            "rows_by_split": {k: len(v) for k, v in splits.items()},
            "missing_pct": {c: round(float(df[c].isna().mean()), 3) for c in num + cat if df[c].isna().any()}}
    assert lab["y"].isin([0, 1]).all(), f"{name}: non-binary labels"
    assert np.isfinite(df[num].select_dtypes("number").fillna(0).to_numpy()).all(), f"{name}: infinite feature values"
    assert info["positives"]["train"] >= 30, f"{name}: too few training positives"
    assert df["age"].dropna().between(19, 45).mean() > 0.98, f"{name}: implausible ages"
    print(f"\n[{name}] rows {info['rows']:,} | positives train/val/test {info['positives']} | rate {info['positive_rate']}")
    return info


def evaluate(y, p, threshold: float, base_rate: float) -> dict:
    y, p = np.asarray(y), np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    pred = (p >= threshold).astype(int)
    brier_ref = brier_score_loss(y, np.full_like(p, base_rate))
    q = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False)
    calib = pd.DataFrame({"p": p, "y": y, "bin": q}).groupby("bin").agg(mean_pred=("p", "mean"), observed=("y", "mean"), n=("y", "size")).round(4)
    top = y[p >= np.quantile(p, 0.9)].mean()
    return {"n": int(len(y)), "positives": int(y.sum()), "base_rate": round(float(y.mean()), 4),
            "roc_auc": round(roc_auc_score(y, p), 4), "pr_auc": round(average_precision_score(y, p), 4),
            "brier": round(brier_score_loss(y, p), 5), "brier_skill_vs_base_rate": round(1 - brier_score_loss(y, p) / brier_ref, 4),
            "log_loss": round(log_loss(y, p), 4), "threshold": round(float(threshold), 4),
            "precision": round(precision_score(y, pred, zero_division=0), 4), "recall": round(recall_score(y, pred, zero_division=0), 4),
            "f1": round(f1_score(y, pred, zero_division=0), 4), "top_decile_rate": round(float(top), 4),
            "top_decile_lift": round(float(top / max(y.mean(), 1e-9)), 2),
            "calibration": calib.reset_index(drop=True).to_dict(orient="records")}


def select_and_fit(df: pd.DataFrame, num: list, cat: list, name: str, threshold_rule) -> dict:
    lab = df.dropna(subset=["y"])
    tr, va, te = (lab[lab.season.isin(s)] for s in (TRAIN, VAL, TEST))
    X = lambda d: d[num + cat]
    candidates = {f"logistic_C{C}": logistic(num, cat, C) for C in (0.03, 0.1, 0.3, 1.0)}
    candidates |= {f"boosted_d{dp}_lr{lr}_it{it}": boosted(num, cat, dp, lr, it) for dp in (2, 3) for lr in (0.03, 0.06) for it in (150, 300)}
    scores = {}
    for k, m in candidates.items():
        m.fit(X(tr), tr["y"])
        scores[k] = log_loss(va["y"], np.clip(m.predict_proba(X(va))[:, 1], 1e-6, 1 - 1e-6))
    best_log = min((k for k in scores if k.startswith("logistic")), key=scores.get)
    best_boost = min((k for k in scores if k.startswith("boosted")), key=scores.get)
    chosen = best_boost if scores[best_boost] < scores[best_log] * 0.995 else best_log
    print(f"[{name}] validation log loss: logistic {scores[best_log]:.4f} | boosted {scores[best_boost]:.4f} -> {chosen}")

    trva = pd.concat([tr, va])
    report = {"validation_log_loss": {k: round(v, 5) for k, v in scores.items()}, "chosen": chosen}
    for label, key in (("baseline_logistic", best_log), ("challenger_boosted", best_boost)):
        m = candidates[key]; m.fit(X(trva), trva["y"])
        p = m.predict_proba(X(te))[:, 1]
        report[label] = {"config": key, "test": evaluate(te["y"], p, threshold_rule(p), trva["y"].mean())}
        if key == chosen:
            chosen_model, chosen_p = m, p
    print(f"[{name}] TEST {TEST[0]}: logistic AUC {report['baseline_logistic']['test']['roc_auc']}, "
          f"boosted AUC {report['challenger_boosted']['test']['roc_auc']}")

    pi = permutation_importance(chosen_model, X(te), te["y"], scoring="neg_log_loss", n_repeats=8, random_state=RANDOM_STATE)
    report["permutation_importance"] = sorted([{"feature": f, "label": FEATURE_LABELS.get(f, f), "mean": round(float(m), 6), "std": round(float(s), 6)}
                                               for f, m, s in zip(num + cat, pi.importances_mean, pi.importances_std)], key=lambda r: -r["mean"])

    cv = []
    for season in sorted(x for x in lab.season.unique() if x < LIVE):
        trn, tst = lab[lab.season != season], lab[lab.season == season]
        m = candidates[chosen]; m.fit(X(trn), trn["y"])
        p = np.clip(m.predict_proba(X(tst))[:, 1], 1e-6, 1 - 1e-6)
        ref = brier_score_loss(tst["y"], np.full(len(tst), trn["y"].mean()))
        cv.append({"season": int(season), "roc_auc": round(roc_auc_score(tst["y"], p), 4),
                   "brier_skill_vs_base_rate": round(1 - brier_score_loss(tst["y"], p) / ref, 4), "positives": int(tst["y"].sum())})
    report["season_cv"] = cv
    report["season_cv_mean"] = {k: round(float(np.mean([c[k] for c in cv])), 4) for k in ("roc_auc", "brier_skill_vs_base_rate")}
    print(f"[{name}] leave-one-season-out mean ROC-AUC {report['season_cv_mean']['roc_auc']} "
          f"(range {min(c['roc_auc'] for c in cv)}-{max(c['roc_auc'] for c in cv)})")

    final = candidates[chosen]; final.fit(X(lab), lab["y"])
    final_log = logistic(num, cat, float(best_log.split("C")[1])); final_log.fit(X(lab), lab["y"])
    return {"report": report, "model": final, "explainer": final_log, "X_train": X(lab), "test_rows": te.assign(p=chosen_p)}


def feature_experiment(nxt: pd.DataFrame) -> dict:
    """Leave-one-season-out (2016-2025, full 3-season history) comparison of next-week setups.
    The key number is ROC-AUC on 'quiet' players: no warning signs going into the week."""
    lab = nxt.dropna(subset=["y"])
    lab = lab[(lab.season >= MODEL_START + 3) & (lab.season < LIVE)]
    old_num = [f for f in NEXT_NUM if f not in HISTORY_NUM]
    setups = {
        "current (no history)": [("all", old_num, ["pos", "era"])],
        "plus history": [("all", NEXT_NUM, NEXT_CAT)],
        "split: quiet + warning models": [("quiet", QUIET_NUM, NEXT_CAT), ("warning", NEXT_NUM, NEXT_CAT)],
    }
    makers = {"logistic": lambda n, c: logistic(n, c, 0.1), "boosted": lambda n, c: boosted(n, c, 3, 0.03, 300)}
    out = {}
    for name, parts in setups.items():
        for mk_name, mk in makers.items():
            preds = []
            for season in sorted(lab.season.unique()):
                tr, te = lab[lab.season != season], lab[lab.season == season]
                p = pd.Series(np.nan, index=te.index)
                for which, num, cat in parts:
                    sel = (lambda d: d) if which == "all" else (lambda d, q=(which == "quiet"): d[d["warning"] == (0 if q else 1)])
                    m = mk(num, cat); m.fit(sel(tr)[num + cat], sel(tr)["y"])
                    tt = sel(te); p.loc[tt.index] = m.predict_proba(tt[num + cat])[:, 1]
                preds.append(te.assign(p=p))
            pr = pd.concat(preds)
            q, w = pr[pr.warning == 0], pr[pr.warning == 1]
            per_season = [roc_auc_score(g.y, g.p) for _, g in pr.groupby("season")]
            per_season_q = [roc_auc_score(g.y, g.p) for _, g in q.groupby("season")]
            out[f"{name} | {mk_name}"] = {
                "auc_all": round(float(np.mean(per_season)), 4), "auc_quiet": round(float(np.mean(per_season_q)), 4),
                "auc_warning": round(roc_auc_score(w.y, w.p), 4),
                "brier_skill": round(1 - brier_score_loss(pr.y, pr.p) / brier_score_loss(pr.y, np.full(len(pr), pr.y.mean())), 4),
                "quiet_share_of_injuries": round(float(q.y.sum() / pr.y.sum()), 3)}
            print(f"[experiment] {name:32s} {mk_name:8s} AUC all {out[f'{name} | {mk_name}']['auc_all']} | quiet "
                  f"{out[f'{name} | {mk_name}']['auc_quiet']} | warning {out[f'{name} | {mk_name}']['auc_warning']} | Brier skill {out[f'{name} | {mk_name}']['brier_skill']}")
    return out


def hypothesis_table(nxt: pd.DataFrame) -> list:
    """Miss rates for quiet players (no warning signs) with and without each injury-history
    signal, 2016-2025, with a two-proportion z-score."""
    lab = nxt.dropna(subset=["y"])
    q = lab[(lab.season >= MODEL_START + 3) & (lab.season < LIVE) & (lab.warning == 0)]
    base = q.y.mean()
    tests = [("Frequently injured (3+ injuries in 3 seasons)", q.frequent_injury == 1),
             ("Already re-injured the same body part", q.reinjured_same_region == 1),
             ("Came back faster than that injury usually takes", q.rushed_return == 1),
             ("Returned from injury within the last 2 weeks", q.weeks_since_return <= 2),
             ("Last injury was a concussion", q.last_injury_group == "concussion"),
             ("Last injury was soft-tissue", q.last_injury_group == "soft_tissue"),
             ("Last injury was a knee", q.last_injury_group == "knee"),
             ("Played 60-85% of his usual snaps last game", (q.snap_drop >= 0.6) & (q.snap_drop < 0.85)),
             ("Short week next (Thursday game)", q.short_week_next == 1),
             ("Next game on artificial turf", q.turf_next == 1),
             ("Heavy recent workload (top 20%)", q.opp_avg3 >= q.opp_avg3.quantile(0.8)),
             ("Age 30+", q.age >= 30)]
    out = []
    for label, m in tests:
        g, o = q[m], q[~m]
        p = q.y.mean(); se = np.sqrt(p * (1 - p) * (1 / max(len(g), 1) + 1 / max(len(o), 1)))
        out.append({"signal": label, "n": int(len(g)), "miss_rate": round(float(g.y.mean()), 4), "lift": round(float(g.y.mean() / base), 2),
                    "z": round(float((g.y.mean() - o.y.mean()) / se), 1) if len(g) else None})
    return [{"base_rate": round(float(base), 4), "n": int(len(q)), "injuries": int(q.y.sum()),
             "share_of_injuries": round(float(q.y.sum() / lab[(lab.season >= MODEL_START + 3) & (lab.season < LIVE)].y.sum()), 3)}] + out


def high_confidence(te: pd.DataFrame) -> list:
    """How often high predictions come true, and how many players per week reach them."""
    weeks = te.groupby(["season", "week"]).ngroups
    out = []
    for cut in (0.25, 0.40, 0.60):
        sel = te[te["p"] >= cut]
        out.append({"threshold": cut, "hit_rate": round(float(sel["y"].mean()), 3) if len(sel) else None,
                    "flags": int(len(sel)), "per_week": round(len(sel) / weeks, 2),
                    "share_of_all_injuries": round(float(sel["y"].sum() / max(te["y"].sum(), 1)), 3)})
    return out


def drivers(explainer: Pipeline, X_train: pd.DataFrame, X_live: pd.DataFrame, num: list, cat: list) -> list:
    """Per-player top risk drivers: exact logistic contributions relative to the average
    player, phrased with the direction of the player's own value."""
    prep, model = explainer.named_steps["prep"], explainer.named_steps["model"]
    names = [n.split("__", 1)[1].replace("missingindicator_", "") for n in prep.get_feature_names_out()]
    orig = [next((c for c in cat if n.startswith(c + "_")), n) for n in names]
    dense = lambda M: np.asarray(M.todense() if hasattr(M, "todense") else M)
    coef = model.coef_[0]
    contrib = dense(prep.transform(X_live)) * coef - dense(prep.transform(X_train)).mean(axis=0) * coef
    med = X_train[num].median()
    out = []
    for i, row in enumerate(contrib):
        agg = pd.Series(row, index=orig).groupby(level=0).sum()
        phrases = []
        for f in agg[agg > 0.05].sort_values(ascending=False).index:
            v = X_live[f].iat[i] if f in X_live else None
            if f in FEATURE_FLAGS:
                if v == 1:
                    phrases.append(FEATURE_FLAGS[f])
            elif f == "snap_drop":
                if v == v and v < 0.6:
                    phrases.append(f"left last game early ({v:.0%} of usual snaps)")
                elif v == v and v < 0.85:
                    phrases.append(f"played {v:.0%} of his usual snaps last game")
            elif f == "pos":
                phrases.append(f"position ({X_live['pos'].iat[i]})")
            elif f in FEATURE_NOUNS and v == v:
                phrases.append(f"{'high' if v > med[f] else 'low'} {FEATURE_NOUNS[f]}")
            if len(phrases) == 3:
                break
        out.append(phrases)
    return out


# ============================================================ time-out distribution
def return_dist(model: Pipeline, row: pd.DataFrame, k0: int, horizon: int = 18) -> np.ndarray:
    """Given he has missed k0 games, P(he misses exactly j MORE games), j = 0, 1, ...
    (j = 0: back next game). The last cell holds 'still out when his season ends'."""
    left = int(row["games_left"].iat[0])
    n = max(1, min(horizon, left))
    fut = pd.concat([row] * n, ignore_index=True)
    ks = k0 + np.arange(n)
    fut["k"], fut["k_log"] = ks, np.log1p(ks)
    fut["week"] = row["week"].iat[0] + np.arange(n)
    fut["games_left"] = np.maximum(left - np.arange(n), 0)
    h = model.predict_proba(fut[DUR_NUM + DUR_CAT])[:, 1]
    probs, alive = [], 1.0
    for x in h:
        probs.append(alive * x); alive *= 1 - x
    probs.append(alive)
    return np.array(probs)


def miss_buckets(m: np.ndarray) -> dict:
    """m[c] = P(misses exactly c of the coming games) -> readable buckets."""
    return {"plays": float(m[0]), "out_1": float(m[1:2].sum()), "out_2_3": float(m[2:4].sum()), "out_4plus": float(m[4:].sum())}


def evaluate_duration(model: Pipeline, dur: pd.DataFrame) -> dict:
    """Episode-level checks on the test season, forecasting from the first missed game."""
    te = dur[dur.season.isin(TEST) & (dur["k"] == 1)].reset_index(drop=True)
    rows = []
    for i in range(len(te)):
        r = te.iloc[[i]]
        dist = return_dist(model, r, 1)  # dist[j]: total games missed = 1 + j
        cdf = np.cumsum(dist)
        rows.append({"actual": int(r["games_missed_total"].iat[0]), "censored": int(r["censored"].iat[0]),
                     "p1": float(dist[0]), "p4": float(dist[3:].sum()),
                     "expected": float(np.sum((np.arange(len(dist)) + 1) * dist)),
                     "median": int(np.searchsorted(cdf, 0.5)) + 1, "region": r["region"].iat[0]})
    ev = pd.DataFrame(rows)
    known4 = ev[(ev.censored == 0) | (ev.actual >= 4)]
    known1 = ev[(ev.censored == 0) | (ev.actual >= 2)]
    unc = ev[ev.censored == 0]
    conc = disc = 0
    arr = ev[["expected", "actual", "censored"]].to_numpy()
    for i in range(len(arr)):
        for j in range(len(arr)):
            if arr[i, 1] < arr[j, 1] and arr[i, 2] == 0:  # i returned before j was last seen
                conc += arr[i, 0] < arr[j, 0]; disc += arr[i, 0] > arr[j, 0]
    err = (unc["median"] - unc["actual"]).abs()
    naive = (unc["actual"].median() - unc["actual"]).abs()
    return {"episodes": int(len(ev)), "c_index": round(conc / max(conc + disc, 1), 4),
            "back_after_one_game_auc": round(roc_auc_score((known1["actual"] == 1).astype(int), known1["p1"]), 4),
            "out_4plus_auc": round(roc_auc_score((known4["actual"] >= 4).astype(int), known4["p4"]), 4),
            "out_4plus_brier": round(brier_score_loss((known4["actual"] >= 4).astype(int), known4["p4"]), 4),
            "mean_abs_error_games": round(float(err.mean()), 2), "naive_mean_abs_error_games": round(float(naive.mean()), 2),
            "within_one_game": round(float((err <= 1).mean()), 3),
            "calibration_out_4plus": (known4.assign(bin=pd.qcut(known4["p4"].rank(method="first"), 5, labels=False))
                                      .groupby("bin").agg(mean_pred=("p4", "mean"), observed=("actual", lambda a: float((a >= 4).mean())), n=("p4", "size"))
                                      .round(3).to_dict(orient="records")),
            "by_region": (unc.groupby("region").agg(n=("actual", "size"), predicted_median=("median", "median"), actual_median=("actual", "median"))
                          .query("n >= 5").reset_index().to_dict(orient="records"))}


# ============================================================ hedge engine
def hedge_inputs(d: dict, cal: dict, eps: pd.DataFrame) -> dict:
    """Fill-in value priors (2020+), per-player fill-in history, waiver level."""
    snaps, stats = d["snaps"], d["stats"]
    grp = {"RB": "RB", "WR": "WR", "TE": "TE"}  # fullbacks aren't fantasy fill-ins
    sn = snaps[snaps["season"] >= 2020].assign(pg=lambda x: x["position"].map(grp)).dropna(subset=["pg"])
    ppr = stats.dropna(subset=["pfr_player_id"]).set_index(["season", "week", "pfr_player_id"])["fantasy_points_ppr"].to_dict()
    fills = []
    for e in eps[(eps.season >= 2020) & (eps.season < LIVE) & eps.pos_group.isin(HEDGE_POS)].itertuples(index=False):
        startable = cal.get((e.season, max(e.start_week - 1, 0)), {})
        if e.pfr_player_id not in startable:
            continue
        for w in range(e.start_week, e.end_week + 1):
            cand = sn[(sn.season == e.season) & (sn.team == e.team) & (sn.week == w) & (sn.pg == e.pos_group)]
            cand = cand[~cand["pfr_player_id"].isin(startable)]
            if len(cand):
                b = cand.sort_values("offense_snaps", ascending=False).iloc[0]
                fills.append((e.pos_group, b["pfr_player_id"], ppr.get((e.season, w, b["pfr_player_id"]), 0.0)))
    fills = pd.DataFrame(fills, columns=["pos", "pfr_player_id", "ppr"])
    s = stats[(stats.season == LIVE - 1) & stats.position.isin(HEDGE_POS)]
    ppg = s.groupby(["pfr_player_id", "position"])["fantasy_points_ppr"].agg(["mean", "size"]).reset_index()
    ppg = ppg[ppg["size"] >= 4]
    waiver = {pos: float(ppg[ppg.position == pos].sort_values("mean", ascending=False)["mean"].to_numpy()[n:n + 12].mean())
              for pos, n in ROSTERED.items()}
    return {"prior": fills.groupby("pos")["ppr"].mean().to_dict(), "fills": fills, "waiver": waiver, "n_fill_games": len(fills)}


def backups_live(d: dict, startable: set, live_week: int) -> dict:
    grp = {"RB": "RB", "WR": "WR", "TE": "TE"}
    sn = d["snaps"][(d["snaps"].season == LIVE) & (d["snaps"].week > live_week - 3)]
    sn = sn.assign(pg=sn["position"].map(grp)).dropna(subset=["pg"])
    ros = load("roster_weekly", [LIVE])
    latest = ros[ros.week == ros.week.max()]
    unavailable = set(latest.loc[latest["status"] != "ACT", "pfr_id"].dropna())
    tot = sn.groupby(["team", "pg", "pfr_player_id", "player"])["offense_snaps"].sum().reset_index()
    tot = tot[~tot.pfr_player_id.isin(startable | unavailable)]
    best = tot.sort_values("offense_snaps", ascending=False).drop_duplicates(["team", "pg"])
    return {(r.team, r.pg): (r.pfr_player_id, r.player) for r in best.itertuples(index=False)}


# ============================================================ live scoring
def score_live(d, A, B, C, nxt, avail, dur, live_out, extra, out_rate, base_rate, off_report_rate) -> pd.DataFrame:
    live_week, cal, stats, games = extra["live_week"], extra["cal"], d["stats"], d["games"]
    startable_now = cal.get((LIVE, live_week), {})
    pool = extra["cal_ext"].get((LIVE, live_week), {})
    live = nxt[(nxt.season == LIVE) & (nxt.week == live_week)].copy()
    live["p_next"] = A["model"].predict_proba(live[NEXT_NUM + NEXT_CAT])[:, 1]
    live["drivers"] = drivers(A["explainer"], A["X_train"], live[NEXT_NUM + NEXT_CAT], NEXT_NUM, NEXT_CAT)
    live_rep = avail[(avail.season == LIVE) & (avail.week > live_week)].copy()
    if len(live_rep):
        live_rep["p_miss"] = np.where(live_rep.report_status == "Out", out_rate, B["model"].predict_proba(live_rep[AVAIL_NUM + AVAIL_CAT])[:, 1])
    hz = hedge_inputs(d, cal, extra["episodes"])
    backups = backups_live(d, set(startable_now), live_week)
    fills = hz["fills"]
    generic = dur[(dur.k == 1) & (dur.season >= 2020)]  # typical fresh injury under current IR rules
    cur_stats = stats[stats.season == LIVE]
    ros = load("roster_weekly", [LIVE])
    on_roster = set(ros.loc[ros.week == ros.week.max(), "pfr_id"].dropna())
    team_final = extra["team_final"]
    roster_team = ros.sort_values("week").dropna(subset=["pfr_id"]).drop_duplicates("pfr_id", keep="last").set_index("pfr_id")["team"].to_dict()

    rows = []
    prof_cache = {}
    for p, (rank_pct, ppg, pos) in pool.items():
        if p not in on_roster and not (len(live_out) and (live_out.pfr_player_id == p).any()):
            continue  # released / no longer on a roster
        me = live[live.pfr_player_id == p]
        out_row = live_out[live_out.pfr_player_id == p] if len(live_out) else live_out
        rep_row = live_rep[live_rep.pfr_player_id == p] if len(live_rep) else live_rep
        mine = cur_stats[cur_stats.pfr_player_id == p]
        src = me if len(me) else out_row if len(out_row) else None
        name = src["player"].iat[0] if src is not None else (mine["player_display_name"].iat[-1] if len(mine) else
                                                             d["players"]["display_name"].get(p, p))
        team = src["team"].iat[0] if src is not None else (mine["team"].iat[-1] if len(mine) else roster_team.get(p, ""))
        row = {"player": name, "pos": pos, "team": team, "ppg": round(ppg, 1), "rank": int(round(rank_pct * STARTER_RANK[pos])),
               "startable": int(p in startable_now), "status": "", "p_next_week": np.nan, "p_next_4": np.nan, "expected_missed_4wk": np.nan, "source": "", "tier": "", "drivers": "", "injury": "",
               "games_missed_so_far": 0, "return_status": "", "report_note": "", "plays": np.nan, "out_1": np.nan, "out_2_3": np.nan, "out_4plus": np.nan,
               "expected_games_out": np.nan, "backup": "", "backup_ppg_when_out": np.nan, "hedge_value_4wk": np.nan, "advice": ""}
        b = base_rate.get(pos, 0.05)
        e4 = np.nan
        if len(out_row):  # already out: forecast from the games missed so far
            r = out_row.iloc[[0]]
            m = return_dist(C["model"], r, int(r["k"].iat[0]))  # m[j] = misses j more
            p_play, note = float(m[0]), f"time-out model ({int(r['k'].iat[0])} games missed)"
            final = (LIVE, team, live_week + 1) in team_final
            if final:  # Friday report is out for his team: it overrides the Tuesday forecast
                if len(rep_row):
                    st = rep_row["report_status"].iat[0]
                    p_play = 1 - float(rep_row["p_miss"].iat[0]); note = f"final report: {st if st != 'None' else 'listed, no game status'}"
                else:
                    p_play, note = off_report_rate, "not on final report (usually still out)"
            elif len(rep_row):
                note += f"; practice report: {rep_row['practice'].iat[0]}"
            if int(r["on_ir"].iat[0]):
                note += "; on IR"
            m = np.concatenate([[p_play], (1 - p_play) * m[1:] / max(1 - m[0], 1e-9)])
            bk = miss_buckets(m)
            row["return_status"] = "On IR" if int(r["on_ir"].iat[0]) and p_play < 0.35 else next(l for c, l in RETURN_LABELS if p_play >= c)
            row["report_note"] = note
            row.update({"status": "Out now", "games_missed_so_far": int(r["k"].iat[0]), "injury": r["region"].iat[0], **bk,
                        "p_next_week": 1 - bk["plays"], "source": "time-out model",
                        "expected_games_out": float(np.sum(np.arange(len(m)) * m))})
            e4 = float(sum(min(j, 4) * m[j] for j in range(len(m))))
        elif len(me):
            h, source = float(me["p_next"].iat[0]), "next-week model"
            if len(rep_row) and (LIVE, team, live_week + 1) in team_final and rep_row["report_status"].iat[0] in ("Questionable", "Doubtful", "Out"):
                h, source = float(rep_row["p_miss"].iat[0]), f"injury report ({rep_row['report_status'].iat[0]})"
            sd = me["snap_drop"].iat[0]
            inj_now = me["injury_region_now"].iat[0] or (rep_row["region"].iat[0] if len(rep_row) else "")
            # if he misses, how long? typical fresh injury at his position (his injury, if known)
            ck = (pos, inj_now or "", int(sd == sd and sd < 0.6), int(round((me["age"].iat[0] or 26) / 2) * 2), games_after(games, LIVE, team, live_week + 1))
            if ck not in prof_cache:
                prof = generic[generic.pos == pos]
                if inj_now:
                    same = prof[prof.region == inj_now]
                    prof = same if len(same) >= 20 else prof
                prof = prof.sample(min(30, len(prof)), random_state=RANDOM_STATE).assign(
                    age=ck[3], left_early=ck[2], week=live_week + 1, games_left=ck[4])
                dists = [return_dist(C["model"], prof.iloc[[i]], 1) for i in range(len(prof))]
                L = max(len(x) for x in dists)
                prof_cache[ck] = np.mean([np.pad(x, (0, L - len(x))) for x in dists], axis=0)
            dist = prof_cache[ck]  # dist[j]: misses 1 + j
            m = np.concatenate([[1 - h], h * dist])
            bk = miss_buckets(m)
            row.update({"status": "Left last game early" if sd == sd and sd < 0.6 else "Active", "p_next_week": h,
                        "p_next_4": 1 - (1 - h) * (1 - b) ** 3, "source": source, "drivers": "; ".join(me["drivers"].iat[0]),
                        "injury": inj_now, **bk, "expected_games_out": float(np.sum(np.arange(len(m)) * m))})
            e_if = float(sum(min(j + 1, 4) * dist[j] for j in range(len(dist))))
            e4 = h * e_if + sum(b * (1 - h) * (1 - b) ** (j - 2) * min(e_if, 5 - j) for j in range(2, 5))
        else:
            row["status"] = "Didn't play (no injury listed)"
        row["expected_missed_4wk"] = e4
        p_now = row["p_next_week"]
        row["tier"] = "" if p_now != p_now else "High" if p_now >= TIER_HIGH else "Elevated" if p_now >= TIER_ELEVATED else "Normal"
        if pos in HEDGE_POS:
            bk_ = backups.get((row["team"], pos))
            if bk_:
                hist = fills.loc[fills.pfr_player_id == bk_[0], "ppr"]
                when_out = (hist.sum() + SHRINK_M * hz["prior"][pos]) / (len(hist) + SHRINK_M)
                gain = max(0.0, when_out - hz["waiver"][pos])
                row.update({"backup": bk_[1], "backup_ppg_when_out": when_out})
                if e4 == e4:
                    row["hedge_value_4wk"] = e4 * gain
                    if row["status"] == "Out now":
                        row["advice"] = "Starter out: add backup" if gain >= 2 else "Starter out: backup ~waiver level"
                    else:
                        row["advice"] = ("Hedge" if row["hedge_value_4wk"] >= HEDGE_STRONG else
                                         "Consider" if row["hedge_value_4wk"] >= HEDGE_CONSIDER else "Not needed")
            else:
                row["advice"] = "No clear backup"
        else:
            row["advice"] = "n/a (QB)"
        rows.append(row)
    order = {"Out now": 0, "Left last game early": 1, "Active": 2}
    scored = (pd.DataFrame(rows).assign(_o=lambda x: x["status"].map(order).fillna(3))
              .sort_values(["_o", "p_next_week"], ascending=[True, False]).drop(columns="_o"))
    return scored, hz, bool(len(live_rep))


# ============================================================ main
def main() -> None:
    REPORTS.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
    print("Loading 2012-2026 data...")
    d = load_inputs()
    print("Building panels (next week, injury report, time out)...")
    nxt, avail, dur, live_out, extra = build_panels(d)
    for name, df in (("next_week", nxt), ("availability", avail), ("duration", dur)):
        df.to_csv(OUT / f"model_panel_{name}.csv", index=False)
    (OUT / "model_panel_risk.csv").unlink(missing_ok=True)

    # 1. validation
    v_next = validate(nxt, NEXT_NUM, NEXT_CAT, "next week")
    avail_m = avail[avail["report_status"] != "Out"]
    v_avail = validate(avail_m, AVAIL_NUM, AVAIL_CAT, "injury report")
    v_dur = validate(dur, DUR_NUM, DUR_CAT, "time out: back next game")
    out_rate = float(avail.loc[(avail.report_status == "Out") & avail.y.notna(), "y"].mean())
    print(f"[injury report] players ruled Out sat {out_rate:.1%} of the time (rule)")

    # 2-6. split, fit, evaluate, explain
    A = select_and_fit(nxt, NEXT_NUM, NEXT_CAT, "next week", lambda p: np.quantile(p, 0.9))
    B = select_and_fit(avail_m, AVAIL_NUM, AVAIL_CAT, "injury report", lambda p: 0.5)
    C = select_and_fit(dur, DUR_NUM, DUR_CAT, "time out", lambda p: 0.5)
    A["report"]["high_confidence"] = high_confidence(A["test_rows"])
    A["report"]["history_signals"] = hypothesis_table(nxt)
    exp_path = OUT / "feature_experiment.json"
    if "--experiment" in sys.argv:
        exp_path.write_text(json.dumps(feature_experiment(nxt), indent=1))
    A["report"]["feature_experiment"] = json.loads(exp_path.read_text()) if exp_path.exists() else None
    C["report"]["episode_eval"] = evaluate_duration(C["model"], dur)
    base_rate = nxt[nxt.season.isin(TRAIN + VAL)].groupby("pos")["y"].mean().to_dict()
    print("[next week] high-confidence flags on test:", A["report"]["high_confidence"])
    print("[time out] episode-level test:", {k: v for k, v in C["report"]["episode_eval"].items() if not isinstance(v, list)})
    for m, n in ((A, "next_week"), (B, "injury_report"), (C, "time_out")):
        joblib.dump(m["model"], MODELS / f"{n}_model.joblib")
    for old in ("risk_model.joblib", "risk_baseline_model.joblib", "availability_model.joblib"):
        (MODELS / old).unlink(missing_ok=True)

    nr = dur[(dur.on_ir == 0) & (dur.season < LIVE)].groupby("next_report")["y"].agg(["mean", "size"])
    off_report_rate = float(nr.loc["Not listed", "mean"])
    print("[time out] return rate by next week's FINAL report:", nr.round(3).to_dict(orient="index"))
    scored, hz, rep_live = score_live(d, A, B, C, nxt, avail, dur, live_out, extra, out_rate, base_rate, off_report_rate)
    scored.to_csv(OUT / "injury_risk_live.csv", index=False)
    live_week = extra["live_week"]
    report = {"live": {"season": LIVE, "scored_after_week": live_week, "predicting_week": live_week + 1,
                       "scored_players": len(scored), "startable_players": int(scored["startable"].sum()),
                       "injury_report_published": rep_live},
              "population": STARTER_RANK, "seasons": {"train": TRAIN, "val": VAL, "test": TEST},
              "recovered_pre2020_ir": d["recovered_ir"],
              "data_validation": {"next_week": v_next, "injury_report": v_avail, "time_out": v_dur, "out_rule_sat_rate": round(out_rate, 4)},
              "next_week_model": A["report"], "injury_report_model": B["report"], "time_out_model": C["report"],
              "position_weekly_base_rate": {k: round(v, 4) for k, v in base_rate.items()},
              "return_by_final_report": nr.round(4).reset_index().to_dict(orient="records"),
              "hedge": {"waiver_ppg": hz["waiver"], "fill_in_prior_ppg": {k: round(v, 2) for k, v in hz["prior"].items()},
                        "fill_in_games": hz["n_fill_games"], "thresholds_4wk_points": [HEDGE_CONSIDER, HEDGE_STRONG]}}
    (OUT / "model_report.json").write_text(json.dumps(report, indent=1, default=float))
    write_markdown(report, scored)
    print(f"\nLive {LIVE} Week {live_week + 1}: {len(scored)} players scored ({int(scored.startable.sum())} startable) | {(scored.status == 'Out now').sum()} out now | "
          f"tiers {scored.tier.value_counts().to_dict()} | advice {scored.advice.value_counts().to_dict()}")


def write_markdown(r: dict, scored: pd.DataFrame) -> None:
    def table(m):
        rows = [("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"), ("Brier", "brier"), ("Brier skill vs. base rate", "brier_skill_vs_base_rate"),
                ("Log loss", "log_loss"), ("Precision @ threshold", "precision"), ("Recall @ threshold", "recall"), ("F1 @ threshold", "f1"),
                ("Top-decile lift", "top_decile_lift")]
        b, c = m["baseline_logistic"]["test"], m["challenger_boosted"]["test"]
        return "\n".join(["| Metric | Logistic (baseline) | Boosted (challenger) |", "|---|---|---|"] +
                         [f"| {n} | {b[k]} | {c[k]} |" for n, k in rows] +
                         ["", f"Chosen: `{m['chosen']}`. Leave-one-season-out ROC-AUC {m['season_cv_mean']['roc_auc']} "
                              "(by season: " + ", ".join(f"{x['season']} {x['roc_auc']}" for x in m["season_cv"]) + ")"])
    A, B, C = r["next_week_model"], r["injury_report_model"], r["time_out_model"]
    ee = C["episode_eval"]
    fmt = lambda v: "—" if v != v else f"{v:.0%}"
    lines = ["# Fantasy injury model report", "",
             f"Startable 12-team PPR players ({', '.join(f'{k} top {v}' for k, v in r['population'].items())}). "
             f"Train {TRAIN[0]}–{TRAIN[-1]}, validate {VAL[0]}–{VAL[-1]}, test {TEST[0]}. "
             f"{r['recovered_pre2020_ir']} pre-2020 IR placements recovered from season-end rosters.", "",
             "## A. Next week: will a player who played last week miss the next game?", "", table(A), "",
             "High-confidence flags on the test season:", "", "| Predicted ≥ | Came true | Flags per week | Share of all injuries caught |", "|---|---|---|---|"] + \
            [f"| {h['threshold']:.0%} | {h['hit_rate']:.0%} | {h['per_week']} | {h['share_of_all_injuries']:.0%} |" for h in A["high_confidence"] if h["hit_rate"] is not None] + \
            ["", "Top features:", ""] + [f"- {f['label']}: {f['mean']:.5f}" for f in A["permutation_importance"][:10]] + \
            ["", "## B. Injury report: will a Questionable/Doubtful player sit?", "", table(B), "",
             f"Players ruled Out sat {r['data_validation']['out_rule_sat_rate']:.1%} of the time (rule).", "",
             "## C. Time out: once out, when does he return?", "",
             "Hazard level (returns next game; one row per game missed):", "", table(C), "",
             f"Episode level, forecast from the first missed game ({ee['episodes']} test episodes): C-index {ee['c_index']}; "
             f"back-after-one-game AUC {ee['back_after_one_game_auc']}; out-4+ AUC {ee['out_4plus_auc']} (Brier {ee['out_4plus_brier']}); "
             f"mean error {ee['mean_abs_error_games']} games vs. {ee['naive_mean_abs_error_games']} for always guessing the median; "
             f"within one game {ee['within_one_game']:.0%}.", "",
             "Top features:", ""] + [f"- {f['label']}: {f['mean']:.5f}" for f in C["permutation_importance"][:8]] + \
            ["", "## Live", "", "| Player | Pos | Status | Misses next week | Source | Plays / out 1 / 2–3 / 4+ | Backup | Advice |", "|---|---|---|---|---|---|---|---|"] + \
            [f"| {x.player} | {x.pos} | {x.status} | {fmt(x.p_next_week)} | {x.source} | {fmt(x.plays)} / {fmt(x.out_1)} / {fmt(x.out_2_3)} / {fmt(x.out_4plus)} | {x.backup or '—'} | {x.advice} |"
             for x in scored.dropna(subset=["p_next_week"]).sort_values("p_next_week", ascending=False).head(20).itertuples()]
    (REPORTS / "injury_model_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
