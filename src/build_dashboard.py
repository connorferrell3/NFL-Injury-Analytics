"""Package processed data into the dashboard, compute headline findings, write summaries.

Reads data/processed/*.csv (from build_dataset.py), injects a compact JSON payload
into dashboard/template.html, and writes dashboard/index.html (self-contained).
Headline findings are computed on completed seasons only; the current season is shown
separately because a few weeks of data isn't comparable to a full season.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC, DASH = ROOT / "data" / "processed", ROOT / "dashboard"
KEY = ["season", "team", "pfr_player_id"]
CURRENT_SEASON = 2026
UNSPEC = "Unspecified (IR)"

# bit flags for key-player criteria, mirrored in template.html
FLAGS = {"is_snap_key": 1, "is_pedigree_key": 2, "first_round_rookie": 4,
         "salary_top30": 8, "pro_bowl_prev": 16, "all_pro_prev": 32}
DEFINITIONS = {
    "Snap share": "is_snap_key",
    "Pedigree": "is_pedigree_key",
    "1st-round rookie contract": "first_round_rookie",
    "Salary top 30%": "salary_top30",
    "Pro Bowl (prior season)": "pro_bowl_prev",
    "All-Pro (prior season)": "all_pro_prev",
    "Everyone who took a snap": None,
}


def p_two_prop(a: pd.Series, b: pd.Series) -> float:
    """Two-sided p-value, two-proportion z-test on boolean series."""
    x1, n1, x2, n2 = a.sum(), len(a), b.sum(), len(b)
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (x1 / n1 - x2 / n2) / se
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def fmt_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.2f}" if p >= 0.01 else f"p = {p:.3f}"


def pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def with_outcomes(players: pd.DataFrame, eps: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = (eps.groupby(keys).agg(absences=("games_missed", "size"), games_missed=("games_missed", "sum"),
                               season_ending=("season_ending", "any"), ongoing=("ongoing", "any")).reset_index())
    out = players.merge(g, on=keys, how="left")
    out["absences"] = out["absences"].fillna(0).astype(int)
    out["games_missed"] = out["games_missed"].fillna(0).astype(int)
    for c in ("season_ending", "ongoing"):
        out[c] = out[c].fillna(False).astype(bool)
    out["injured"] = out["absences"] > 0
    return out


def next_season_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Player-level injured flag in season Y joined to season Y+1 (both seasons present)."""
    py = df.groupby(["pfr_player_id", "season"])["injured"].max().reset_index()
    return py.merge(py.assign(season=py["season"] - 1), on=["pfr_player_id", "season"], suffixes=("", "_next"))


def same_region(eps: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """For consecutive absences by the same player, how often is the next one the same region?"""
    e = eps[eps["body_region"] != UNSPEC].sort_values(["pfr_player_id", "season", "start_week"]).copy()
    e["next_region"] = e.groupby("pfr_player_id")["body_region"].shift(-1)
    pairs = e.dropna(subset=["next_region"]).assign(same=lambda d: d["body_region"] == d["next_region"])
    share = e["body_region"].value_counts(normalize=True)
    by = pairs.groupby("body_region").agg(repeat_pairs=("same", "size"), same_region=("same", "mean"))
    by["chance"] = share.reindex(by.index)
    by = by[by["repeat_pairs"] >= 40].sort_values("same_region", ascending=False).round(3)
    return by, pairs["same"].mean(), float((share ** 2).sum())


def findings(ps: pd.DataFrame, ep: pd.DataFrame, fan: pd.DataFrame) -> dict:
    done = ps[ps["season"] < CURRENT_SEASON]
    sk = done[done["is_snap_key"]]
    keyset = set(zip(sk["season"], sk["team"], sk["pfr_player_id"]))
    sk_eps = ep[[k in keyset for k in zip(ep["season"], ep["team"], ep["pfr_player_id"])]]
    by_season = sk.groupby("season")["injured"].mean()

    nxt = next_season_pairs(sk)
    hurt, fine = nxt[nxt["injured"]]["injured_next"], nxt[~nxt["injured"]]["injured_next"]
    reg, same_all, chance = same_region(sk_eps)
    pb, not_pb = sk[sk["pro_bowl_prev"]]["injured"], sk[~sk["pro_bowl_prev"]]["injured"]
    rk, not_rk = sk[sk["first_round_rookie"]]["injured"], sk[~sk["first_round_rookie"]]["injured"]
    regions = sk_eps.groupby("body_region")["games_missed"].agg(["size", "sum"])
    regions["share"] = regions["size"] / regions["size"].sum()
    ach = sk_eps[sk_eps["body_region"] == "Achilles"]["season_ending"].mean()
    knee = regions.loc["Knee"]
    se_eps = sk_eps[sk_eps["season_ending"]]
    se_by = se_eps.groupby("season").size()
    se_rate = (se_by / sk.groupby("season").size()).round(3)
    se_unspec = (se_eps["body_region"] == UNSPEC).mean()

    f = fan[fan["season"] < CURRENT_SEASON].assign(missed4=lambda d: d["games_missed"] >= 4,
                                                   rnd=lambda d: (d["adp_rank"] - 1) // 12 + 1)
    qb, rb, wr = (f[f["pos_group"] == p]["injured"] for p in ("QB", "RB", "WR"))
    r1, r24 = f[f["rnd"] == 1]["injured"], f[f["rnd"] > 1]["injured"]
    fn = next_season_pairs(f)
    fh, ff = fn[fn["injured"]]["injured_next"], fn[~fn["injured"]]["injured_next"]

    cur = ps[(ps["season"] == CURRENT_SEASON) & ps["is_snap_key"]]
    cf = fan[fan["season"] == CURRENT_SEASON]
    cur_week = int(ep.loc[ep["season"] == CURRENT_SEASON, "end_week"].max()) if (ep["season"] == CURRENT_SEASON).any() else 0

    cards = [
        {"group": "NFL", "id": "starter_rate", "value": pct(sk["injured"].mean()), "title": "of starters miss at least one game to injury each season",
         "detail": f"Every season from 2020 to 2025 landed between {pct(by_season.min())} and {pct(by_season.max())}.",
         "bars": [[str(y), round(v, 3)] for y, v in by_season.items()]},
        {"group": "NFL", "id": "season_ending", "value": f"~{se_by.mean():.0f}", "title": "starters suffer a season-ending injury each season",
         "detail": f"Between {se_by.min()} and {se_by.max()} a year, or {pct(se_rate.min())}–{pct(se_rate.max())} of starters. "
                   f"{pct(se_unspec)} of them never get a body part on the report because the player goes straight to IR.",
         "bars": [[str(y), round(v, 3)] for y, v in se_rate.items()]},
        {"group": "NFL", "id": "recurrence", "value": f"+{(hurt.mean() - fine.mean()) * 100:.0f} pts", "title": "more likely to get hurt next season if hurt this season",
         "detail": f"{pct(hurt.mean())} vs. {pct(fine.mean())} for starters who stayed healthy ({fmt_p(p_two_prop(hurt, fine))}, {len(nxt):,} player pairs).",
         "bars": [["Hurt last season", round(hurt.mean(), 3)], ["Healthy last season", round(fine.mean(), 3)]]},
        {"group": "NFL", "id": "same_region", "value": f"{same_all / chance:.1f}×", "title": "more likely than chance that a repeat injury hits the same body part",
         "detail": f"{pct(same_all)} of repeat absences are the same region, vs. {pct(chance)} by chance. Knees repeat {pct(reg.loc['Knee', 'same_region'])} of the time, hamstrings {pct(reg.loc['Hamstring', 'same_region'])}.",
         "bars": [["Same region", round(same_all, 3)], ["Chance", round(chance, 3)]]},
        {"group": "NFL", "id": "pro_bowl", "value": pct(pb.mean()), "title": "of prior-season Pro Bowlers miss time, the highest of any group",
         "detail": f"vs. {pct(not_pb.mean())} of other starters ({fmt_p(p_two_prop(pb, not_pb))}). Stars play more snaps, so they're exposed more.",
         "bars": [["Pro Bowlers", round(pb.mean(), 3)], ["Other starters", round(not_pb.mean(), 3)]]},
        {"group": "NFL", "id": "rookies", "value": pct(rk.mean()), "title": "of first-round picks on rookie deals miss time, the same as everyone else",
         "detail": f"vs. {pct(not_rk.mean())} of other starters ({fmt_p(p_two_prop(rk, not_rk))}). Youth doesn't protect them.",
         "bars": [["1st-rd rookies", round(rk.mean(), 3)], ["Other starters", round(not_rk.mean(), 3)]]},
        {"group": "NFL", "id": "knee", "value": pct(knee["share"]), "title": "of starter absences are knee injuries, the most common and costliest",
         "detail": f"Knees cost {int(knee['sum']):,} games over six seasons. Achilles injuries are rarer but {pct(ach)} of them end the season."},
        {"group": "Fantasy", "id": "fan_rate", "value": pct(f["injured"].mean()), "title": "of top-48 fantasy picks miss at least one week to injury",
         "detail": f"{pct(f['missed4'].mean())} miss four weeks or more. In a 12-team league, that's about {f['missed4'].mean() * 4:.0f} of each manager's first four picks.",
         "bars": [["Missed 1+ week", round(f["injured"].mean(), 3)], ["Missed 4+ weeks", round(f["missed4"].mean(), 3)]]},
        {"group": "Fantasy", "id": "fan_pos", "value": f"{pct(qb.mean())} vs {pct(rb.mean())}", "title": "QBs vs. RBs: quarterbacks are the safest early picks",
         "detail": f"QB vs. RB {fmt_p(p_two_prop(qb, rb))}. RBs and WRs ({pct(wr.mean())}) are statistically the same ({fmt_p(p_two_prop(rb, wr))}).",
         "bars": [["QB", round(qb.mean(), 3)], ["RB", round(rb.mean(), 3)], ["WR", round(wr.mean(), 3)]]},
        {"group": "Fantasy", "id": "fan_round", "value": "No edge", "title": "Round 1 picks get hurt as often as Rounds 2–4",
         "detail": f"{pct(r1.mean())} vs. {pct(r24.mean())} ({fmt_p(p_two_prop(r1, r24))}). Paying up doesn't buy durability.",
         "bars": [["Round 1", round(r1.mean(), 3)], ["Rounds 2–4", round(r24.mean(), 3)]]},
        {"group": "Fantasy", "id": "fan_recur", "value": "Weak signal", "title": "“He was hurt last year” barely predicts fantasy injuries",
         "detail": f"{pct(fh.mean())} vs. {pct(ff.mean())} ({fmt_p(p_two_prop(fh, ff))}, only {len(fn)} repeat picks). The NFL-wide effect is real, but too small to see in a 48-player pool.",
         "bars": [["Hurt last year", round(fh.mean(), 3)], ["Healthy last year", round(ff.mean(), 3)]]},
        {"group": f"{CURRENT_SEASON} so far", "id": "live", "value": str(int(cur["ongoing"].sum())), "title": f"snap-share starters are out as of Week {cur_week}",
         "detail": f"{pct(cur['injured'].mean())} of starters have already missed a game, and {int(cf['ongoing'].sum())} of the top 48 fantasy picks are currently out."},
    ]
    reg_out = reg.reset_index().rename(columns={"body_region": "region"})
    return {"cards": cards, "same_region": reg_out.values.tolist(), "current_week": cur_week,
            "recur": {"hurt": round(hurt.mean(), 3), "fine": round(fine.mean(), 3)}}, reg_out


def summaries(ps: pd.DataFrame, fan: pd.DataFrame, reg: pd.DataFrame) -> None:
    done = ps[ps["season"] < CURRENT_SEASON]
    rows = []
    for name, col in DEFINITIONS.items():
        d = done if col is None else done[done[col]]
        rows.append({"definition": name, "players_per_season": round(len(d) / d["season"].nunique()),
                     "pct_injured": d["injured"].mean(), "games_lost_per_player": d["games_missed"].mean(),
                     "season_ending_rate": d["season_ending"].mean(), "avg_snap_share": d["avg_unit_pct"].mean(),
                     "share_also_snap_starter": d["is_snap_key"].mean()})
    comp = pd.DataFrame(rows).round(3)
    comp.to_csv(PROC / "summary_definition_comparison.csv", index=False)
    print(comp.to_string(index=False))

    snap = done[done["is_snap_key"]]
    snap.groupby(["season", "side"])["injured"].mean().round(3).unstack().to_csv(PROC / "summary_pct_injured_by_side.csv")
    reg.to_csv(PROC / "summary_same_region_recurrence.csv", index=False)

    f = fan[fan["season"] < CURRENT_SEASON].assign(missed_4plus=lambda d: d["games_missed"] >= 4,
                                                   adp_round=lambda d: (d["adp_rank"] - 1) // 12 + 1)
    for by in ("season", "pos_group", "adp_round", "team"):
        (f.groupby(by).agg(picks=("injured", "size"), pct_injured=("injured", "mean"),
                           pct_missed_4plus=("missed_4plus", "mean"), games_lost_per_pick=("games_missed", "mean"))
          .round(3).to_csv(PROC / f"summary_fantasy_by_{by}.csv"))


def main() -> None:
    ps = pd.read_csv(PROC / "player_seasons.csv")
    ep = pd.read_csv(PROC / "injury_episodes.csv")
    fan = pd.read_csv(PROC / "fantasy_top48.csv")
    fep = pd.read_csv(PROC / "fantasy_episodes.csv")

    ps_o = with_outcomes(ps, ep, KEY)
    fan_o = with_outcomes(fan, fep, ["season", "pfr_player_id"])
    found, reg = findings(ps_o, ep, fan_o)
    summaries(ps_o, fan_o, reg)
    for c in found["cards"]:
        print(f"[{c['group']}] {c['value']} {c['title']} -- {c['detail']}")

    ids = sorted(set(ps["pfr_player_id"]) | set(fan["pfr_player_id"]))
    pid = {p: i for i, p in enumerate(ids)}
    flags = sum(ps[c].astype(int) * bit for c, bit in FLAGS.items())

    def ep_rows(e: pd.DataFrame) -> list:
        # [season, team, playerIdx, player, pos_group, start_week, games, region,
        #  season_ending, went_on_ir, label, ongoing]
        return e.assign(p=e["pfr_player_id"].map(pid), injury_label=e["injury_label"].fillna(""),
                        season_ending=e["season_ending"].astype(int), went_on_ir=e["went_on_ir"].astype(int),
                        ongoing=e["ongoing"].astype(int))[
            ["season", "team", "p", "player", "pos_group", "start_week", "games_missed",
             "body_region", "season_ending", "went_on_ir", "injury_label", "ongoing"]].values.tolist()

    payload = {
        "current": {"season": CURRENT_SEASON, "week": found["current_week"],
                    "adp_source": fan.loc[fan["season"] == CURRENT_SEASON, "source"].iat[0]},
        "findings": found["cards"],
        "same_region": found["same_region"],
        "recur": found["recur"],
        # [season, team, pos_group, playerIdx, flags, avg_unit_pct x100]
        "ps": ps[["season", "team", "pos_group"]].assign(
            p=ps["pfr_player_id"].map(pid), f=flags, u=(ps["avg_unit_pct"] * 100).round().astype(int)
        ).values.tolist(),
        "ep": ep_rows(ep),
        # [season, adp_rank, adp_formatted, player, pos_group, team, playerIdx]
        "fan": fan.assign(p=fan["pfr_player_id"].map(pid))[
            ["season", "adp_rank", "adp_formatted", "player", "pos_group", "team", "p"]].values.tolist(),
        "fep": ep_rows(fep),
    }
    html = (DASH / "template.html").read_text()
    html = html.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":"), default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
    # full document so it renders correctly as a standalone file (GitHub Pages, local browser)
    html = ('<!doctype html>\n<html lang="en">\n<head>\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + html.replace("</style>", "</style>\n</head>\n<body>", 1) + "\n</body>\n</html>\n")
    (DASH / "index.html").write_text(html)
    print(f"Wrote dashboard/index.html ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
