"""Scrape Pro Bowl, All-Pro and NFL Top 100 lists from Wikipedia (PFR blocks scripted access).

Output: data/raw/awards.csv with one row per (season, award, name, team_text).
`season` is the NFL season the honor recognizes: the 2024 Pro Bowl Games honor 2023, and
the "Top 100 Players of 2024" (player-voted, released before the 2024 season) ranks 2023.
Names are matched to player IDs later, in build_dataset.py.
"""
import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
HEADERS = {"User-Agent": "NFL-Injury-Analytics/1.0 (personal research project)"}
SEASONS = range(2019, 2026)  # honors from the prior season feed 2020-2026 key players

# Pro Bowl page title for the game honoring each season
PRO_BOWL_PAGES = {2019: "2020_Pro_Bowl", 2020: "2021_Pro_Bowl", 2021: "2022_Pro_Bowl",
                  2022: "2023_Pro_Bowl_Games", 2023: "2024_Pro_Bowl_Games",
                  2024: "2025_Pro_Bowl_Games", 2025: "2026_Pro_Bowl_Games"}

TEAM_WORDS = {"Arizona", "Atlanta", "Baltimore", "Buffalo", "Carolina", "Chicago", "Cincinnati",
              "Cleveland", "Dallas", "Denver", "Detroit", "Green Bay", "Houston", "Indianapolis",
              "Jacksonville", "Kansas City", "Las Vegas", "LA Chargers", "LA Rams", "Los Angeles",
              "Miami", "Minnesota", "New England", "New Orleans", "NY Giants", "NY Jets",
              "New York", "Philadelphia", "Pittsburgh", "San Francisco", "Seattle", "Tampa Bay",
              "Tennessee", "Washington"}
NICKNAMES = ("Cardinals Falcons Ravens Bills Panthers Bears Bengals Browns Cowboys Broncos Lions "
             "Packers Texans Colts Jaguars Chiefs Raiders Chargers Rams Dolphins Vikings Patriots "
             "Saints Giants Jets Eagles Steelers 49ers Seahawks Buccaneers Titans Commanders "
             "Football Team Redskins").split()


def is_team_link(a) -> bool:
    text, title = a.get_text(strip=True), a.get("title", "")
    return (text in TEAM_WORDS or any(n in title for n in NICKNAMES)
            or "season" in title or "Pro Bowl" in title)


def roster_links(page: str) -> list[tuple[str, str]]:
    """(player name, team text) pairs from every wikitable on a page."""
    html = requests.get(f"https://en.wikipedia.org/wiki/{page}", headers=HEADERS, timeout=60)
    html.raise_for_status()
    soup = BeautifulSoup(html.text, "lxml")
    out = []
    for table in soup.select("table.wikitable"):
        header = table.get_text(" ", strip=True)[:200].lower()
        if not any(k in header for k in ("position", "offense", "defense", "special")):
            continue
        for td in table.select("td"):
            links = [a for a in td.select("a[href*='/wiki/']") if "/wiki/File:" not in a["href"]
                     and "/wiki/Help:" not in a["href"] and not a.find_parent("sup")]
            for i, a in enumerate(links):
                if is_team_link(a) or not re.search(r"[A-Z][a-z'.]+", a.get_text()):
                    continue
                team = links[i + 1].get_text(strip=True) if i + 1 < len(links) and is_team_link(links[i + 1]) else ""
                out.append((a.get_text(strip=True), team))
    return out


def top100(page: str) -> list[tuple[str, str]]:
    """(player name, prior-season team) pairs from the ranked table on a Top 100 page."""
    html = requests.get(f"https://en.wikipedia.org/wiki/{page}", headers=HEADERS, timeout=60)
    html.raise_for_status()
    soup = BeautifulSoup(html.text, "lxml")
    for table in soup.select("table.wikitable"):
        head = [th.get_text(" ", strip=True).lower() for th in table.select("tr")[0].select("th")]
        if not head or head[0] != "rank" or "player" not in head:
            continue
        out = []
        for tr in table.select("tr")[1:]:
            cells = tr.select("td, th")
            if len(cells) > 3:
                link = cells[1].select_one("a[href*='/wiki/']")
                out.append(((link or cells[1]).get_text(strip=True), cells[3].get_text(" ", strip=True)))
        return out
    raise ValueError(f"no ranked table on {page}")


def main() -> None:
    rows = []
    for season in SEASONS:
        for award, page, parse in [("pro_bowl", PRO_BOWL_PAGES[season], roster_links),
                                   ("all_pro", f"{season}_All-Pro_Team", roster_links),
                                   ("top100", f"NFL_Top_100_Players_of_{season + 1}", top100)]:
            pairs = parse(page)
            rows += [{"season": season, "award": award, "name": n, "team_text": t} for n, t in pairs]
            print(f"  {season} {award:8s} {len(pairs):3d} names  ({page})")
    df = pd.DataFrame(rows).drop_duplicates()
    df.to_csv(RAW / "awards.csv", index=False)
    print(f"Saved data/raw/awards.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
