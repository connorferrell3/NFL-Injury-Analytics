# Fantasy injury model report

Startable 12-team PPR players (QB top 12, RB top 30, WR top 36, TE top 12). Train 2013–2022, validate 2023–2024, test 2025. 197 pre-2020 IR placements recovered from season-end rosters.

## A. Next week: will a player who played last week miss the next game?

| Metric | Logistic (baseline) | Boosted (challenger) |
|---|---|---|
| ROC-AUC | 0.7867 | 0.7978 |
| PR-AUC | 0.288 | 0.3115 |
| Brier | 0.03962 | 0.03947 |
| Brier skill vs. base rate | 0.1299 | 0.1333 |
| Log loss | 0.1608 | 0.158 |
| Precision @ threshold | 0.2295 | 0.2377 |
| Recall @ threshold | 0.4828 | 0.5 |
| F1 @ threshold | 0.3111 | 0.3222 |
| Top-decile lift | 4.8 | 4.97 |

Chosen: `logistic_C0.03`. Leave-one-season-out ROC-AUC 0.7615 (by season: 2013 0.7867, 2014 0.7625, 2015 0.7956, 2016 0.7793, 2017 0.7255, 2018 0.7179, 2019 0.7795, 2020 0.7477, 2021 0.7371, 2022 0.7615, 2023 0.7941, 2024 0.7263, 2025 0.7862)

High-confidence flags on the test season:

| Predicted ≥ | Came true | Flags per week | Share of all injuries caught |
|---|---|---|---|
| 25% | 48% | 1.47 | 21% |
| 40% | 50% | 0.71 | 10% |
| 60% | 67% | 0.18 | 3% |

Top features:

- left last game early (snap drop): 0.03311
- workload last game: 0.00237
- coming off a bye: 0.00178
- recent snap share: 0.00159
- played through an injury last game: 0.00141
- workload last season: 0.00138
- last injury was soft-tissue: 0.00103
- played through a soft-tissue injury: 0.00090
- games missed last season: 0.00089
- was listed Questionable/Doubtful last week: 0.00085

## B. Injury report: will a Questionable/Doubtful player sit?

| Metric | Logistic (baseline) | Boosted (challenger) |
|---|---|---|
| ROC-AUC | 0.905 | 0.9168 |
| PR-AUC | 0.5077 | 0.5522 |
| Brier | 0.05954 | 0.05353 |
| Brier skill vs. base rate | 0.2335 | 0.3109 |
| Log loss | 0.1953 | 0.1799 |
| Precision @ threshold | 0.4286 | 0.6667 |
| Recall @ threshold | 0.3 | 0.3 |
| F1 @ threshold | 0.3529 | 0.4138 |
| Top-decile lift | 4.96 | 5.46 |

Chosen: `logistic_C1.0`. Leave-one-season-out ROC-AUC 0.9273 (by season: 2013 0.9705, 2014 0.9563, 2015 0.9364, 2016 0.9139, 2017 0.8857, 2018 0.9308, 2019 0.9394, 2020 0.9301, 2021 0.9005, 2022 0.9033, 2023 0.9263, 2024 0.9562, 2025 0.9059)

Players ruled Out sat 100.0% of the time (rule).

## C. Time out: once out, when does he return?

Hazard level (returns next game; one row per game missed):

| Metric | Logistic (baseline) | Boosted (challenger) |
|---|---|---|
| ROC-AUC | 0.8739 | 0.9006 |
| PR-AUC | 0.6055 | 0.6909 |
| Brier | 0.09977 | 0.09163 |
| Brier skill vs. base rate | 0.3235 | 0.3787 |
| Log loss | 0.3152 | 0.2971 |
| Precision @ threshold | 0.625 | 0.8462 |
| Recall @ threshold | 0.303 | 0.3333 |
| F1 @ threshold | 0.4082 | 0.4783 |
| Top-decile lift | 2.94 | 4.41 |

Chosen: `boosted_d2_lr0.03_it300`. Leave-one-season-out ROC-AUC 0.8415 (by season: 2013 0.8646, 2014 0.8431, 2015 0.866, 2016 0.8638, 2017 0.8491, 2018 0.8358, 2019 0.7619, 2020 0.7928, 2021 0.7971, 2022 0.8371, 2023 0.8976, 2024 0.8291, 2025 0.9009)

Episode level, forecast from the first missed game (58 test episodes): C-index 0.8512; back-after-one-game AUC 0.9395; out-4+ AUC 0.9567 (Brier 0.1086); mean error 0.97 games vs. 1.42 for always guessing the median; within one game 82%.

Top features:

- games left in season: 0.09870
- game status: 0.07219
- on Injured Reserve: 0.02873
- type of injury: 0.01934
- practice participation: 0.01598
- games missed so far: 0.01136
- games missed last season: 0.00426
- week of season: 0.00162

## Live

| Player | Pos | Status | Misses next week | Source | Plays / out 1 / 2–3 / 4+ | Backup | Advice |
|---|---|---|---|---|---|---|---|
| Josh Oliver | TE | Out now | 97% | time-out model | 3% / 3% / 7% / 87% | Gavin Bartholomew | Starter out: backup ~waiver level |
| David Njoku | TE | Out now | 97% | time-out model | 3% / 3% / 9% / 86% | Oronde Gadsden II | Starter out: backup ~waiver level |
| De'Von Achane | RB | Out now | 97% | time-out model | 3% / 3% / 8% / 87% | Ollie Gordon | Starter out: add backup |
| Jack Bech | WR | Out now | 97% | time-out model | 3% / 3% / 8% / 86% | Jalen Nailor | Starter out: backup ~waiver level |
| A.J. Brown | WR | Out now | 97% | time-out model | 3% / 5% / 9% / 83% | Mack Hollins | Starter out: add backup |
| Ja'Kobi Lane | WR | Out now | 97% | time-out model | 3% / 5% / 9% / 83% | Rashod Bateman | Starter out: add backup |
| Ronnie Rivers | RB | Out now | 97% | time-out model | 3% / 3% / 10% / 84% | Blake Corum | Starter out: add backup |
| Arian Smith | WR | Out now | 97% | time-out model | 3% / 3% / 10% / 84% | Isaiah Williams | Starter out: add backup |
| KeAndre Lambert-Smith | WR | Out now | 97% | time-out model | 3% / 5% / 10% / 82% | Quentin Johnston | Starter out: add backup |
| Jake Tonges | TE | Out now | 97% | time-out model | 3% / 5% / 10% / 82% | Luke Farrell | Starter out: backup ~waiver level |
| Dylan Sampson | RB | Out now | 97% | time-out model | 3% / 5% / 10% / 82% | Raheim Sanders | Starter out: add backup |
| Jonathon Brooks | RB | Out now | 97% | time-out model | 3% / 3% / 11% / 83% | AJ Dillon | Starter out: add backup |
| Jordan Mason | RB | Out now | 97% | time-out model | 3% / 6% / 10% / 81% | DeeJay Dallas | Starter out: add backup |
| Demarcus Robinson | WR | Out now | 95% | time-out model | 5% / 5% / 15% / 75% | Jacob Cowing | Starter out: add backup |
| Jalen McMillan | WR | Out now | 94% | time-out model | 6% / 6% / 14% / 74% | Emeka Egbuka | Starter out: add backup |
| Jonah Coleman | RB | Out now | 94% | time-out model | 6% / 5% / 17% / 72% | J.K. Dobbins | Starter out: add backup |
| Andrei Iosivas | WR | Out now | 94% | time-out model | 6% / 6% / 18% / 70% | Dohnte Meyers | Starter out: add backup |
| Alec Pierce | WR | Out now | 94% | time-out model | 6% / 6% / 18% / 69% | Laquon Treadwell | Starter out: backup ~waiver level |
| Jayden Reed | WR | Out now | 94% | time-out model | 6% / 6% / 18% / 69% | Skyy Moore | Starter out: add backup |
| Omar Cooper | WR | Out now | 94% | time-out model | 6% / 10% / 17% / 67% | Isaiah Williams | Starter out: add backup |
