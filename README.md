# Who delivers when it matters?

**Ranking the most reliable T20 batters and bowlers under pressure**
SupeRR Selector Season 2 (Rajasthan Royals) · Round 1 · Bishal Roy

This repository holds the full, reproducible method behind my Round 1 submission: code, settings, and technical notes. It contains no competition data (see [Data policy](#data-policy)).

---

## The question

The brief: build a data-driven system that measures how batters and bowlers perform in high-pressure match situations, and rank the ten best of each. The aim is to identify the players Rajasthan Royals should back at the IPL 2027 auction.

The data: ball-by-ball records of the IPL and T20 internationals, 2022 to 2026 (883 matches, about 201,000 balls), supplied by the organisers. For testing only, I also used 3,993 public matches from [Cricsheet](https://cricsheet.org).

## The answer

| # | Batter | Role | Strike rate under pressure | Typical batter, same situations | Stays top 10* |
|---:|---|---|---:|---:|---:|
| 1 | Abhishek Sharma | Opener | 184 | 156 | 93% |
| 2 | Suryakumar Yadav | No. 3–4 | 167 | 149 | 93% |
| 3 | Vaibhav Sooryavanshi | Opener | 236 | 176 | 87% |
| 4 | Phil Salt | Opener | 176 | 152 | 82% |
| 5 | Tim David | Finisher | 195 | 152 | 66% |
| 6 | Yashasvi Jaiswal | Opener | 180 | 157 | 73% |
| 7 | Heinrich Klaasen | No. 5–6 | 200 | 152 | 67% |
| 8 | Rovman Powell | No. 5–6 | 177 | 145 | 56% |
| 9 | Josh Inglis | No. 3–4 | 164 | 149 | 53% |
| 10 | Priyansh Arya | Opener | 202 | 172 | 37% |

| # | Bowler | Role | Economy under pressure | Typical bowler, same type and situations | Stays top 10* |
|---:|---|---|---:|---:|---:|
| 1 | Jasprit Bumrah | Pace · new ball & death | 8.20 | 10.03 | 93% |
| 2 | Lungi Ngidi | Pace · new ball & death | 8.96 | 9.95 | 64% |
| 3 | Josh Hazlewood | Pace · new ball | 8.69 | 9.77 | 74% |
| 4 | Benjamin Sears | Pace · middle overs | 9.97 | 10.02 | 61% |
| 5 | Nathan Ellis | Pace · death | 8.42 | 10.05 | 64% |
| 6 | Rashid Khan | Spin · middle overs | 7.41 | 8.53 | 62% |
| 7 | Sunil Narine | Spin · middle overs | 7.10 | 8.77 | 54% |
| 8 | Brydon Carse | Pace · middle overs | 9.00 | 9.08 | 36% |
| 9 | Mitchell Santner | Spin · middle overs | 7.30 | 8.48 | 46% |
| 10 | Jamie Overton | Pace · middle overs | 7.48 | 9.36 | 26% |

\* Share of 1,000 rebuilds of the ranking on reshuffled matches (a match-level bootstrap) in which the player stayed in the top 10.

The order comes from the full rating, which counts runs, dismissals or wickets, and the weight of each ball. It does not come from the strike rate or economy columns alone. Sears and Carse, for example, are in the list for wickets: each takes one every 16 pressure balls.

## Method

The method follows the five items the brief asks the report to cover.

### 1. Pressure definition

- Before every ball, a win-probability model estimates the batting side's chance of winning. **Impact** is how far the next ball could move that chance, averaged over the possible outcomes and weighted by how likely each one is.
- **Pressure = impact × match importance.** Importance is 1.5 for knockouts, 1.25 for must-win games, 1 for normal games and 0.75 for dead rubbers.
- The top 20% of balls are **pressure balls**. In the ratings every ball counts, weighted by its pressure percentile squared.
- **How the definition was chosen.**
  - I started with 38 candidate pressure signals: scoreboard, recent overs, the bowler's spell, the match and the player.
  - Each signal faced three pre-registered tests: does it change how players perform, does it decide matches, and does it hold in every season and competition?
  - Eight combined definitions were then scored against each other, including two published ones (Bhattacharjee & Lemmer, Thomson et al.), clustering and PCA. Impact scored highest.
- **Evidence.** On pressure balls the win chance moves 2.2 times as much as on an average ball, and batters get out 8% more often than at the same stage of an innings.
- **Model choice.** The win-probability model is a logistic regression with curves on the game state. It beat ten machine-learning alternatives on unseen matches; see [docs/win_probability_model.md](docs/win_probability_model.md).

Code: `src/p03_context_models.py`, `src/p03b_model_zoo.py`, `src/p04_pressure.py`.

### 2. Performance measurement

- Every ball is compared with what a **typical player would have done in exactly that situation**.
  - The typical outcome comes from a player-blind LightGBM model over 41 facts about the situation: over, score, wickets, target, recent wickets, how long the batter has been in, ground, season and competition.
  - Bowlers are compared with bowlers of the same type (pace or spin).
- **Batters** gain runs above typical and lose the situational cost of a dismissal.
- **Bowlers** gain runs saved and their own wickets (run-outs excluded). Wides and no-balls count against them.
- **Rating.**
  - Pressure-weighted value per ball, shrunk towards average by empirical Bayes: `rating = mean × n / (n + k)`.
  - At least 60 pressure balls are needed to be ranked.
- **Why not each player's own usual level?** The brief mentions comparing performance under pressure with a player's usual level, so I measured that first.
  - The pressure-minus-usual gap does not repeat: one half of a player's matches says nothing about the other half.
  - Random relabelling of pressure balls (200 shuffles) produces the same spread between players.
  - The ranking therefore uses the level under pressure against a typical player. The gap gets zero weight, a weight chosen by out-of-sample tests.

Code: `src/p05_value.py`, `src/p06_ratings.py`, `src/ratings_lib.py`.

### 3. Fair comparisons

| Brief item | Treatment |
|---|---|
| Sample sizes | Empirical-Bayes shrinkage, plus a floor of 60 pressure balls. |
| Player eligibility | All players are rated. The final lists include only players with a match in 2025 or 2026 (configurable in `config.yaml`, `eligibility`). Excluded players remain in every baseline and every test. |
| Match contexts | The baseline conditions on ground (venue par), season, competition level, tournament stage, toss and day/night. Bowlers are compared with their own type (before that rule, 9 of the top 10 bowlers were spinners). |
| IPL and T20I data | Both are used, with four competition levels (IPL, full-member T20I, associate T20I, A-team), each with its own typical player. All seasons are weighted equally. |

### 4. Evidence for the top 10

- Every pick beat a typical player in the same situations (charts in the submitted report).
- Settings (value metric, rating model, shrinkage, minimum sample) were chosen on **52 out-of-sample splits**: 2 rolling-origin and 50 random match splits. The yardstick was the players' performance on pressure balls in matches the method never saw.
- On those splits:
  - the batting top 10 beat a typical batter in 51 of 52 tests (about +12 runs per 100 pressure balls);
  - the bowling top 10 saved runs in 50 of 52 tests (about 8 runs per 100 pressure balls);
  - higher-ranked players did better in 51 of 52.

Code: `src/p09_experiments.py`, `src/p10_rank.py`, `src/p14b_family_check.py`.

### 5. Reliability and limitations

- **External exam.** The models trained on the supplied data were applied unchanged to 3,993 public matches (T20 World Cups, franchise and domestic leagues, other T20Is). On pressure balls in those matches:
  - the top 10s beat a typical player by +4.2 runs (batters) and +10.8 runs saved (bowlers) per 100 pressure balls;
  - matches present in both sources (766), plus 12 duplicates within the public data, were removed.
- **Robustness.** I changed one choice at a time: pressure definition, threshold (10% to 30%), value metric, rating model and data scope. A median of 8 of the 10 names stay.
- **Known limits, stated openly.**
  - Batters are much harder to judge than bowlers. Only about 6% of the spread between batters in one season is skill, and the 2026 batting list did not hold on 2026 alone.
  - On the public matches, plain strike rate under pressure picked a better batting top 10. On the 52 internal splits my ratings did about five times better.
  - On the public matches, a bowling top 10 built on all balls (ignoring pressure) did slightly better at the very top (+12.7 against +10.8 runs saved per 100 pressure balls). My ranking still ordered the whole pool of bowlers better (rank correlation 0.34 against 0.24).
  - First-innings pressure is measured less sharply (see the technical note).
  - Rebuilt on IPL matches only, 4 of the 10 bowlers stay.
  - There is no fielding, dropped-catch, pitch or weather data.
  - The 2026 season was examined more than once during development, so it is not a perfectly clean holdout.

Code: `src/p11_robustness.py`, `src/p14_external.py`.

## Technical notes

- [docs/win_probability_model.md](docs/win_probability_model.md): why a logistic regression beat decision trees, random forests, extra trees, XGBoost, LightGBM, CatBoost, k-nearest neighbours and three ensembles. It covers the match-fingerprint problem, learning curves, the logistic design and end-game calibration.
- `docs/results/`: the model-comparison tables behind that note.
- `docs/figures/`: model comparison, win-probability calibration, the chase pressure map and the external exam (in win-chance points).

## Repository layout

```
config.yaml                every threshold, weight, seed and path
requirements.txt
src/
  common.py                paths, config, IO, eligibility helpers
  ratings_lib.py           rating models (ridge APM, empirical Bayes, composite)
  run_all.py               runs the pipeline in order
  p00_audit.py             data audit and cleaning
  p00x_external.py         Cricsheet parsing and player linking
  p01_state.py             ball-state reconstruction, match context (Elo, venue par, tables)
  p02_features.py          feature store and catalogue
  p03_context_models.py    ball-outcome, win-probability and run-expectancy models
  p03b_model_zoo.py        overfitting check: 10 alternatives, fingerprint test, learning curves
  p04_pressure.py          pressure signals, pre-registered tests, definition choice
  p05_value.py             value per ball (four candidates)
  p06_ratings.py           rating models and the "rise above your usual level" test
  p07_unsupervised.py      usage-based roles, pressure regimes, anomaly checks (insight only)
  p08_selection.py         feature selection scorecard
  p09_experiments.py       52-split experiment grid and pre-registered selection
  p10_rank.py              final rankings, bootstrap intervals, P(top 10)
  p11_robustness.py        sensitivity, ablations, negative controls, noise checks
  p12_figures.py           figures from saved tables
  p12_report.py            plain-language methodology report (Markdown)
  p13_notebook.py          evidence notebook (every number next to its source table)
  p14_external.py          external exam on public data
  p14b_family_check.py     evidence for the rating-model choice
  p16_slides_pdf.py        the submitted slide deck (needs local image assets, not included)
docs/
  win_probability_model.md
  results/                 model-comparison tables
  figures/
```

## Reproducing

1. Install Python 3.12 and the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place the supplied data pack (not distributed) in `6af3166e7b922_data_pack/` at the repository root. The path is set in `config.yaml` under `paths.raw`.
3. For the external exam, download these men's T20 JSON zips from [cricsheet.org/downloads](https://cricsheet.org/downloads/) into `external/cricsheet_raw/`, together with `people.csv` and `names.csv`:
   `it20s_male_json.zip`, `t20s_male_json.zip`, `icc_mens_t20_world_cup_male_json.zip`, `ipl_male_json.zip`, `bbl_male_json.zip`, `bpl_male_json.zip`, `cpl_male_json.zip`, `ilt_male_json.zip`, `lpl_male_json.zip`, `mlc_male_json.zip`, `ntb_male_json.zip`, `psl_male_json.zip`, `sat_male_json.zip`, `sma_male_json.zip`.
4. Run the pipeline:
   ```bash
   python src/run_all.py            # or run the scripts one by one in the order listed in run_all.py
   ```
   Outputs go to `outputs/` (tables, figures, logs, models). That folder is not tracked.

Every random step uses the seed in `config.yaml` (2026). The evidence notebook (`src/p13_notebook.py`) shows every reported number next to the table it comes from.

## Data policy

- **The supplied competition data is not in this repository**, and nothing derived from it at ball level is either. The organisers' rules do not allow it to be shared. The `.gitignore` is an allow-list, so only code and documentation can be committed.
- **Cricsheet data** is © Cricsheet, under the Open Data Commons Attribution License. It is used for testing only and is not redistributed here.
- Player images and team branding used in the submitted deck are not part of this repository.

## Author

Bishal Roy · SupeRR Selector Season 2, Round 1 (November 2026)
