# Outside matches (Cricsheet)

Data from **[Cricsheet](https://cricsheet.org)**, licensed under the
[Open Data Commons Attribution License (ODC-By 1.0)](https://opendatacommons.org/licenses/by/1-0/). Thanks to Cricsheet for making it free.

These matches are used **only to test** the final lists (notebook 14). They were never used to build, tune or choose anything.

| Folder | What it is |
|---|---|
| `cricsheet_raw/` | The exact men's T20 downloads used, taken from cricsheet.org on 11 Nov 2026: `t20s`, `it20s`, `icc_mens_t20_world_cup`, `ipl`, `bbl`, `bpl`, `cpl`, `ilt`, `lpl`, `mlc`, `ntb` (Vitality Blast), `psl`, `sat` (SA20) and `sma` (Syed Mushtaq Ali), each as `<name>_male_json.zip`, plus Cricsheet's `people.csv` and `names.csv`. |
| `parsed/` | The same matches from 2022 on, flattened by notebook 03: `matches.parquet` (4,791 matches) and `balls.parquet` (one row per delivery). |

Notebook 03 removes matches that also appear in the competition's supplied data, and 12 duplicates, which leaves the **3,993 outside matches (907,349 balls)** used in the test.

The competition's supplied data is **not** in this repository and never will be.
