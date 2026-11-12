"""P2 - Feature store.
Adds player-history pointers (pre-match information only) to the ball-state table and writes the
feature catalogue (family, description, which models may use it)."""
import numpy as np
import pandas as pd
from common import load, save, save_table, save_json, get_logger

import os
TAG = os.environ.get("RR_TAG", "")
log = get_logger("p02_features" + TAG)
b = load("ball_state" + TAG)
mt = load("matches_ctx" + TAG)[["match_id", "match_date"]]
b["lvl_group"] = np.select([b.level.eq("IPL"), b.level.eq("A-team")], ["IPL", "A"], "INT")

# ---------------------------------------------------------------- batter history (pre-match only)
bat_inn = (b.groupby(["match_id", "striker_id"])
           .agg(runs=("runs", "sum"), balls=("batter_faced", "sum"), pos=("batting_order", "median"),
                lvl=("lvl_group", "first"), date=("match_date", "first"))
           .reset_index().rename(columns={"striker_id": "pid"}).sort_values(["date", "match_id"]))
gb = bat_inn.groupby("pid")
bat_inn["n_prior"] = gb.cumcount()
bat_inn["n_prior_lvl"] = bat_inn.groupby(["pid", "lvl"]).cumcount()
bat_inn["last3_runs"] = gb.runs.transform(lambda s: s.shift(1).rolling(3).mean())
bat_inn["prior_med_runs"] = gb.runs.transform(lambda s: s.shift(1).expanding().median())
bat_inn["prior_med_pos"] = gb.pos.transform(lambda s: s.shift(1).expanding().median())
bat_inn["bat_early_career"] = (bat_inn.n_prior_lvl < 5).astype(int)
bat_inn["bat_slump"] = ((bat_inn.n_prior >= 6) & (bat_inn.last3_runs < bat_inn.prior_med_runs)).astype(int)
pos_ok = bat_inn.n_prior >= 5
bat_inn["promoted"] = (pos_ok & (bat_inn.pos <= bat_inn.prior_med_pos - 2)).astype(int)
bat_inn["demoted"] = (pos_ok & (bat_inn.pos >= bat_inn.prior_med_pos + 2)).astype(int)
bat_inn["out_of_position"] = bat_inn.promoted | bat_inn.demoted
b = b.merge(bat_inn[["match_id", "pid", "bat_early_career", "bat_slump", "promoted", "demoted", "out_of_position"]]
            .rename(columns={"pid": "striker_id"}), on=["match_id", "striker_id"], how="left")

# ---------------------------------------------------------------- bowler history (pre-match only)
spell = (b.groupby(["match_id", "bowler_id"])
         .agg(runs=("bowler_runs", "sum"), balls=("legal", "sum"), lvl=("lvl_group", "first"), date=("match_date", "first"))
         .reset_index().rename(columns={"bowler_id": "pid"}).sort_values(["date", "match_id"]))
spell["econ"] = spell.runs * 6 / spell.balls.clip(lower=1)
gs = spell.groupby("pid")
spell["n_prior"] = gs.cumcount()
spell["n_prior_lvl"] = spell.groupby(["pid", "lvl"]).cumcount()
spell["last3_econ"] = gs.econ.transform(lambda s: s.shift(1).rolling(3).mean())
spell["prior_med_econ"] = gs.econ.transform(lambda s: s.shift(1).expanding().median())
spell["bwl_early_career"] = (spell.n_prior_lvl < 5).astype(int)
spell["bwl_slump"] = ((spell.n_prior >= 6) & (spell.last3_econ > spell.prior_med_econ)).astype(int)
b = b.merge(spell[["match_id", "pid", "bwl_early_career", "bwl_slump"]].rename(columns={"pid": "bowler_id"}),
            on=["match_id", "bowler_id"], how="left")
for c in ["bat_early_career", "bat_slump", "promoted", "demoted", "out_of_position", "bwl_early_career", "bwl_slump"]:
    b[c] = b[c].fillna(0).astype(int)

# ---------------------------------------------------------------- simple derived context
# public leagues map onto the nearest supplied level (franchise -> IPL-like, Indian domestic -> A-team-like);
# their own scoring level is removed later by re-centring values within each competition
b["lvl_code"] = b.level.map({"IPL": 0, "T20I-full": 1, "T20I-assoc": 2, "A-team": 3, "SMAT": 3, "BBL": 0, "PSL": 0,
                             "CPL": 0, "SA20": 0, "ILT20": 0, "MLC": 0, "BPL": 0, "LPL": 0, "Blast": 0})
b["stage_ko"] = b.knock_out.astype(int)
# unknown type (public-data players not in the pack) stays missing, never silently "spin" / "right-handed"
b["is_pace"] = np.where(b.pace_or_spin.isna(), np.nan, b.pace_or_spin.eq("Pace").astype(float))
b["lhb"] = np.where(b.batting_type.isna(), np.nan, b.batting_type.eq("LHB").astype(float))
b["left_arm"] = np.where(b.bowling_type.isna(), np.nan, b.bowling_type.astype(str).str.startswith("Left").astype(float))
b["same_side"] = np.where(b.lhb.isna() | b.left_arm.isna(), np.nan, (b.lhb == b.left_arm).astype(float))
b["milestone"] = (b.str_runs.between(40, 49) | b.str_runs.between(90, 99)).astype(int)
b["tail_partner"] = ((b.str_balls >= 20) & (b.wkts_before >= 7)).astype(int)
b["slow_self"] = ((b.str_balls >= 10) & (b.str_sr < 100)).astype(int)

# outcome classes for M1: 0 dot, 1, 2, 3 (incl 5), 4, 6, W (striker or run-out of either), X (wide / no-ball)
cls = np.select([b.is_wicket, b.is_wide | b.is_noball, b.runs >= 6, b.runs.isin([4]), b.runs.isin([3, 5]), b.runs == 2, b.runs == 1],
                [6, 7, 5, 4, 3, 2, 1], 0)
b["y_class"] = cls
save(b, "features" + TAG)
if TAG:
    log.info(f"P2{TAG} done: {len(b):,} balls")
    raise SystemExit(0)

# ---------------------------------------------------------------- feature families / catalogue
FAM = {
    "A_core": ["innings", "legal_before", "balls_left", "score_before", "wkts_before", "crr", "need", "rrr",
               "maxb", "target", "phase"],
    "A_derived": ["rrr_minus_crr", "rrr_over_crr", "need_per_wkt", "balls_per_wkt", "proj_total", "chase_eq",
                  "last_over", "last_two", "ball_in_over"],
    "B_momentum": ["runs_last6", "runs_last12", "runs_last18", "runs_last30", "wkts_last6", "wkts_last12",
                   "wkts_last24", "wkts_last36", "team_dot_streak", "team_balls_since_bdry", "balls_since_wkt",
                   "pship_runs", "pship_balls", "str_balls", "str_runs", "str_dot_streak", "str_balls_since_bdry",
                   "ns_balls", "free_hit", "rebowl", "bwl_balls", "over_runs_so_far", "over_bdry_so_far",
                   "prev_over_runs", "bwl_prev_bdry", "bwl_prev_wkt", "bwl_prev_dot"],
    "C_context": ["lvl_code", "year", "impact_era", "stage_ko", "venue_par", "toss_bat", "home_bat", "dls", "night",
                  "month"],
    "C_strength": ["elo_bat", "elo_bowl", "elo_diff"],
    "D_matchup": ["is_pace", "lhb", "left_arm", "same_side"],
}
DESC = {
    "innings": "innings (1 = setting, 2 = chasing)", "legal_before": "legal balls bowled before this ball",
    "balls_left": "legal balls remaining (revision-aware)", "score_before": "team score before the ball",
    "wkts_before": "wickets lost", "crr": "current run rate", "need": "runs still needed (chase)",
    "rrr": "required run rate (chase)", "maxb": "innings length in balls (DLS-aware)", "target": "target (DLS-aware)",
    "phase": "powerplay / middle / death", "rrr_minus_crr": "required minus current run rate",
    "rrr_over_crr": "required / current run rate", "need_per_wkt": "runs needed per wicket in hand",
    "balls_per_wkt": "balls left per wicket in hand", "proj_total": "projected total at current rate",
    "chase_eq": "runs needed minus balls left", "last_over": "last over of innings", "last_two": "last two overs",
    "ball_in_over": "legal ball number in the over", "free_hit": "free hit", "rebowl": "re-bowl after wide/no-ball",
    "lvl_code": "competition level (IPL / T20I full / T20I associate / A-team)", "year": "season",
    "impact_era": "IPL Impact Player era (2023+)", "stage_ko": "knockout match", "venue_par": "venue par (LOO, shrunk)",
    "toss_bat": "batting side won toss", "home_bat": "batting side at home", "dls": "rain-reduced innings",
    "night": "night match (dew proxy)", "month": "month", "elo_bat": "batting team Elo (pre-match)",
    "elo_bowl": "bowling team Elo", "elo_diff": "Elo gap", "is_pace": "bowler is pace", "lhb": "left-hand batter",
    "left_arm": "left-arm bowler", "same_side": "batter hand matches bowling arm",
}
rows = []
for fam, cols in FAM.items():
    for c in cols:
        used = {"A_core": "M1-M4, pressure", "A_derived": "M1, pressure", "B_momentum": "M1, pressure pointers",
                "C_context": "M1-M4", "C_strength": "M2/M3 (win prob) only; excluded from player baselines",
                "D_matchup": "tested in M1 ablation only; excluded from baselines (player attributes)"}[fam]
        rows.append({"feature": c, "family": fam, "description": DESC.get(c, c.replace("_", " ")), "used_by": used,
                     "missing_pct": round(100 * b[c].isna().mean(), 2)})
cat = pd.DataFrame(rows)
save_table(cat, "feature_catalogue_ball")
save_json(FAM, "feature_families")
log.info(f"Ball-level features: {len(cat)} in {len(FAM)} families")
log.info(f"Player-history pointers: early career bat {b.bat_early_career.mean():.3f} / bowl {b.bwl_early_career.mean():.3f}, "
         f"slump bat {b.bat_slump.mean():.3f} / bowl {b.bwl_slump.mean():.3f}, out of position {b.out_of_position.mean():.3f}")
log.info(f"Outcome classes: {pd.Series(cls).value_counts(normalize=True).sort_index().round(4).to_dict()}")
log.info("P2 done")
