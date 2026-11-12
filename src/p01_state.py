"""P1 - Ball-state reconstruction + match-level context (Elo, venue par, points tables, series scores).
Every ball feature uses only information known BEFORE that ball is bowled."""
import numpy as np
import pandas as pd
from common import CFG, load, save, save_json, get_logger

import os
TAG = os.environ.get("RR_TAG", "")          # "" = supplied data; "_ext" = public Cricsheet matches
log = get_logger("p01_state" + TAG)
b = load("balls_clean" + TAG)
m = load("matches_clean" + TAG)
pe = load("playing_eleven_clean" + TAG)

b = b[~b.is_super_over].sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)
b["dot_team"] = b.legal & (b.team_runs == 0)
IK = ["match_id", "innings"]
g = b.groupby(IK, sort=False)

# ---------------------------------------------------------------- core scoreboard (pre-ball)
b["ball_idx"] = g.cumcount()
b["legal_before"] = g.legal.cumsum() - b.legal
b["score_before"] = g.team_runs.cumsum() - b.team_runs
b["wkts_before"] = g.is_wicket.cumsum() - b.is_wicket
b["wih"] = 10 - b.wkts_before

inn = b.groupby(IK).agg(total=("team_runs", "sum"), legal_total=("legal", "sum"), wk_total=("is_wicket", "sum"),
                        bat_team=("batting_team_id", "first")).reset_index()
i1 = inn[inn.innings == 1].set_index("match_id")
mi = m.set_index("match_id")

def innings_max(row):
    mid, k = row.match_id, row.innings
    r1, r2 = mi.at[mid, "first_inn_revised_overs"], mi.at[mid, "second_inn_revised_overs"]
    if k == 1:
        mx = r1 if pd.notna(r1) else 120
    else:
        mx = r2 if pd.notna(r2) else (r1 if pd.notna(r1) else 120)
    inferred = False
    won = mi.at[mid, "win_team_id"] == row.bat_team
    if row.wk_total < 10 and row.legal_total < mx:
        if k == 1 and pd.isna(r1):
            mx, inferred = row.legal_total, True
        if k == 2 and not won and not mi.at[mid, "is_tie"] and pd.isna(r2):
            mx, inferred = row.legal_total, True
    return pd.Series({"maxb": mx, "maxb_inferred": inferred})

inn[["maxb", "maxb_inferred"]] = inn.apply(innings_max, axis=1)
inn["target"] = np.nan
is2 = inn.innings == 2
inn.loc[is2, "target"] = [mi.at[mid, "revised_target"] if pd.notna(mi.at[mid, "revised_target"])
                          else (i1.at[mid, "total"] + 1 if mid in i1.index else np.nan) for mid in inn.loc[is2, "match_id"]]
# chase target unknown: DLS mentioned in result but no revised target recorded
dls_txt = m.set_index("match_id").result.str.contains("D/L|DLS", case=False, na=False)
inn["irregular"] = inn.maxb_inferred | ((inn.innings == 2) & inn.match_id.map(dls_txt) & inn.match_id.map(mi.revised_target.isna()))
b = b.merge(inn[IK + ["maxb", "maxb_inferred", "target", "irregular", "total", "legal_total"]], on=IK, how="left")
b["balls_left"] = (b.maxb - b.legal_before).clip(lower=1)
b["over_no"] = b.overid  # 1-based
b["phase"] = np.select([b.legal_before < 36, b.legal_before < 90], [0, 1], 2)  # PP / middle / death (by balls bowled)
b["crr"] = b.score_before * 6 / b.legal_before.clip(lower=1)
b["need"] = np.where(b.innings == 2, b.target - b.score_before, np.nan)
b["rrr"] = b.need * 6 / b.balls_left
b["rrr_minus_crr"] = b.rrr - b.crr
b["rrr_over_crr"] = b.rrr / b.crr.clip(lower=1)
b["need_per_wkt"] = b.need / b.wih.clip(lower=1)
b["balls_per_wkt"] = b.balls_left / b.wih.clip(lower=1)
b["proj_total"] = b.score_before + b.crr * b.balls_left / 6
b["chase_eq"] = b.need - b.balls_left
b["last_over"] = (b.balls_left <= 6).astype(int)
b["last_two"] = (b.balls_left <= 12).astype(int)

# ---------------------------------------------------------------- momentum (pre-ball)
g = b.groupby(IK, sort=False)
cs_r = g.team_runs.cumsum()
cs_w = g.is_wicket.cumsum().astype(int)
for n in (6, 12, 18, 30):
    b[f"runs_last{n}"] = (cs_r.groupby([b.match_id, b.innings]).shift(1).fillna(0)
                          - cs_r.groupby([b.match_id, b.innings]).shift(n + 1).fillna(0))
for n in (6, 12, 24, 36):
    b[f"wkts_last{n}"] = (cs_w.groupby([b.match_id, b.innings]).shift(1).fillna(0)
                          - cs_w.groupby([b.match_id, b.innings]).shift(n + 1).fillna(0))

def streak_before(flag, keys, count=None):
    """number of consecutive previous rows (within keys) where flag is True (counting `count` weights)."""
    flag = flag.astype(bool)
    reset = (~flag).astype(int).groupby([b[k] for k in keys]).cumsum()
    c = (flag if count is None else (flag & count)).astype(int)
    run = c.groupby([b[k] for k in keys] + [reset]).cumsum()
    return run.groupby([b[k] for k in keys]).shift(1).fillna(0)

b["team_dot_streak"] = streak_before(b.dot_team | ~b.legal & (b.team_runs == 0), IK, count=b.legal)
b["team_balls_since_bdry"] = streak_before(~b.boundary, IK, count=b.legal)
b["balls_since_wkt"] = streak_before(~b.is_wicket, IK, count=b.legal)

# partnership
b["pship_id"] = b.wkts_before
gp = b.groupby(IK + ["pship_id"], sort=False)
b["pship_runs"] = gp.team_runs.cumsum() - b.team_runs
b["pship_balls"] = gp.legal.cumsum() - b.legal

# striker within innings
gs = b.groupby(IK + ["striker_id"], sort=False)
b["str_balls"] = gs.batter_faced.cumsum() - b.batter_faced
b["str_runs"] = gs.runs.cumsum() - b.runs
b["str_sr"] = np.where(b.str_balls > 0, 100 * b.str_runs / b.str_balls.clip(lower=1), np.nan)
sk = IK + ["striker_id"]
b["str_dot_streak"] = streak_before(b.runs.eq(0) | ~b.batter_faced, sk, count=b.batter_faced)
b["str_balls_since_bdry"] = streak_before(~b.boundary, sk, count=b.batter_faced)

# non-striker balls faced so far: last cumulative value of that player as striker before this ball
cum = b[IK + ["ball_idx", "striker_id"]].copy()
cum["faced_after"] = gs.batter_faced.cumsum()
cum = cum.rename(columns={"striker_id": "pid"})
ns = b[IK + ["ball_idx", "non_striker_id"]].rename(columns={"non_striker_id": "pid"}).reset_index()
ns["key"] = ns.match_id.astype(np.int64) * 10 + ns.innings
cum["key"] = cum.match_id.astype(np.int64) * 10 + cum.innings
ns = ns.sort_values("ball_idx")
cum = cum.sort_values("ball_idx")
merged = pd.merge_asof(ns, cum[["key", "pid", "ball_idx", "faced_after"]], on="ball_idx", by=["key", "pid"],
                       allow_exact_matches=False, direction="backward")
b["ns_balls"] = merged.set_index("index").faced_after.reindex(b.index).fillna(0).values

b["new_batter"] = (b.str_balls <= 5).astype(int)
b["ns_new"] = (b.ns_balls <= 5).astype(int)

# bowler within match
gb = b.groupby(["match_id", "bowler_id"], sort=False)
b["bwl_balls"] = gb.legal.cumsum() - b.legal
b["bwl_runs"] = gb.bowler_runs.cumsum() - b.bowler_runs
b["bwl_wkts"] = gb.bowler_wicket.cumsum() - b.bowler_wicket
b["bwl_econ"] = np.where(b.bwl_balls >= 6, b.bwl_runs * 6 / b.bwl_balls.clip(lower=1), np.nan)
b["bwl_prev_bdry"] = gb.boundary.shift(1).fillna(False).astype(int)
b["bwl_prev_wkt"] = gb.is_wicket.shift(1).fillna(False).astype(int)
b["bwl_prev_dot"] = gb.dot_team.shift(1).fillna(False).astype(int)
go = b.groupby(IK + ["overid"], sort=False)
b["over_runs_so_far"] = go.bowler_runs.cumsum() - b.bowler_runs
b["over_bdry_so_far"] = (go.boundary.cumsum() - b.boundary).astype(int)
b["ball_in_over"] = go.legal.cumsum() - b.legal + 1
b["prev_ball_bdry_same_over"] = go.boundary.shift(1).fillna(False).astype(int)
ovr = b.groupby(IK + ["overid"]).team_runs.sum().rename("ov_runs").reset_index()
ovr["prev_over_runs"] = ovr.groupby(IK).ov_runs.shift(1)
b = b.merge(ovr[IK + ["overid", "prev_over_runs"]], on=IK + ["overid"], how="left")
quota = (np.ceil(b.maxb / 30) * 6)  # max balls per bowler (1/5 of the overs)
over_start_balls = b.groupby(["match_id", "bowler_id", "overid"]).bwl_balls.transform("min")
b["bwl_last_quota_over"] = (over_start_balls >= quota - 6).astype(int)
b["final_over_of_inns"] = (b.groupby(IK + ["overid"]).balls_left.transform("max") <= 6).astype(int)

prev_type = b.groupby(IK).extras_type.shift(1).fillna("none")
prev2 = b.groupby(IK).extras_type.shift(2).fillna("none")
b["free_hit"] = (prev_type.isin(CFG["data"]["noball_extras"]) |
                 (prev_type.eq("wide") & prev2.isin(CFG["data"]["noball_extras"]))).astype(int)
b["rebowl"] = prev_type.isin(CFG["data"]["illegal_extras"]).astype(int)

# ---------------------------------------------------------------- match-level context
m = m.sort_values(["match_date", "match_id"]).reset_index(drop=True)

# Elo (data-derived, pre-match). Pools: IPL, internationals, A-team are connected only where teams meet.
full = set(CFG["data"]["full_members"])
def init_elo(name, level):
    if level == "IPL": return 1500.0
    if name.endswith(" A"): return 1450.0
    return 1550.0 if name in full else 1400.0
elo, K = {}, 24.0
pre_h, pre_a = [], []
for r in m.itertuples():
    for tid, nm in ((r.home_team_id, r.home_team), (r.away_team_id, r.away_team)):
        elo.setdefault(tid, init_elo(nm, r.level))
    eh, ea = elo[r.home_team_id], elo[r.away_team_id]
    pre_h.append(eh); pre_a.append(ea)
    if pd.isna(r.win_team_id) and not r.is_tie:
        continue
    s = 0.5 if r.is_tie else float(r.win_team_id == r.home_team_id)
    e = 1 / (1 + 10 ** ((ea - eh) / 400))
    elo[r.home_team_id] = eh + K * (s - e)
    elo[r.away_team_id] = ea - K * (s - e)
m["elo_home"], m["elo_away"] = pre_h, pre_a

# venue par: first-innings totals of complete, unrevised innings; leave-one-match-out, shrunk to level-year mean
i1t = inn[(inn.innings == 1) & (inn.maxb == 120) & ~inn.maxb_inferred][["match_id", "total"]].merge(
    m[["match_id", "venue_id", "level", "year"]], on="match_id")
ly = i1t.groupby(["level", "year"]).total.agg(["sum", "count"])
vs = i1t.groupby("venue_id").total.agg(["sum", "count"])
KV = 5.0
def venue_par(mid, vid, lev, yr, own):
    s, n = (vs.loc[vid, "sum"], vs.loc[vid, "count"]) if vid in vs.index else (0.0, 0)
    ls, ln = (ly.loc[(lev, yr), "sum"], ly.loc[(lev, yr), "count"]) if (lev, yr) in ly.index else (np.nan, 0)
    if own is not None:
        s, n, ls, ln = s - own, n - 1, ls - own, ln - 1
    prior = ls / ln if ln > 0 else i1t.total.mean()
    return (s + KV * prior) / (n + KV)
own_tot = i1t.set_index("match_id").total
m["venue_par"] = [venue_par(r.match_id, r.venue_id, r.level, r.year, own_tot.get(r.match_id)) for r in m.itertuples()]

# IPL points table rebuilt before every league match
m["race_home"] = 0; m["race_away"] = 0; m["dead_home"] = 0; m["dead_away"] = 0
for yr, gm in m[m.level == "IPL"].groupby("year"):
    lg = gm[gm.stage.eq("League")].sort_values(["match_date", "match_id"])
    tms = pd.unique(lg[["home_team_id", "away_team_id"]].values.ravel())
    pts = {t: 0 for t in tms}; played = {t: 0 for t in tms}
    n_games = int(round(2 * len(lg) / len(tms)))
    for r in lg.itertuples():
        p4 = sorted(pts.values(), reverse=True)[3]
        for side, t in (("home", r.home_team_id), ("away", r.away_team_id)):
            rem = n_games - played[t]
            alive = pts[t] + 2 * rem >= p4
            must = alive and pts[t] + 2 * (rem - 1) < p4 + 2  # a loss leaves the team needing others to slip
            close = alive and rem <= 6 and pts[t] <= p4 + 2
            m.loc[m.match_id == r.match_id, f"race_{side}"] = 2 if (must and rem <= 6) else int(close)
            m.loc[m.match_id == r.match_id, f"dead_{side}"] = int(not alive)
        for t in (r.home_team_id, r.away_team_id):
            played[t] += 1
        if pd.isna(r.win_team_id):
            pts[r.home_team_id] += 1; pts[r.away_team_id] += 1
        else:
            pts[int(r.win_team_id)] = pts.get(int(r.win_team_id), 0) + 2

# T20I bilateral series score rebuilt before every match
m["series_must_home"] = 0; m["series_must_away"] = 0; m["series_dead"] = 0; m["series_decider"] = 0
for cid, gm in m[m.level.str.startswith("T20I") & m.comp_name.str.contains("Series", case=False)
                 & ~m.comp_name.str.contains("Tri", case=False)].groupby("comp_id"):
    tms = pd.unique(gm[["home_team_id", "away_team_id"]].values.ravel())
    if len(tms) != 2 or len(gm) < 2:
        continue
    N = len(gm); need = N // 2 + 1
    wins = {t: 0 for t in tms}
    for i, r in enumerate(gm.sort_values(["match_date", "match_id"]).itertuples()):
        h, a = r.home_team_id, r.away_team_id
        decided = max(wins.values()) >= need
        if decided:
            m.loc[m.match_id == r.match_id, "series_dead"] = 1
        else:
            m.loc[m.match_id == r.match_id, "series_must_home"] = int(wins[a] + 1 >= need)
            m.loc[m.match_id == r.match_id, "series_must_away"] = int(wins[h] + 1 >= need)
            m.loc[m.match_id == r.match_id, "series_decider"] = int(wins[h] + 1 >= need and wins[a] + 1 >= need)
        if pd.notna(r.win_team_id) and r.win_team_id in wins:
            wins[int(r.win_team_id)] += 1

# ---------------------------------------------------------------- attach match context to balls
mc = m[["match_id", "level", "year", "impact_era", "knock_out", "stage", "venue_id", "venue_country", "home_team_id",
        "away_team_id", "home_team", "away_team", "toss_win_team_id", "toss_opted", "win_team_id", "is_tie",
        "no_result", "night", "elo_home", "elo_away", "venue_par", "race_home", "race_away", "dead_home",
        "dead_away", "series_must_home", "series_must_away", "series_dead", "series_decider", "match_date",
        "comp_id", "comp_name"]]
b = b.merge(mc, on="match_id", how="left")
bat_home = b.batting_team_id == b.home_team_id
b["elo_bat"] = np.where(bat_home, b.elo_home, b.elo_away)
b["elo_bowl"] = np.where(bat_home, b.elo_away, b.elo_home)
b["elo_diff"] = b.elo_bat - b.elo_bowl
b["race_bat"] = np.where(bat_home, b.race_home, b.race_away)
b["race_bowl"] = np.where(bat_home, b.race_away, b.race_home)
b["dead_bat"] = np.where(bat_home, b.dead_home, b.dead_away)
b["dead_bowl"] = np.where(bat_home, b.dead_away, b.dead_home)
b["series_must_bat"] = np.where(bat_home, b.series_must_home, b.series_must_away)
b["series_must_bowl"] = np.where(bat_home, b.series_must_away, b.series_must_home)
country_alias = {"England": "United Kingdom", "Scotland": "United Kingdom", "United States Of America": "United States"}
bat_name = np.where(bat_home, b.home_team, b.away_team)
bat_country = pd.Series(bat_name).replace(country_alias).str.replace(r" A$", "", regex=True).values
b["home_bat"] = np.where(b.level.eq("IPL"), bat_home.astype(int), (bat_country == b.venue_country.values).astype(int))
b["toss_bat"] = (b.toss_win_team_id == b.batting_team_id).astype(int)
b["chasing"] = (b.innings == 2).astype(int)
b["dls"] = (b.maxb < 120).astype(int)
b["month"] = pd.to_datetime(b.match_date).dt.month
b["bat_won"] = np.where(b.is_tie, 0.5, (b.win_team_id == b.batting_team_id).astype(float))
b.loc[b.no_result & ~b.is_tie, "bat_won"] = np.nan

# playing XI flags for striker / non-striker / bowler
pex = pe[["match_id", "player_id", "is_captain", "is_impact", "batting_order"]].rename(columns={"batting_order": "xi_order"})
b = b.merge(pex.add_prefix("s_").rename(columns={"s_match_id": "match_id", "s_player_id": "striker_id"}), on=["match_id", "striker_id"], how="left")
b = b.merge(pex[["match_id", "player_id", "is_captain", "is_impact"]].add_prefix("b_").rename(
    columns={"b_match_id": "match_id", "b_player_id": "bowler_id"}), on=["match_id", "bowler_id"], how="left")
for c in ["s_is_captain", "s_is_impact", "b_is_captain", "b_is_impact"]:
    b[c] = b[c].fillna(0).astype(int)

b = b.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)
b["ball_uid"] = np.arange(len(b))

# ---------------------------------------------------------------- validation gates
ov = b[b.overid < b.groupby(IK).overid.transform("max")].groupby(IK + ["overid"]).legal.sum()
gate1 = float((ov == 6).mean())
if TAG:
    gate2, iqr = float("nan"), []
else:
    chk = load("margin_check")
    gate2 = float(chk[chk.type == "runs"].err.median())
    iqr = chk[chk.type == "runs"].err.quantile([.25, .75]).tolist()
log.info(f"Gate: complete overs with 6 legal balls = {100 * gate1:.3f}% (need >= 99.9%)")
log.info(f"Gate: margin error median {gate2}, IQR {iqr} (need median 0, IQR within +-3)")
log.info(f"Innings with inferred max balls: {int(inn.maxb_inferred.sum())}; irregular innings: {int(inn.irregular.sum())}")
log.info(f"IPL playoff-race flags (team-matches): race=1 {int((m[['race_home','race_away']]==1).sum().sum())}, "
         f"must-win=2 {int((m[['race_home','race_away']]==2).sum().sum())}; dead {int(m[['dead_home','dead_away']].sum().sum())}")
log.info(f"Series: must-win team-matches {int(m[['series_must_home','series_must_away']].sum().sum())}, "
         f"deciders {int(m.series_decider.sum())}, dead rubbers {int(m.series_dead.sum())}")
log.info(f"Elo range {m.elo_home.min():.0f}-{m.elo_home.max():.0f}; venue par mean {m.venue_par.mean():.1f}")
save(b, "ball_state" + TAG)
save(m, "matches_ctx" + TAG)
save(inn, "innings" + TAG)
save_json({"gate_6_legal": gate1, "gate_margin_median": gate2, "gate_margin_iqr": iqr,
           "n_balls": len(b), "n_matches": int(b.match_id.nunique())}, "p01_summary" + TAG)
log.info(f"P1 done: {len(b):,} balls, {b.shape[1]} columns")
