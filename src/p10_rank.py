"""P10 - Final rankings with the configuration chosen in P9, fitted on all seasons (2022-2026, equal weight).
Bootstrap (resampling whole matches) gives a 90% rank interval and P(top-10) for every player.
Ties (score gap < 0.02 z) are broken by P(top-10), then by the narrower rank interval."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from common import CFG, SEED, load, raw, save_table, save_json, load_json, get_logger, Timer, excluded_reasons
import ratings_lib as RL

log = get_logger("p10_rank")
v = load("values")
comp = load_json("p05_components")
chosen = load_json("p09_summary")["chosen"]
players = load("players").set_index("player_id")
roles = load("roles")
EXR = excluded_reasons()
EXCL = set(EXR)
teams = raw("teams").drop_duplicates("team_id").set_index("team_id").team_name
f = load("features", columns=["ball_uid", "y_class", "str_balls"])
v = v.merge(f, on="ball_uid")
# 'average player in the same moments' (the comparison the method makes), for the report tables: expected batter runs
# per ball, and expected team runs per legal ball from the same-type ball-outcome model, calibrated per bowling type so
# that it matches the actual league economy (the outcome classes count a wide or no-ball as one run and miss byes)
_cx = load("ctx_preds", columns=["ball_uid", "x_bat_runs", "p_1", "p_2", "p_3", "p_4", "p_6", "p_x"])
v = v.merge(_cx, on="ball_uid")
v["e_team"] = v.p_1 + 2 * v.p_2 + 3 * v.p_3 + 4 * v.p_4 + 6 * v.p_6 + v.p_x
v["e_legal"] = 1 - v.p_x
_cal = (v.groupby("is_pace").team_runs.sum() / v.groupby("is_pace").legal.sum()) / (v.groupby("is_pace").e_team.sum() / v.groupby("is_pace").e_legal.sum())
v["e_team"] = v.e_team * v.is_pace.map(_cal).values
B = CFG["ratings"]["bootstrap_reps"]

# per-match row index for fast bootstrap
order = np.argsort(v.match_id.values, kind="stable")
mids, starts, counts = np.unique(v.match_id.values[order], return_index=True, return_counts=True)


def boot_sample(seed):
    r = np.random.RandomState(seed)
    draw = r.randint(0, len(mids), len(mids))
    idx = np.concatenate([order[starts[d]:starts[d] + counts[d]] for d in draw])
    newmid = np.repeat(np.arange(len(draw)), counts[draw])
    s = v.iloc[idx].copy()
    s["match_id"] = newmid
    return s


def boot_scores(seed, side, cfg, kfix):
    RL.K_FIXED = kfix
    s = boot_sample(seed)
    sc = RL.build_scores(s, side, cfg, comp)
    return sc.score, sc.rating


def delivery_styles():
    """delivery_style per ball, aligned to ball_uid (P1 drops super overs and sorts by match, innings, ball)."""
    bc = load("balls_clean", columns=["match_id", "innings", "innings_ball_num", "is_super_over", "delivery_style"])
    bc = bc[~bc.is_super_over].sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)
    assert len(bc) == len(v) and (bc.match_id.values == v.match_id.values).all()
    return bc.delivery_style.values


def job_notes(side, ids):
    """One statistic for the job each role does, against the same role (batters) or same bowling type and phase
    (bowlers). Explanation only: the ranking already counts the runs these skills save or make."""
    rl = roles[roles.side == side].set_index("player_id").role
    pc = RL.PCOL[side]
    out = {}
    if side == "bat":
        b = v[v.faced].assign(role=lambda x: x.striker_id.map(rl))
        jobs = {
            "Opener": (lambda x: x[x.phase == 0], lambda x: 100 * x.runs.mean(), "powerplay strike rate {p:.0f} (openers' average {a:.0f})"),
            "Top order (No. 3-4)": (lambda x: x[(x.innings == 2) & (x.phase == 1)], lambda x: 100 * (x.runs > 0).mean(),
                                    "in chases, overs 7-15, scores off {p:.0f}% of balls (role average {a:.0f}%)"),
            "Middle order (No. 5-6)": (lambda x: x[(x.innings == 2) & (x.phase == 1)], lambda x: 100 * (x.runs > 0).mean(),
                                       "in chases, overs 7-15, scores off {p:.0f}% of balls (role average {a:.0f}%)"),
            "Finisher": (lambda x: x[x.str_balls < 10], lambda x: 100 * x.runs.mean(),
                         "strike rate {p:.0f} in his first 10 balls (finishers' average {a:.0f})"),
            "Lower order": (lambda x: x[x.phase == 2], lambda x: 100 * x.boundary.mean(),
                            "boundary off {p:.0f}% of death-over balls (role average {a:.0f}%)")}
        for pid in ids:
            r = rl.get(pid)
            if r not in jobs:
                continue
            flt, met, txt = jobs[r]
            mine, grp = flt(b[b.striker_id == pid]), flt(b[b.role == r])
            if len(mine) >= 60:
                out[pid] = txt.format(p=met(mine), a=met(grp))
    else:
        b = v.assign(role=lambda x: x.bowler_id.map(rl), ds=DS)
        york = lambda x: 100 * x.ds.isin(["Yorker", "Wide Yorker"]).mean()
        dots = lambda x: 100 * (x.y_class == 0).mean()
        econ = lambda x: 6 * x.team_runs.sum() / max(x.legal.sum(), 1)
        jobs = {
            "Pace - death": (2, york, "yorkers {p:.0f}% of his death-over balls (pace average {a:.0f}%)"),
            "Pace - new ball + death": (2, york, "yorkers {p:.0f}% of his death-over balls (pace average {a:.0f}%)"),
            "Pace - new ball": (0, dots, "powerplay dot balls {p:.0f}% (pace average {a:.0f}%)"),
            "Pace - middle overs": (1, dots, "middle-overs dot balls {p:.0f}% (pace average {a:.0f}%)"),
            "Spin - middle overs": (1, dots, "middle-overs dot balls {p:.0f}% (spin average {a:.0f}%)"),
            "Spin - powerplay": (0, econ, "powerplay economy {p:.2f} (spin average {a:.2f})")}
        for pid in ids:
            r = rl.get(pid)
            if r not in jobs:
                continue
            ph, met, txt = jobs[r]
            pace = r.startswith("Pace")
            mine = b[(b.bowler_id == pid) & (b.phase == ph)]
            grp = b[(b.is_pace == int(pace)) & (b.phase == ph)]
            if len(mine) >= 60:
                out[pid] = txt.format(p=met(mine), a=met(grp))
    return pd.Series(out, dtype=object)


DS = delivery_styles()


def describe(side, ids):
    pc = RL.PCOL[side]
    d = v[v[pc].isin(ids)]
    tcol = "batting_team_id" if side == "bat" else "bowling_team_id"
    tm = d.groupby([pc, tcol]).year.agg(["min", "max"]).reset_index()
    tm["t"] = tm[tcol].map(teams) + " (" + tm["min"].astype(str) + np.where(tm["min"] != tm["max"], "-" + tm["max"].astype(str), "") + ")"
    team_str = tm.groupby(pc).t.apply(lambda s: "; ".join(s))
    yrs = d.groupby(pc).year.agg(lambda s: f"{s.min()}-{s.max()}")
    lv = d.groupby(pc)["level"].agg(lambda s: ", ".join(f"{k} {100 * x:.0f}%" for k, x in s.value_counts(normalize=True).items()))
    if side == "bat":
        dd = d[d.faced]
        g, gp = dd.groupby(pc), dd[dd.is_pressure].groupby(pc)
        stats = pd.DataFrame({
            "balls": g.size(), "pressure_balls": gp.size(), "matches": g.match_id.nunique(),
            "SR_usual": 100 * dd[~dd.is_pressure].groupby(pc).runs.mean(), "SR_pressure": 100 * gp.runs.mean(),
            "balls_per_out_pressure": gp.size() / gp.striker_out.sum().clip(lower=1),
            "runs_above_avg_per100_pressure": 100 * gp.v1_bat.mean(),
            "winprob_added_per100_pressure_pct": 100 * 100 * gp.v2_bat.mean(),
            "SR_expected_pressure": 100 * gp.x_bat_runs.mean()})
    else:
        g, gp = d.groupby(pc), d[d.is_pressure].groupby(pc)
        dn = d[~d.is_pressure]
        stats = pd.DataFrame({
            "balls": g.size(), "pressure_balls": gp.size(), "matches": g.match_id.nunique(),
            "econ_usual": 6 * dn.groupby(pc).team_runs.sum() / dn.groupby(pc).legal.sum(),
            "econ_pressure": 6 * gp.team_runs.sum() / gp.legal.sum(),
            "balls_per_wkt_pressure": gp.legal.sum() / gp.bowler_wicket.sum().clip(lower=1),
            "dot_pct_pressure": 100 * (d[d.is_pressure].y_class == 0).groupby(d[d.is_pressure][pc]).mean(),
            "runs_saved_per100_pressure": 100 * gp.v1_bowl.mean(),
            "winprob_added_per100_pressure_pct": 100 * 100 * gp.v2_bowl.mean(),
            "econ_expected_pressure": 6 * gp.e_team.sum() / gp.e_legal.sum()})
    stats["teams"] = team_str; stats["seasons"] = yrs; stats["competition_mix"] = lv
    stats["job_note"] = job_notes(side, list(ids)).reindex(stats.index)
    # strongest phase vs the league (for a short, data-based "how he helps RR" note)
    PH = {0: "powerplay", 1: "middle overs", 2: "death overs"}
    base = v[v.faced] if side == "bat" else v
    if side == "bat":
        lg = base.groupby("phase").runs.mean() * 100
        pp = d[d.faced].groupby([pc, "phase"]).agg(x=("runs", "mean"), n=("runs", "size"))
        pp["rel"] = pp.x * 100 - pp.index.get_level_values("phase").map(lg)
    else:
        lg = base.groupby("phase").team_runs.sum() * 6 / base.groupby("phase").legal.sum()
        pp = d.groupby([pc, "phase"]).agg(r=("team_runs", "sum"), l=("legal", "sum"), n=("legal", "size"))
        pp["x"] = pp.r * 6 / pp.l.clip(lower=1)
        pp["rel"] = pp.index.get_level_values("phase").map(lg) - pp.x
    pp = pp[pp.n >= 60].reset_index()
    best = pp.sort_values("rel", ascending=False).drop_duplicates(pc).set_index(pc)
    unit = "SR" if side == "bat" else "economy"
    stats["best_phase_note"] = [
        (f"best in the {PH[int(best.at[i, 'phase'])]}: {unit} {best.at[i, 'x'] * (100 if side == 'bat' else 1):.{0 if side == 'bat' else 2}f} "
         f"vs league {lg[int(best.at[i, 'phase'])]:.{0 if side == 'bat' else 2}f}") if i in best.index else "" for i in stats.index]
    return stats


finals = {}
byrole = []
excluded_rows = []
for side in ("bat", "bowl"):
    cfg = chosen[side]
    log.info(f"{side}: config {cfg}")
    RL.K_RECORD = {}
    main = RL.build_scores(v, side, cfg, comp)
    kfix = dict(RL.K_RECORD); RL.K_RECORD = None
    log.info(f"{side}: shrinkage strengths held fixed in the bootstrap: { {str(k): round(x, 1) for k, x in kfix.items()} }")
    cons = RL.consistency(v, side, cfg.get("value", "v1"))
    # availability: players not available for selection are rated like everyone else but left out of the lists;
    # 'chance of top 10' is then computed among available players only
    ex = main[main.index.isin(EXCL)].copy()
    if len(ex):
        ex["would_be_rank"] = [int((main.score > s_).sum()) + 1 for s_ in ex.score]
        ex["name"] = players.name.reindex(ex.index)
        ex["role"] = roles[roles.side == side].set_index("player_id").role.reindex(ex.index)
        ex["reason"] = [EXR[i] for i in ex.index]
        ex.insert(0, "side", side)
        excluded_rows.append(ex.sort_values("would_be_rank").reset_index().rename(columns={"index": "player_id", "pid": "player_id"}))
        log.info(f"{side}: {len(ex)} players left out (availability rules); best would-be ranks: "
                 + ", ".join(f"{n} #{r} ({why})" for n, r, why in ex.sort_values('would_be_rank')[['name', 'would_be_rank', 'reason']].head(6).values))
    main = main[~main.index.isin(EXCL)].copy()
    pool = main.index
    with Timer(log, f"{side} bootstrap x{B}"):
        res = Parallel(n_jobs=CFG["n_jobs"], batch_size=4)(delayed(boot_scores)(SEED + i, side, cfg, kfix) for i in range(B))
    S = pd.concat([r[0].reindex(pool) for r in res], axis=1)
    RT = pd.concat([r[1].reindex(pool) for r in res], axis=1)
    ranks = S.rank(ascending=False)
    main["rank_point"] = main.score.rank(ascending=False).astype(int)
    main["rank_lo90"] = ranks.quantile(0.05, axis=1)
    main["rank_hi90"] = ranks.quantile(0.95, axis=1)
    main["rank_median"] = ranks.median(axis=1)
    main["p_top10"] = (ranks <= 10).mean(axis=1)
    main["p_top20"] = (ranks <= 20).mean(axis=1)
    main["rating_lo90"] = RT.quantile(0.05, axis=1)
    main["rating_hi90"] = RT.quantile(0.95, axis=1)
    main = main.join(cons[["consistency", "pos_share", "n_inns", "p10"]], how="left")
    main["width"] = main.rank_hi90 - main.rank_lo90
    # tie rule
    main = main.sort_values("score", ascending=False)
    sc = main.score.values
    blocks = np.concatenate([[0], np.cumsum(np.diff(sc) < -0.02)])
    main["block"] = blocks
    main = main.sort_values(["block", "p_top10", "width"], ascending=[True, False, True])
    main["final_rank"] = np.arange(1, len(main) + 1)
    top = main.head(30).copy()
    info = describe(side, top.index)
    top = top.join(info)
    top["name"] = players.name.reindex(top.index)
    top["shared_name_warning"] = players.shared_name.reindex(top.index).fillna(False)
    rl = roles[roles.side == side].set_index("player_id").role
    top["role"] = rl.reindex(top.index).fillna("(<120 balls)")
    top["rise_or_dip"] = np.where(top.gap_shrunk > 0.005, "rises", np.where(top.gap_shrunk < -0.005, "dips", "steady"))
    cols = ["final_rank", "name", "teams", "seasons", "role", "score", "rating", "rating_lo90", "rating_hi90",
            "gap_shrunk", "rise_or_dip", "consistency", "pos_share", "n_inns", "p_top10", "p_top20", "rank_lo90",
            "rank_hi90", "balls", "pressure_balls", "matches", "competition_mix", "shared_name_warning"]
    extra = [c for c in info.columns if c not in cols and c not in ("teams", "seasons", "competition_mix")]
    top = top[cols + extra]
    top.index.name = "player_id"
    save_table(top.reset_index(), f"final_top30_{side}")
    full = main.copy(); full["name"] = players.name.reindex(full.index)
    full.index.name = "player_id"
    save_table(full.reset_index(), f"final_all_{side}")
    # best by role: the overall rank decides the order; the role (usage-based, P7) only groups the list
    full["role"] = rl.reindex(full.index).fillna("(<120 balls)")
    br = full[full.role != "(<120 balls)"].sort_values("final_rank").groupby("role").head(5)
    br = br[["role", "final_rank", "name", "score", "p_top10", "rank_lo90", "rank_hi90", "n_p"]].copy()
    br.insert(0, "side", side)
    br["rank_in_role"] = br.groupby("role").cumcount() + 1
    byrole.append(br.reset_index())
    finals[side] = top.head(10)
    log.info(f"\n{side.upper()} TOP 10\n" + top.head(10)[["final_rank", "name", "teams", "score", "p_top10", "rank_lo90", "rank_hi90", "pressure_balls"]].to_string())

save_table(pd.concat(byrole, ignore_index=True), "best_by_role")
if excluded_rows:
    save_table(pd.concat(excluded_rows, ignore_index=True), "excluded_unavailable")
save_json({s: [{"rank": int(r.final_rank), "player_id": int(i), "name": r["name"], "teams": r.teams}
               for i, r in finals[s].iterrows()] for s in finals}, "final_top10")
log.info("P10 done")
