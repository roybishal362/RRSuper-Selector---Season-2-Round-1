"""P11 - Robustness and stress tests for the final rankings.
1 sensitivity matrix: top-10 overlap / Kendall tau under alternative pressure definitions, thresholds, value metrics,
  rating models and competition scopes
2 ablations: IPL only, T20I only, no associates / A-teams, drop 2022, leave-one-season-out, no knockouts
3 negative control: pressure labels shuffled within each player's balls -> null distribution of the clutch spread
4 synthetic injection: plant a known pressure boost in 10 random players and see whether they surface
5 bias audit: role / competition / era / position mix of the top-10 vs the eligible pool"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from joblib import Parallel, delayed
from common import CFG, SEED, DATA, load, load_table, save_table, save_json, load_json, get_logger, Timer
import ratings_lib as RL

log = get_logger("p11_robustness")
v0 = load("values")
pr = load("pressure")
comp = load_json("p05_components")
chosen = load_json("p09_summary")["chosen"]
p04 = load_json("p04_summary")
roles = load("roles")
players = load("players").set_index("player_id")
q_star = p04["q_star"]
cand_cols = [c for c in pr.columns if c.startswith("cand_")]
assert (pr.ball_uid.values == v0.ball_uid.values).all()
FINAL = {s: load_table(f"final_all_{s}").set_index("player_id") for s in ("bat", "bowl")}
from common import excluded_players
EXCL = excluded_players()


def with_pressure(v, score, q):
    v = v.copy()
    pct = pd.Series(score).rank(pct=True).values
    v["pressure_pct"] = pct
    v["is_pressure"] = pct >= 1 - q
    return v


def compare(side, sc, ref):
    sc = sc[~sc.index.isin(EXCL)]            # same availability rule as the final lists
    top_ref = ref.sort_values("final_rank").index[:10]
    top = sc.sort_values(ascending=False).index[:10]
    pool = ref.index[ref.n_p >= 30].intersection(sc.index)
    tau = kendalltau(ref.score.reindex(pool), sc.reindex(pool))[0] if len(pool) > 10 else np.nan
    return {"top10_overlap": len(set(top) & set(top_ref)), "kendall_tau_pool": tau,
            "new_names": ", ".join(players.name.reindex(list(set(top) - set(top_ref))).fillna("?"))}


rows = []
def run_variant(group, label, v, side, cfg):
    try:
        sc = RL.build_scores(v, side, cfg, comp).score
        r = compare(side, sc, FINAL[side])
    except Exception as e:  # tiny subsets can break a model; record it
        r = {"top10_overlap": np.nan, "kendall_tau_pool": np.nan, "new_names": f"failed: {e}"}
    rows.append({"group": group, "variant": label, "side": side, **r})


with Timer(log, "sensitivity matrix"):
    for side in ("bat", "bowl"):
        cfg = chosen[side]
        for c in cand_cols:
            run_variant("pressure definition", c.replace("cand_", ""), with_pressure(v0, pr[c].values, q_star), side, cfg)
        # match-importance weights (knockout / must-win / dead rubber)
        imp = pr.importance.values
        lvl = np.select([imp == CFG["pressure"]["importance"]["knockout"], imp == CFG["pressure"]["importance"]["must_win"],
                         imp == CFG["pressure"]["importance"]["dead_rubber"]], ["ko", "must", "dead"], "normal")
        for nm_, w_ in {"none (swing only)": (1, 1, 1), "mild (1.25 / 1.1 / 0.9)": (1.25, 1.1, 0.9),
                        "strong (2.0 / 1.5 / 0.5)": (2.0, 1.5, 0.5)}.items():
            mult = np.select([lvl == "ko", lvl == "must", lvl == "dead"], list(w_), 1.0)
            run_variant("match importance", nm_, with_pressure(v0, pr.cand_CA.values * mult, q_star), side, cfg)
        # the previous, type-blind method (every bowler compared with a mixed pace/spin average)
        tb = os.path.join(DATA, "values_typeblind.parquet")
        if os.path.exists(tb):
            cfg_tb = load_json("p09_summary_typeblind")["chosen"][side]
            run_variant("baseline", "type-blind (previous method)", pd.read_parquet(tb), side, cfg_tb)
        for q in CFG["pressure"]["hard_cuts"]:
            run_variant("pressure threshold", f"top {int(q * 100)}%", with_pressure(v0, pr.pressure.values, q), side, cfg)
        for met in ("v1", "v2", "v3", "v4"):
            c2 = dict(cfg, value=met)
            lab = met
            if c2["model"] == "R3":          # R-3 reads components, not the value column
                c2["model"], lab = "R2-joint", f"{met} (with R2-joint)"
            run_variant("value metric", lab, v0, side, c2)
        for m in ("R2-p_only", "R2-joint", "R2-all", "R1", "R3"):
            run_variant("rating model", m, v0, side, dict(cfg, model=m))
        run_variant("rating model", "R2-weighted (alpha 0.5)", v0, side, dict(cfg, model="R2-weighted", alpha=0.5))
        for wd in (0.0, 0.5, 1.0):
            run_variant("gap weight w_d", str(wd), v0, side, dict(cfg, w_d=wd))
        if side == "bowl":
            # bowling-type-relative baseline: compare pace with pace and spin with spin
            vt = v0.copy()
            for c_ in RL.COMPS["bowl"] + ["v1_bowl", "v2_bowl", "v3_bowl", "v4_bowl"]:
                vt[c_] = vt[c_] - vt.groupby("is_pace")[c_].transform("mean")
            run_variant("type-relative baseline", "pace vs pace, spin vs spin", vt, side, cfg)
        # ablations / scopes
        abl = {"IPL only": v0.level.eq("IPL"), "T20I only": v0.level.str.startswith("T20I"),
               "no associates / A-teams": v0.level.isin(["IPL", "T20I-full"]), "drop 2022": v0.year != 2022,
               "no knockouts": v0.knock_out.eq(0)}
        for y in sorted(v0.year.unique()):
            abl[f"leave out {y}"] = v0.year != y
        for nm, m in abl.items():
            run_variant("ablation", nm, v0[m.values], side, cfg)
S = pd.DataFrame(rows)
# best pace bowlers under the main method (spinners dominate the overall list; shown for transparency)
pace_share = v0.groupby("bowler_id").is_pace.mean()
FB = FINAL["bowl"].copy()
FB["pace"] = pace_share.reindex(FB.index) >= 0.5
tp = FB[FB.pace].sort_values("final_rank").head(10)[["final_rank", "name", "score", "p_top10", "n_p"]]
tp.insert(0, "pace_rank", range(1, len(tp) + 1))
save_table(tp.reset_index(), "top_pace_bowlers")
log.info("Best pace bowlers under the main method:" + chr(10) + tp.to_string())
save_table(S, "sensitivity_matrix")
log.info("Sensitivity (top-10 overlap with final, Kendall tau on pool):\n" + S.drop(columns="new_names").round(2).to_string())


# ---------------------------------------------------------------- negative control (clutch spread vs shuffled labels)
def raw_tau_d(d):
    """split-half covariance estimate of the true clutch variance (unclipped, so the null is centred near 0)."""
    return RL.split_half_cov(d)[0]


def shuffled(d, seed):
    r = np.random.RandomState(seed)
    d = d.copy()
    d["pres"] = d.groupby("pid").pres.transform(lambda s: r.permutation(s.values))
    return raw_tau_d(d)


neg = {}
with Timer(log, "negative control"):
    for side in ("bat", "bowl"):
        met = chosen[side].get("value", "v1")
        d = RL.side_frame(v0, side, met)
        obs = raw_tau_d(d)
        null = np.array(Parallel(n_jobs=CFG["n_jobs"])(delayed(shuffled)(d, SEED + i) for i in range(200)))
        p = float((null >= obs).mean())
        neg[side] = {"observed_clutch_var": obs, "null_mean": float(null.mean()), "null_95pct": float(np.quantile(null, 0.95)),
                     "p_value": p, "clutch_sd_if_real": float(np.sqrt(max(obs, 0)))}
        log.info(f"{side}: clutch variance observed {obs:.3e} vs shuffled null mean {null.mean():.3e} "
                 f"(95% {np.quantile(null, .95):.3e}); p = {p:.3f}")
save_json(neg, "negative_control")


# ---------------------------------------------------------------- synthetic injection
def inject_once(seed, side, cfg, boost, cand):
    r = np.random.RandomState(seed)
    picks = r.choice(cand, 10, replace=False)
    v = v0.copy()
    col = f"{cfg.get('value', 'v1')}_{side}"
    m = v[RL.PCOL[side]].isin(picks).values & v.is_pressure.values & v[col].notna().values
    v.loc[m, col] = v.loc[m, col] + boost
    if cfg.get("model") == "R3":
        # R-3 reads the component residuals: plant the boost through the boundary component
        cb = RL.COMPS[side][0]
        rel, beta = comp[side]["reliability"][cb], comp[side]["beta"][cb]
        v.loc[m, cb] = v.loc[m, cb] + boost / (rel * beta)
    if cfg.get("model") == "R1":
        bcol = f"{cfg.get('value', 'v1')}_bowl"
        if side == "bat":
            v.loc[m, bcol] = v.loc[m, bcol] - boost
        else:
            v.loc[m, bcol] = v.loc[m, bcol] + boost
    sc = RL.build_scores(v, side, cfg, comp)
    sc = sc[~sc.index.isin(EXCL)]
    base_rank = FINAL[side].final_rank.reindex(picks)
    new_rank = sc.score.rank(ascending=False).reindex(picks)
    return {"in_top10": int((new_rank <= 10).sum()), "in_top25": int((new_rank <= 25).sum()),
            "median_rank_gain": float((base_rank - new_rank).median()),
            "gap_detected": float((sc.gap_shrunk.reindex(picks) > 0).mean()) if "gap_shrunk" in sc else np.nan}


inj = []
with Timer(log, "synthetic injection"):
    for side in ("bat", "bowl"):
        cfg = chosen[side]
        met = cfg.get("value", "v1")
        d = RL.side_frame(v0, side, met)
        _, _, tau2p = RL.eb_shrink(d[d.pres])
        tau = np.sqrt(tau2p)
        cand = FINAL[side].index[FINAL[side].n_p >= 60].values
        for mult in (0.5, 1.0, 2.0):
            reps = 30 if cfg.get("model") in ("R1", "blend") else 60
            res = Parallel(n_jobs=CFG["n_jobs"])(delayed(inject_once)(SEED + i, side, cfg, mult * tau, cand) for i in range(reps))
            R = pd.DataFrame(res)
            inj.append({"side": side, "boost_in_true_sd": mult, "boost_value": mult * tau, "reps": reps,
                        "avg_planted_in_top10": R.in_top10.mean(), "avg_planted_in_top25": R.in_top25.mean(),
                        "median_rank_gain": R.median_rank_gain.mean(), "gap_detected_share": R.gap_detected.mean()})
            log.info(f"{side} boost {mult} SD: {inj[-1]}")
save_table(pd.DataFrame(inj), "synthetic_injection")

# ---------------------------------------------------------------- bias audit
aud = []
for side in ("bat", "bowl"):
    F = FINAL[side]
    pool = F[F.n_p >= max(30, chosen[side].get("min_n", 0))]
    top = F.sort_values("final_rank").head(10)
    pc = RL.PCOL[side]
    vv = v0[v0[pc].isin(pool.index)]
    if side == "bat":
        vv = vv[vv.faced]
    pl = vv.groupby(pc).agg(ipl_share=("level", lambda s: (s == "IPL").mean()), mean_year=("year", "mean"),
                            pressure_share=("is_pressure", "mean"), position=("batting_order", "median"),
                            pace_share=("is_pace", "mean"))
    rl = roles[roles.side == side].set_index("player_id").role
    pl["role"] = rl.reindex(pl.index).fillna("unassigned")
    for col in ("ipl_share", "mean_year", "pressure_share", "position"):
        aud.append({"side": side, "attribute": col, "top10_mean": pl.loc[pl.index.intersection(top.index), col].mean(),
                    "pool_mean": pl[col].mean()})
    rc_top = pl.loc[pl.index.intersection(top.index), "role"].value_counts(normalize=True)
    rc_pool = pl.role.value_counts(normalize=True)
    for r_ in rc_pool.index:
        aud.append({"side": side, "attribute": f"role: {r_}", "top10_mean": float(rc_top.get(r_, 0)), "pool_mean": float(rc_pool[r_])})
A = pd.DataFrame(aud)
save_table(A, "bias_audit")
log.info("Bias audit:\n" + A.round(3).to_string())

# ---------------------------------------------------------------- where does pressure fall, and is the innings balance fair?
# 1) share of balls vs share of pressure balls vs share of ranking weight (pressure percentile^(4 alpha)), by innings/phase
#    and by batting position; 2) an innings-balanced alternative (percentile within each innings) judged with P4's own
#    yardsticks (win-chance swing, extra wickets vs the stage baseline) and by how much the top 10 would change.
with Timer(log, "pressure balance check"):
    need_f = [c for c in ["y_class", "batter_faced", "wkts_before"] if c not in v0.columns]
    need_c = [c for c in ["fold", "wp_now", "wp_next"] if c not in v0.columns]
    ff = load("features", columns=["ball_uid"] + need_f)
    cx = load("ctx_preds", columns=["ball_uid"] + need_c)
    vb = v0.merge(ff, on="ball_uid").merge(cx, on="ball_uid")
    vb["w"] = vb.pressure_pct.clip(1e-3, 1) ** 2   # soft pressure weight (percentile squared; the R2-weighted alpha 0.5 form)
    vb["innings_lbl"] = np.where(vb.innings == 1, "1st innings (setting)", "2nd innings (chasing)")
    vb["phase_lbl"] = vb.phase.map({0: "Powerplay (1-6)", 1: "Middle (7-15)", 2: "Death (16-20)"})
    vb["position"] = pd.cut(vb.batting_order, [0, 2, 4, 6, 11], labels=["1-2", "3-4", "5-6", "7-11"]).astype(str)
    rows_w = []
    for lab, d_, keys in (("all balls", vb, ["innings_lbl", "phase_lbl"]), ("balls faced", vb[vb.batter_faced], ["position", "innings_lbl"])):
        for kk in (keys[:1], keys):
            g = d_.groupby(kk, observed=True)
            t = pd.DataFrame({"share_of_balls": g.size() / len(d_), "share_of_pressure_balls": g.is_pressure.sum() / d_.is_pressure.sum(),
                              "share_of_soft_pressure_weight": g.w.sum() / d_.w.sum()}).reset_index()
            t.insert(0, "group", " x ".join(kk)); t.insert(0, "balls", lab)
            t = t[~t.isin(["nan"]).any(axis=1)]   # balls with no batting position recorded
            rows_w.append(t)
    save_table(pd.concat(rows_w, ignore_index=True), "pressure_where_it_falls")
    K_ = CFG["cv"]["n_folds"]
    key = (vb.innings.astype(str) + "_" + vb.phase.astype(str) + "_" + vb.wkts_before.clip(upper=7).astype(str) + "_"
           + vb.lvl_code.astype(str) + "_" + vb.is_pace.astype(str))
    stage = np.zeros((len(vb), 8))
    for k in range(K_):
        tr_, te_ = (vb.fold != k).values, (vb.fold == k).values
        tab = pd.crosstab(key[tr_], vb.y_class.values[tr_]).reindex(columns=range(8), fill_value=0) + 1
        stage[te_] = tab.div(tab.sum(1), axis=0).reindex(key[te_]).fillna(1 / 8).values
    yc = vb.y_class.values
    wk, dt = (yc == 6).astype(float), (yc == 0).astype(float)
    dwp = (vb.wp_next - vb.wp_now).abs().values
    okw = ~np.isnan(dwp)
    VAR = {"CHOSEN: percentile over all balls": vb.pressure_pct.values,
           "Innings-balanced: percentile within each innings": vb.groupby("innings").pressure.rank(pct=True).values}
    rows_b, tops = [], {}
    for nm, pct in VAR.items():
        fl = pct >= 0.8
        for scope, m in (("all pressure balls", fl), ("1st-innings pressure balls", fl & (vb.innings == 1).values),
                         ("chase pressure balls", fl & (vb.innings == 2).values)):
            rows_b.append({"variant": nm, "scope": scope, "share_of_pressure_balls": m.sum() / fl.sum(),
                           "win_chance_swing_vs_normal": dwp[m & okw].mean() / dwp[okw].mean(),
                           "wickets_vs_expected": wk[m].sum() / stage[m, 6].sum(), "dot_ball_shift": (dt[m] - stage[m, 0]).mean()})
        vv = v0.assign(pressure_pct=pct, is_pressure=fl)
        for side in ("bat", "bowl"):
            t_ = RL.build_scores(vv, side, chosen[side], comp)
            tops[(nm, side)] = t_[~t_.index.isin(EXCL)]
    names_ = list(VAR)
    for side in ("bat", "bowl"):
        a_, b_ = tops[(names_[0], side)], tops[(names_[1], side)]
        cm = a_.index.intersection(b_.index)
        rows_b.append({"variant": names_[1], "scope": f"{side}: top-10 overlap with chosen",
                       "top10_overlap": len(set(a_.index[:10]) & set(b_.index[:10])),
                       "rank_correlation": a_.score.reindex(cm).corr(b_.score.reindex(cm), method="spearman"),
                       "joins": ", ".join(players.name.reindex(list(set(b_.index[:10]) - set(a_.index[:10]))).astype(str)),
                       "leaves": ", ".join(players.name.reindex(list(set(a_.index[:10]) - set(b_.index[:10]))).astype(str))})
    PB = pd.DataFrame(rows_b)
    save_table(PB, "pressure_innings_balance")
    log.info("Pressure innings-balance check:\n" + PB.round(3).to_string())

# ---------------------------------------------------------------- is the chase share real? model-free swing check
# In the same match situation, how much does the ACTUAL final result differ after a wicket (or a boundary instead of a
# dot), compared with what the win-probability model says? Ratio > 1 = the model under-states that swing.
with Timer(log, "swing check"):
    fs = load("features", columns=["ball_uid", "over_no", "wkts_before", "score_before", "need", "balls_left", "wih", "y_class", "irregular"])
    cs = load("ctx_preds", columns=["ball_uid", "wp_next", "bat_won"])
    ds = v0[["ball_uid", "innings", "phase", "is_wicket", "boundary", "legal"]].merge(fs, on="ball_uid").merge(cs, on="ball_uid")
    ds = ds[ds.bat_won.isin([0, 1]) & ~ds.irregular & ds.wp_next.notna() & (ds.innings <= 2) & (ds.legal == 1)].copy()
    ds["y"] = ds.bat_won.astype(float)
    i1 = ds.innings.eq(1)
    ds["bucket"] = np.where(i1, "1_" + ds.over_no.astype(str) + "_" + ds.wkts_before.clip(upper=6).astype(str) + "_" + (ds.score_before // 15).astype(str),
                            "2_" + (ds.balls_left // 6).astype(str) + "_" + ds.wih.clip(lower=4).astype(str) + "_" + (ds.need.clip(upper=200) // 12).astype(str))
    sw = []
    for event, evm, bsm in (("wicket vs no wicket", ds.is_wicket.astype(bool), ~ds.is_wicket.astype(bool)),
                            ("boundary vs dot", ds.boundary.astype(bool), ds.y_class.eq(0))):
        for (inn, ph), g in ds.groupby(["innings", "phase"]):
            e_, o_ = g[evm.loc[g.index]], g[bsm.loc[g.index]]
            j = e_.groupby("bucket").agg(ye=("y", "mean"), pe=("wp_next", "mean"), n_e=("y", "size")).join(
                o_.groupby("bucket").agg(yo=("y", "mean"), po=("wp_next", "mean"), n_o=("y", "size")), how="inner")
            j = j[(j.n_e >= 3) & (j.n_o >= 10)]
            emp, mod = np.average(j.ye - j.yo, weights=j.n_e), np.average(j.pe - j.po, weights=j.n_e)
            sw.append({"event": event, "innings": int(inn), "phase": {0: "powerplay", 1: "middle", 2: "death"}[ph], "events": int(j.n_e.sum()),
                       "actual_result_change": emp, "model_says": mod, "actual_over_model": emp / mod})
    SW = pd.DataFrame(sw)
    save_table(SW, "pressure_swing_check")
    log.info("Model-free swing check:\n" + SW.round(3).to_string())

# ---------------------------------------------------------------- luck vs skill: how much can one season tell us?
from scipy.stats import spearmanr as _sp
with Timer(log, "season noise check"):
    def _pf(d, side, met):
        pc = RL.PCOL[side]
        x = d[d.is_pressure & d[f"{met}_{side}"].notna()]
        return x[[pc, "match_id", f"{met}_{side}"]].rename(columns={pc: "pid", f"{met}_{side}": "val"})

    def _half_rel(x, min_half=10, reps=30):
        mids = x.match_id.unique(); rs = []
        for r in range(reps):
            h = pd.Series(np.random.RandomState(SEED + r).randint(0, 2, len(mids)), index=mids)
            g = x.assign(h=x.match_id.map(h).values).groupby(["pid", "h"]).val.agg(["mean", "size"]).unstack("h")
            g = g[(g[("size", 0)] >= min_half) & (g[("size", 1)] >= min_half)]
            if len(g) >= 15:
                rs.append(_sp(g[("mean", 0)], g[("mean", 1)]).correlation)
        r_ = float(np.mean(rs)) if rs else np.nan
        return 2 * r_ / (1 + r_) if rs else np.nan

    nz = []
    for side in ("bat", "bowl"):
        for yr in (2024, 2025, 2026):
            nz.append({"side": side, "check": f"one season agrees with itself ({yr}, split-half)", "value": _half_rel(_pf(v0[v0.year == yr], side, "v2"))})
        tr_ = v0[v0.year < 2026]; mids_ = tr_.match_id.unique(); rs = []
        n_p = tr_[tr_.is_pressure].groupby(RL.PCOL[side]).size()
        for r in range(6):
            h = pd.Series(np.random.RandomState(100 + r).randint(0, 2, len(mids_)), index=mids_)
            a_ = RL.build_scores(tr_[tr_.match_id.map(h) == 0], side, chosen[side], comp).rating
            b_ = RL.build_scores(tr_[tr_.match_id.map(h) == 1], side, chosen[side], comp).rating
            ids = a_.index.intersection(b_.index).intersection(n_p[n_p >= 60].index)
            rs.append(_sp(a_.reindex(ids), b_.reindex(ids)).correlation)
        rr_ = float(np.mean(rs)); nz.append({"side": side, "check": "our ratings agree between two halves of 2022-25", "value": 2 * rr_ / (1 + rr_)})
        x = _pf(v0, side, "v2"); g = x.groupby("pid").val.agg(["sum", "count"]); s2 = float(np.var(x.val.values))
        tau2, k_, _ = RL.eb_params(g["sum"].values, g["count"].values.astype(float), s2)
        n26 = _pf(v0[v0.year == 2026], side, "v2").groupby("pid").size(); n_typ = float(n26[n26 >= 20].median())
        nz.append({"side": side, "check": f"skill share of the spread in one season (median {n_typ:.0f} pressure balls)", "value": tau2 / (tau2 + s2 / n_typ)})
        nz.append({"side": side, "check": "pressure balls needed before skill is half of what you see", "value": k_})
        sv = []
        for yr in (2022, 2023, 2024, 2025):
            a_ = _pf(v0[v0.year == yr], side, "v2").groupby("pid").val.agg(["mean", "size"])
            b_ = _pf(v0[v0.year == yr + 1], side, "v2").groupby("pid").val.agg(["mean", "size"])
            j = a_.join(b_, lsuffix="_a", rsuffix="_b", how="inner"); j = j[(j.size_a >= 20) & (j.size_b >= 20)]
            sv.append(_sp(j.mean_a, j.mean_b).correlation)
        nz.append({"side": side, "check": "raw pressure value, one season vs the next (no model)", "value": float(np.mean(sv))})
    NZ = pd.DataFrame(nz)
    for side in ("bat", "bowl"):
        ry = NZ[(NZ.side == side) & NZ.check.str.contains("2026")].value.iloc[0]
        rr_ = NZ[(NZ.side == side) & NZ.check.str.startswith("our ratings")].value.iloc[0]
        NZ.loc[len(NZ)] = {"side": side, "check": "ceiling on a 2026 rank correlation (sqrt of the two reliabilities)", "value": float(np.sqrt(max(ry, 0) * max(rr_, 0)))}
    save_table(NZ, "season_noise_check")
    log.info("Season noise check:\n" + NZ.round(3).to_string())

summ = {"sensitivity_median_overlap": S.groupby("side").top10_overlap.median().to_dict(),
        "negative_control": neg, "injection": inj}
save_json(summ, "p11_summary")
log.info("P11 done")
