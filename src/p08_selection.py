"""P8 - Feature selection, two tracks, one scorecard.
Track 1 (context model M1-base): redundancy (|rho|>0.9), VIF, SHAP + grouped permutation importance (OOF fold),
         null-importance test (shuffled labels), stability selection (subsampled refits), family ablation.
Track 2 (player profiles E/F, usual / pressure / delta): split-half reliability filter (>=0.3), redundancy clusters,
         mutual information + Spearman vs FUTURE pressure performance (profiles 2022-24 -> yardstick 2025)."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import shap
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import log_loss
from sklearn.feature_selection import mutual_info_regression
from common import CFG, SEED, NJ, MODELS, load, save_table, save_json, load_json, get_logger, Timer

log = get_logger("p08_selection")
f = load("features")
c = load("ctx_preds")
fam = load_json("feature_families")
CM = joblib.load(f"{MODELS}/context_models.joblib")
BASE = CM["feats"]["m1"]["base"]
FULL = CM["feats"]["m1"]["full"]
y = f.y_class.values
folds = f.match_id.map(load("folds").set_index("match_id").fold).values
LGBP = dict(CFG["models"]["lgb_common"], n_jobs=NJ, random_state=SEED)
rows1 = {}

# ------------------------------------------------ redundancy + VIF
S = f[FULL].sample(50000, random_state=SEED)
rho = S.corr(method="spearman")
pairs = [(a, b, rho.loc[a, b]) for i, a in enumerate(FULL) for b in FULL[i + 1:] if abs(rho.loc[a, b]) > 0.9]
save_table(pd.DataFrame(pairs, columns=["feature_a", "feature_b", "spearman"]), "fs_redundant_pairs")
log.info(f"Redundant pairs (|rho|>0.9): {len(pairs)}")
Xv = f[BASE].sample(30000, random_state=SEED).fillna(0).astype(float)
Xv = (Xv - Xv.mean()) / (Xv.std() + 1e-9)
vif = {}
for col in BASE:
    others = Xv.drop(columns=col).values
    beta, *_ = np.linalg.lstsq(np.c_[np.ones(len(Xv)), others], Xv[col].values, rcond=None)
    r2 = 1 - np.var(Xv[col].values - np.c_[np.ones(len(Xv)), others] @ beta) / max(np.var(Xv[col].values), 1e-12)
    vif[col] = 1 / max(1 - r2, 1e-6)
for c_ in BASE:
    rows1[c_] = {"feature": c_, "track": "context (M1)", "vif": vif[c_],
                 "redundant_with": ", ".join(b for a, b, _ in pairs if a == c_) + ", ".join(a for a, b, _ in pairs if b == c_)}

# ------------------------------------------------ importance on fold 0 model (OOF rows)
mdl = CM["m1"][("base", 0)]
te = folds == 0
Xte, yte = f.loc[te, BASE], y[te]
base_ll = log_loss(yte, mdl.predict(Xte, num_iteration=mdl.best_iteration), labels=range(8))
rng = np.random.RandomState(SEED)
with Timer(log, "permutation importance"):
    mids_te = f.match_id.values[te]
    for col in BASE:
        Xp = Xte.copy()
        # grouped permutation: shuffle whole matches' values of the feature against other matches
        um = np.unique(mids_te); mp = dict(zip(um, rng.permutation(um)))
        order = pd.Series(np.arange(len(Xp)), index=mids_te)
        Xp[col] = rng.permutation(Xp[col].values)
        rows1[col]["perm_dlogloss"] = log_loss(yte, mdl.predict(Xp, num_iteration=mdl.best_iteration), labels=range(8)) - base_ll
with Timer(log, "SHAP"):
    smp = Xte.sample(1500, random_state=SEED)
    sv = shap.TreeExplainer(mdl).shap_values(smp)
    sv = np.array(sv)
    if sv.ndim == 3 and sv.shape[0] == 8:
        sv = np.transpose(sv, (1, 2, 0))
    mean_abs = np.abs(sv).sum(axis=2).mean(axis=0)
    for col, val in zip(BASE, mean_abs):
        rows1[col]["shap_mean_abs"] = float(val)

# ------------------------------------------------ null importance (shuffled labels)
tr = folds != 0
Xtr, ytr = f.loc[tr, BASE], y[tr]
P = dict(LGBP, objective="multiclass", num_class=8, learning_rate=0.1)
with Timer(log, "null importance"):
    real = lgb.train(P, lgb.Dataset(Xtr, ytr), 150).feature_importance("gain")
    nulls = np.array([lgb.train(dict(P, seed=s), lgb.Dataset(Xtr, rng.permutation(ytr)), 150).feature_importance("gain")
                      for s in range(8)])
for i, col in enumerate(BASE):
    rows1[col]["gain_real"] = float(real[i]); rows1[col]["gain_null_max"] = float(nulls[:, i].max())
    rows1[col]["null_test_pass"] = bool(real[i] > nulls[:, i].max())

# ------------------------------------------------ stability selection (20 x 50% of matches)
with Timer(log, "stability selection"):
    um = np.unique(f.match_id[tr])
    sel = np.zeros(len(BASE))
    for s in range(20):
        pick = set(rng.choice(um, len(um) // 2, replace=False))
        m_ = f.match_id[tr].isin(pick).values
        g = lgb.train(dict(P, seed=s), lgb.Dataset(Xtr[m_], ytr[m_]), 100).feature_importance("gain")
        sel += (g / g.sum()) > 0.005
for i, col in enumerate(BASE):
    rows1[col]["stability_freq"] = float(sel[i] / 20)

# ------------------------------------------------ family ablation (5-fold OOF log-loss)
EXO_B = [x for x in BASE if x in fam["B_momentum"]]
abl = {"base (all)": BASE, "drop A_derived": [x for x in BASE if x not in fam["A_derived"]],
       "drop exogenous momentum": [x for x in BASE if x not in EXO_B],
       "drop C_context": [x for x in BASE if x not in fam["C_context"]]}
ab_rows = []
with Timer(log, "family ablation"):
    for nm, feats in abl.items():
        oof = np.zeros((len(f), 8))
        for k in range(5):
            trk, tek = folds != k, folds == k
            m_ = lgb.train(dict(P, learning_rate=0.08), lgb.Dataset(f.loc[trk, feats], y[trk]), 250)
            oof[tek] = m_.predict(f.loc[tek, feats])
        ab_rows.append({"variant": nm, "n_features": len(feats), "oof_logloss": log_loss(y, oof, labels=range(8))})
AB = pd.DataFrame(ab_rows)
AB["delta_vs_base"] = AB.oof_logloss - AB.oof_logloss.iloc[0]
AB["family_kept"] = AB.delta_vs_base > 0
save_table(AB, "fs_family_ablation")
log.info("Family ablation:\n" + AB.to_string())
fam_keep = dict(zip(["A_derived", "B_momentum", "C_context"], AB.family_kept.iloc[1:].tolist()))
for col, r in rows1.items():
    fm = next((k for k, v_ in fam.items() if col in v_), "?")
    r["family"] = fm
    passes = sum([r["null_test_pass"], r["stability_freq"] >= 0.7, r["perm_dlogloss"] > 0])
    r["decision"] = "keep" if passes >= 2 else "drop (weak)"
    r["reason"] = f"null {'pass' if r['null_test_pass'] else 'fail'}, stability {r['stability_freq']:.2f}, perm dLL {r['perm_dlogloss']:.5f}"
T1 = pd.DataFrame(rows1.values())

# =================================================== Track 2: player profiles
v = load("values")
fx = f[["ball_uid", "phase", "is_pace", "lhb", "str_balls", "str_dot_streak", "delivery_style", "prev_ball_bdry_same_over",
        "y_class", "is_four", "is_six", "bowler_runs"]]
v = v.merge(fx[["ball_uid"] + [c_ for c_ in fx.columns if c_ not in v.columns]], on="ball_uid")
v["dot"] = (v.y_class == 0).astype(float)
v["four"] = v.is_four.astype(float); v["six"] = v.is_six.astype(float); v["bdry"] = v.boundary.astype(float)
v["rot"] = (v.runs.between(1, 3)).astype(float)
v["out"] = v.striker_out.astype(float)
v["extras"] = (v.is_wide | v.is_noball).astype(float)
v["wkt"] = v.bowler_wicket.astype(float)
v["yorker"] = v.delivery_style.isin(["Yorker", "Wide Yorker"]).astype(float)
v["slower"] = v.delivery_style.isin(["Slow Ball", "Slow Bouncer", "Knuckle", "Off Cutter", "Leg Cut"]).astype(float) * v.is_pace
v["bouncer"] = v.delivery_style.isin(["Bouncer", "Slow Bouncer"]).astype(float)
v["fulltoss"] = v.delivery_style.isin(["Full Toss", "Beamer"]).astype(float)
v["one"] = 1.0
BAT_RATIOS = {"sr": ("runs", 100), "dot_pct": ("dot", 1), "bdry_pct": ("bdry", 1), "four_pct": ("four", 1), "six_pct": ("six", 1),
              "rotation_pct": ("rot", 1), "out_rate": ("out", 1), "runs_above_exp": ("r_runs_bat", 1),
              "bdry_above_exp": ("r_bdry_bat", 1), "dot_above_exp": ("r_dot_bat", 1), "outs_above_exp": ("r_wkt_bat", 1),
              "v1_per_ball": ("v1_bat", 1), "v2_per_ball": ("v2_bat", 1)}
BOWL_RATIOS = {"econ": ("bowler_runs", 6), "dot_pct": ("dot", 1), "bdry_pct": ("bdry", 1), "six_pct": ("six", 1),
               "extras_pct": ("extras", 1), "wkt_rate": ("wkt", 1), "bdry_above_exp": ("r_bdry_bowl", 1),
               "dot_above_exp": ("r_dot_bowl", 1), "wkt_above_exp": ("r_wkt_bowl", 1), "extras_above_exp": ("r_extra_bowl", 1),
               "v1_per_ball": ("v1_bowl", 1), "v2_per_ball": ("v2_bowl", 1)}
BAT_MASKS = {"pp": v.phase == 0, "middle": v.phase == 1, "death": v.phase == 2, "vs_pace": v.is_pace == 1,
             "vs_spin": v.is_pace == 0, "chasing": v.innings == 2, "setting": v.innings == 1, "startup": v.str_balls < 10,
             "set": v.str_balls >= 20, "after_2_dots": v.str_dot_streak >= 2}
BOWL_MASKS = {"pp": v.phase == 0, "middle": v.phase == 1, "death": v.phase == 2, "vs_lhb": v.lhb == 1, "vs_rhb": v.lhb == 0,
              "after_conceding_bdry": v.prev_ball_bdry_same_over == 1}


def profile(d, side):
    pcol = "striker_id" if side == "bat" else "bowler_id"
    d = d[d.faced] if side == "bat" else d
    R = BAT_RATIOS if side == "bat" else BOWL_RATIOS
    out, cnt = {}, {}
    g_all = d.groupby(pcol)
    n_all = g_all.one.sum(); dp = d[d.is_pressure]; n_p = dp.groupby(pcol).one.sum()
    for nm, (col, mult) in R.items():
        u = g_all[col].sum() / n_all * mult
        pr = dp.groupby(pcol)[col].sum() / n_p * mult
        out[f"{nm}_usual"] = u; out[f"{nm}_pressure"] = pr; out[f"{nm}_delta"] = pr - u
        cnt[f"{nm}_usual"] = n_all; cnt[f"{nm}_pressure"] = n_p; cnt[f"{nm}_delta"] = n_p
    masks = BAT_MASKS if side == "bat" else BOWL_MASKS
    key = "runs" if side == "bat" else "bowler_runs"
    mult = 100 if side == "bat" else 6
    for mk, msk in masks.items():
        dm = d[msk.reindex(d.index).values]
        n_m = dm.groupby(pcol).one.sum()
        out[f"{'sr' if side == 'bat' else 'econ'}_{mk}"] = dm.groupby(pcol)[key].sum() / n_m * mult
        cnt[f"{'sr' if side == 'bat' else 'econ'}_{mk}"] = n_m
    if side == "bat":
        st = d[d.str_balls < 10].groupby(pcol)
        out["startup_out_rate"] = st.out.sum() / st.one.sum(); cnt["startup_out_rate"] = st.one.sum()
        out["acceleration"] = out["sr_set"] - out["sr_startup"]; cnt["acceleration"] = cnt["sr_set"]
        out["death_share"] = (d.phase == 2).groupby(d[pcol]).mean(); cnt["death_share"] = n_all
        out["position"] = d.groupby(pcol).batting_order.median(); cnt["position"] = n_all
    else:
        for x in ("yorker", "slower", "bouncer", "fulltoss"):
            out[f"{x}_pct"] = g_all[x].mean(); cnt[f"{x}_pct"] = n_all
        ent = d.groupby([pcol, "delivery_style"]).size().groupby(level=0).apply(lambda s: -(s / s.sum() * np.log(s / s.sum())).sum())
        out["variation_entropy"] = ent; cnt["variation_entropy"] = n_all
        out["death_share"] = (d.phase == 2).groupby(d[pcol]).mean(); cnt["death_share"] = n_all
        out["new_ball_share"] = (d.phase == 0).groupby(d[pcol]).mean(); cnt["new_ball_share"] = n_all
    return pd.DataFrame(out), pd.DataFrame(cnt)


rows2 = []
mids = v.match_id.unique()
half = pd.Series(np.random.RandomState(SEED).randint(0, 2, len(mids)), index=mids)
v["half"] = v.match_id.map(half)
for side in ("bat", "bowl"):
    with Timer(log, f"{side} profiles"):
        full_p, full_n = profile(v, side)
        h0, n0 = profile(v[v.half == 0], side)
        h1, n1 = profile(v[v.half == 1], side)
        tr_p, tr_n = profile(v[v.year <= 2024], side)
        pcol = "striker_id" if side == "bat" else "bowler_id"
        fut = v[(v.year == 2025) & v.is_pressure & v[f"v2_{side}"].notna()].groupby(pcol)[f"v2_{side}"].agg(["mean", "count"])
        fut = fut[fut["count"] >= 30]
    # redundancy clusters on the full-data profile (players with >= 200 balls)
    big = full_n.iloc[:, 0] >= 200
    C = full_p[big].corr(method="spearman").fillna(0)
    Dm = 1 - C.abs().values
    np.fill_diagonal(Dm, 0)
    cl = fcluster(linkage(squareform(Dm, checks=False), "average"), t=0.2, criterion="distance")
    grp = dict(zip(C.columns, cl))
    for col in full_p.columns:
        minh = 30 if ("pressure" in col or "delta" in col) else 60
        okk = (n0[col] >= minh) & (n1[col] >= minh)
        a, b_ = h0[col][okk], h1[col][okk]
        okk2 = a.notna() & b_.notna()
        r = np.corrcoef(a[okk2], b_[okk2])[0, 1] if okk2.sum() >= 15 else np.nan
        rsb = 2 * r / (1 + r) if r == r and r > -1 else np.nan
        x = tr_p[col].reindex(fut.index)
        okf = x.notna() & (tr_n[col].reindex(fut.index) >= minh)
        sp_ = spearmanr(x[okf], fut["mean"][okf])[0] if okf.sum() >= 15 else np.nan
        mi = mutual_info_regression(x[okf].values.reshape(-1, 1), fut["mean"][okf].values, random_state=SEED)[0] if okf.sum() >= 15 else np.nan
        rows2.append({"feature": f"{side}:{col}", "track": f"player profile ({side})", "reliability": rsb,
                      "n_players_reliability": int(okk2.sum()), "spearman_future_pressure": sp_, "mi_future_pressure": mi,
                      "n_players_future": int(okf.sum()), "redundancy_group": f"{side}-{grp.get(col, -1)}"})
T2 = pd.DataFrame(rows2)
T2["rel_ok"] = T2.reliability >= 0.3
T2["group_best"] = T2.groupby("redundancy_group").reliability.transform("max") == T2.reliability
T2["decision"] = np.select([~T2.rel_ok, ~T2.group_best], ["drop (noise: reliability < 0.3)", "drop (redundant; group has a more reliable feature)"], "keep")
T2["reason"] = ("reliability " + T2.reliability.round(2).astype(str) + ", future-pressure rho " + T2.spearman_future_pressure.round(2).astype(str))
SC = pd.concat([T1, T2], ignore_index=True)
save_table(SC, "feature_scorecard")
log.info(f"Track 1: {int((T1.decision == 'keep').sum())}/{len(T1)} context features kept")
log.info(f"Track 2: {int((T2.decision == 'keep').sum())}/{len(T2)} profile features kept; "
         f"reliability >= 0.3: {int(T2.rel_ok.sum())}")
log.info("Most reliable profile features:\n" + T2.sort_values("reliability", ascending=False).head(15)[["feature", "reliability", "spearman_future_pressure", "decision"]].round(3).to_string())
log.info("Delta (pressure - usual) features reliability:\n" + T2[T2.feature.str.contains("_delta")][["feature", "reliability", "spearman_future_pressure"]].round(3).to_string())
log.info("P8 done")
