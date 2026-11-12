"""P3 - Player-blind context models, all out-of-fold by match.
M1 ball outcome (8 classes)   -> expected outcome probabilities per ball (core / base / full feature sets)
M2 chase win probability      -> monotone LightGBM (Optuna, nested) vs logistic (curves for game state, straight lines
                                  for match context) vs XGBoost; cross-fitted isotonic
M3 first-innings win prob     -> same recipe
M4 run expectancy             -> Poisson LightGBM (monotone) vs lookup table
Then derives, per ball, everything later phases need: WP now / next, expected WP next, leverage, RE now / next,
expected run value. Hypothetical next states use the SAME fold's models, so every number is out-of-fold."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import optuna
import joblib
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, mean_absolute_error, mean_poisson_deviance
from common import CFG, SEED, NJ, MODELS, load, save, save_json, save_table, load_json, get_logger, Timer

optuna.logging.set_verbosity(optuna.logging.WARNING)
log = get_logger("p03_context_models")
f = load("features")
FAM = load_json("feature_families")
K = CFG["cv"]["n_folds"]
LGBP = dict(CFG["models"]["lgb_common"], n_jobs=NJ, random_state=SEED)

# one match-level fold assignment shared by every model
mids = np.sort(f.match_id.unique())
rng = np.random.RandomState(SEED)
perm = rng.permutation(len(mids))
fold_of = pd.Series(perm % K, index=mids)
f["fold"] = f.match_id.map(fold_of).values
save(pd.DataFrame({"match_id": mids, "fold": fold_of.values}), "folds")


def ece(y, p, bins=20):
    q = np.clip((p * bins).astype(int), 0, bins - 1)
    df = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((df.n * (df.y - df.p).abs()).sum() / df.n.sum())


def inner_split(idx_groups, frac=0.15):
    g = np.unique(idx_groups)
    r = np.random.RandomState(SEED).permutation(g)
    val = set(r[: int(len(g) * frac)])
    m = np.array([x in val for x in idx_groups])
    return ~m, m


# =========================================================================== M1 ball outcome
EXO_B = ["wkts_last6", "wkts_last12", "wkts_last24", "balls_since_wkt", "pship_balls", "str_balls", "ns_balls",
         "free_hit", "rebowl", "bwl_balls"]
M1_SETS = {
    "core": FAM["A_core"] + FAM["C_context"],
    "base": FAM["A_core"] + FAM["A_derived"] + EXO_B + FAM["C_context"],
}
if CFG.get("baseline", {}).get("bowler_type", False):
    # role, not skill: expected outcomes for an average bowler of the same type (pace / spin) in the same situation
    M1_SETS["core"] = M1_SETS["core"] + ["is_pace"]
    M1_SETS["base"] = M1_SETS["base"] + ["is_pace"]
M1_SETS["full"] = M1_SETS["base"] + [c for c in FAM["B_momentum"] if c not in EXO_B] + FAM["C_strength"] +     [c for c in FAM["D_matchup"] if c not in M1_SETS["base"]]
NC = 8
y1 = f.y_class.values
m1_scores, m1_models = [], {}
oof1 = {}
for name, feats in M1_SETS.items():
    oof = np.zeros((len(f), NC))
    with Timer(log, f"M1-{name} ({len(feats)} feats)"):
        for k in range(K):
            tr, te = (f.fold != k).values, (f.fold == k).values
            a, v = inner_split(f.match_id.values[tr])
            Xtr = f.loc[tr, feats]
            dtr = lgb.Dataset(Xtr[a], y1[tr][a]); dva = lgb.Dataset(Xtr[v], y1[tr][v])
            params = dict(LGBP, objective="multiclass", num_class=NC, learning_rate=0.06)
            mdl = lgb.train(params, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
            oof[te] = mdl.predict(f.loc[te, feats], num_iteration=mdl.best_iteration)
            m1_models[(name, k)] = mdl
    oof1[name] = oof
    ll = log_loss(y1, oof, labels=range(NC))
    m1_scores.append({"model": f"LightGBM-{name}", "n_features": len(feats), "multi_logloss": ll})
    log.info(f"M1-{name}: OOF multi-logloss {ll:.4f}")

# baselines: lookup table and multinomial spline-logistic (core numeric features)
with Timer(log, "M1 baselines"):
    key = f.innings.astype(str) + "_" + f.phase.astype(str) + "_" + f.wkts_before.clip(upper=7).astype(str) + "_" + f.lvl_code.astype(str)
    oof_lu = np.zeros((len(f), NC))
    for k in range(K):
        tr, te = (f.fold != k).values, (f.fold == k).values
        tab = pd.crosstab(key[tr], y1[tr]).reindex(columns=range(NC), fill_value=0) + 1
        tab = tab.div(tab.sum(1), axis=0)
        glob = np.bincount(y1[tr], minlength=NC) / tr.sum()
        oof_lu[te] = tab.reindex(key[te]).fillna(pd.Series(glob)).values
    m1_scores.append({"model": "Lookup table (innings x phase x wickets x level)", "n_features": 4,
                      "multi_logloss": log_loss(y1, oof_lu, labels=range(NC))})
    num = ["innings", "legal_before", "balls_left", "wkts_before", "score_before", "lvl_code", "year", "venue_par"]
    Xl = f[num].copy()
    Xl["need"] = f.need.fillna(0); Xl["rrr"] = f.rrr.fillna(0).clip(0, 36)
    oof_lr = np.zeros((len(f), NC))
    for k in range(K):
        tr, te = (f.fold != k).values, (f.fold == k).values
        sub = np.where(tr)[0]
        sub = np.random.RandomState(SEED).choice(sub, min(60000, len(sub)), replace=False)
        pipe = make_pipeline(StandardScaler(), SplineTransformer(n_knots=5, degree=3),
                             LogisticRegression(max_iter=300, C=1.0))
        pipe.fit(Xl.iloc[sub], y1[sub])
        pr = np.zeros((te.sum(), NC)); pr[:, pipe[-1].classes_] = pipe.predict_proba(Xl[te])
        oof_lr[te] = np.clip(pr, 1e-6, 1)
    m1_scores.append({"model": "Multinomial logistic + splines (core)", "n_features": len(Xl.columns),
                      "multi_logloss": log_loss(y1, oof_lr / oof_lr.sum(1, keepdims=True), labels=range(NC))})

# per-class calibration of the chosen baseline model (M1-base)
cal_rows = []
for c, lab in enumerate(["dot", "1", "2", "3", "4", "6", "wicket", "wide/nb"]):
    yc = (y1 == c).astype(int)
    cal_rows.append({"class": lab, "observed": yc.mean(), "predicted_mean": oof1["base"][:, c].mean(),
                     "ece": ece(yc, oof1["base"][:, c]), "brier": brier_score_loss(yc, oof1["base"][:, c])})
save_table(pd.DataFrame(m1_scores), "m1_model_comparison")
save_table(pd.DataFrame(cal_rows), "m1_calibration")
log.info("M1 comparison:\n" + pd.DataFrame(m1_scores).to_string())


# =========================================================================== WP models (M2 chase, M3 first innings)
def wp_logit(feats, state):
    """Curves (4 quantile knots) for the in-game state; straight lines for match-level context. Curves on match-level
    inputs (target, venue par, Elo gap, season) let the model memorise individual matches: in P3b the curves-on-every-
    input version had a 3x larger train-test gap and calibration slope 0.86 (over-confident) on unseen matches."""
    ctx = [c for c in feats if c not in state]
    ct = ColumnTransformer([("state", make_pipeline(StandardScaler(), SplineTransformer(n_knots=4, degree=3, knots="quantile",
                                                                                         extrapolation="linear")), state),
                            ("context", StandardScaler(), ctx)])
    return make_pipeline(ct, LogisticRegression(max_iter=2000, C=1.0))


def fit_wp(name, d, feats, mono, n_trials, state=None):
    """Nested: Optuna tunes on inner grouped folds of the training part only; outer fold scores the result."""
    y = d.y.values
    oof = {"lgb": np.zeros(len(d)), "xgb": np.zeros(len(d)), "logit": np.zeros(len(d))}
    models = {}
    for k in range(K):
        tr, te = (d.fold != k).values, (d.fold == k).values
        Xtr, ytr, gtr = d.loc[tr, feats], y[tr], d.match_id.values[tr]
        inner = pd.Series(gtr).map(pd.Series(np.arange(len(np.unique(gtr))) % 3,
                                             index=np.random.RandomState(SEED + k).permutation(np.unique(gtr)))).values

        def objective(trial):
            p = dict(LGBP, objective="binary", monotone_constraints=mono, monotone_constraints_method="intermediate",
                     learning_rate=trial.suggest_float("lr", 0.03, 0.15, log=True),
                     num_leaves=trial.suggest_int("num_leaves", 7, 63),
                     min_child_samples=trial.suggest_int("mcs", 50, 1000, log=True),
                     reg_lambda=trial.suggest_float("l2", 0.1, 50, log=True))
            ls = []
            for j in range(3):
                a, v = inner != j, inner == j
                mdl = lgb.train(p, lgb.Dataset(Xtr[a], ytr[a]), 1500, valid_sets=[lgb.Dataset(Xtr[v], ytr[v])],
                                callbacks=[lgb.early_stopping(50, verbose=False)])
                ls.append(mdl.best_score["valid_0"]["binary_logloss"])
                trial.set_user_attr(f"it{j}", mdl.best_iteration)
            trial.set_user_attr("iters", int(np.mean([trial.user_attrs[f"it{j}"] for j in range(3)])))
            return float(np.mean(ls))

        st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
        st.optimize(objective, n_trials=n_trials)
        bp = st.best_trial
        p = dict(LGBP, objective="binary", monotone_constraints=mono, monotone_constraints_method="intermediate",
                 learning_rate=bp.params["lr"], num_leaves=bp.params["num_leaves"],
                 min_child_samples=bp.params["mcs"], reg_lambda=bp.params["l2"])
        mdl = lgb.train(p, lgb.Dataset(Xtr, ytr), max(50, int(bp.user_attrs["iters"] * 1.1)))
        oof["lgb"][te] = mdl.predict(d.loc[te, feats])
        models[("lgb", k)] = mdl
        # XGBoost monotone, fixed sensible config
        xm = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=4, min_child_weight=50,
                               subsample=0.8, colsample_bytree=0.8, monotone_constraints=tuple(mono),
                               n_jobs=NJ, random_state=SEED, tree_method="hist")
        xm.fit(Xtr.fillna(-1), ytr)
        oof["xgb"][te] = xm.predict_proba(d.loc[te, feats].fillna(-1))[:, 1]
        # logistic: curves for the in-game state, straight lines for match context
        lr = wp_logit(feats, state)
        lr.fit(Xtr.fillna(0), ytr)
        oof["logit"][te] = lr.predict_proba(d.loc[te, feats].fillna(0))[:, 1]
        models[("logit", k)] = lr
        log.info(f"{name} fold {k}: best inner logloss {bp.value:.4f} params {bp.params} iters {bp.user_attrs['iters']}")
    # cross-fitted isotonic calibration of the LightGBM model
    cal = np.zeros(len(d))
    isos = {}
    for k in range(K):
        tr, te = (d.fold != k).values, (d.fold == k).values
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(oof["lgb"][tr], y[tr])
        cal[te] = iso.predict(oof["lgb"][te])
        isos[k] = iso
    oof["lgb_cal"] = cal
    rows = []
    for nm, pr in oof.items():
        pr = np.clip(pr, 1e-4, 1 - 1e-4)
        rows.append({"model": nm, "logloss": log_loss(y, pr), "brier": brier_score_loss(y, pr), "auc": roc_auc_score(y, pr),
                     "ece": ece(y, pr)})
    sc = pd.DataFrame(rows)
    log.info(f"{name} comparison:\n{sc.to_string()}")
    return models, isos, oof, sc


# Chase difficulty, in cricket terms: runs needed vs the runs normally scored from these balls with these wickets
# (a DLS-like resource table learned from first innings; it never sees chase results), scaled by how much that varies.
# Last ball, 4 needed -> about 14%; 6 needed -> about 3%. Without it the spline model was badly wrong in the last overs.
_r1 = f[(f.innings == 1) & (f.maxb == 120) & ~f.irregular & ~f.no_result]
RES = lgb.LGBMRegressor(objective="poisson", n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=100,
                        monotone_constraints=[1, 1], verbose=-1, random_state=SEED, n_jobs=NJ)
RES.fit(_r1[["balls_left", "wih"]], _r1.total - _r1.score_before)
_pr = RES.predict(_r1[["balls_left", "wih"]])
RES_PHI = float((((_r1.total - _r1.score_before) - _pr) ** 2).sum() / _pr.sum())


def add_chase_feats(S):
    S = S.copy()
    S["rrr"] = S.rrr.clip(0, 36)
    r = RES.predict(pd.DataFrame({"balls_left": S.balls_left.clip(lower=0), "wih": S.wih.clip(lower=0)}))
    S["z_chase"] = (S.need - r) / np.sqrt(RES_PHI * np.clip(r, 0.05, None))
    return S


M2F = ["need", "balls_left", "wih", "rrr", "target", "maxb", "lvl_code", "year", "venue_par", "elo_diff", "z_chase"]
M2MONO = [-1, 1, 1, -1, 0, 0, 0, 0, 0, 0, -1]
M3F = ["score_before", "balls_left", "wih", "maxb", "lvl_code", "year", "venue_par", "elo_diff"]
M3MONO = [1, 1, 1, 0, 0, 0, 0, 0]
NT = CFG["models"]["optuna_trials"]

d2 = add_chase_feats(f[(f.innings == 2) & f.bat_won.isin([0, 1]) & ~f.irregular & (f.need > 0)].copy())
d2["y"] = d2.bat_won.astype(int)
with Timer(log, "M2"):
    m2_models, m2_iso, m2_oof, m2_sc = fit_wp("M2-chase", d2, M2F, M2MONO, NT, state=["need", "balls_left", "wih", "rrr", "z_chase"])
# Plan C for the last two overs of a chase (WASP-style backward calculation, "M5" in the plan): from how often a dot,
# 1, 2, 3, 4, 6, wicket or wide happens in final-overs chases (by required rate and wickets in hand, learned per fold),
# compute the exact chance of getting the runs ball by ball. Blended with the main model between 24 and 12 balls left.
USE_DP = False   # tested and rejected (see wp_endgame_variants); kept for the record
DP_B, DP_N = 24, 150
RRB = np.array([-1, 6, 9, 12, 15, 18, 24, 36, 1e9]); WB = np.array([0, 2, 4, 6, 10])
DP_R = np.array([0, 1, 2, 3, 4, 6, 0, 1]); DP_L = np.array([1, 1, 1, 1, 1, 1, 1, 0]); DP_W = np.array([0, 0, 0, 0, 0, 0, 1, 0])


def build_dp(rows):
    prior = np.bincount(rows.y_class, minlength=8) + 1.0
    prior = prior / prior.sum()
    rb = np.digitize(rows.rrr.clip(0, 1e6), RRB[1:-1]); wb = np.digitize(rows.wih, WB[1:-1], right=True)
    probs = np.zeros((len(RRB) - 1, len(WB) - 1, 8))
    for a in range(probs.shape[0]):
        for c in range(probs.shape[1]):
            cnt = np.bincount(rows.y_class[(rb == a) & (wb == c)], minlength=8)
            probs[a, c] = (cnt + 50 * prior) / (cnt.sum() + 50)
    V = np.zeros((DP_B + 1, 11, DP_N + 1))
    for b_ in range(DP_B + 1):
        for w_ in range(11):
            for n_ in range(1, DP_N + 1):
                if b_ == 0 or w_ == 0:
                    V[b_, w_, n_] = 0.5 if n_ == 1 else 0.0
                    continue
                p_ = probs[np.digitize(n_ * 6 / b_, RRB[1:-1]), np.digitize(w_, WB[1:-1], right=True)]
                val = 0.0
                for k_ in range(8):
                    n2 = n_ - DP_R[k_]
                    if n2 <= 0:
                        val += p_[k_]; continue
                    b2, w2 = b_ - DP_L[k_], w_ - DP_W[k_]
                    val += p_[k_] * V[b2, w2, n2]
                V[b_, w_, n_] = val
    return V


dp_rows = f[(f.innings == 2) & (f.balls_left <= 30) & (f.need > 0) & ~f.irregular]
DPV = {k: build_dp(dp_rows[dp_rows.fold != k]) for k in range(K)}
log.info(f"Plan C end-game tables built from {len(dp_rows):,} final-overs chase balls (per fold)")
d3 = f[(f.innings == 1) & f.bat_won.isin([0, 1]) & ~f.irregular].copy()
d3["y"] = d3.bat_won.astype(int)
with Timer(log, "M3"):
    m3_models, m3_iso, m3_oof, m3_sc = fit_wp("M3-first", d3, M3F, M3MONO, NT, state=["score_before", "balls_left", "wih"])
m2_sc.insert(0, "target", "M2 chase"); m3_sc.insert(0, "target", "M3 first innings")
save_table(pd.concat([m2_sc, m3_sc]), "wp_model_comparison")
for nm, d, oo in (("m2", d2, m2_oof), ("m3", d3, m3_oof)):
    r = pd.DataFrame({"y": d.y.values, "p": oo["lgb_cal"]})
    r["bin"] = (r.p * 10).clip(0, 9.999).astype(int)
    save_table(r.groupby("bin").agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size")).reset_index(), f"{nm}_reliability")

# gates (Plan A -> B -> C)
g2 = m2_sc.set_index("model")
m2_pass = bool(g2.at["lgb_cal", "logloss"] <= g2.at["logit", "logloss"] and g2.at["lgb_cal", "ece"] <= CFG["models"]["wp_calibration_max_ece"])
g3 = m3_sc.set_index("model")
m3_pass = bool(g3.at["lgb_cal", "auc"] >= CFG["models"]["first_innings_auc_gate"])
log.info(f"Gate M2 (Plan A LightGBM beats logistic and ECE<=0.03): {m2_pass}")
log.info(f"Gate M3 (AUC >= 0.72 for first-innings leverage): {m3_pass}  (AUC {g3.at['lgb_cal','auc']:.3f})")

# =========================================================================== M4 run expectancy
d4 = f[~f.irregular & ~f.no_result].copy()
d4["y"] = d4.total - d4.score_before
M4F = ["innings", "balls_left", "wih", "score_before", "need", "maxb", "lvl_code", "year", "venue_par"]
M4MONO = [0, 1, 1, 0, 0, 0, 0, 0, 0]
oof4 = np.zeros(len(d4)); oof4_lu = np.zeros(len(d4)); m4_models = {}
with Timer(log, "M4"):
    for k in range(K):
        tr, te = (d4.fold != k).values, (d4.fold == k).values
        a, v = inner_split(d4.match_id.values[tr])
        Xtr, ytr = d4.loc[tr, M4F], d4.y.values[tr]
        p = dict(LGBP, objective="poisson", monotone_constraints=M4MONO, monotone_constraints_method="intermediate")
        mdl = lgb.train(p, lgb.Dataset(Xtr[a], ytr[a]), 3000, valid_sets=[lgb.Dataset(Xtr[v], ytr[v])],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        oof4[te] = mdl.predict(d4.loc[te, M4F], num_iteration=mdl.best_iteration)
        m4_models[k] = mdl
        kk = d4.innings.astype(str) + "_" + (d4.balls_left // 6).astype(str) + "_" + d4.wih.astype(str)
        tab = d4[tr].groupby(kk[tr]).y.mean()
        oof4_lu[te] = kk[te].map(tab).fillna(d4.y[tr].mean()).values
m4_sc = pd.DataFrame([
    {"model": "LightGBM Poisson (monotone)", "poisson_dev": mean_poisson_deviance(d4.y, np.clip(oof4, .1, None)), "mae": mean_absolute_error(d4.y, oof4)},
    {"model": "Lookup (innings x over x wickets)", "poisson_dev": mean_poisson_deviance(d4.y, np.clip(oof4_lu, .1, None)), "mae": mean_absolute_error(d4.y, oof4_lu)}])
save_table(m4_sc, "m4_model_comparison")
log.info(f"M4 comparison:\n{m4_sc.to_string()}")

joblib.dump({"dpv": DPV, "dp_b": DP_B, "dp_n": DP_N, "res": RES, "res_phi": RES_PHI, "m1": {k: v for k, v in m1_models.items() if k[0] in ("base", "core")}, "m2": m2_models, "m2_iso": m2_iso,
             "m3": m3_models, "m3_iso": m3_iso, "m4": m4_models,
             "feats": {"m1": M1_SETS, "m2": M2F, "m3": M3F, "m4": M4F}}, f"{MODELS}/context_models.joblib")

# =========================================================================== per-ball derived quantities
P = oof1["base"]          # expected outcome probs used for values and leverage
PC = oof1["core"]         # scoreboard-only baseline (pressure tests T1)
R_K = np.array([0, 1, 2, 3, 4, 6, 0, 1])
L_K = np.array([1, 1, 1, 1, 1, 1, 1, 0])
W_K = np.array([0, 0, 0, 0, 0, 0, 1, 0])


M2_CHOICE = "lgb_cal" if m2_pass else "logit"      # Plan A -> Plan B (logistic, curves for game state)
M3_CHOICE = "lgb_cal" if m3_pass else "plugin"     # Plan A -> Plan B (run expectancy mapped through the chase model)
log.info(f"Chosen chase WP model: {M2_CHOICE}; first-innings stakes: {M3_CHOICE}")


def wp2_eval(S, folds):
    """chase WP for non-terminal states S (need>0, balls_left>0, wih>0), using the gate-chosen model."""
    S = add_chase_feats(S)
    out = np.zeros(len(S))
    for k in range(K):
        mk = folds == k
        if mk.any():
            if M2_CHOICE == "logit":
                out[mk] = m2_models[("logit", k)].predict_proba(S.loc[mk, M2F].fillna(0))[:, 1]
            else:
                out[mk] = m2_iso[k].predict(m2_models[("lgb", k)].predict(S.loc[mk, M2F]))
            bl = S.loc[mk, "balls_left"].values
            we = np.clip((24 - bl) / 12, 0, 1) * USE_DP
            if (we > 0).any():
                sub = we > 0
                bi = np.clip(bl[sub], 0, DP_B).astype(int)
                wi = np.clip(S.loc[mk, "wih"].values[sub], 0, 10).astype(int)
                ni = np.clip(np.ceil(S.loc[mk, "need"].values[sub]), 1, DP_N).astype(int)
                dp = DPV[k][bi, wi, ni]
                o = out[mk]; o[sub] = (1 - we[sub]) * o[sub] + we[sub] * dp; out[mk] = o
    return out


PHI = float(((d4.y - oof4) ** 2).sum() / np.clip(oof4, 0.1, None).sum())   # dispersion of runs-to-come
GH_X, GH_W = np.polynomial.hermite.hermgauss(9)
log.info(f"Runs-to-come dispersion (variance / mean): {PHI:.2f}")


def wp1_plugin(S, folds):
    """Plan B first-innings WP: batting-first side's chance = E_R[1 - chase WP at the start of a chase of
    (score + R + 1)], R = runs still to come ~ Normal(RE, PHI * RE) integrated by 9-point Gauss-Hermite.
    At the end of the innings RE = 0, so it meets the chase-start value exactly (no jump)."""
    re = re_eval(S, folds)
    sd = np.sqrt(PHI * np.clip(re, 0, None))
    out = np.zeros(len(S))
    for x, w in zip(GH_X, GH_W):
        T = S.copy()
        T["target"] = T.score_before + np.clip(re + np.sqrt(2) * sd * x, 0, None) + 1
        T["need"] = T.target; T["balls_left"] = T.maxb; T["wih"] = 10
        T["rrr"] = T.need * 6 / T.balls_left; T["elo_diff"] = -T.elo_diff
        out += w / np.sqrt(np.pi) * (1 - wp2_eval(T, folds))
    return out


def wp3_eval(S, folds):
    out = np.zeros(len(S))
    for k in range(K):
        mk = folds == k
        if mk.any():
            out[mk] = m3_iso[k].predict(m3_models[("lgb", k)].predict(S.loc[mk, M3F]))
    return out


def re_eval(S, folds):
    out = np.zeros(len(S))
    for k in range(K):
        mk = folds == k
        if mk.any():
            out[mk] = m4_models[k].predict(S.loc[mk, M4F])
    return out


def state_values(S):
    """WP (batting side) and RE for arbitrary states incl. terminal ones. S has innings, score_before, legal_before,
    balls_left, wih, need, target, maxb + context + fold."""
    S = S.copy()
    S["wih"] = S.wih.clip(lower=0)
    folds = S.fold.values
    wp = np.full(len(S), np.nan); re = np.zeros(len(S))
    i2 = S.innings.values == 2
    won = i2 & (S.need.values <= 0)
    over = (S.wih.values <= 0) | (S.balls_left.values <= 0)
    lost2 = i2 & ~won & over
    live2 = i2 & ~won & ~over
    wp[won] = 1.0
    wp[lost2] = np.where(S.need.values[lost2] == 1, 0.5, 0.0)
    if live2.any():
        T = S[live2].copy(); T["rrr"] = T.need * 6 / T.balls_left
        wp[live2] = wp2_eval(T, folds[live2])
    end1 = (~i2) & over
    live1 = (~i2) & ~over
    if live1.any():
        wp[live1] = (wp3_eval(S[live1], folds[live1]) if M3_CHOICE == "lgb_cal"
                     else wp1_plugin(S[live1], folds[live1]))
    if end1.any():
        T = S[end1].copy()
        T["target"] = T.score_before + 1; T["need"] = T.target; T["balls_left"] = T.maxb; T["wih"] = 10
        T["rrr"] = T.need * 6 / T.balls_left; T["elo_diff"] = -T.elo_diff
        wp[end1] = 1 - wp2_eval(T, folds[end1])
    live = live1 | live2
    if live.any():
        re[live] = re_eval(S[live], folds[live])
    return wp, re


CTX = ["fold", "innings", "maxb", "target", "lvl_code", "year", "venue_par", "elo_diff"]
with Timer(log, "derived per-ball values"):
    base = f[["ball_uid", "match_id", "score_before", "legal_before", "balls_left", "wih", "need"] + CTX].copy()
    base["fold"] = f.fold.values
    wp_now, re_now = state_values(base)
    # expected next-state values over the 8 outcome classes
    E_wp = np.zeros(len(f)); E_absdwp = np.zeros(len(f)); E_rv_team = np.zeros(len(f)); E_rv_bat = np.zeros(len(f))
    wp_next_k = np.zeros((len(f), NC)); re_next_k = np.zeros((len(f), NC))
    for c in range(NC):
        S = base.copy()
        S["score_before"] = S.score_before + R_K[c]
        S["legal_before"] = S.legal_before + L_K[c]
        S["balls_left"] = S.balls_left - L_K[c]
        S["wih"] = S.wih - W_K[c]
        S["need"] = S.need - R_K[c]
        wk, rk = state_values(S)
        wp_next_k[:, c] = wk; re_next_k[:, c] = rk
    E_wp = (P * wp_next_k).sum(1)
    LI_raw = (P * np.abs(wp_next_k - wp_now[:, None])).sum(1)
    E_team = (P * (R_K[None, :] + re_next_k)).sum(1)
    # batter expectation: faced balls only -> renormalise over classes 0..6 (no wides); bat runs exclude the extra
    Pb = P[:, :7] / P[:, :7].sum(1, keepdims=True)
    E_bat = (Pb * (R_K[None, :7] + re_next_k[:, :7])).sum(1)
    Pcb = PC[:, :7] / PC[:, :7].sum(1, keepdims=True)

# actual next-state values: next ball's "now" values in the same match (flip side at innings change)
out = f[["ball_uid", "match_id", "innings", "fold", "bat_won", "no_result", "is_tie", "irregular"]].copy()
out["wp_now"] = wp_now; out["re_now"] = re_now
nxt_match = out.match_id.shift(-1); nxt_inn = out.innings.shift(-1)
same_inn = (nxt_match == out.match_id) & (nxt_inn == out.innings)
to_chase = (nxt_match == out.match_id) & (nxt_inn == 2) & (out.innings == 1)
wp_next = np.where(same_inn, out.wp_now.shift(-1), np.nan)
wp_next = np.where(to_chase, 1 - out.wp_now.shift(-1), wp_next)
last = ~same_inn & ~to_chase
# last ball of the match (or of innings 1 with no chase): result from this batting side's view
wp_next = np.where(last & (out.innings == 2), out.bat_won, wp_next)
wp_next = np.where(last & (out.innings == 1) & out.bat_won.notna(), out.bat_won, wp_next)
out["wp_next"] = wp_next
out["re_next"] = np.where(same_inn, out.re_now.shift(-1), 0.0)
out["E_wp_next"] = E_wp
out["LI_raw"] = LI_raw
out["E_team_rv"] = E_team
out["E_bat_rv"] = E_bat
for c, lab in enumerate(["p_dot", "p_1", "p_2", "p_3", "p_4", "p_6", "p_wkt", "p_x"]):
    out[lab] = P[:, c]
    out["c" + lab] = PC[:, c]
out["x_bat_runs"] = (Pb * R_K[None, :7]).sum(1)
out["cx_bat_runs"] = (Pcb * R_K[None, :7]).sum(1)
save(out, "ctx_preds")

# sanity: expected vs actual WP movement (drift) and RE consistency
ok = out.wp_next.notna()
drift = (out.wp_next - out.E_wp_next)[ok]
log.info(f"WPA drift: mean actual-expected {drift.mean():+.5f} per ball (should be ~0)")
tr_ = f.team_runs.values + out.re_next.values - out.E_team_rv.values
log.info(f"Run-value drift: mean actual-expected {np.nanmean(tr_):+.4f} runs per ball (should be ~0)")
log.info(f"LI_raw mean {LI_raw.mean():.4f}; quantiles {np.quantile(LI_raw, [.5, .8, .95, .99]).round(4)}")
# Plan B checks: first-innings plug-in WP quality on the same rows M3 was scored on, and monotonicity of chase WP
d3e = out[(f.innings == 1).values & f.bat_won.isin([0, 1]).values & ~f.irregular.values]
plug_auc = roc_auc_score(d3e.bat_won, d3e.wp_now)
plug_ll = log_loss(d3e.bat_won, d3e.wp_now.clip(1e-4, 1 - 1e-4))
log.info(f"First-innings Plan B (plug-in) OOF: AUC {plug_auc:.3f}, logloss {plug_ll:.4f}, ECE {ece(d3e.bat_won.values, d3e.wp_now.values):.4f}")
eg = out[(f.innings == 2).values & f.bat_won.isin([0, 1]).values & ~f.irregular.values & (f.balls_left <= 12).values & (f.need > 0).values]
egb = pd.cut(eg.wp_now, [0, .1, .3, .5, .7, .9, 1.0])
EG = eg.groupby(egb).agg(predicted=("wp_now", "mean"), actual=("bat_won", "mean"), balls=("wp_now", "size")).reset_index()
EG["wp_bin"] = EG.wp_now.astype(str); EG = EG.drop(columns="wp_now")
save_table(EG, "wp_endgame_calibration")
eg_ece = float((EG.balls * (EG.predicted - EG.actual).abs()).sum() / EG.balls.sum())
log.info(f"End-game (last 2 overs of chases) calibration error {eg_ece:.3f}:" + chr(10) + EG.round(3).to_string())
# record: the same end-game states scored with the Plan C (backward calculation) blend switched on
egs = f[(f.innings == 2).values & f.bat_won.isin([0, 1]).values & ~f.irregular.values & (f.balls_left <= 12).values & (f.need > 0).values]
S_eg = egs[["need", "balls_left", "wih", "target", "maxb", "lvl_code", "year", "venue_par", "elo_diff"]].copy()
S_eg["rrr"] = S_eg.need * 6 / S_eg.balls_left
rows_v = []
for flag in (False, True):
    USE_DP = flag
    pw = wp2_eval(S_eg, egs.fold.values)
    bb = pd.cut(pw, [0, .1, .3, .5, .7, .9, 1.0], include_lowest=True)
    g_ = pd.DataFrame({"p": pw, "y": egs.bat_won.values, "b": bb}).groupby("b").agg(p=("p", "mean"), y=("y", "mean"), n=("p", "size"))
    rows_v.append({"variant": "Plan C blend (backward calculation)" if flag else "CHOSEN: main model + runs-needed-vs-available input",
                   "endgame_calibration_error": float((g_.n * (g_.p - g_.y).abs()).sum() / g_.n.sum())})
USE_DP = False
save_table(pd.DataFrame(rows_v), "wp_endgame_variants")
log.info(f"End-game variants: {rows_v}")
# reality check: actual chase outcomes from common end-game situations
rc = []
cc = f[(f.innings == 2) & f.bat_won.isin([0, 1]) & ~f.irregular & (f.need > 0) & (f.wih >= 4)]
for bl_, lo, hi in ((1, 4, 4), (1, 5, 6), (6, 8, 12), (6, 13, 16), (12, 15, 20), (12, 21, 26), (18, 25, 32)):
    x = cc[(cc.balls_left == bl_) & cc.need.between(lo, hi)]
    S_ = x[["need", "balls_left", "wih", "target", "maxb", "lvl_code", "year", "venue_par", "elo_diff"]].copy(); S_["rrr"] = S_.need * 6 / S_.balls_left
    rc.append({"situation": f"{lo}-{hi} needed off {bl_} ball(s), 4+ wickets", "times_seen": len(x), "actual_win_rate": x.bat_won.mean(),
               "model_win_prob": float(wp2_eval(S_, x.fold.values).mean()) if len(x) else np.nan})
save_table(pd.DataFrame(rc), "endgame_reality_check")
log.info("End-game reality check:" + chr(10) + pd.DataFrame(rc).round(2).to_string())
for need_, bl_ in ((4, 1), (6, 1), (10, 6), (20, 12)):
    S_ = pd.DataFrame({"need": [need_], "balls_left": [bl_], "wih": [5], "target": [170], "maxb": [120], "lvl_code": [0],
                       "year": [2024], "venue_par": [170.0], "elo_diff": [0.0]})
    S_["rrr"] = S_.need * 6 / S_.balls_left
    log.info(f"  WP check: {need_} needed off {bl_} balls, 5 wickets in hand -> {np.mean([wp2_eval(S_, np.array([k]))[0] for k in range(K)]):.3f}")
grid = pd.DataFrame([(n, b_, w) for n in range(1, 201, 3) for b_ in range(1, 121, 3) for w in range(1, 11)],
                    columns=["need", "balls_left", "wih"])
grid = grid.assign(target=160, maxb=120, lvl_code=0, year=2024, venue_par=170.0, elo_diff=0.0)
grid["rrr"] = grid.need * 6 / grid.balls_left
viol = {}
for k in range(K):
    g_ = grid.copy(); g_["wp"] = wp2_eval(g_, np.full(len(g_), k))
    for col, sgn in (("need", -1), ("balls_left", 1), ("wih", 1)):
        others = [c_ for c_ in ["need", "balls_left", "wih"] if c_ != col]
        dd = g_.sort_values(others + [col]).groupby(others).wp.diff()
        viol.setdefault(col, []).append(float((sgn * dd < -0.005).mean()))
viol = {k_: float(np.mean(v_)) for k_, v_ in viol.items()}
log.info(f"Chase WP monotonicity violations (>0.5pp, share of adjacent grid steps): {viol}")
save_table(pd.DataFrame([{"model": "M3 LightGBM (calibrated)", **g3.loc["lgb_cal"].to_dict()},
                         {"model": "M3 spline logistic", **g3.loc["logit"].to_dict()},
                         {"model": "Plan B plug-in (M4 -> M2)", "logloss": plug_ll, "auc": plug_auc,
                          "ece": ece(d3e.bat_won.values, d3e.wp_now.values)}]), "first_innings_wp_planB")
save_json({"endgame_ece": eg_ece, "m2_planA_pass": m2_pass, "m3_auc_gate_pass": m3_pass, "m2_choice": M2_CHOICE, "m3_choice": M3_CHOICE,
           "first_inns_plugin_auc": plug_auc, "monotonicity_violations": viol, "wpa_drift": float(drift.mean()),
           "rv_drift": float(np.nanmean(tr_))}, "p03_summary")
log.info("P3 done")
