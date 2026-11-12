"""P3b - Is the win-probability model overfitting? A model zoo under the SAME match-grouped folds as P3.

Every model is scored on whole matches it never saw. Each model's complexity setting is chosen on 3 inner match-grouped
folds of the training part only; boosted models also early-stop on a 15% match-level slice of their own training data.
For each model we report:
  - train vs test log-loss (the overfitting gap: overfitting = good on train, worse on unseen matches)
  - test Brier, AUC, ECE, calibration slope (slope < 1 = predictions too extreme = the classic overfitting sign)
  - end-game ECE (chase, last two overs)
Plus: stacking (meta = logistic, meta = Extra Trees) on inner out-of-fold predictions, a learning curve, and the
'match fingerprint' check (match-constant inputs identify a match, so a flexible model can memorise its result)."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from common import CFG, SEED, NJ, load, save_table, save_json, get_logger, Timer

log = get_logger("p03b_model_zoo")
K = CFG["cv"]["n_folds"]
EPS = 1e-4

f = load("features")
c = load("ctx_preds", columns=["ball_uid", "fold", "wp_now"])
assert (f.ball_uid.values == c.ball_uid.values).all()
f["fold"] = c.fold.values
f["wp_cur"] = c.wp_now.values

# same chase-difficulty input as P3
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
MATCH_CONST = {"chase": ["target", "maxb", "lvl_code", "year", "venue_par", "elo_diff"],
               "first": ["maxb", "lvl_code", "year", "venue_par", "elo_diff"]}

d2 = add_chase_feats(f[(f.innings == 2) & f.bat_won.isin([0, 1]) & ~f.irregular & (f.need > 0)].copy())
d3 = f[(f.innings == 1) & f.bat_won.isin([0, 1]) & ~f.irregular].copy()
import os
if os.environ.get("ZOO_SMOKE"):   # quick bug check on a small sample of matches
    keep = np.random.RandomState(0).choice(f.match_id.unique(), 120, replace=False)
    d2, d3 = d2[d2.match_id.isin(keep)].copy(), d3[d3.match_id.isin(keep)].copy()
    save_table = lambda *a, **k: None
for d in (d2, d3):
    d["y"] = d.bat_won.astype(int)
DATA = {"chase": (d2, M2F, M2MONO), "first": (d3, M3F, M3MONO)}


# ----------------------------------------------------------------------------- helpers
def clipp(p):
    return np.clip(p, EPS, 1 - EPS)


def logit(p):
    p = clipp(p)
    return np.log(p / (1 - p))


def ece(y, p, bins=20):
    q = np.clip((p * bins).astype(int), 0, bins - 1)
    df = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((df.n * (df.y - df.p).abs()).sum() / df.n.sum())


def cal_slope(y, p):
    """Slope of a logistic fit of the outcome on the model's log-odds. 1 = right; < 1 = too extreme (overfit)."""
    return float(LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y).coef_[0, 0])


def es_split(groups, seed=SEED, frac=0.15):
    g = np.unique(groups)
    val = np.isin(groups, np.random.RandomState(seed).permutation(g)[: int(len(g) * frac)])
    return ~val, val


def grouped_folds(groups, k, seed):
    g = np.unique(groups)
    lab = pd.Series(np.arange(len(g)) % k, index=np.random.RandomState(seed).permutation(g))
    return lab.reindex(groups).values


# ----------------------------------------------------------------------------- model wrappers
# Each: fit(X, y, groups) -> self ; predict(X) -> P(batting side wins)
class Sk:
    def __init__(self, make, fill=0.0):
        self.make, self.fill = make, fill

    def fit(self, X, y, g):
        self.m = self.make().fit(X.fillna(self.fill).values, y)
        return self

    def predict(self, X):
        return self.m.predict_proba(X.fillna(self.fill).values)[:, 1]


class LGBW:
    def __init__(self, num_leaves, mono):
        self.p = dict(objective="binary", learning_rate=0.05, num_leaves=num_leaves, min_child_samples=200, subsample=0.8,
                      subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, n_jobs=NJ, seed=SEED,
                      monotone_constraints=mono, monotone_constraints_method="intermediate")

    def fit(self, X, y, g):
        a, v = es_split(g)
        m = lgb.train(self.p, lgb.Dataset(X[a], y[a]), 3000, valid_sets=[lgb.Dataset(X[v], y[v])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        self.iters = max(20, int(m.best_iteration * 1.1))
        self.m = lgb.train(self.p, lgb.Dataset(X, y), self.iters)
        return self

    def predict(self, X):
        return self.m.predict(X)


class XGBW:
    def __init__(self, depth, mono):
        self.kw = dict(learning_rate=0.05, max_depth=depth, min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
                       reg_lambda=5.0, monotone_constraints=tuple(mono), n_jobs=NJ, random_state=SEED, tree_method="hist",
                       eval_metric="logloss")

    def fit(self, X, y, g):
        a, v = es_split(g)
        m = xgb.XGBClassifier(n_estimators=3000, early_stopping_rounds=50, **self.kw)
        m.fit(X[a].fillna(-1), y[a], eval_set=[(X[v].fillna(-1), y[v])], verbose=False)
        self.iters = max(20, int((m.best_iteration + 1) * 1.1))
        self.m = xgb.XGBClassifier(n_estimators=self.iters, **self.kw).fit(X.fillna(-1), y)
        return self

    def predict(self, X):
        return self.m.predict_proba(X.fillna(-1))[:, 1]


class CatW:
    def __init__(self, depth):
        self.kw = dict(learning_rate=0.08, depth=depth, l2_leaf_reg=5.0, loss_function="Logloss", random_seed=SEED,
                       thread_count=NJ, verbose=False, allow_writing_files=False)

    def fit(self, X, y, g):
        a, v = es_split(g)
        m = CatBoostClassifier(iterations=2000, od_type="Iter", od_wait=50, **self.kw)
        m.fit(X[a], y[a], eval_set=(X[v], y[v]), use_best_model=True)
        self.iters = max(20, int(m.get_best_iteration() * 1.1))
        self.m = CatBoostClassifier(iterations=self.iters, **self.kw).fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict_proba(X)[:, 1]


def spline_logit():
    return make_pipeline(StandardScaler(), SplineTransformer(n_knots=6, degree=3, knots="quantile", extrapolation="linear"),
                         LogisticRegression(max_iter=1000, C=1.0))


def zoo(mono):
    """name -> list of (setting label, factory)"""
    return {
        "Logistic (straight lines)": [("C=1", lambda: Sk(lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))))],
        "Spline logistic, curves on every input (previous)": [("6 knots", lambda: Sk(spline_logit))],
        "KNN": [(f"k={k}", (lambda k=k: Sk(lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k, n_jobs=NJ)))))
                for k in (100, 400, 1600)],
        "Decision tree": [(f"depth={dp}", (lambda dp=dp: Sk(lambda: DecisionTreeClassifier(max_depth=dp, min_samples_leaf=100, random_state=SEED), -1)))
                          for dp in (4, 6, 8)],
        "Random forest": [(f"leaf={lf}", (lambda lf=lf: Sk(lambda: RandomForestClassifier(150, min_samples_leaf=lf, max_features=0.5, n_jobs=NJ, random_state=SEED), -1)))
                          for lf in (50, 300)],
        "Extra trees": [(f"leaf={lf}", (lambda lf=lf: Sk(lambda: ExtraTreesClassifier(150, min_samples_leaf=lf, max_features=0.7, n_jobs=NJ, random_state=SEED), -1)))
                        for lf in (50, 300)],
        "XGBoost (monotone)": [(f"depth={dp}", (lambda dp=dp: XGBW(dp, mono))) for dp in (3, 6)],
        "LightGBM (monotone)": [(f"leaves={nl}", (lambda nl=nl: LGBW(nl, mono))) for nl in (7, 31)],
        "CatBoost": [(f"depth={dp}", (lambda dp=dp: CatW(dp))) for dp in (4, 6)],
    }


STACK_BASE = ["Spline logistic, curves on every input (previous)", "LightGBM (monotone)", "CatBoost", "XGBoost (monotone)", "Random forest",
              "Extra trees", "KNN"]


def metrics(y, p, extra_mask=None):
    p = clipp(p)
    r = {"test_logloss": log_loss(y, p), "test_brier": brier_score_loss(y, p), "test_auc": roc_auc_score(y, p),
         "test_ece": ece(y, p), "calibration_slope": cal_slope(y, p)}
    if extra_mask is not None and extra_mask.sum() > 200:
        r["endgame_ece"] = ece(y[extra_mask], p[extra_mask])
    return r


# ----------------------------------------------------------------------------- main comparison
def run(target):
    d, feats, mono = DATA[target]
    y, groups, folds = d.y.values, d.match_id.values, d.fold.values
    X = d[feats].reset_index(drop=True)
    Z = zoo(mono)
    oof = {nm: np.zeros(len(d)) for nm in list(Z) + ["Stacking (meta: logistic)", "Stacking (meta: extra trees)",
                                                     "Average of spline + LightGBM + CatBoost"]}
    train_ll = {nm: [] for nm in oof}
    chosen = {nm: [] for nm in Z}
    for k in range(K):
        tr, te = folds != k, folds == k
        Xtr, ytr, gtr = X[tr].reset_index(drop=True), y[tr], groups[tr]
        inner = grouped_folds(gtr, 3, SEED + k)
        inner_oof = {}
        with Timer(log, f"{target} outer fold {k}"):
            for nm, grid in Z.items():
                # choose the complexity setting on inner folds; keep its inner out-of-fold predictions for stacking
                best = None
                for lab, make in grid:
                    po = np.zeros(len(Xtr))
                    for j in range(3):
                        a, v = inner != j, inner == j
                        po[v] = make().fit(Xtr[a].reset_index(drop=True), ytr[a], gtr[a]).predict(Xtr[v])
                    ll = log_loss(ytr, clipp(po))
                    if best is None or ll < best[0]:
                        best = (ll, lab, make, po)
                chosen[nm].append(best[1])
                inner_oof[nm] = best[3]
                m = best[2]().fit(Xtr, ytr, gtr)
                oof[nm][te] = m.predict(X[te])
                ptr = m.predict(Xtr)
                train_ll[nm].append(log_loss(ytr, clipp(ptr)))
                inner_oof[nm + "__train"] = ptr
                inner_oof[nm + "__test"] = oof[nm][te]
                log.info(f"  {target} fold {k} {nm:28s} {best[1]:10s} inner {best[0]:.4f} train {train_ll[nm][-1]:.4f} "
                         f"test {log_loss(y[te], clipp(oof[nm][te])):.4f}")
            # stacking: meta-learners trained on inner out-of-fold base predictions
            Mtr = np.column_stack([logit(inner_oof[b]) for b in STACK_BASE])
            Mte = np.column_stack([logit(inner_oof[b + "__test"]) for b in STACK_BASE])
            for nm, meta in (("Stacking (meta: logistic)", LogisticRegression(C=1.0, max_iter=1000)),
                             ("Stacking (meta: extra trees)", ExtraTreesClassifier(300, min_samples_leaf=300, max_features=1.0,
                                                                                   n_jobs=NJ, random_state=SEED))):
                meta.fit(Mtr, ytr)
                oof[nm][te] = meta.predict_proba(Mte)[:, 1]
                train_ll[nm].append(log_loss(ytr, clipp(meta.predict_proba(Mtr)[:, 1])))   # the data the meta-model was fitted on
            av = ["Spline logistic, curves on every input (previous)", "LightGBM (monotone)", "CatBoost"]
            oof["Average of spline + LightGBM + CatBoost"][te] = np.mean([inner_oof[b + "__test"] for b in av], axis=0)
            train_ll["Average of spline + LightGBM + CatBoost"].append(
                log_loss(ytr, clipp(np.mean([inner_oof[b + "__train"] for b in av], axis=0))))
    # current pipeline output (and, for the first innings, a cross-fitted recalibration of it)
    extra = {}
    if target == "first":
        cur = d.wp_cur.values
        extra["Plan B plug-in (CURRENT first-innings model)"] = cur
        rec = np.zeros(len(d))
        for k in range(K):
            tr, te = folds != k, folds == k
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(cur[tr]).reshape(-1, 1), y[tr])
            rec[te] = lr.predict_proba(logit(cur[te]).reshape(-1, 1))[:, 1]
        extra["Plan B plug-in + recalibration"] = rec
    eg = (d.balls_left.values <= 12) if target == "chase" else None
    rows = []
    for nm, p in list(oof.items()) + list(extra.items()):
        r = {"target": target, "model": nm, "train_logloss": np.mean(train_ll[nm]) if train_ll.get(nm) else np.nan}
        r.update(metrics(y, p, eg))
        r["overfit_gap"] = r["test_logloss"] - r["train_logloss"]
        r["setting_chosen"] = ", ".join(pd.Series(chosen[nm]).value_counts().index) if nm in chosen else ""
        rows.append(r)
    res = pd.DataFrame(rows).sort_values("test_logloss").reset_index(drop=True)
    return res, oof


# ----------------------------------------------------------------------------- fingerprint check
def fingerprint(target):
    d, feats, mono = DATA[target]
    mc = MATCH_CONST[target]
    n_match = d.match_id.nunique()
    n_combo = len(d[mc].round(6).drop_duplicates())
    ball_per_match = len(d) / n_match
    y, groups, folds = d.y.values, d.match_id.values, d.fold.values
    rows = []
    for lab, make in (("Spline logistic, match-constant inputs only", lambda: Sk(spline_logit)),
                      ("Flexible LightGBM, match-constant inputs only",
                       lambda: Sk(lambda: lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=20,
                                                             verbose=-1, n_jobs=NJ, random_state=SEED)))):
        tr_ll, te_p = [], np.zeros(len(d))
        for k in range(K):
            tr, te = folds != k, folds == k
            m = make().fit(d.loc[tr, mc], y[tr], groups[tr])
            tr_ll.append(log_loss(y[tr], clipp(m.predict(d.loc[tr, mc]))))
            te_p[te] = m.predict(d.loc[te, mc])
        rows.append({"target": target, "check": lab, "train_logloss": np.mean(tr_ll), "test_logloss": log_loss(y, clipp(te_p)),
                     "matches": n_match, "distinct_match_input_combos": n_combo, "balls_per_match": ball_per_match})
    return rows


# ----------------------------------------------------------------------------- learning curve
def learning_curve(target):
    d, feats, mono = DATA[target]
    y, groups, folds = d.y.values, d.match_id.values, d.fold.values
    X = d[feats].reset_index(drop=True)
    models = {"Spline logistic, curves on every input (previous)": lambda: Sk(spline_logit), "LightGBM (monotone)": lambda: LGBW(31, mono),
              "CatBoost": lambda: CatW(6), "Random forest": lambda: Sk(lambda: RandomForestClassifier(
                  150, min_samples_leaf=50, max_features=0.5, n_jobs=NJ, random_state=SEED), -1)}
    rows = []
    for frac in (0.1, 0.25, 0.5, 1.0):
        for nm, make in models.items():
            trl, tel, nmat = [], [], []
            for k in range(K):
                tr, te = folds != k, folds == k
                gm = np.unique(groups[tr])
                keep = np.isin(groups, np.random.RandomState(SEED + k).permutation(gm)[: max(10, int(len(gm) * frac))]) & tr
                m = make().fit(X[keep].reset_index(drop=True), y[keep], groups[keep])
                trl.append(log_loss(y[keep], clipp(m.predict(X[keep]))))
                tel.append(log_loss(y[te], clipp(m.predict(X[te]))))
                nmat.append(int(len(np.unique(groups[keep]))))
            rows.append({"target": target, "model": nm, "train_matches": int(np.mean(nmat)), "share_of_training": frac,
                         "train_logloss": np.mean(trl), "test_logloss": np.mean(tel)})
            log.info(f"  LC {target} {frac:.2f} {nm:28s} train {np.mean(trl):.4f} test {np.mean(tel):.4f}")
    return rows


# ----------------------------------------------------------------------------- logistic designs (chase)
def logistic_designs():
    """Where does the logistic model's flexibility belong? Curves on every input vs curves for the game state only."""
    from sklearn.compose import ColumnTransformer
    d, feats, _ = DATA["chase"]
    y, folds = d.y.values, d.fold.values
    X = d[feats].fillna(0).reset_index(drop=True)
    STATE = ["need", "balls_left", "wih", "rrr", "z_chase"]
    CTX = [c for c in feats if c not in STATE]

    def hybrid(knots=4, C=1.0, ctx=True):
        parts = [("s", make_pipeline(StandardScaler(), SplineTransformer(n_knots=knots, degree=3, knots="quantile",
                                                                          extrapolation="linear")), STATE)]
        if ctx:
            parts.append(("c", StandardScaler(), CTX))
        return make_pipeline(ColumnTransformer(parts), LogisticRegression(max_iter=2000, C=C))

    V = {"Plain logistic (straight lines)": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
         "Previous: curves on every input (6 knots)": spline_logit,
         "CHOSEN: curves for game state, straight lines for match context (4 knots)": lambda: hybrid(4),
         "Curves for game state (6 knots), straight lines for match context": lambda: hybrid(6),
         "  same, stronger penalty (C=0.1)": lambda: hybrid(6, 0.1),
         "Curves for game state only (no match context)": lambda: hybrid(6, ctx=False)}
    eg = d.balls_left.values <= 12
    cc = d.wih.values >= 4
    SIT = ((1, 4, 4), (1, 5, 6), (6, 8, 12), (6, 13, 16), (12, 15, 20), (12, 21, 26), (18, 25, 32))
    rows, sit = [], []
    for nm, mk in V.items():
        oof = np.zeros(len(d)); trl = []
        for k in range(K):
            tr, te = folds != k, folds == k
            m = mk().fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
            trl.append(log_loss(y[tr], clipp(m.predict_proba(X[tr])[:, 1])))
        r = {"model": nm, "train_logloss": np.mean(trl)}
        r.update(metrics(y, oof))
        r["overfit_gap"] = r["test_logloss"] - r["train_logloss"]
        b = pd.cut(oof[eg], [0, .1, .3, .5, .7, .9, 1.0], include_lowest=True)
        g = pd.DataFrame({"p": oof[eg], "y": y[eg], "b": b}).groupby("b", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("p", "size"))
        r["endgame_calibration_error"] = float((g.n * (g.p - g.y).abs()).sum() / g.n.sum())
        num = den = 0.0
        for bl_, lo, hi in SIT:
            m_ = cc & (d.balls_left.values == bl_) & (d.need.values >= lo) & (d.need.values <= hi)
            num += abs(oof[m_].mean() - y[m_].mean()) * m_.sum(); den += m_.sum()
            sit.append({"model": nm, "situation": f"{lo}-{hi} needed off {bl_} ball(s), 4+ wickets", "times_seen": int(m_.sum()),
                        "actual_win_rate": y[m_].mean(), "model_win_prob": oof[m_].mean()})
        r["reality_check_error"] = num / den
        rows.append(r)
    R = pd.DataFrame(rows)
    save_table(R, "wp_logistic_designs")
    save_table(pd.DataFrame(sit), "wp_logistic_designs_situations")
    log.info("logistic designs:\n" + R.round(4).to_string())


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "designs":
        logistic_designs()
        sys.exit(0)
    logistic_designs()
    fp = []
    for t in ("chase", "first"):
        fp += fingerprint(t)
    fp = pd.DataFrame(fp)
    save_table(fp, "model_zoo_fingerprint")
    log.info("fingerprint:\n" + fp.to_string())
    allres = []
    for t in ("chase", "first"):
        with Timer(log, f"zoo {t}"):
            res, _ = run(t)
        allres.append(res)
        save_table(pd.concat(allres), "model_zoo")
        log.info(f"zoo {t}:\n" + res.to_string())
    lc = []
    for t in ("chase", "first"):
        with Timer(log, f"learning curve {t}"):
            lc += learning_curve(t)
    save_table(pd.DataFrame(lc), "model_zoo_learning_curve")
    log.info("P3b done")
