"""P7 - Unsupervised learning (fairness and insight only; never used to pick players).
U-1 roles by usage (semi-supervised: cricket-rule seeds -> multinomial logistic; checked vs KMeans / GMM / Ward / HDBSCAN / UMAP+HDBSCAN)
U-2 pressure regimes (GMM from P4) summarised
U-3 anomalies: Isolation Forest on player-match lines + 'one-match wonder' concentration of pressure value
U-4 latent skill factors: PCA of player profiles (2-D skill map)"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.ensemble import IsolationForest
from common import SEED, load, save, save_table, save_json, get_logger

log = get_logger("p07_unsupervised")
f = load("features", columns=["ball_uid", "match_id", "striker_id", "bowler_id", "batting_order", "phase", "over_no",
                              "is_pace", "innings", "runs", "batter_faced", "striker_out", "boundary", "is_six",
                              "team_runs", "bowler_runs", "legal", "bowler_wicket", "is_wide", "is_noball", "y_class",
                              "str_balls", "legal_before"])
v = load("values", columns=["ball_uid", "v1_bat", "v1_bowl", "is_pressure"])
p = load("pressure", columns=["ball_uid", "regime", "LI", "pressure"])
f = f.merge(v, on="ball_uid").merge(p, on="ball_uid")
players = load("players").set_index("player_id")
MINB = 120

# ---------------------------------------------------------------- profiles
fb = f[f.batter_faced]
gb = fb.groupby("striker_id")
entry = fb[fb.str_balls == 0].groupby("striker_id").legal_before.mean() / 6 + 1
bat = pd.DataFrame({
    "balls": gb.size(),
    "sr": 100 * gb.runs.sum() / gb.size(),
    "bdry_pct": gb.boundary.mean(), "six_share": gb.is_six.sum() / gb.boundary.sum().clip(lower=1),
    "dot_pct": (fb.y_class == 0).groupby(fb.striker_id).mean(),
    "balls_per_out": gb.size() / gb.striker_out.sum().clip(lower=1),
    "pp_share": (fb.phase == 0).groupby(fb.striker_id).mean(), "death_share": (fb.phase == 2).groupby(fb.striker_id).mean(),
    "position": gb.batting_order.median(), "entry_over": entry, "vs_spin_share": (1 - fb.is_pace).groupby(fb.striker_id).mean(),
})
bat = bat[bat.balls >= MINB].dropna()
gw = f.groupby("bowler_id")
bowl = pd.DataFrame({
    "balls": gw.legal.sum(),
    "econ": 6 * gw.bowler_runs.sum() / gw.legal.sum().clip(lower=1),
    "dot_pct": (f.y_class == 0).groupby(f.bowler_id).mean(), "bdry_pct": f.boundary.groupby(f.bowler_id).mean(),
    "wkt_rate": gw.bowler_wicket.mean(), "extras_pct": (f.is_wide | f.is_noball).groupby(f.bowler_id).mean(),
    "pp_share": (f.phase == 0).groupby(f.bowler_id).mean(), "death_share": (f.phase == 2).groupby(f.bowler_id).mean(),
    "pace": gw.is_pace.mean(), "avg_over": gw.over_no.mean(),
})
bowl = bowl[bowl.balls >= MINB].dropna()


def compare_clusterings(X, name):
    Z = StandardScaler().fit_transform(X)
    Zp = PCA(0.9, random_state=SEED).fit_transform(Z)
    rows = []
    rng = np.random.RandomState(SEED)
    for k in range(3, 9):
        for meth in ("KMeans", "GMM", "Ward"):
            if meth == "KMeans":
                lab = KMeans(k, n_init=20, random_state=SEED).fit_predict(Zp); bic = np.nan
            elif meth == "GMM":
                gm = GaussianMixture(k, random_state=SEED, n_init=5).fit(Zp); lab = gm.predict(Zp); bic = gm.bic(Zp)
            else:
                lab = AgglomerativeClustering(k, linkage="ward").fit_predict(Zp); bic = np.nan
            aris = []
            for _ in range(20):
                ix = rng.choice(len(Zp), len(Zp), replace=True)
                if meth == "KMeans":
                    l2 = KMeans(k, n_init=5, random_state=rng.randint(1e6)).fit(Zp[ix]).predict(Zp)
                elif meth == "GMM":
                    l2 = GaussianMixture(k, random_state=rng.randint(1e6)).fit(Zp[ix]).predict(Zp)
                else:
                    from sklearn.neighbors import KNeighborsClassifier
                    l_ix = AgglomerativeClustering(k, linkage="ward").fit_predict(Zp[ix])
                    l2 = KNeighborsClassifier(1).fit(Zp[ix], l_ix).predict(Zp)
                aris.append(adjusted_rand_score(lab, l2))
            rows.append({"set": name, "method": meth, "k": k, "silhouette": silhouette_score(Zp, lab),
                         "davies_bouldin": davies_bouldin_score(Zp, lab), "bic": bic, "stability_ari": np.mean(aris)})
    hd = HDBSCAN(min_cluster_size=15).fit_predict(Zp)
    nk = len(set(hd)) - (1 if -1 in hd else 0)
    rows.append({"set": name, "method": "HDBSCAN", "k": nk, "silhouette": silhouette_score(Zp[hd >= 0], hd[hd >= 0]) if nk > 1 else np.nan,
                 "davies_bouldin": np.nan, "bic": np.nan, "stability_ari": np.nan, "noise_share": float((hd < 0).mean())})
    R = pd.DataFrame(rows)
    # pre-registered choice: among solutions with bootstrap ARI >= 0.6, best silhouette (ties -> fewer clusters)
    cand = R[(R.method != "HDBSCAN") & (R.stability_ari >= 0.6)]
    if cand.empty:
        cand = R[R.method != "HDBSCAN"]
    best = cand.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]
    if best.method == "KMeans":
        lab = KMeans(int(best.k), n_init=20, random_state=SEED).fit_predict(Zp)
    elif best.method == "GMM":
        lab = GaussianMixture(int(best.k), random_state=SEED, n_init=5).fit(Zp).predict(Zp)
    else:
        lab = AgglomerativeClustering(int(best.k), linkage="ward").fit_predict(Zp)
    return R, best, lab


# ---------------------------------------------------------------- U-1 roles by USAGE (semi-supervised)
# A role is a player's JOB (when and where he bats or bowls), not how well he does it. So roles use usage inputs only:
# batting position mix, entry over and phase shares (bowlers: phase shares, pace and spin handled separately).
# Version 1 clustered on performance too (strike rate, economy...), which put finishers such as Tim David and Klaasen
# in "Anchor / middle order" and made the by-role fairness check partly circular (roles encoding quality).
# Method: (1) cricket rules label the clear-cut players ("seeds"); (2) a multinomial logistic model trained on the
# seeds assigns everyone else and gives every player a confidence; (3) checked against unsupervised views
# (k-means / GMM / Ward / HDBSCAN, and UMAP + HDBSCAN) and a bootstrap stability test.
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold

fbi = fb[fb.innings <= 2]
pos_inn = fbi.groupby(["striker_id", "match_id"]).batting_order.first()
ub = pd.crosstab(pos_inn.index.get_level_values(0), pd.cut(pos_inn.values, [0, 2, 4, 6, 7, 11],
                 labels=["pos_1_2", "pos_3_4", "pos_5_6", "pos_7", "pos_8_11"]), normalize="index")
ub.columns = ub.columns.astype(str); ub.index.name = None; ub.columns.name = None
ub["entry_over"] = fbi[fbi.str_balls == 0].groupby("striker_id").legal_before.median() / 6 + 1
for k_, ph in (("pp", 0), ("mid", 1), ("death", 2)):
    ub[f"{k_}_share"] = (fbi.phase == ph).groupby(fbi.striker_id).mean()
ub = ub.reindex(bat.index).dropna()
fi = f[f.innings <= 2]
ubw = pd.DataFrame({"pace": fi.groupby("bowler_id").is_pace.mean()})
for k_, ph in (("pp", 0), ("mid", 1), ("death", 2)):
    ubw[f"{k_}_share"] = (fi.phase == ph).groupby(fi.bowler_id).mean()
ubw["avg_over"] = fi.groupby("bowler_id").over_no.mean()
ubw = ubw.reindex(bowl.index).dropna()


def seed_bat(u):
    if u.pos_8_11 >= 0.5: return "Lower order"
    if u.pos_1_2 >= 0.7: return "Opener"
    if u.pos_3_4 >= 0.6 and u.entry_over <= 9: return "Top order (No. 3-4)"
    if u.entry_over >= 13.5 and u.death_share >= 0.45: return "Finisher"
    if u.pos_5_6 >= 0.5 and 9 < u.entry_over < 12.5 and u.death_share < 0.4: return "Middle order (No. 5-6)"
    return None


def seed_bowl(u):
    if u.pace >= 0.5:
        if u.pp_share >= 0.5 and u.death_share < 0.3: return "Pace - new ball"
        if u.death_share >= 0.38 and u.pp_share < 0.3: return "Pace - death"
        if u.pp_share >= 0.4 and u.death_share >= 0.3: return "Pace - new ball + death"
        if u.pp_share < 0.35 and u.death_share < 0.3: return "Pace - middle overs"
    else:
        if u.pp_share >= 0.3: return "Spin - powerplay"
        if u.pp_share < 0.2 and u.mid_share >= 0.65: return "Spin - middle overs"
    return None


def semi_supervised(U, seeds, feats):
    """seeds -> multinomial logistic on usage inputs -> role + confidence for everyone; CV agreement on the seeds."""
    s = seeds.dropna()
    keep = s.value_counts()[lambda c: c >= 3].index
    s = s[s.isin(keep)]
    if s.nunique() < 2:   # one clear role only: everyone gets it
        lab1 = s.iloc[0] if len(s) else "Unassigned"
        return (pd.Series(lab1, index=U.index), pd.Series(1.0, index=U.index), None,
                {"seeds": int(len(s)), "players": int(len(U)), "cv_agreement_on_seeds": np.nan, "bootstrap_ari": np.nan, "classes": [lab1]})
    sc = StandardScaler().fit(U[feats])
    mdl = LogisticRegression(C=1.0, max_iter=2000)
    nfold = int(min(5, s.value_counts().min()))
    cv_pred = cross_val_predict(mdl, sc.transform(U.loc[s.index, feats]), s.values,
                                cv=StratifiedKFold(nfold, shuffle=True, random_state=SEED)) if nfold >= 2 else s.values
    mdl.fit(sc.transform(U.loc[s.index, feats]), s.values)
    pr = pd.DataFrame(mdl.predict_proba(sc.transform(U[feats])), index=U.index, columns=mdl.classes_)
    role = pr.idxmax(axis=1)
    role.loc[s.index] = s                      # the rules are the definition for clear-cut players
    conf = pr.max(axis=1)
    # bootstrap stability: refit on resampled seeds, re-assign the non-seed players
    rng = np.random.RandomState(SEED)
    aris = []
    for _ in range(30):
        ix = rng.choice(len(s), len(s), replace=True)
        sb = s.iloc[ix]
        if sb.nunique() < s.nunique():
            continue
        m2 = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(U.loc[sb.index, feats]), sb.values)
        r2 = pd.Series(m2.predict(sc.transform(U[feats])), index=U.index)
        r2.loc[s.index] = s
        aris.append(adjusted_rand_score(role, r2))
    return role, conf, pr, {"seeds": int(len(s)), "players": int(len(U)), "cv_agreement_on_seeds": float((cv_pred == s.values).mean()),
                            "bootstrap_ari": float(np.mean(aris)) if aris else np.nan, "classes": list(mdl.classes_)}


BF = ["pos_1_2", "pos_3_4", "pos_5_6", "pos_7", "pos_8_11", "entry_over", "pp_share", "mid_share", "death_share"]
WF = ["pp_share", "mid_share", "death_share", "avg_over"]
roles, conf_all, role_info, detail = {}, {}, [], []
rb, cb, _, ib = semi_supervised(ub, ub.apply(seed_bat, axis=1), BF)
roles["batters"], conf_all["batters"] = rb, cb
role_info.append(dict(set="batters", **ib))
ubw["is_pace"] = ubw.pace >= 0.5
parts_r, parts_c = [], []
for kind, sub in ubw.groupby("is_pace"):           # pace and spin never mix (same rule as the baseline)
    r_, c_, _, i_ = semi_supervised(sub, sub.apply(seed_bowl, axis=1), WF)
    parts_r.append(r_); parts_c.append(c_)
    role_info.append(dict(set="bowlers - " + ("pace" if kind else "spin"), **i_))
roles["bowlers"], conf_all["bowlers"] = pd.concat(parts_r).reindex(ubw.index), pd.concat(parts_c).reindex(ubw.index)

# unsupervised views on the same usage inputs (are the roles natural groups?) + UMAP + HDBSCAN
umap_ok = True
try:
    import umap
except Exception:
    umap_ok = False
for name, U, feats in (("batters", ub, BF), ("bowlers", ubw, WF + ["pace"])):
    R, best, lab = compare_clusterings(U[feats], name)
    Z = StandardScaler().fit_transform(U[feats])
    k_roles = roles[name].nunique()
    extra = {"GMM (k = number of roles)": GaussianMixture(k_roles, random_state=SEED, n_init=5).fit(Z).predict(Z),
             f"best by silhouette ({best.method} k={int(best.k)})": lab}
    if umap_ok:
        emb = umap.UMAP(n_neighbors=15, min_dist=0.0, random_state=SEED).fit_transform(Z)
        hl = HDBSCAN(min_cluster_size=10).fit_predict(emb)
        extra["UMAP + HDBSCAN"] = hl
        R = pd.concat([R, pd.DataFrame([{"set": name, "method": "UMAP+HDBSCAN", "k": len(set(hl)) - (1 if -1 in hl else 0),
                                         "noise_share": float((hl < 0).mean())}])])
    save_table(R, f"clustering_comparison_{name}")
    for nm_, lb in extra.items():
        role_info.append({"set": name, "unsupervised_view": nm_, "ari_with_roles": adjusted_rand_score(roles[name].reindex(U.index).values, lb),
                          "groups": int(len(set(lb)) - (1 if -1 in lb else 0))})
    cent = U[feats].assign(role=roles[name]).groupby("role").mean()
    cent["n"] = roles[name].value_counts()
    cent["mean_confidence"] = conf_all[name].groupby(roles[name]).mean()
    save_table(cent.reset_index(), f"role_centroids_{name}")
    log.info(f"{name} roles (usage, semi-supervised):\n" + cent.round(2).to_string())
    d_ = U[feats].copy()
    d_["role"], d_["confidence"], d_["name"] = roles[name], conf_all[name], players.name.reindex(U.index)
    d_["side"] = "bat" if name == "batters" else "bowl"
    detail.append(d_.reset_index().rename(columns={"index": "player_id", "striker_id": "player_id", "bowler_id": "player_id"}))
save_table(pd.DataFrame(role_info), "role_method_check")
save_table(pd.concat(detail, ignore_index=True), "roles_detail")
log.info("Role method check:\n" + pd.DataFrame(role_info).to_string())

role_tab = pd.concat([roles["batters"].rename("role").to_frame().assign(side="bat"),
                      roles["bowlers"].rename("role").to_frame().assign(side="bowl")]).reset_index()
role_tab.columns = ["player_id", "role", "side"]
save(role_tab, "roles")

# ---------------------------------------------------------------- U-2 regimes
reg = f.groupby("regime").agg(share=("ball_uid", "size"), mean_LI=("LI", "mean"), mean_pressure=("pressure", "mean"),
                              pressure_ball_share=("is_pressure", "mean"), chase_share=("innings", lambda s: (s == 2).mean()),
                              avg_over=("over_no", "mean"))
reg["share"] = reg.share / reg.share.sum()
save_table(reg.reset_index().sort_values("mean_LI"), "pressure_regimes")
log.info("Pressure regimes:\n" + reg.round(3).to_string())

# ---------------------------------------------------------------- U-3 anomalies / one-match wonders
anom = []
for side, pcol, val in (("bat", "striker_id", "v1_bat"), ("bowl", "bowler_id", "v1_bowl")):
    d = f[f[val].notna()].assign(pv=lambda x: x[val] * x.is_pressure)
    pm = d.groupby([pcol, "match_id"]).agg(balls=(val, "size"), value=(val, "sum"), pval=("pv", "sum"),
                                          pballs=("is_pressure", "sum")).reset_index()
    X = pm[["balls", "value", "pval", "pballs"]].values
    iso = IsolationForest(n_estimators=300, contamination=0.01, random_state=SEED).fit(X)
    pm["anomaly"] = iso.predict(X) == -1
    tot = pm.groupby(pcol).agg(pval_total=("pval", "sum"), pval_abs=("pval", lambda s: s.abs().sum()),
                               best_match=("pval", "max"), n_matches=("match_id", "nunique"), pballs=("pballs", "sum"))
    tot["top_match_share"] = tot.best_match / tot.pval_abs.clip(lower=1e-9)
    tot["one_match_wonder"] = (tot.pval_total > 0) & (tot.best_match > tot.pval_total) & (tot.pballs >= 30)
    tot["side"] = side
    anom.append(tot.reset_index().rename(columns={pcol: "player_id"}))
    log.info(f"{side}: anomalous player-match lines {int(pm.anomaly.sum())}; players whose best single pressure match "
             f"exceeds their total pressure value: {int(tot.one_match_wonder.sum())}")
save_table(pd.concat(anom), "one_match_wonders")

# ---------------------------------------------------------------- U-4 latent skill factors
for name, X in (("batters", bat.drop(columns="balls")), ("bowlers", bowl.drop(columns="balls"))):
    Z = StandardScaler().fit_transform(X)
    pca = PCA(random_state=SEED).fit(Z)
    load_ = pd.DataFrame(pca.components_[:3].T, index=X.columns, columns=["PC1", "PC2", "PC3"])
    log.info(f"{name}: explained variance {pca.explained_variance_ratio_[:4].round(3)}\n{load_.round(2).to_string()}")
    save_table(load_.reset_index(), f"pca_loadings_{name}")
    coords = pd.DataFrame(pca.transform(Z)[:, :2], index=X.index, columns=["PC1", "PC2"])
    coords["role"] = roles[name].reindex(coords.index)
    coords["name"] = players.name.reindex(coords.index)
    save_table(coords.reset_index().rename(columns={"index": "player_id"}), f"skill_map_{name}")
log.info("P7 done")
