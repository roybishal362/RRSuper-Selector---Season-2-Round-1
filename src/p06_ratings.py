"""P6 - Player rating models (R-1 ridge APM, R-2 empirical Bayes x3, R-3 composite) fitted on all data.
Also: the clutch-spread test (is there real between-player variance in the pressure-vs-usual gap?),
model agreement (Kendall tau), and a grouped-CV check of the ridge penalties."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from common import CFG, load, save_table, save_json, load_json, get_logger, Timer
import ratings_lib as RL

log = get_logger("p06_ratings")
v = load("values")
players = load("players").set_index("player_id")
comp = load_json("p05_components")

summary = {}
all_tabs = {}
for side in ("bat", "bowl"):
    tabs = {}
    with Timer(log, f"{side} ratings"):
        for met in ("v1", "v2", "v3", "v4"):
            d = RL.side_frame(v, side, met)
            e, k, tau2 = RL.eb_shrink(d)
            dl, tau_d, mu_d = RL.delta_shrink(d)
            ep, kp, tau2p = RL.eb_shrink(d[d.pres])
            summary[f"{side}_{met}"] = {"k_all": k, "tau_all": np.sqrt(tau2), "k_pressure": kp, "tau_pressure": np.sqrt(tau2p),
                                        "tau_clutch": np.sqrt(tau_d), "league_gap": mu_d,
                                        "sigma_ball": float(d.val.std()),
                                        "clutch_share_of_pressure_var": tau_d / max(tau2p, 1e-12)}
        for model in ("R2-p_only", "R2-joint", "R2-weighted"):
            tabs[model] = RL.r2_ratings(v, side, "v1", model.split("-")[1])
        tabs["R1"] = RL.r1_ratings(v, side, "v1")
        tabs["R3"] = RL.r3_ratings(v, side, comp)
    # agreement among players with >= 60 pressure balls
    base = tabs["R2-joint"]
    pool = base.index[base.n_p >= 60]
    names = list(tabs)
    M = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b_ in names:
            x, y = tabs[a].rating.reindex(pool), tabs[b_].rating.reindex(pool)
            okk = x.notna() & y.notna()
            M.loc[a, b_] = kendalltau(x[okk], y[okk])[0]
    save_table(M.reset_index().rename(columns={"index": "model"}), f"rating_agreement_{side}")
    log.info(f"{side} model agreement (Kendall tau, players with >=60 pressure balls, n={len(pool)}):\n{M.round(2).to_string()}")
    # combined table
    T = pd.DataFrame(index=base.index)
    T["name"] = players.name.reindex(T.index)
    T["n_pressure"] = base.n_p; T["n_other"] = base.n_u
    for nm, t in tabs.items():
        T[nm] = t.rating.reindex(T.index)
    T["R1_usual"] = tabs["R1"].usual.reindex(T.index); T["R1_clutch"] = tabs["R1"].clutch.reindex(T.index)
    T["gap_raw"] = base.gap_raw; T["gap_shrunk"] = base.gap_shrunk; T["gap_reliability"] = base.gap_rel
    T = T.sort_values("R2-joint", ascending=False)
    save_table(T.reset_index().rename(columns={"index": "player_id", "pid": "player_id"}), f"ratings_{side}_all_models")
    all_tabs[side] = T
    log.info(f"{side}: top 12 by R2-joint\n" + T[T.n_pressure >= 30].head(12)[["name", "n_pressure", "R2-joint", "R1", "R3", "R2-p_only", "gap_shrunk"]].round(3).to_string())

S = pd.DataFrame(summary).T
save_table(S.reset_index().rename(columns={"index": "side_metric"}), "clutch_spread")
log.info("Variance decomposition (per ball, runs or WP units):\n" + S.round(4).to_string())

# ridge penalty check: grouped 5-fold CV of held-out ball value, multipliers of the Bayesian penalty
folds = load("folds").set_index("match_id").fold
v["fold"] = v.match_id.map(folds)
rows = []
with Timer(log, "R1 penalty CV"):
    for km in (0.25, 0.5, 1.0, 2.0, 4.0):
        se, se0 = 0.0, 0.0
        for k in range(5):
            tr, te = v[v.fold != k], v[v.fold == k]
            fit = RL.r1_fit(tr, "v1", km)
            yb = -te.v1_bowl.values
            yb = yb - (-tr.v1_bowl.values).mean()
            pb = (fit["bat"].usual.reindex(te.striker_id).fillna(0).values * te.faced.values
                  + fit["bat"].clutch.reindex(te.striker_id).fillna(0).values * te.faced.values * te.is_pressure.values
                  - fit["bowl"].usual.reindex(te.bowler_id).fillna(0).values
                  - fit["bowl"].clutch.reindex(te.bowler_id).fillna(0).values * te.is_pressure.values)
            se += ((yb - pb) ** 2).sum(); se0 += (yb ** 2).sum()
        rows.append({"penalty_multiplier": km, "heldout_mse_reduction_pct": 100 * (1 - se / se0)})
        log.info(f"R1 penalty x{km}: held-out MSE reduction {100 * (1 - se / se0):.4f}%")
save_table(pd.DataFrame(rows), "r1_penalty_cv")
save_json(summary, "p06_summary")
log.info("P6 done")
