"""P4 - Pressure: a library of pointers, five pre-registered tests, and a choice between combined definitions.
Pointers use pre-ball information only. Tests:
  T1 real effect  - outcome shift for the average player vs a baseline (stage baseline for scoreboard pointers,
                    full-scoreboard model M1-core for every other pointer); match-clustered bootstrap, BH-FDR
  T2 it matters   - realised |dWP| lift on pointer balls (volatility pointers); big-match pointers are stakes by design
  T3 enough data  - >=2% of balls and >=50 players with >=30 such balls
  T4 separates    - split-half reliability of players' pointer-vs-other run-value gap (informational)
  T5 stable       - same direction IPL vs T20I and in most seasons
Combined definitions C-A..C-G are judged on stakes capture + strain + coverage + stability (never on the ranking)."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from common import CFG, SEED, NJ, load, save, save_table, save_json, load_json, get_logger, Timer, split_half_reliability

log = get_logger("p04_pressure")
PC = CFG["pressure"]
f = load("features")
c = load("ctx_preds")
assert (f.ball_uid.values == c.ball_uid.values).all()
for col in c.columns:
    if col not in f.columns:
        f[col] = c[col].values
K = CFG["cv"]["n_folds"]
f["dwp"] = (f.wp_next - f.wp_now).abs()
f["LI"] = f.LI_raw / f.LI_raw.mean()

# ---------------------------------------------------------------- baselines for T1
# stage baseline: innings x phase x wickets x level, out-of-fold by match
key = f.innings.astype(str) + "_" + f.phase.astype(str) + "_" + f.wkts_before.clip(upper=7).astype(str) + "_" + f.lvl_code.astype(str)
if CFG.get("baseline", {}).get("bowler_type", False):
    key = key + "_" + f.is_pace.astype(str)
stage = np.zeros((len(f), 8))
for k in range(K):
    tr, te = (f.fold != k).values, (f.fold == k).values
    tab = pd.crosstab(key[tr], f.y_class.values[tr]).reindex(columns=range(8), fill_value=0) + 1
    tab = tab.div(tab.sum(1), axis=0)
    stage[te] = tab.reindex(key[te]).fillna(1 / 8).values
yc = f.y_class.values
OUTC = {"wkt": (yc == 6).astype(float), "dot": (yc == 0).astype(float), "bdry": np.isin(yc, [4, 5]).astype(float),
        "extra": (yc == 7).astype(float)}
EXP_STAGE = {"wkt": stage[:, 6], "dot": stage[:, 0], "bdry": stage[:, 4] + stage[:, 5], "extra": stage[:, 7]}
EXP_CORE = {"wkt": f.cp_wkt.values, "dot": f.cp_dot.values, "bdry": (f.cp_4 + f.cp_6).values, "extra": f.cp_x.values}

# run value above expected (V-1 style) for T4
f["rv_bat"] = np.where(f.batter_faced, f.runs + f.re_next - f.E_bat_rv, np.nan)
f["rv_bowl"] = -(f.team_runs + f.re_next - f.E_team_rv)

# ---------------------------------------------------------------- resources (data-estimated DLS-style table)
d1 = f[(f.innings == 1) & (f.maxb == 120) & ~f.irregular & ~f.no_result]
rmod = lgb.LGBMRegressor(objective="poisson", n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=100,
                         monotone_constraints=[1, 1], verbose=-1, random_state=SEED, n_jobs=NJ)
rmod.fit(d1[["balls_left", "wih"]], d1.total - d1.score_before)
full_res = rmod.predict(pd.DataFrame({"balls_left": [120], "wih": [10]}))[0]
f["res_left_pct"] = 100 * rmod.predict(f[["balls_left", "wih"]]) / full_res
f["res_start_pct"] = 100 * rmod.predict(pd.DataFrame({"balls_left": f.maxb, "wih": 10})) / full_res
f["res_used_pct"] = (f.res_start_pct - f.res_left_pct).clip(lower=0) / f.res_start_pct * 100

# ---------------------------------------------------------------- published indices
WW = np.array([1.30, 1.35, 1.40, 1.45, 1.38, 1.18, 0.98, 0.79, 0.59, 0.39, 0.19])
cumw = np.concatenate([[0], np.cumsum(WW)])
ch = f.innings.eq(2).values
irrr = 6 * f.target / f.maxb
crrr = 6 * f.need.clip(lower=0) / f.balls_left
f["bl_pi"] = np.where(ch, (crrr / irrr) * 0.5 * (np.exp(f.res_used_pct / 100) + np.exp(cumw[f.wkts_before.clip(0, 10).astype(int)] / 11)), np.nan)
f["thomson_rho"] = np.where(ch, f.need / (0.566 * f.res_left_pct.clip(lower=1)), np.nan)

# ---------------------------------------------------------------- pointer library
I = lambda x: np.asarray(x, dtype=bool)
li80 = f.LI >= f.LI.quantile(0.8)
ko = f.knock_out.eq(1) | f.stage.isin(["Final", "Qualifier", "Eliminator", "Semi Final", "Super 4"])
P = {}   # id -> (name, family, perspective, stakes_type, flag_bat, flag_bowl)
def add(pid, name, fam, persp, stakes, flag_bat, flag_bowl=None):
    P[pid] = (name, fam, persp, stakes, I(flag_bat), I(flag_bat if flag_bowl is None else flag_bowl))
add("S1", "Leverage index top 20%", "S", "both", "volatility", li80)
add("S2", "Close game: win prob 30-70%", "S", "both", "volatility", f.wp_now.between(0.3, 0.7))
add("S3", "Bhattacharjee-Lemmer PI top 20% of chase balls", "S", "both", "volatility", ch & (f.bl_pi >= np.nanquantile(f.bl_pi, 0.8)))
add("S4", "Thomson clutch zone (rho 2.27-3.33)", "S", "both", "volatility", ch & f.thomson_rho.between(2.27, 3.33))
add("S5", "Required rate >= current + 2 and >= 10", "S", "both", "volatility", ch & (f.rrr_minus_crr >= 2) & (f.rrr >= 10))
add("S6", "Setting: projected total >= 15 below venue par", "S", "both", "volatility",
    (f.innings == 1) & (f.score_before + f.re_now <= f.venue_par - 15) & (f.legal_before >= 12))
add("S7", "Collapse: 2+ wickets in last 12 balls", "S", "both", "volatility", f.wkts_last12 >= 2)
add("S8", "Tail exposed: 6+ wickets down", "S", "both", "volatility", f.wkts_before >= 6)
add("S10", "Endgame: last two overs", "S", "both", "volatility", f.balls_left <= 12)
add("S11", "Big chase: target >= venue par + 20", "S", "both", "volatility", ch & (f.target >= f.venue_par + 20))
add("M1", "Team dot streak 2+", "M", "both", "volatility", f.team_dot_streak >= 2)
add("M1b", "Team dot streak 3+", "M", "both", "volatility", f.team_dot_streak >= 3)
add("M2", "Boundary drought 12+ balls", "M", "both", "volatility", f.team_balls_since_bdry >= 12)
add("M3", "New batter (<=5 balls)", "M", "bat", "volatility", f.str_balls <= 5)
add("M4", "Wicket in previous 6 balls", "M", "both", "volatility", f.wkts_last6 >= 1)
add("M5", "Two new batters, 3+ down", "M", "both", "volatility", (f.pship_balls < 6) & (f.wkts_before >= 3) & (f.str_balls <= 5) & (f.ns_balls <= 5))
add("M6", "Bowler just hit for 4/6", "M", "bowl", "volatility", f.prev_ball_bdry_same_over == 1)
add("M7", "Expensive over in progress (10+)", "M", "bowl", "volatility", f.over_runs_so_far >= 10)
add("M8", "Bowler economy so far >= 10", "M", "bowl", "volatility", (f.bwl_balls >= 12) & (f.bwl_econ >= 10))
add("M9", "Free hit", "M", "bowl", "volatility", f.free_hit == 1)
add("M10", "Re-bowl after wide/no-ball", "M", "bowl", "volatility", f.rebowl == 1)
add("M11", "Bowler's last quota over / final over", "M", "bowl", "volatility", (f.bwl_last_quota_over == 1) | (f.final_over_of_inns == 1))
add("M12", "Milestone nerves (40-49 / 90-99)", "M", "bat", "volatility", f.milestone == 1)
add("M13", "Set batter with tail partner", "M", "bat", "volatility", f.tail_partner == 1)
add("M14", "Self-made: own SR < 100 after 10 balls", "M", "bat", "volatility", f.slow_self == 1)
add("B1", "Knockout match", "B", "both", "importance", ko)
add("B2", "IPL playoff race (batting/bowling team)", "B", "both", "importance", f.race_bat >= 1, f.race_bowl >= 1)
add("B2b", "IPL must-win", "B", "both", "importance", f.race_bat >= 2, f.race_bowl >= 2)
add("B3", "T20I series must-win", "B", "both", "importance", f.series_must_bat == 1, f.series_must_bowl == 1)
add("B4", "Strong opposition (Elo gap >= 50)", "B", "both", "control", f.elo_bowl - f.elo_bat >= 50, f.elo_bat - f.elo_bowl >= 50)
add("B6", "Dead rubber (anti-pressure)", "B", "both", "control", (f.dead_bat == 1) | (f.series_dead == 1), (f.dead_bowl == 1) | (f.series_dead == 1))
add("P1", "Early career (first 5 matches at level)", "P", "both", "volatility", f.bat_early_career == 1, f.bwl_early_career == 1)
add("P2", "Captain", "P", "both", "volatility", f.s_is_captain == 1, f.b_is_captain == 1)
add("P3", "Impact Player substitute", "P", "both", "volatility", f.s_is_impact == 1, f.b_is_impact == 1)
add("P4", "Form slump (last 3 below own median)", "P", "both", "volatility", f.bat_slump == 1, f.bwl_slump == 1)
add("P5", "Batting out of usual position", "P", "bat", "volatility", f.out_of_position == 1)
add("C1", "Night chase (dew)", "C", "both", "control", (f.night == 1) & ch)
add("C2", "Batting at home", "C", "both", "control", f.home_bat == 1)
log.info(f"Pointer library: {len(P)} pointers")

# ---------------------------------------------------------------- tests
B = PC["boot_reps"]
mids = f.match_id.values
mu, minv = np.unique(mids, return_inverse=True)
W = np.random.RandomState(SEED).multinomial(len(mu), np.ones(len(mu)) / len(mu), size=B).astype(float)  # B x M
faced = f.batter_faced.values
def msum(x, mask):
    return np.bincount(minv[mask], weights=x[mask], minlength=len(mu))

def boot_diff(res, flag, base_mask):
    a = flag & base_mask; o = ~flag & base_mask
    Sa, Na, So, No = msum(res, a), msum(np.ones(len(res)), a), msum(res, o), msum(np.ones(len(res)), o)
    est = Sa.sum() / max(Na.sum(), 1) - So.sum() / max(No.sum(), 1)
    bd = (W @ Sa) / np.maximum(W @ Na, 1) - (W @ So) / np.maximum(W @ No, 1)
    sd = bd.std()
    return est, sd

def boot_ratio(x, flag):
    Sa, Na, So, No = msum(x, flag), msum(np.ones(len(x)), flag), msum(x, ~flag), msum(np.ones(len(x)), ~flag)
    est = (Sa.sum() / max(Na.sum(), 1)) / (So.sum() / max(No.sum(), 1))
    br = ((W @ Sa) / np.maximum(W @ Na, 1)) / ((W @ So) / np.maximum(W @ No, 1))
    return est, np.quantile(br, 0.025), np.quantile(br, 0.975)

dwp = f.dwp.fillna(0).values
valid_wp = f.dwp.notna().values
rows = []
with Timer(log, "pointer tests"):
    for pid, (name, fam, persp, stakes, fb, fw) in P.items():
        flag_all = fb if persp != "bowl" else fw
        base_mask = faced if persp == "bat" else np.ones(len(f), bool)
        expd = EXP_STAGE if fam == "S" else EXP_CORE
        r = {"id": pid, "pointer": name, "family": fam, "for": persp, "stakes_type": stakes,
             "share_balls": float(flag_all.mean())}
        for comp in ["wkt", "dot", "bdry", "extra"]:
            if persp == "bat" and comp == "extra":
                r[f"t1_{comp}"] = np.nan; r[f"p_{comp}"] = np.nan; continue
            res = OUTC[comp] - expd[comp]
            est, sd = boot_diff(res, flag_all, base_mask)
            r[f"t1_{comp}"] = est
            r[f"rel_{comp}"] = est / max(OUTC[comp][base_mask].mean(), 1e-9)
            r[f"p_{comp}"] = 2 * norm.sf(abs(est) / max(sd, 1e-12))
        # T2
        if stakes == "volatility":
            est, lo, hi = boot_ratio(dwp, flag_all & valid_wp)
            r.update(t2_lift=est, t2_lo=lo, t2_hi=hi)
        else:
            r.update(t2_lift=np.nan, t2_lo=np.nan, t2_hi=np.nan)
        # T3
        n_ok = {}
        for side, pcol, fl in (("bat", "striker_id", fb & faced), ("bowl", "bowler_id", fw)):
            cnt = pd.Series(fl).groupby(f[pcol].values).sum()
            n_ok[side] = int((cnt >= PC["t3_min_balls"]).sum())
        r["t3_bat_players"], r["t3_bowl_players"] = n_ok["bat"], n_ok["bowl"]
        need_sides = ["bat", "bowl"] if persp == "both" else [persp]
        r["T3"] = bool(r["share_balls"] >= PC["t3_min_share"] and all(n_ok[s] >= PC["t3_min_players"] for s in need_sides))
        # T4: reliability of player gap (pointer - other) in run value above expected
        t4 = []
        for side, pcol, val, fl in (("bat", "striker_id", "rv_bat", fb), ("bowl", "bowler_id", "rv_bowl", fw)):
            if side not in need_sides:
                continue
            d = f[[pcol, "match_id", val]].copy()
            d["v"] = np.where(fl, d[val], np.nan)
            d["o"] = np.where(~fl, d[val], np.nan)
            # per player per half: mean pointer value - mean other value
            mm = np.unique(d.match_id)
            half = pd.Series(np.random.RandomState(SEED).randint(0, 2, len(mm)), index=mm)
            d["h"] = d.match_id.map(half)
            gg = d.groupby([pcol, "h"]).agg(v=("v", "mean"), nv=("v", "count"), o=("o", "mean"), no=("o", "count"))
            gg["gap"] = gg.v - gg.o
            pv = gg.unstack("h")
            okp = (pv[("nv", 0)] >= 15) & (pv[("nv", 1)] >= 15)
            pv = pv[okp]
            if len(pv) >= 20:
                rr = np.corrcoef(pv[("gap", 0)], pv[("gap", 1)])[0, 1]
                bs = []
                rs = np.random.RandomState(SEED)
                for _ in range(200):
                    ix = rs.randint(0, len(pv), len(pv))
                    bs.append(np.corrcoef(pv[("gap", 0)].values[ix], pv[("gap", 1)].values[ix])[0, 1])
                t4.append((side, 2 * rr / (1 + rr), np.nanquantile(bs, 0.05), len(pv)))
            else:
                t4.append((side, np.nan, np.nan, len(pv)))
        r["t4"] = "; ".join(f"{s}: r={v:.2f} (lo90 {lo:.2f}, n={n})" if not np.isnan(v) else f"{s}: n={n} too few" for s, v, lo, n in t4)
        r["T4"] = bool(any((not np.isnan(lo)) and lo > 0 for s, v, lo, n in t4))
        # T5 stability of the main component (largest |z|)
        comps = [x for x in ["wkt", "dot", "bdry", "extra"] if not np.isnan(r.get(f"p_{x}", np.nan))]
        main = min(comps, key=lambda x: r[f"p_{x}"])
        r["main_component"] = main
        res = OUTC[main] - expd[main]
        signs = []
        for lab, msk in (("IPL", f.level.eq("IPL").values), ("nonIPL", ~f.level.eq("IPL").values)):
            a = flag_all & base_mask & msk; o = ~flag_all & base_mask & msk
            signs.append(np.sign(res[a].mean() - res[o].mean()) if a.sum() > 50 else 0)
        yrs = []
        for y in sorted(f.year.unique()):
            msk = f.year.eq(y).values
            a = flag_all & base_mask & msk; o = ~flag_all & base_mask & msk
            if a.sum() > 50:
                yrs.append(np.sign(res[a].mean() - res[o].mean()))
        same = np.sign(r[f"t1_{main}"])
        r["t5_ipl_vs_t20i_same_sign"] = bool(signs[0] == signs[1] == same)
        r["t5_seasons_agree"] = f"{int(sum(s == same for s in yrs))}/{len(yrs)}"
        r["T5"] = bool(r["t5_ipl_vs_t20i_same_sign"] and sum(s == same for s in yrs) >= np.ceil(0.6 * len(yrs)))
        rows.append(r)

T = pd.DataFrame(rows)
pcols = [c for c in T.columns if c.startswith("p_")]
allp = T[pcols].values.ravel()
okp = ~np.isnan(allp)
adj = np.full(allp.shape, np.nan); adj[okp] = __import__("common").bh_fdr(allp[okp])
T[[c.replace("p_", "q_") for c in pcols]] = adj.reshape(T[pcols].shape)
T["T1"] = (T[[c.replace("p_", "q_") for c in pcols]] < PC["fdr_q"]).any(axis=1)
T["T2"] = np.where(T.stakes_type.eq("volatility"), T.t2_lo > 1.0, T.stakes_type.eq("importance"))
T["verdict"] = np.select(
    [T.stakes_type.eq("control"), ~T.T1, ~T.T3, ~T.T2 & T.stakes_type.eq("volatility"), ~T.T2, T.T4],
    ["control (adjustment only)", "dropped: no real effect on players", "real but too rare to measure",
     "STRAIN only (real effect, not higher stakes on its own)", "dropped: not consequential",
     "PRESSURE + separates players"], "PRESSURE")
save_table(T, "pressure_pointer_tests")
log.info("\n" + T[["id", "pointer", "share_balls", "t1_wkt", "q_wkt", "t1_dot", "q_dot", "t2_lift", "T1", "T2", "T3", "T4", "T5", "verdict"]].round(4).to_string())
surv = T[T.verdict.str.startswith("PRESSURE")].id.tolist()
log.info(f"Surviving pointers ({len(surv)}): {surv}")

# ---------------------------------------------------------------- combined definitions
flagb = {pid: P[pid][4] for pid in P}
flagw = {pid: P[pid][5] for pid in P}
imp_ids = [p for p in surv if P[p][3] == "importance"]
big = np.zeros(len(f))
for p in imp_ids:
    big = np.maximum(big, (flagb[p] | flagw[p]).astype(float))
cands = {}
cands["C-A leverage"] = f.LI.values
cands["C-B leverage x big match"] = f.LI.values * (1 + 0.5 * big)
big_all = np.zeros(len(f))
for p in ["B1", "B2b", "B3"]:
    big_all = np.maximum(big_all, (flagb[p] | flagw[p]).astype(float))
cands["C-B2 leverage x all big-match flags (by design)"] = f.LI.values * (1 + 0.5 * big_all)
# C-C stakes x strain: strain learned from surviving volatility pointers (M/S/P), Poisson rate models with
# stage-baseline offsets for dismissal and dot; strain = geometric mean of the two predicted rate ratios
strain_ids = T[T.verdict.str.startswith("PRESSURE") | T.verdict.str.startswith("STRAIN")].id.tolist()
vol_ids = [p for p in strain_ids if P[p][3] == "volatility" and p != "S1"]
log.info(f"Strain learned from {len(vol_ids)} pointers with a real effect: {vol_ids}")
X = np.column_stack([(flagb[p] | flagw[p]).astype(float) for p in vol_ids]) if vol_ids else np.zeros((len(f), 1))
coef = {}
lin = np.zeros(len(f))
for comp, wgt in (("wkt", 0.5), ("dot", 0.5)):
    e = np.clip(EXP_STAGE[comp], 1e-4, None)
    pr = PoissonRegressor(alpha=1e-4, max_iter=300).fit(X, OUTC[comp] / e, sample_weight=e)
    coef[comp] = dict(zip(vol_ids, pr.coef_.round(4)))
    lin += wgt * (X @ pr.coef_)
strain = np.clip(np.exp(lin), 0.5, 2.5)
f["strain"] = strain
cands["C-C stakes x strain"] = f.LI.values * strain * (1 + 0.5 * big)
log.info(f"Strain coefficients: {coef}")
# C-D first principal component of surviving pointers (+LI)
Z = StandardScaler().fit_transform(np.column_stack([X, f.LI.values]))
pc1 = PCA(1, random_state=SEED).fit_transform(Z)[:, 0]
if np.corrcoef(pc1, f.LI.values)[0, 1] < 0:
    pc1 = -pc1
cands["C-D PCA factor"] = pc1
# C-E GMM pressure regimes on ball state, regimes scored by mean realised |dWP|
st = np.column_stack([f.balls_left, f.wih, f.wp_now.clip(0.001, 0.999), np.log1p(f.LI), f.innings])
Zs = StandardScaler().fit_transform(st)
gm = GaussianMixture(8, random_state=SEED, covariance_type="full").fit(Zs[::4])
lab = gm.predict(Zs)
reg_score = pd.Series(dwp).groupby(lab).mean()
cands["C-E GMM regimes"] = pd.Series(lab).map(reg_score).values + 1e-6 * f.LI.values
f["regime"] = lab
# C-F published (chase only; first innings gets 0)
cands["C-F Bhattacharjee-Lemmer"] = np.nan_to_num(f.bl_pi.values, nan=0.0)
cands["C-F Thomson zone"] = np.where(ch, -np.abs(np.nan_to_num(f.thomson_rho.values, nan=99) - 2.8), -99)
# C-G simple rules
rules = ((f.legal_before >= 90).astype(int) + (ch & (f.rrr >= 9)).astype(int) + f.wp_now.between(0.2, 0.8).astype(int)
         + (f.wkts_before >= 5).astype(int))
cands["C-G simple rules"] = rules.values + 1e-6 * f.LI.values
SIMPLICITY = {"C-G simple rules": 0, "C-A leverage": 1, "C-B leverage x big match": 2, "C-B2 leverage x all big-match flags (by design)": 2, "C-F Bhattacharjee-Lemmer": 2,
              "C-F Thomson zone": 2, "C-C stakes x strain": 3, "C-D PCA factor": 4, "C-E GMM regimes": 5}

wkt_res_stage = OUTC["wkt"] / np.clip(EXP_STAGE["wkt"], 1e-4, None)
evals = []
mean_dwp = dwp[valid_wp].mean()
for nm, sc in cands.items():
    for q in PC["hard_cuts"]:
        thr = np.quantile(sc, 1 - q)
        fl = sc >= thr
        lift = dwp[fl & valid_wp].mean() / mean_dwp
        capture = dwp[fl & valid_wp].sum() / dwp[valid_wp].sum()
        wr = OUTC["wkt"][fl].sum() / EXP_STAGE["wkt"][fl].sum()
        dot_shift = (OUTC["dot"][fl] - EXP_STAGE["dot"][fl]).mean()
        ext = OUTC["extra"][fl].sum() / EXP_STAGE["extra"][fl].sum()
        nb = (pd.Series(fl & faced).groupby(f.striker_id.values).sum() >= 30).sum()
        nw = (pd.Series(fl).groupby(f.bowler_id.values).sum() >= 30).sum()
        li_ipl = dwp[fl & valid_wp & f.level.eq("IPL").values].mean() / dwp[valid_wp & f.level.eq("IPL").values].mean()
        li_int = dwp[fl & valid_wp & ~f.level.eq("IPL").values].mean() / dwp[valid_wp & ~f.level.eq("IPL").values].mean()
        evals.append({"definition": nm, "form": f"hard top {int(q * 100)}%", "q": q, "stakes_lift": lift,
                      "stakes_capture": capture, "strain_wkt_ratio": wr, "strain_dot_shift": dot_shift,
                      "extras_ratio": ext, "batters_30": int(nb), "bowlers_30": int(nw),
                      "lift_IPL": li_ipl, "lift_T20I": li_int, "first_inns_share": float((f.innings[fl] == 1).mean())})
    for a in PC["alphas"]:
        w = np.clip(sc - sc.min(), 0, None) ** a if nm != "C-D PCA factor" else np.exp(a * (sc - sc.mean()) / sc.std())
        w = w / w.mean()
        lift = np.average(dwp[valid_wp], weights=w[valid_wp]) / mean_dwp
        wr = (w * OUTC["wkt"]).sum() / (w * EXP_STAGE["wkt"]).sum()
        evals.append({"definition": nm, "form": f"continuous alpha={a}", "q": np.nan, "stakes_lift": lift,
                      "strain_wkt_ratio": wr, "strain_dot_shift": float(np.average(OUTC["dot"] - EXP_STAGE["dot"], weights=w)),
                      "extras_ratio": (w * OUTC["extra"]).sum() / (w * EXP_STAGE["extra"]).sum()})
E = pd.DataFrame(evals)
# pre-registered choice: rank definitions by mean rank of stakes lift and strain over the hard cuts (T3 coverage
# must hold at q=20%); ties within 2% go to the simpler definition
H = E[E.q.notna()].copy()
agg = H.groupby("definition").agg(stakes=("stakes_lift", "mean"), strain=("strain_wkt_ratio", "mean"),
                                  cov_bat=("batters_30", lambda s: s.iloc[2]), cov_bowl=("bowlers_30", lambda s: s.iloc[2]))
agg["rank_stakes"] = agg.stakes.rank(ascending=False)
agg["rank_strain"] = agg.strain.rank(ascending=False)
agg["score"] = (agg.stakes / agg.stakes.max() + agg.strain / agg.strain.max()) / 2
agg["coverage_ok"] = (agg.cov_bat >= PC["t3_min_players"]) & (agg.cov_bowl >= PC["t3_min_players"])
agg["simplicity"] = agg.index.map(SIMPLICITY)
okd = agg[agg.coverage_ok]
best = okd.score.max()
tied = okd[okd.score >= best * 0.98].sort_values(["simplicity", "score"], ascending=[True, False])
chosen = tied.index[0]
log.info("Definition comparison:\n" + agg.sort_values("score", ascending=False).round(3).to_string())
log.info(f"Chosen definition: {chosen}")
# threshold: the largest q where pressure balls still carry >= 2x normal win-prob movement and >= 1.2x stage wickets
hc = H[H.definition == chosen].sort_values("q")
okq = hc[(hc.stakes_lift >= 2.0) & (hc.strain_wkt_ratio >= 1.2)]
q_star = float(okq.q.max()) if len(okq) else 0.20
log.info(f"Chosen hard cut: top {int(q_star * 100)}% of balls\n{hc.round(3).to_string()}")
save_table(E, "pressure_definition_comparison")
save_table(agg.reset_index(), "pressure_definition_summary")

# match importance by design: the swing test above measures how much a ball can change THIS match, so it cannot
# see that a final matters more than a league game. Pressure = within-match swing x match importance.
IMP = PC["importance"]
must = (f.race_bat >= 2) | (f.race_bowl >= 2) | (f.series_must_bat == 1) | (f.series_must_bowl == 1)
dead = (f.series_dead == 1) | ((f.dead_bat == 1) & (f.dead_bowl == 1))
f["importance"] = np.select([ko.values, must.values, dead.values], [IMP["knockout"], IMP["must_win"], IMP["dead_rubber"]], 1.0)
log.info(f"Match importance: share of balls knockout {ko.mean():.3f}, must-win {(must & ~ko).mean():.3f}, "
         f"dead rubber {(dead & ~ko & ~must).mean():.3f}")
score = cands[chosen] * f.importance.values
f["pressure"] = score
f["pressure_pct"] = pd.Series(score).rank(pct=True).values
f["is_pressure"] = f.pressure_pct >= 1 - q_star
# per-ball pressure pieces kept for later phases / sensitivity
keep = ["ball_uid", "match_id", "LI", "strain", "regime", "importance", "pressure", "pressure_pct", "is_pressure", "bl_pi",
        "thomson_rho", "res_left_pct"]
outp = f[keep].copy()
for nm, sc in cands.items():
    outp["cand_" + nm.split()[0].replace("-", "") + ("_" + nm.split()[1] if nm.startswith("C-F") else "")] = sc
for pid in P:
    outp["ptr_" + pid] = (flagb[pid] | flagw[pid]).astype(np.int8)
save(outp, "pressure")

# face validity: 20 highest and 20 lowest pressure situations (one per match for variety)
fv = f[["match_id", "innings", "legal_before", "over_no", "ball_in_over", "score_before", "wkts_before", "target", "need",
        "balls_left", "wp_now", "LI", "strain", "pressure", "level", "year", "stage"]].copy()
top = fv.sort_values("pressure", ascending=False).drop_duplicates("match_id").head(20)
bot = fv[fv.legal_before > 0].sort_values("pressure").drop_duplicates("match_id").head(20)
save_table(pd.concat([top.assign(group="highest"), bot.assign(group="lowest")]), "pressure_face_validity")
# heatmap data: chase pressure by over x wickets and by required-rate band
hm = f[f.innings == 2].assign(rrr_band=pd.cut(f.rrr, [0, 6, 8, 10, 12, 15, 99])).groupby(["over_no", "wkts_before"]).pressure_pct.mean().unstack()
save_table(hm.reset_index(), "pressure_heatmap_chase")
hm1 = f[f.innings == 1].groupby(["over_no", "wkts_before"]).pressure_pct.mean().unstack()
save_table(hm1.reset_index(), "pressure_heatmap_first")
save_json({"chosen_definition": chosen, "importance": IMP, "q_star": q_star, "surviving_pointers": surv, "strain_coef": coef,
           "pressure_share_first_innings": float((f.innings[f.is_pressure] == 1).mean()),
           "importance_pointers": imp_ids}, "p04_summary")
log.info(f"Pressure balls: {f.is_pressure.mean():.3f} of all; first-innings share among them "
         f"{(f.innings[f.is_pressure] == 1).mean():.3f}")
log.info("P4 done")
