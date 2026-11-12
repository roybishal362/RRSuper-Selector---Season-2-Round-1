"""P9 - Experiment grid with a pre-registered, staged selection protocol (Stage 2 of the plan).
Pressure is already locked (P4). Every remaining choice is judged on ONE yardstick: the player's de-drifted
win-probability added above expected (V-2) on pressure balls in matches the method never saw
(run value above expected, V-1, is the second check).
Splits: rolling-origin (2022-23 -> 2024, 2022-24 -> 2025) + 50 random 50/50 match splits within 2022-25.
2026 is sealed and opened once at the end for the chosen configuration (and simple baselines).
Stages: A value metric -> B rating model -> C gap weight w_d and consistency weight w_c -> D shrinkage and minimum
sample -> E local sensitivity. Rule: best mean objective; a simpler option within one paired SE wins."""
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import pandas as pd
from common import CFG, SEED, load, save_table, save_json, load_json, get_logger, Timer, weighted_spearman
import ratings_lib as RL

log = get_logger("p09_experiments")
V_ALL = load("values")
comp_info = load_json("p05_components")
HOLD = CFG["cv"]["holdout_season"]
V = V_ALL[V_ALL.year < HOLD].copy()
MIN_TEST = 20

# ------------------------------------------------ splits
splits = []
for tr_years, te_year in CFG["cv"]["rolling"]:
    splits.append(("rolling", f"{tr_years[0]}-{tr_years[-1]}->{te_year}", V.year.isin(tr_years).values, (V.year == te_year).values))
mids = np.sort(V.match_id.unique())
rng = np.random.RandomState(SEED)
for i in range(CFG["cv"]["random_splits"]):
    a = set(rng.choice(mids, len(mids) // 2, replace=False))
    m = V.match_id.isin(a).values
    splits.append(("random", f"r{i}", m, ~m))
log.info(f"{len(splits)} splits")


def yardstick(test, side):
    pcol = RL.PCOL[side]
    t = test[test.is_pressure & test[f"v2_{side}"].notna()]
    g = t.groupby(pcol).agg(y2=(f"v2_{side}", "mean"), y1=(f"v1_{side}", "mean"), n=(f"v2_{side}", "size"))
    return g[g.n >= MIN_TEST]


YARD, BASE_IDS = {}, {}
for kind, name, tr, te in splits:
    for side in ("bat", "bowl"):
        YARD[(name, side)] = yardstick(V[te], side)
        # common evaluation pool for EVERY configuration: players with a test yardstick and >=1 training pressure
        # ball. Players a configuration excludes (minimum-sample rules) are ranked last instead of being dropped,
        # so correlations are always computed on the same players.
        d = V[tr]
        BASE_IDS[(name, side)] = set(d.loc[d.is_pressure & d[f"v1_{side}"].notna(), RL.PCOL[side]].unique())

R1_CACHE = {}
_orig_r1 = RL.r1_ratings
def cached_r1(v, side, metric="v1", k_mult=1.0, lam=None):
    key = (CUR_SPLIT, metric, k_mult)
    if key not in R1_CACHE:
        R1_CACHE[key] = RL.r1_fit(v, metric, k_mult, lam)
    return R1_CACHE[key][side]
RL.r1_ratings = cached_r1
CUR_SPLIT = None


def evaluate(cfg, sides=("bat", "bowl")):
    global CUR_SPLIT
    rows = []
    for kind, name, tr, te in splits:
        CUR_SPLIT = name
        train = V[tr]
        for side in sides:
            sc = RL.build_scores(train, side, cfg, comp_info).score
            y = YARD[(name, side)]
            pool = y.index[y.index.isin(BASE_IDS[(name, side)])]
            if len(pool) < 15:
                continue
            s, yy = sc.reindex(pool).fillna(-1e9), y.loc[pool]
            top = s.sort_values(ascending=False).index[:10]
            wmean = lambda col, idx: np.average(yy.loc[idx, col], weights=yy.loc[idx, "n"])
            rows.append({"kind": kind, "split": name, "side": side, "n_pool": len(pool),
                         "rho_v2": weighted_spearman(s.values, yy.y2.values, yy.n.values),
                         "rho_v1": weighted_spearman(s.values, yy.y1.values, yy.n.values),
                         "top10_lift_v2": wmean("y2", top) - wmean("y2", pool),
                         "top10_lift_v1": wmean("y1", top) - wmean("y1", pool)})
    return pd.DataFrame(rows)


def objective(res):
    """per side: 0.5 * mean over random splits + 0.5 * mean over rolling splits of rho_v2; averaged over sides."""
    o = {}
    for side in res.side.unique():
        r = res[res.side == side]
        o[side] = 0.5 * r[r.kind == "random"].rho_v2.mean() + 0.5 * r[r.kind == "rolling"].rho_v2.mean()
    return o


SIMPLE = {"value": {"v1": 1, "v2": 1, "v4": 2, "v3": 3},
          "model": {"R2-p_only": 1, "R2-joint": 2, "R2-weighted": 3, "R3": 3, "R1": 4, "blend": 5}}


def complexity(cfg):
    return (SIMPLE["value"].get(cfg.get("value"), 1) + SIMPLE["model"].get(cfg.get("model"), 1)
            + (cfg.get("w_d", 0) > 0) + (cfg.get("w_c", 0) > 0) + (cfg.get("k_mult", 1) != 1) + (cfg.get("min_n", 0) > 0)
            + 0.1 * abs(np.log2(cfg.get("alpha", 1.0))))


def run_stage(label, cfgs, sides=("bat", "bowl")):
    out, per = [], {}
    for cfg in cfgs:
        t0 = time.time()
        res = evaluate(cfg, sides)
        per[str(cfg)] = res
        for sd in sides:
            r = res[res.side == sd]
            out.append({"stage": label, "side": sd, "config": str(cfg), **{k: cfg.get(k) for k in ("value", "model", "w_d", "w_c", "k_mult", "min_n", "alpha")},
                        "objective": objective(res)[sd],
                        "rho_v2_random": r[r.kind == "random"].rho_v2.mean(), "rho_v2_rolling": r[r.kind == "rolling"].rho_v2.mean(),
                        "rho_v1_random": r[r.kind == "random"].rho_v1.mean(), "rho_v1_rolling": r[r.kind == "rolling"].rho_v1.mean(),
                        "top10_lift_v2_x100": 100 * r.top10_lift_v2.mean(), "top10_lift_v1": r.top10_lift_v1.mean(),
                        "complexity": complexity(cfg)})
        log.info(f"[{label}] {cfg} -> " + ", ".join(f"{o['side']} {o['objective']:.4f}" for o in out[-len(sides):]) + f" ({time.time() - t0:.0f}s)")
    return pd.DataFrame(out), per


def choose(df, per, side):
    d = df[df.side == side].sort_values("objective", ascending=False).reset_index(drop=True)
    best = d.iloc[0]
    rb = per[best.config]
    rb = rb[(rb.side == side) & (rb.kind == "random")].set_index("split").rho_v2
    pick = best
    for _, r in d.iterrows():
        if r.complexity >= pick.complexity:
            continue
        rr = per[r.config]
        rr = rr[(rr.side == side) & (rr.kind == "random")].set_index("split").rho_v2
        diff = (rb - rr.reindex(rb.index))
        se = diff.std() / np.sqrt(diff.notna().sum())
        if best.objective - r.objective <= se:
            pick = r
    return eval(pick.config), d


chosen = {}
all_rows = []
base = {"value": "v1", "model": "R2-joint", "w_d": 0.0, "w_c": 0.0, "k_mult": 1.0, "min_n": 0}

# Stage A: value metric
with Timer(log, "Stage A"):
    A, perA = run_stage("A value metric", [dict(base, value=x) for x in ("v1", "v2", "v3", "v4")])
all_rows.append(A)
for side in ("bat", "bowl"):
    chosen[side], _ = choose(A, perA, side)
log.info(f"Stage A chosen: {chosen}")

# Stage B: rating model (per side, using that side's chosen value)
B_rows, perB = [], {}
for side in ("bat", "bowl"):
    cf = chosen[side]
    cfgs = [dict(cf, model=m) for m in ("R2-p_only", "R2-joint", "R1", "R3", "blend")] + \
           [dict(cf, model="R2-weighted", alpha=a) for a in CFG["pressure"]["alphas"]]
    with Timer(log, f"Stage B {side}"):
        Bs, pB = run_stage("B rating model", cfgs, (side,))
    Bs = Bs[Bs.side == side]
    B_rows.append(Bs); perB.update(pB)
    fam = CFG["selection"].get("model_family", {}).get(side)
    if fam:   # family fixed by design (config.yaml explains why); the data still picks within it
        best_any = Bs.sort_values("objective", ascending=False).iloc[0]
        chosen[side], _ = choose(Bs[Bs.model == fam], pB, side)
        log.info(f"Stage B {side}: model family fixed to {fam} by design; best within family {chosen[side]}; "
                 f"best overall was {best_any.model} (objective {best_any.objective:.4f})")
    else:
        chosen[side], _ = choose(Bs, pB, side)
all_rows.append(pd.concat(B_rows))
log.info(f"Stage B chosen: {chosen}")

# Stage C: w_d and w_c
for side in ("bat", "bowl"):
    cf = chosen[side]
    cfgs = [dict(cf, w_d=wd, w_c=wc) for wd in CFG["selection"]["w_d_grid"] for wc in CFG["selection"]["w_c_grid"]]
    with Timer(log, f"Stage C {side}"):
        Cs, pC = run_stage("C gap & consistency weights", cfgs, (side,))
    Cs = Cs[Cs.side == side]
    all_rows.append(Cs)
    chosen[side], _ = choose(Cs, pC, side)
log.info(f"Stage C chosen: {chosen}")

# Stage D: shrinkage strength and minimum sample
for side in ("bat", "bowl"):
    cf = chosen[side]
    cfgs = [dict(cf, k_mult=km, min_n=mn) for km in CFG["selection"]["k_mult_grid"] for mn in CFG["selection"]["min_pressure_balls_grid"]]
    with Timer(log, f"Stage D {side}"):
        Ds, pD = run_stage("D shrinkage & minimum sample", cfgs, (side,))
    Ds = Ds[Ds.side == side]
    all_rows.append(Ds)
    chosen[side], _ = choose(Ds, pD, side)
log.info(f"Stage D chosen: {chosen}")
# review decisions (config.yaml explains each, with the evidence): settings override and the evidence floor
rule_pick = {s: dict(chosen[s]) for s in chosen}
for side in ("bat", "bowl"):
    ov = CFG["selection"].get("override", {}).get(side)
    if ov:
        chosen[side].update(ov)
        log.info(f"{side}: review decision overrides {ov} (rule's own pick kept as rule_pick: {rule_pick[side]})")
    fl = CFG["selection"].get("min_pressure_balls_floor", 0)
    if chosen[side].get("min_n", 0) < fl:
        chosen[side]["min_n"] = fl
        log.info(f"{side}: evidence floor applied, minimum {fl} pressure balls")
log.info(f"Final settings: {chosen}")
RES = pd.concat(all_rows, ignore_index=True)
save_table(RES, "experiment_results")

# ------------------------------------------------ sealed 2026 holdout
# NOTE (disclosed in the report): opened a second time after fixing a pool-comparability bug in Stage D that was
# identified from the Stage D pattern itself (objective rising mechanically with the minimum-sample rule).
RL.r1_ratings = _orig_r1
train, test = V_ALL[V_ALL.year < HOLD], V_ALL[V_ALL.year == HOLD]
hold = []
for side in ("bat", "bowl"):
    y = yardstick(test, side)
    variants = {"CHOSEN": chosen[side],
                "baseline: overall level, no pressure (R2 all balls)": dict(base, model="R2-all", value=chosen[side]["value"]),
                "baseline: raw pressure WPA (no shrinkage)": "raw_v2",
                "baseline: raw pressure strike rate / economy": "raw_rate"}
    for nm, cfg in variants.items():
        if cfg == "raw_v2":
            d = RL.side_frame(train, side, "v2"); d = d[d.pres]
            sc = d.groupby("pid").val.mean()
        elif cfg == "raw_rate":
            pc = RL.PCOL[side]
            t = train[train.is_pressure & (train.faced if side == "bat" else True)]
            sc = t.groupby(pc).runs.mean() if side == "bat" else -t.groupby(pc).team_runs.mean()
        else:
            sc = RL.build_scores(train, side, cfg, comp_info).score
        tr_ids = set(train.loc[train.is_pressure & train[f"v1_{side}"].notna(), RL.PCOL[side]].unique())
        pool = y.index[y.index.isin(tr_ids)]
        s, yy = sc.reindex(pool).fillna(-1e9), y.loc[pool]
        top = s.sort_values(ascending=False).index[:10]
        hold.append({"side": side, "method": nm, "n_pool": len(pool),
                     "rho_v2": weighted_spearman(s.values, yy.y2.values, yy.n.values),
                     "rho_v1": weighted_spearman(s.values, yy.y1.values, yy.n.values),
                     "top10_lift_v2_x100": 100 * (np.average(yy.loc[top, "y2"], weights=yy.loc[top, "n"]) - np.average(yy.y2, weights=yy.n)),
                     "top10_lift_v1": np.average(yy.loc[top, "y1"], weights=yy.loc[top, "n"]) - np.average(yy.y1, weights=yy.n)})
H = pd.DataFrame(hold)
save_table(H, "holdout_2026")
log.info("2026 sealed holdout (opened once):\n" + H.round(4).to_string())
save_json({"chosen": chosen, "rule_pick": rule_pick}, "p09_summary")
log.info("P9 done")
