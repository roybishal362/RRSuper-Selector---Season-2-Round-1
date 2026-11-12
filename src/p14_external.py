"""P14 - Independent exam on public data (Cricsheet) + combined-data ranking.
The context models trained on the supplied data are applied UNCHANGED to ~4,000 public matches they never saw
(T20 World Cups 2022/24/26, franchise leagues, Syed Mushtaq Ali, other T20Is). Pressure balls use the supplied data's
threshold. Then:
 A. External validation: do the supplied-data ratings predict players' pressure performance in public matches?
    Yardstick: raw win probability added per pressure ball (de-drifted within each competition) - it does not depend
    on any baseline choice, so it is fair to every method compared.
 B. Combined ranking: supplied + public balls together (players from the supplied data only), compared with the main.
 C. World Cup knockouts: how the top-10s did in World Cup semi-finals and finals (descriptive; small samples)."""
import re
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
from common import CFG, MODELS, LOGS, load, load_table, save, save_table, save_json, load_json, get_logger, weighted_spearman
import ratings_lib as RL

log = get_logger("p14_external")
K = CFG["cv"]["n_folds"]
CM = joblib.load(f"{MODELS}/context_models.joblib")
F1 = CM["feats"]["m1"]["base"]; M2F = CM["feats"]["m2"]; M4F = CM["feats"]["m4"]
PHI = float(re.search(r"dispersion \(variance / mean\): ([0-9.]+)", open(f"{LOGS}/p03_context_models.log", encoding="utf-8").read()).group(1))
f = load("features_ext")
mx = load("matches_ctx_ext")
# duplicates the P0x linking step missed: a public match on the same date as a supplied match with >= 11 of the same
# players in the two playing XIs is the same match -> removed, so the exam stays strictly out-of-sample
_pm = load("matches_clean")[["match_id", "match_date"]]; _xm = load("matches_clean_ext")[["match_id", "match_date"]]
_pm["d"] = pd.to_datetime(_pm.match_date).dt.date; _xm["d"] = pd.to_datetime(_xm.match_date).dt.date
_pairs = _xm.merge(_pm, on="d", suffixes=("_x", "_p"))
_P = load("playing_eleven_clean")[["match_id", "player_id"]].groupby("match_id").player_id.apply(set)
_X = load("playing_eleven_clean_ext")[["match_id", "player_id"]].groupby("match_id").player_id.apply(set)
_pairs["shared"] = [len(_X.get(a, set()) & _P.get(b, set())) for a, b in zip(_pairs.match_id_x, _pairs.match_id_p)]
DUP = set(_pairs.loc[_pairs.shared >= 11, "match_id_x"])
log.info(f"Duplicate check: {len(DUP)} public matches share a date and >= 11 players with a supplied match -> removed")
f = f[~f.match_id.isin(DUP)].reset_index(drop=True)
mx = mx[~mx.match_id.isin(DUP)]
f["fold"] = (f.match_id % K).values
NC = 8
R_K = np.array([0, 1, 2, 3, 4, 6, 0, 1]); L_K = np.array([1, 1, 1, 1, 1, 1, 1, 0]); W_K = np.array([0, 0, 0, 0, 0, 0, 1, 0])
GH_X, GH_W = np.polynomial.hermite.hermgauss(9)


def per_fold(S, fn):
    out = np.zeros(len(S)); fo = S.fold.values
    for k in range(K):
        m = fo == k
        if m.any():
            out[m] = fn(k, S[m])
    return out


def _chase(S):
    S = S.copy(); S["rrr"] = S.rrr.clip(0, 36)
    r = CM["res"].predict(pd.DataFrame({"balls_left": S.balls_left.clip(lower=0), "wih": S.wih.clip(lower=0)}))
    S["z_chase"] = (S.need - r) / np.sqrt(CM["res_phi"] * np.clip(r, 0.05, None))
    return S


def _wp2_fold(k, X):
    o = CM["m2"][("logit", k)].predict_proba(X[M2F].fillna(0))[:, 1]
    bl = X.balls_left.values
    we = np.clip((24 - bl) / 12, 0, 1)
    if (we > 0).any():
        sub = we > 0
        bi = np.clip(bl[sub], 0, CM["dp_b"]).astype(int)
        wi = np.clip(X.wih.values[sub], 0, 10).astype(int)
        ni = np.clip(np.ceil(X.need.values[sub]), 1, CM["dp_n"]).astype(int)
        o[sub] = (1 - we[sub]) * o[sub] + we[sub] * CM["dpv"][k][bi, wi, ni]
    return o


wp2 = lambda S: per_fold(_chase(S), _wp2_fold)
re_ = lambda S: per_fold(S, lambda k, X: CM["m4"][k].predict(X[M4F]))


def wp1(S):
    re0 = re_(S); sd = np.sqrt(PHI * np.clip(re0, 0, None)); out = np.zeros(len(S))
    for x, w in zip(GH_X, GH_W):
        T = S.copy(); T["target"] = T.score_before + np.clip(re0 + np.sqrt(2) * sd * x, 0, None) + 1
        T["need"] = T.target; T["balls_left"] = T.maxb; T["wih"] = 10; T["rrr"] = T.need * 6 / T.balls_left; T["elo_diff"] = -T.elo_diff
        out += w / np.sqrt(np.pi) * (1 - wp2(T))
    return out


def state_values(S):
    S = S.copy(); S["wih"] = S.wih.clip(lower=0)
    wp = np.full(len(S), np.nan); rv = np.zeros(len(S))
    i2 = S.innings.values == 2
    won = i2 & (S.need.values <= 0); over = (S.wih.values <= 0) | (S.balls_left.values <= 0)
    lost2 = i2 & ~won & over; live2 = i2 & ~won & ~over; end1 = ~i2 & over; live1 = ~i2 & ~over
    wp[won] = 1.0; wp[lost2] = np.where(S.need.values[lost2] == 1, 0.5, 0.0)
    if live2.any():
        T = S[live2].copy(); T["rrr"] = T.need * 6 / T.balls_left; wp[live2] = wp2(T)
    if live1.any():
        wp[live1] = wp1(S[live1])
    if end1.any():
        T = S[end1].copy(); T["target"] = T.score_before + 1; T["need"] = T.target; T["balls_left"] = T.maxb; T["wih"] = 10
        T["rrr"] = T.need * 6 / T.balls_left; T["elo_diff"] = -T.elo_diff; wp[end1] = 1 - wp2(T)
    live = live1 | live2
    if live.any():
        rv[live] = re_(S[live])
    return wp, rv


log.info(f"Scoring {len(f):,} public deliveries with the supplied-data models")

P = np.zeros((len(f), NC))
for k in range(K):
    m = f.fold.values == k
    P[m] = CM["m1"][("base", k)].predict(f.loc[m, F1], num_iteration=CM["m1"][("base", k)].best_iteration)
base = f[["score_before", "legal_before", "balls_left", "wih", "need", "fold", "innings", "maxb", "target", "lvl_code", "year", "venue_par", "elo_diff"]].copy()
wp_now, re_now = state_values(base)
wpk = np.zeros((len(f), NC)); rek = np.zeros((len(f), NC))
for c in range(NC):
    S = base.copy(); S["score_before"] += R_K[c]; S["legal_before"] += L_K[c]; S["balls_left"] -= L_K[c]; S["wih"] -= W_K[c]; S["need"] -= R_K[c]
    wpk[:, c], rek[:, c] = state_values(S)
LI_raw = (P * np.abs(wpk - wp_now[:, None])).sum(1)
E_wp = (P * wpk).sum(1); E_team = (P * (R_K[None, :] + rek)).sum(1)
Pb = P[:, :7] / P[:, :7].sum(1, keepdims=True)
E_bat = (Pb * (R_K[None, :7] + rek[:, :7])).sum(1)
o = f[["match_id", "innings"]].copy(); o["wp_now"] = wp_now; o["re_now"] = re_now
nm, ni = o.match_id.shift(-1), o.innings.shift(-1)
same = (nm == o.match_id) & (ni == o.innings); chase = (nm == o.match_id) & (ni == 2) & (o.innings == 1)
wpn = np.where(same, o.wp_now.shift(-1), np.nan); wpn = np.where(chase, 1 - o.wp_now.shift(-1), wpn)
last = ~same & ~chase
wpn = np.where(last & f.bat_won.notna().values, f.bat_won.values, wpn)
f["wp_now"], f["wp_next"] = wp_now, wpn
f["re_next"] = np.where(same, o.re_now.shift(-1), 0.0)

# pressure on the supplied data's scale and threshold
ctx = load("ctx_preds", columns=["LI_raw"])
pr = load("pressure", columns=["pressure"])
p4 = load_json("p04_summary")
thr = float(np.quantile(pr.pressure, 1 - p4["q_star"]))
IMP = CFG["pressure"]["importance"]
ko = f.knock_out.eq(1) | f.stage.isin(["Final", "Semi Final", "Qualifier", "Eliminator", "Quarter Final"])
f["importance"] = np.where(ko, IMP["knockout"], 1.0)
f["pressure"] = LI_raw / ctx.LI_raw.mean() * f.importance
f["is_pressure"] = f.pressure >= thr
f["pressure_pct"] = np.searchsorted(np.sort(pr.pressure.values), f.pressure.values) / len(pr)
log.info(f"Public pressure balls: {f.is_pressure.mean():.3f} of deliveries (supplied data: {p4['q_star']:.2f})")

# values (same formulas as P5; component weights from the supplied data)
comp = load_json("p05_components")
faced = f.batter_faced.values & ~f.nonstriker_out.values
yc = f.y_class.values
comp_key = f.level.astype(str) + "_" + f.year.astype(str)
v = f[["match_id", "innings", "year", "level", "striker_id", "bowler_id", "is_pressure", "pressure", "pressure_pct", "phase",
       "is_pace", "knock_out", "runs", "team_runs", "legal", "bowler_wicket", "striker_out", "batting_order"]].copy()
v["faced"] = faced
v["comp_key"] = comp_key.values
v["is_wc"] = f.match_id.map(mx.set_index("match_id").is_wc).values if "is_wc" in mx else 0
v["v1_bat"] = np.where(faced, f.runs + f.re_next - E_bat, np.nan)
v["v1_bowl"] = -(f.team_runs + f.re_next - E_team)
wpa_raw = f.wp_next - f.wp_now                      # raw win probability added (no baseline)
wpa = f.wp_next - E_wp
binid = f.innings.astype(str) + "_" + (f.wp_now.clip(0, 0.9999) * 10).astype(int).astype(str)
v["v2_bat"] = np.where(faced, wpa - wpa.groupby(binid).transform("mean"), np.nan)
v["v2_bowl"] = -(wpa - wpa.groupby(binid).transform("mean"))
v["wpa_bat"] = np.where(faced, wpa_raw, np.nan)
v["wpa_bowl"] = -wpa_raw
v["r_bdry_bat"] = np.where(faced, np.isin(yc, [4, 5]) - (Pb[:, 4] + Pb[:, 5]), np.nan)
v["r_dot_bat"] = np.where(faced, (yc == 0) - Pb[:, 0], np.nan)
v["r_wkt_bat"] = np.where(faced, f.striker_out.astype(float) - Pb[:, 6], np.nan)
share_bw = (f.bowler_wicket.groupby(f.phase).sum() / f.is_wicket.groupby(f.phase).sum()).reindex(f.phase).values
v["r_bdry_bowl"] = np.isin(yc, [4, 5]) - (P[:, 4] + P[:, 5]); v["r_dot_bowl"] = (yc == 0) - P[:, 0]
v["r_wkt_bowl"] = f.bowler_wicket.astype(float) - P[:, 6] * share_bw; v["r_extra_bowl"] = (yc == 7) - P[:, 7]
for side in ("bat", "bowl"):
    cols = RL.COMPS[side]; beta, rel = comp[side]["beta"], comp[side]["reliability"]
    v[f"v3_{side}"] = sum(rel[c] * beta[c] * v[c] for c in cols)
    wk = [c for c in cols if "wkt" in c][0]
    v[f"v4_{side}"] = v[f"v1_{side}"] - (1 - rel[wk]) * beta[wk] * v[wk]
v.loc[~faced, ["v3_bat", "v4_bat"]] = np.nan
# league-relative: remove each competition-season's average (scoring levels differ between leagues)
for c in [x for x in v.columns if x[:2] in ("v1", "v2", "v3", "v4", "wp", "r_")]:
    v[c] = v[c] - v.groupby("comp_key")[c].transform("mean")
v["ball_uid"] = 10_000_000 + np.arange(len(v))
save(v, "values_ext")

# ---------------------------------------------------------------- A. external validation
players = load("players").set_index("player_id")
chosen = load_json("p09_summary")["chosen"]
from common import excluded_players
EXCL = excluded_players()
V0 = load("values")
GROUPS = {"all public matches": np.ones(len(v), bool), "T20 World Cups": v.is_wc.eq(1).values,
          "franchise leagues": v.level.isin(["BBL", "PSL", "CPL", "SA20", "ILT20", "MLC", "BPL", "LPL", "Blast"]).values,
          "full-member T20Is (incl. World Cups)": v.level.eq("T20I-full").values}
rows = []
methods = {"CHOSEN (same-type baseline)": {s: load_table(f"final_all_{s}").set_index("player_id").score for s in ("bat", "bowl")}}
try:
    methods["previous type-blind method"] = {s: load_table(f"final_all_{s}_typeblind").set_index("player_id").score for s in ("bat", "bowl")}
except FileNotFoundError:
    pass
for s in ("bat", "bowl"):
    pc = RL.PCOL[s]
    t = V0[V0.is_pressure & (V0.faced if s == "bat" else True)]
    methods.setdefault("raw pressure strike rate / economy (supplied data)", {})[s] = (t.groupby(pc).runs.mean() if s == "bat" else -t.groupby(pc).team_runs.mean())
    methods.setdefault("overall level, no pressure (supplied data)", {})[s] = RL.build_scores(V0, s, dict(chosen[s], model="R2-all", min_n=0)).score
for gname, gm in GROUPS.items():
    for s in ("bat", "bowl"):
        pc = RL.PCOL[s]
        d = v[gm & v.is_pressure & v[f"wpa_{s}"].notna()]
        y = d.groupby(pc).agg(y=(f"wpa_{s}", "mean"), y1=(f"v1_{s}", "mean"), n=(f"wpa_{s}", "size"))
        # common pool for every method: players with a public yardstick and >= 1 supplied-data pressure ball;
        # players a method excludes (e.g. minimum-sample rules) are ranked last, never dropped
        base_ids = set(V0.loc[V0.is_pressure & V0[f"v1_{s}"].notna(), pc].unique())
        y = y[(y.n >= 20) & y.index.isin(players.index) & y.index.isin(base_ids) & ~y.index.isin(EXCL)]   # available players only, for every method
        for mname, sc in methods.items():
            pool = y.index
            if len(pool) < 15:
                continue
            ss, yy = sc[s].reindex(pool).fillna(-1e9), y.loc[pool]
            top = ss.sort_values(ascending=False).index[:10]
            rows.append({"group": gname, "side": s, "method": mname, "players": len(pool),
                         "rank_corr_winprob": weighted_spearman(ss.values, yy.y.values, yy.n.values),
                         "rank_corr_runs": weighted_spearman(ss.values, yy.y1.values, yy.n.values),
                         "top10_winprob_pts_per_100_pressure_balls": 100 * 100 * (np.average(yy.loc[top, "y"], weights=yy.loc[top, "n"]) - np.average(yy.y, weights=yy.n)),
                         "top10_runs_per_100_pressure_balls": 100 * (np.average(yy.loc[top, "y1"], weights=yy.loc[top, "n"]) - np.average(yy.y1, weights=yy.n))})
EV = pd.DataFrame(rows)
save_table(EV, "external_validation")
log.info("External validation (supplied-data ratings -> public-match pressure performance):\n" + EV.round(3).to_string())

# ---------------------------------------------------------------- B. combined ranking (supplied-data players only)
keep = ["ball_uid", "match_id", "year", "level", "striker_id", "bowler_id", "is_pressure", "pressure_pct", "faced", "runs", "team_runs",
        "legal", "phase", "is_pace"] + [c for c in V0.columns if c[:3] in ("v1_", "v2_", "v3_", "v4_")] + sum(RL.COMPS.values(), [])
keep = [c for c in dict.fromkeys(keep) if c in V0.columns and c in v.columns]
VC = pd.concat([V0[keep], v[keep]], ignore_index=True)
comb = {}
for s in ("bat", "bowl"):
    sc = RL.build_scores(VC, s, chosen[s], comp)
    sc = sc[sc.index.isin(players.index) & ~sc.index.isin(EXCL)].copy()
    sc["combined_rank"] = np.arange(1, len(sc) + 1)
    main = load_table(f"final_all_{s}").set_index("player_id")
    sc["name"] = players.name.reindex(sc.index)
    sc["main_rank"] = main.final_rank.reindex(sc.index)
    sc["pressure_balls_supplied"] = main.n_p.reindex(sc.index)
    comb[s] = sc
    sc.head(30).reset_index().rename(columns={"pid": "player_id"}).to_csv(f"{__import__('common').TABLES}/combined_top30_{s}.csv", index=False)
    top_main = set(main.sort_values("final_rank").index[:10]); top_c = set(sc.index[:10])
    log.info(f"{s}: combined-data top 10 shares {len(top_main & top_c)}/10 with the main list\n"
             + sc.head(15)[["name", "combined_rank", "main_rank", "n_p", "pressure_balls_supplied"]].to_string())

# ---------------------------------------------------------------- C. World Cup knockouts (descriptive)
wk = []
for s in ("bat", "bowl"):
    pc = RL.PCOL[s]
    main = load_table(f"final_all_{s}").set_index("player_id").sort_values("final_rank").head(10)
    d = v[(v.is_wc == 1) & (v.knock_out == 1) & v[pc].isin(main.index)]
    if s == "bat":
        d = d[d.faced]
        g = d.groupby(pc).agg(balls=("runs", "size"), runs=("runs", "sum"), outs=("striker_out", "sum"), wpa=("wpa_bat", "sum"))
        g["SR"] = 100 * g.runs / g.balls
    else:
        g = d.groupby(pc).agg(balls=("legal", "sum"), runs=("team_runs", "sum"), wkts=("bowler_wicket", "sum"), wpa=("wpa_bowl", "sum"))
        g["econ"] = 6 * g.runs / g.balls.clip(lower=1)
    g["win_prob_added_pts"] = 100 * g.wpa
    g["name"] = players.name.reindex(g.index); g["side"] = s; g["main_rank"] = main.final_rank.reindex(g.index)
    wk.append(g.reset_index().rename(columns={pc: "player_id"}))
WK = pd.concat(wk)
save_table(WK, "world_cup_knockouts_top10")
log.info("World Cup knockout matches (2022-26 semis/finals) for the top-10s:\n" + WK.round(2).to_string())
# chase share of pressure balls: supplied vs public data, by competition (is the ~90% a property of the data or of one source?)
_cs = [{"dataset": "supplied", "competition": "all", "share_of_pressure_balls_from_chases": float((V0[V0.is_pressure].innings == 2).mean()),
        "share_of_balls_from_chases": float((V0.innings == 2).mean()),
        "share_of_soft_weight_from_chases": float((V0.pressure_pct.clip(1e-3, 1) ** 2)[V0.innings == 2].sum() / (V0.pressure_pct.clip(1e-3, 1) ** 2).sum())},
       {"dataset": "public", "competition": "all", "share_of_pressure_balls_from_chases": float((v[v.is_pressure].innings == 2).mean()),
        "share_of_balls_from_chases": float((v.innings == 2).mean()),
        "share_of_soft_weight_from_chases": float((v.pressure_pct.clip(1e-3, 1) ** 2)[v.innings == 2].sum() / (v.pressure_pct.clip(1e-3, 1) ** 2).sum())}]
for ds_, dd_ in (("supplied", V0), ("public", v)):
    for lv_, g_ in dd_[dd_.is_pressure].groupby("level"):
        _cs.append({"dataset": ds_, "competition": lv_, "share_of_pressure_balls_from_chases": float((g_.innings == 2).mean())})
save_table(pd.DataFrame(_cs), "chase_share_by_dataset")
_dup = len(DUP)
save_json({"duplicate_matches_removed": _dup, "public_pressure_share": float(v.is_pressure.mean()), "public_matches": int(v.match_id.nunique()),
           "public_deliveries": len(v)}, "p14_summary")
log.info("P14 done")
