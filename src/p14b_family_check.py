"""P14b - Evidence for the rating-model family decision (2026-11-12 review), on ONE yardstick, three ways:
 (i) P9's 52 unseen-match splits, (ii) the 2026 season, (iii) the public-data exam (duplicates removed in P14).
CHOSEN = the pressure-weighted settings in p09_summary; ALTERNATIVE = what P9 picked when every model family was allowed
(batters: R2-joint, bowlers: R3). Players not available for selection are left out of the exam pools, as in P14."""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "p09_experiments.py"), encoding="utf-8").read()
head = src[:src.index("chosen = {}")].replace('get_logger("p09_experiments")', 'get_logger("p14b_family_check")')
exec(compile(head, "p09_head", "exec"))   # splits, yardsticks and evaluate() exactly as in P9
from common import load, load_json, save_table, weighted_spearman, excluded_players

EXCL = excluded_players()
_ch = load_json("p09_summary")["chosen"]
CFGS = {
    ("bat", "ALTERNATIVE: all balls (R2-joint)"): {"value": "v3", "model": "R2-joint", "w_d": 0.0, "w_c": 0.0, "k_mult": 2.0, "min_n": _ch["bat"].get("min_n", 0)},
    ("bat", "CHOSEN: pressure-weighted"): _ch["bat"],
    ("bowl", "ALTERNATIVE: pressure balls only (R3)"): {"value": "v1", "model": "R3", "w_d": 0.0, "w_c": 0.0, "k_mult": 2.0, "min_n": _ch["bowl"].get("min_n", 0)},
    # every option uses the same evidence floor as the chosen one, so the comparison is like for like
    ("bowl", "CHOSEN: pressure-weighted"): _ch["bowl"],
}
# the stage-by-stage rule's own pick within the family, where a review decision changed it
_rp = load_json("p09_summary").get("rule_pick", {})
for _s in ("bat", "bowl"):
    if _s in _rp and {k: v for k, v in _rp[_s].items() if k != "min_n"} != {k: v for k, v in _ch[_s].items() if k != "min_n"}:
        CFGS[(_s, f"RULE_PICK: pressure-weighted (a{_rp[_s].get('alpha')}, k{_rp[_s].get('k_mult')})")] = dict(_rp[_s], min_n=_ch[_s].get("min_n", 0))
rows, per = [], {}
# (i) 52 unseen-match splits
for (side, nm), cfg in CFGS.items():
    res = evaluate(cfg, (side,))
    per[(side, nm)] = res
    r = res[res.side == side]
    rows.append({"side": side, "setting": nm, "config": str(cfg), "test": "(i) 52 unseen-match splits",
                 "rank_corr": 0.5 * r[r.kind == "random"].rho_v2.mean() + 0.5 * r[r.kind == "rolling"].rho_v2.mean(),
                 "rank_corr_random50": r[r.kind == "random"].rho_v2.mean(), "rank_corr_rolling2": r[r.kind == "rolling"].rho_v2.mean(),
                 "top10_lift_winprob_pts_per_ball": 100 * r.top10_lift_v2.mean()})
    log.info(str(rows[-1]))
for side in ("bat", "bowl"):
    kc = [k for k in per if k[0] == side and k[1].startswith("CHOSEN")][0]
    a = per[kc].query("kind == 'random'").set_index("split").rho_v2
    for other in ("ALTERNATIVE", "RULE_PICK"):
        ko_ = [k for k in per if k[0] == side and k[1].startswith(other)]
        if not ko_:
            continue
        ko = ko_[0]
        dd = a - per[ko].query("kind == 'random'").set_index("split").rho_v2.reindex(a.index)
        rows.append({"side": side, "setting": f"CHOSEN minus {other}", "test": "(i) paired, 50 random splits",
                     "rank_corr": dd.mean(), "paired_se": dd.std() / np.sqrt(len(dd)), "chosen_better_share": (dd > 0).mean()})
        log.info(str(rows[-1]))

# (ii) the 2026 season (fit on 2022-25)
train, test = V_ALL[V_ALL.year < HOLD], V_ALL[V_ALL.year == HOLD]
for (side, nm), cfg in CFGS.items():
    y = yardstick(test, side)
    sc = RL.build_scores(train, side, cfg, comp_info).score
    tr_ids = set(train.loc[train.is_pressure & train[f"v1_{side}"].notna(), RL.PCOL[side]].unique())
    pool = y.index[y.index.isin(tr_ids)]
    s, yy = sc.reindex(pool).fillna(-1e9), y.loc[pool]
    top = s.sort_values(ascending=False).index[:10]
    rs = np.random.RandomState(SEED); bs = []
    for _ in range(500):
        ix = rs.randint(0, len(pool), len(pool))
        bs.append(weighted_spearman(s.values[ix], yy.y2.values[ix], yy.n.values[ix]))
    rows.append({"side": side, "setting": nm, "config": str(cfg), "test": "(ii) 2026 season", "players": len(pool),
                 "rank_corr": weighted_spearman(s.values, yy.y2.values, yy.n.values),
                 "rank_corr_lo90": np.quantile(bs, 0.05), "rank_corr_hi90": np.quantile(bs, 0.95),
                 "top10_lift_winprob_pts_per_ball": 100 * (np.average(yy.loc[top, "y2"], weights=yy.loc[top, "n"]) - np.average(yy.y2, weights=yy.n))})
    log.info(str(rows[-1]))

# (iii) public-data exam: same pools and yardstick as P14
vx = load("values_ext")
players = load("players").set_index("player_id")
G = {"all public matches": np.ones(len(vx), bool), "T20 World Cups": vx.is_wc.eq(1).values,
     "franchise leagues": vx.level.isin(["BBL", "PSL", "CPL", "SA20", "ILT20", "MLC", "BPL", "LPL", "Blast"]).values,
     "full-member T20Is": vx.level.eq("T20I-full").values}
full_scores = {k: RL.build_scores(V_ALL, k[0], cfg, comp_info).score for k, cfg in CFGS.items()}
for gname, gm in G.items():
    for (side, nm), sc in full_scores.items():
        pc = RL.PCOL[side]
        d = vx[gm & vx.is_pressure & vx[f"wpa_{side}"].notna()]
        y = d.groupby(pc).agg(y=(f"wpa_{side}", "mean"), n=(f"wpa_{side}", "size"))
        base_ids = set(V_ALL.loc[V_ALL.is_pressure & V_ALL[f"v1_{side}"].notna(), pc].unique())
        y = y[(y.n >= 20) & y.index.isin(players.index) & y.index.isin(base_ids) & ~y.index.isin(EXCL)]
        ss = sc.reindex(y.index).fillna(-1e9)
        top = ss.sort_values(ascending=False).index[:10]
        rows.append({"side": side, "setting": nm, "config": str(CFGS[(side, nm)]), "test": f"(iii) public: {gname}", "players": len(y),
                     "rank_corr": weighted_spearman(ss.values, y.y.values, y.n.values),
                     "top10_lift_winprob_pts_per_ball": 100 * (np.average(y.loc[top, "y"], weights=y.loc[top, "n"]) - np.average(y.y, weights=y.n))})
R = pd.DataFrame(rows)
save_table(R, "rating_family_check")
log.info("Rating family check:\n" + R.drop(columns="config").round(4).to_string(index=False))
