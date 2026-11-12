"""P5 - Performance value per ball (4 candidates), batter and bowler mirror versions.
V-1 run value above expected   : (runs + RE_next) - E[runs + RE_next | state]   (a wicket costs its situational run value)
V-2 win-prob added above exp.  : WP_next - E[WP_next | state], de-drifted within WP bins
V-3 component profile          : reliability-weighted run value of boundary / dot / dismissal / extras residuals
V-4 hybrid                     : V-1 with the dismissal part shrunk by the dismissal component's reliability
Batter values use balls faced (no wides); non-striker run-outs are removed from the striker's ledger.
Bowler values use every delivery and team runs (byes included)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from common import load, save, save_table, save_json, get_logger, split_half_reliability

log = get_logger("p05_value")
f = load("features")
c = load("ctx_preds")
p = load("pressure")
for col in c.columns.difference(f.columns):
    f[col] = c[col].values
for col in p.columns.difference(f.columns):
    f[col] = p[col].values

faced = f.batter_faced.values & ~f.nonstriker_out.values
yc = f.y_class.values
v = f[["ball_uid", "match_id", "innings", "year", "level", "lvl_code", "striker_id", "bowler_id", "batting_team_id",
       "bowling_team_id", "venue_id", "batting_order", "is_pressure", "pressure", "pressure_pct", "LI", "strain",
       "wp_now", "phase", "is_pace", "knock_out", "comp_id", "match_date", "runs", "team_runs", "is_wicket",
       "striker_out", "legal", "boundary", "is_wide", "is_noball", "bowler_wicket"]].copy()
v["faced"] = faced

# ---------------- V-1
v["v1_team"] = f.team_runs + f.re_next - f.E_team_rv
v["v1_bat"] = np.where(faced, f.runs + f.re_next - f.E_bat_rv, np.nan)
v["v1_bowl"] = -v.v1_team

# ---------------- V-2 (de-drifted within innings x WP-decile bins)
wpa = f.wp_next - f.E_wp_next
binid = f.innings.astype(str) + "_" + (f.wp_now.clip(0, 0.9999) * 10).astype(int).astype(str)
drift = wpa.groupby(binid).transform("mean")
v["v2_team"] = wpa - drift
v["v2_bat"] = np.where(faced, v.v2_team, np.nan)
v["v2_bowl"] = -v.v2_team
log.info(f"V-2 drift removed: max |bin mean| {wpa.groupby(binid).mean().abs().max():.4f}")

# ---------------- component residuals (vs M1-base expectation)
pb = np.column_stack([f[c_] for c_ in ["p_dot", "p_1", "p_2", "p_3", "p_4", "p_6", "p_wkt"]])
pb = pb / pb.sum(1, keepdims=True)  # faced-ball (no wide) renormalisation for batters
v["r_bdry_bat"] = np.where(faced, np.isin(yc, [4, 5]) - (pb[:, 4] + pb[:, 5]), np.nan)
v["r_dot_bat"] = np.where(faced, (yc == 0) - pb[:, 0], np.nan)
v["r_wkt_bat"] = np.where(faced, f.striker_out.astype(float) - pb[:, 6], np.nan)
v["r_runs_bat"] = np.where(faced, f.runs - f.x_bat_runs, np.nan)
v["r_bdry_bowl"] = np.isin(yc, [4, 5]) - (f.p_4 + f.p_6)
v["r_dot_bowl"] = (yc == 0) - f.p_dot
# bowler-credited wickets only (run-outs are not the bowler's); expected = P(wicket) x share credited to bowlers in
# that phase (run-outs are more common at the death)
share_bw = (f.bowler_wicket.groupby(f.phase).sum() / f.is_wicket.groupby(f.phase).sum()).reindex(f.phase).values
v["r_wkt_bowl"] = f.bowler_wicket.astype(float) - f.p_wkt * share_bw
v["r_extra_bowl"] = (yc == 7) - f.p_x

# run value of each component (regression of V-1 on the residuals) and their reliabilities (all balls)
comps = {"bat": ["r_bdry_bat", "r_dot_bat", "r_wkt_bat"], "bowl": ["r_bdry_bowl", "r_dot_bowl", "r_wkt_bowl", "r_extra_bowl"]}
info = {}
for side, cols in comps.items():
    pcol = "striker_id" if side == "bat" else "bowler_id"
    d = v.dropna(subset=cols + [f"v1_{side}"])
    lr = LinearRegression(fit_intercept=False).fit(d[cols], d[f"v1_{side}"])
    beta = dict(zip(cols, lr.coef_))
    rel = {}
    for cc in cols:
        r, n = split_half_reliability(d, pcol, "match_id", cc, min_n=60)
        rel[cc] = max(0.0, r)
    info[side] = {"beta": beta, "reliability": rel}
    log.info(f"{side}: component run values {({k: round(x, 2) for k, x in beta.items()})}; "
             f"reliabilities {({k: round(x, 2) for k, x in rel.items()})}")
    v[f"v3_{side}"] = sum(rel[cc] * beta[cc] * v[cc] for cc in cols)
    wk = [cc for cc in cols if "wkt" in cc][0]
    v[f"v4_{side}"] = v[f"v1_{side}"] - (1 - rel[wk]) * beta[wk] * v[wk]
v["v3_bat"] = np.where(faced, v.v3_bat, np.nan)
v["v4_bat"] = np.where(faced, v.v4_bat, np.nan)

# ---------------- reliability table of each value metric (split-half, all vs pressure balls)
rows = []
for side, pcol in (("bat", "striker_id"), ("bowl", "bowler_id")):
    for met in ["v1", "v2", "v3", "v4"]:
        col = f"{met}_{side}"
        d = v.dropna(subset=[col])
        r_all, n_all = split_half_reliability(d, pcol, "match_id", col, min_n=60)
        r_p, n_p = split_half_reliability(d[d.is_pressure], pcol, "match_id", col, min_n=20)
        rows.append({"side": side, "metric": met, "rel_all": r_all, "players_all": n_all, "rel_pressure": r_p,
                     "players_pressure": n_p, "mean": d[col].mean(), "sd": d[col].std()})
R = pd.DataFrame(rows)
save_table(R, "value_metric_reliability")
log.info("\n" + R.round(3).to_string())
save(v, "values")
save_json(info, "p05_components")
log.info("P5 done")
