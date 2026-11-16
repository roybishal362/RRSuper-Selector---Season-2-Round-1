"""Player rating models shared by P6 (fit), P9 (validation), P10 (final) and P11 (robustness).
All functions take a ball-level values frame (from p05) and return one row per player.

R-1 adjusted plus-minus (ridge): team value ~ batter + bowler + pressure x (batter + bowler).
    Penalties are the empirical-Bayes noise/skill variance ratios (Bayesian ridge), so no hand tuning.
R-2 empirical Bayes: shrink each player's mean towards the average by n / (n + k), k = sigma^2 / tau^2.
    'p_only'  : pressure balls only
    'joint'   : overall level (all balls) + shrunk pressure-vs-usual gap
    'weighted': every ball weighted by pressure^alpha
R-3 composite: shrunk component residuals (boundary, dot, dismissal, extras) on pressure balls, each turned into runs
    and weighted by its reliability.
"""
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from scipy.sparse.linalg import lsqr

_CFG = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"), encoding="utf-8"))
# where empirical Bayes shrinks each player: 'data_mean' = the real average of the value being rated (its ball-weighted
# mean in the data; 16 Nov 2026 review, config ratings.shrink_to) or 'zero' (the original: the average player = 0)
SHRINK_TO = _CFG.get("ratings", {}).get("shrink_to", "zero")

PCOL = {"bat": "striker_id", "bowl": "bowler_id"}
K_MAX = 5000.0
# Bootstrap support: the shrinkage strength k is recorded on the full data and held fixed inside bootstrap re-runs.
# (Resampling matches adds noise to every player's mean, so re-estimating k there under-shrinks small samples.)
K_RECORD = None
K_FIXED = None
COMPS = {"bat": ["r_bdry_bat", "r_dot_bat", "r_wkt_bat"],
         "bowl": ["r_bdry_bowl", "r_dot_bowl", "r_wkt_bowl", "r_extra_bowl"]}


def side_frame(v, side, metric):
    col = f"{metric}_{side}"
    d = v.loc[v[col].notna()]
    out = pd.DataFrame({"pid": d[PCOL[side]].values, "match_id": d.match_id.values, "val": d[col].values,
                        "pres": d.is_pressure.values.astype(bool), "pw": d.pressure_pct.values})
    return out


def eb_params(sums, ns, s2, min_n=30):
    """method-of-moments between-player variance tau^2 and k = sigma^2 / tau^2."""
    ok = ns >= min_n
    if ok.sum() < 10:
        ok = ns >= max(5, np.quantile(ns, 0.5))
    m = sums[ok] / ns[ok]
    w = ns[ok]
    mu = np.average(m, weights=w)
    var_obs = np.average((m - mu) ** 2, weights=w)
    noise = np.average(s2 / ns[ok], weights=w)
    tau2 = max(var_obs - noise, s2 / K_MAX)   # floor: prior never stronger than K_MAX balls
    return float(tau2), float(s2 / tau2), float(tau2 / var_obs if var_obs > 0 else 0)


def eb_shrink(d, k_mult=1.0, prior=0.0, weight=None, key=None):
    """d has pid, val (+ optional weight col). returns DataFrame pid, n, mean, shrunk, k."""
    if weight is None:
        g = d.groupby("pid").val.agg(["sum", "count"])
        n = g["count"].values.astype(float)
        s = g["sum"].values
    else:
        dd = d.assign(wv=d.val * d[weight], w2=d[weight] ** 2)
        g = dd.groupby("pid").agg(sum=("wv", "sum"), w=(weight, "sum"), w2=("w2", "sum"))
        n = (g.w ** 2 / g.w2).values  # effective n
        s = (g["sum"] / g.w).values * n
    s2 = float(np.var(d.val.values))
    if SHRINK_TO == "data_mean" and prior == 0.0:
        prior = float(np.average(d.val.values, weights=None if weight is None else d[weight].values))
    tau2, k, rel = eb_params(s, n, s2)
    k *= k_mult
    if key is not None and K_FIXED is not None and key in K_FIXED:
        k = K_FIXED[key]
    if key is not None and K_RECORD is not None:
        K_RECORD[key] = k
    mean = s / n
    shr = prior + n / (n + k) * (mean - prior)
    return pd.DataFrame({"pid": g.index.values, "n": n, "mean": mean, "shrunk": shr}).set_index("pid"), k, tau2


def _gaps(d):
    """per player: pressure mean, usual mean, counts and own per-ball variances (shrunk to the global ones)."""
    g = d.groupby(["pid", "pres"]).val.agg(["sum", "count", "var"]).unstack("pres")
    out = pd.DataFrame(index=g.index)
    for flag, nm in ((True, "p"), (False, "u")):
        if ("count", flag) in g:
            out[f"n_{nm}"] = g[("count", flag)].fillna(0).values
            out[f"m_{nm}"] = (g[("sum", flag)] / g[("count", flag)]).values
            out[f"v_{nm}"] = g[("var", flag)].values
        else:
            out[f"n_{nm}"], out[f"m_{nm}"], out[f"v_{nm}"] = 0.0, np.nan, np.nan
    return out


def split_half_cov(d, min_p=15, min_u=30, reps=4, seed=0):
    """true between-player variance of the pressure-minus-usual gap = covariance of the gap between random match
    halves (noise is independent between halves, so it cancels whatever its size or shape)."""
    mids = d.match_id.unique()
    covs = []
    for r in range(reps):
        h = pd.Series(np.random.RandomState(seed + r).randint(0, 2, len(mids)), index=mids)
        hv = d.match_id.map(h).values
        a, b = _gaps(d[hv == 0]), _gaps(d[hv == 1])
        j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
        ok = (j.n_p_a >= min_p) & (j.n_p_b >= min_p) & (j.n_u_a >= min_u) & (j.n_u_b >= min_u)
        j = j[ok]
        if len(j) < 20:
            continue
        ga, gb = j.m_p_a - j.m_u_a, j.m_p_b - j.m_u_b
        covs.append(float(np.cov(ga, gb)[0, 1]))
    return float(np.mean(covs)) if covs else 0.0, len(covs)


def delta_shrink(d, k_mult=1.0, key=None):
    """shrunk pressure-minus-usual gap per player. True gap variance from split-half covariance (clipped at 0);
    each player's noise from his own per-ball variances. The league-wide average gap is removed (it belongs to the
    situation, not the player)."""
    G = _gaps(d)
    s2p = float(np.var(d.val.values[d.pres.values])) if d.pres.any() else float(np.var(d.val.values))
    s2u = float(np.var(d.val.values[~d.pres.values])) if (~d.pres).any() else float(np.var(d.val.values))
    vp = ((G.n_p - 1).clip(lower=0) * G.v_p.fillna(s2p) + 20 * s2p) / ((G.n_p - 1).clip(lower=0) + 20)
    vu = ((G.n_u - 1).clip(lower=0) * G.v_u.fillna(s2u) + 20 * s2u) / ((G.n_u - 1).clip(lower=0) + 20)
    se2 = vp / G.n_p.clip(lower=1) + vu / G.n_u.clip(lower=1)
    gap = (G.m_p - G.m_u)
    both = (G.n_p > 0) & (G.n_u > 0)
    w = 1 / se2[both & (G.n_p >= 30)]
    mu = float(np.average(gap[w.index], weights=w)) if len(w) else 0.0
    tau2, _ = split_half_cov(d)
    tau2 = max(tau2, 0.0)
    # In a match bootstrap a match drawn twice gets two new ids, so identical copies can sit in both halves and the
    # split-half covariance becomes spuriously positive (100x+ inflation). The bootstrap therefore holds tau2 fixed at
    # its full-data value, exactly as it holds the shrinkage strengths k fixed.
    if key is not None and K_FIXED is not None and key in K_FIXED:
        tau2 = K_FIXED[key]
    if key is not None and K_RECORD is not None:
        K_RECORD[key] = tau2
    shrink = np.where(both, tau2 / (tau2 + k_mult * se2), 0.0)
    shr = shrink * (gap.fillna(mu) - mu)
    out = pd.DataFrame({"n_p": G.n_p.values, "n_u": G.n_u.values, "gap_raw": gap.values, "gap_shrunk": shr,
                        "gap_rel": shrink}, index=G.index)
    out.index.name = "pid"
    return out, float(tau2), mu


def r2_ratings(v, side, metric="v1", variant="joint", k_mult=1.0, alpha=1.0):
    d = side_frame(v, side, metric)
    dl, tau_d, mu_d = delta_shrink(d, k_mult, key=("tau_d", side, metric))
    if variant == "p_only":
        e, k, tau2 = eb_shrink(d[d.pres], k_mult, key=("r2", side, metric, variant))
        rating = e.shrunk
    elif variant == "joint":
        e, k, tau2 = eb_shrink(d, k_mult, key=("r2", side, metric, variant))
        share_p = (dl.n_p / (dl.n_p + dl.n_u)).reindex(e.index).fillna(0)
        rating = e.shrunk + (1 - share_p) * dl.gap_shrunk.reindex(e.index).fillna(0)
    elif variant == "all":
        e, k, tau2 = eb_shrink(d, k_mult, key=("r2", side, metric, variant))
        rating = e.shrunk
    elif variant == "weighted":
        d = d.assign(w=(d.pw.clip(1e-3, 1) ** (4 * alpha)))  # emphasis on high-pressure balls grows with alpha
        e, k, tau2 = eb_shrink(d, k_mult, weight="w", key=("r2", side, metric, variant, alpha))
        rating = e.shrunk
    else:
        raise ValueError(variant)
    out = pd.DataFrame({"rating": rating})
    out = out.join(dl[["n_p", "n_u", "gap_raw", "gap_shrunk", "gap_rel"]], how="left")
    out.attrs.update(k=k, tau2=tau2, tau_d2=tau_d, mu_d=mu_d)
    return out


def r1_ratings(v, side, metric="v1", k_mult=1.0, lam=None):
    """Ridge APM with Bayesian penalties. Fits batters and bowlers together on team value; returns `side`."""
    return r1_fit(v, metric, k_mult, lam)[side]


def r1_fit(v, metric="v1", k_mult=1.0, lam=None):
    col = f"{metric}_bowl"
    d = v.loc[v[col].notna(), ["striker_id", "bowler_id", "faced", "is_pressure", col, "match_id"]]
    y = -d[col].values  # team (batting side) value
    y = y - y.mean()
    bats, bi = np.unique(d.striker_id.values, return_inverse=True)
    bwls, wi = np.unique(d.bowler_id.values, return_inverse=True)
    n = len(d)
    fb = d.faced.values.astype(float)
    pr = d.is_pressure.values.astype(float)
    nb, nw = len(bats), len(bwls)
    rows = np.arange(n)
    Xb = sp.csr_matrix((fb, (rows, bi)), shape=(n, nb))
    Xw = sp.csr_matrix((np.ones(n), (rows, wi)), shape=(n, nw))
    Xbp = sp.csr_matrix((fb * pr, (rows, bi)), shape=(n, nb))
    Xwp = sp.csr_matrix((pr, (rows, wi)), shape=(n, nw))
    if lam is None:
        s2 = float(np.var(y))
        db = side_frame(v, "bat", metric); dw = side_frame(v, "bowl", metric)
        _, kb, _ = eb_shrink(db); _, kw, _ = eb_shrink(dw)
        _, tdb, _ = delta_shrink(db); _, tdw, _ = delta_shrink(dw)
        lam = {"bat": kb, "bowl": kw, "bat_p": s2 / max(tdb, 1e-9), "bowl_p": s2 / max(tdw, 1e-9)}
    lam = {k_: min(x * k_mult, 1e7) for k_, x in lam.items()}
    # penalty via column scaling: beta = gamma / sqrt(lam), ridge on gamma with penalty 1
    sc = np.concatenate([np.full(nb, lam["bat"]), np.full(nw, lam["bowl"]), np.full(nb, lam["bat_p"]),
                         np.full(nw, lam["bowl_p"])]) ** -0.5
    X = sp.hstack([Xb, Xw, Xbp, Xwp]).tocsc() @ sp.diags(sc)
    gam = lsqr(X, y, damp=1.0, atol=1e-8, btol=1e-8, iter_lim=3000)[0]
    beta = gam * sc
    main_b, main_w = beta[:nb], beta[nb:nb + nw]
    int_b, int_w = beta[nb + nw:2 * nb + nw], beta[2 * nb + nw:]
    res = {}
    for s_, ids, main, inter in (("bat", bats, main_b, int_b), ("bowl", bwls, -main_w, -int_w)):
        o = pd.DataFrame({"usual": main, "clutch": inter, "rating": main + inter}, index=pd.Index(ids, name="pid"))
        o.attrs.update(lam=lam)
        res[s_] = o
    return res


def r3_ratings(v, side, comp_info, k_mult=1.0):
    """reliability-weighted sum of shrunk component residuals on pressure balls (in runs per ball)."""
    beta, rel = comp_info[side]["beta"], comp_info[side]["reliability"]
    d0 = v.loc[v[f"v1_{side}"].notna() & v.is_pressure]
    tot = None
    for c in COMPS[side]:
        d = pd.DataFrame({"pid": d0[PCOL[side]].values, "val": d0[c].values})
        e, k, _ = eb_shrink(d, k_mult, key=("r3", side, c))
        part = rel[c] * beta[c] * e.shrunk
        tot = part if tot is None else tot.add(part, fill_value=0)
    return pd.DataFrame({"rating": tot})


def consistency(v, side, metric="v1", min_balls=3):
    """per pressure innings / spell: share with positive value (beta-binomial shrunk), 10th pct, mean - SD/sqrt(n)."""
    d = side_frame(v, side, metric)
    d = d[d.pres]
    pm = d.groupby(["pid", "match_id"]).val.agg(["sum", "count"])
    pm = pm[pm["count"] >= min_balls]
    g = pm.groupby("pid")["sum"]
    pos = (pm["sum"] > 0).groupby("pid").sum()
    n = g.count()
    p0 = pos.sum() / n.sum()
    ok = n >= 8
    if ok.sum() > 10:
        r = (pos[ok] / n[ok])
        var_obs = np.average((r - p0) ** 2, weights=n[ok])
        bin_var = np.average(p0 * (1 - p0) / n[ok], weights=n[ok])
        tau2 = max(var_obs - bin_var, 1e-6)
        k = p0 * (1 - p0) / tau2
    else:
        k = 20.0
    out = pd.DataFrame({"n_inns": n, "pos_share": pos / n, "consistency": (pos + k * p0) / (n + k),
                        "p10": g.quantile(0.10), "lcb": g.mean() - g.std().fillna(0) / np.sqrt(n)})
    out.attrs.update(k=k, p0=p0)
    return out


def zscore(s, w=None):
    s = pd.Series(s)
    if w is None:
        return (s - s.mean()) / (s.std() + 1e-12)
    m = np.average(s, weights=w)
    sd = np.sqrt(np.average((s - m) ** 2, weights=w))
    return (s - m) / (sd + 1e-12)


def build_scores(v, side, cfg, comp_info=None):
    """one configuration -> final score per player. cfg keys: value, model, w_d, w_c, k_mult, min_n, alpha."""
    met, model = cfg.get("value", "v1"), cfg.get("model", "R2-joint")
    km = cfg.get("k_mult", 1.0)
    if model == "R1":
        r = r1_ratings(v, side, met, km)
        dl, _, _ = delta_shrink(side_frame(v, side, met), km, key=("tau_d", side, met))
        r = r.join(dl[["n_p", "n_u", "gap_shrunk"]], how="left")
    elif model.startswith("R2"):
        r = r2_ratings(v, side, met, model.split("-")[1], km, cfg.get("alpha", 1.0))
    elif model == "R3":
        r = r3_ratings(v, side, comp_info, km)
        dl, _, _ = delta_shrink(side_frame(v, side, "v1"), km, key=("tau_d", side, "v1"))
        r = r.join(dl[["n_p", "n_u", "gap_shrunk"]], how="left")
    elif model == "blend":
        parts = []
        for mm in cfg.get("blend_of", ["R1", "R2-joint", "R3"]):
            sub = build_scores(v, side, dict(cfg, model=mm, w_d=0, w_c=0, min_n=0), comp_info)
            parts.append(sub.score.rank(pct=True).rename(mm))
        P = pd.concat(parts, axis=1)
        base = build_scores(v, side, dict(cfg, model="R2-joint", w_d=0, w_c=0, min_n=0), comp_info)
        r = base[["n_p", "n_u", "gap_shrunk"]].copy()
        r["rating"] = P.mean(1).reindex(r.index)
    else:
        raise ValueError(model)
    r["n_p"] = r.n_p.fillna(0)
    r = r[r.n_p >= max(1, cfg.get("min_n", 0))].copy()
    w_d, w_c = cfg.get("w_d", 0.0), cfg.get("w_c", 0.0)
    score = zscore(r.rating)
    if w_d:
        score = score + w_d * zscore(r.gap_shrunk.fillna(0))
    if w_c:
        cs = consistency(v, side, met)
        score = score + w_c * zscore(cs.consistency.reindex(r.index).fillna(cs.attrs["p0"]))
    r["score"] = score
    return r.sort_values("score", ascending=False)
