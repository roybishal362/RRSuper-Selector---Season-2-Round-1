"""Shared paths, config, IO and small cricket helpers for every phase script."""
import os, sys, json, time, logging
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "14")
import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))
RAW = os.path.join(ROOT, CFG["paths"]["raw"])
# RR_OUT lets a full re-run write somewhere else (used to check the notebooks reproduce the results)
OUT = os.environ.get("RR_OUT") or os.path.join(ROOT, CFG["paths"]["out"])
DATA, MODELS, TABLES, FIGS, LOGS = (os.path.join(OUT, d) for d in ("data", "models", "tables", "figures", "logs"))
for _d in (DATA, MODELS, TABLES, FIGS, LOGS):
    os.makedirs(_d, exist_ok=True)

SEED = CFG["seed"]
NJ = CFG["n_jobs"]
np.random.seed(SEED)


def get_logger(name):
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
        fh = logging.FileHandler(os.path.join(LOGS, f"{name}.log"), mode="w", encoding="utf-8")
        sh = logging.StreamHandler(sys.stdout)
        for h in (fh, sh):
            h.setFormatter(fmt)
            log.addHandler(h)
    return log


def raw(name):
    return pd.read_csv(os.path.join(RAW, name + ".csv"), low_memory=False)


def save(df, name):
    df.to_parquet(os.path.join(DATA, name + ".parquet"), index=False)


def load(name, columns=None):
    return pd.read_parquet(os.path.join(DATA, name + ".parquet"), columns=columns)


def save_table(df, name, index=False):
    df.to_csv(os.path.join(TABLES, name + ".csv"), index=index)


def load_table(name, **kw):
    return pd.read_csv(os.path.join(TABLES, name + ".csv"), **kw)


def save_json(obj, name):
    with open(os.path.join(TABLES, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def load_json(name):
    with open(os.path.join(TABLES, name + ".json"), encoding="utf-8") as f:
        return json.load(f)


class Timer:
    def __init__(self, log, label):
        self.log, self.label = log, label

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        self.log.info(f"[{self.label}] {time.time() - self.t:.1f}s")


def split_half_reliability(df, player_col, match_col, value_col, weight_col=None, min_n=20, seed=SEED):
    """Match-level split-half reliability (odd/even matches per player), Spearman-Brown corrected.
    Returns (r_sb, n_players). Players need >= min_n balls in each half."""
    d = df[[player_col, match_col, value_col] + ([weight_col] if weight_col else [])].copy()
    rng = np.random.RandomState(seed)
    mids = d[match_col].unique()
    half = pd.Series(rng.randint(0, 2, len(mids)), index=mids)
    d["half"] = d[match_col].map(half).values
    w = d[weight_col] if weight_col else 1.0
    d["wv"] = d[value_col] * w
    d["w"] = w
    g = d.groupby([player_col, "half"]).agg(s=("wv", "sum"), n=("w", "sum"), c=(value_col, "size"))
    g["m"] = g.s / g.n
    piv = g.unstack("half")
    ok = (piv[("c", 0)] >= min_n) & (piv[("c", 1)] >= min_n)
    piv = piv[ok]
    if len(piv) < 10:
        return np.nan, int(len(piv))
    r = np.corrcoef(piv[("m", 0)], piv[("m", 1)])[0, 1]
    r_sb = 2 * r / (1 + r) if r > -1 else np.nan
    return float(r_sb), int(len(piv))


def bh_fdr(pvals, q=0.05):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1)
    return out


def weighted_spearman(x, y, w):
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    w = np.asarray(w, float)
    mx, my = np.average(rx, weights=w), np.average(ry, weights=w)
    cov = np.average((rx - mx) * (ry - my), weights=w)
    return float(cov / np.sqrt(np.average((rx - mx) ** 2, weights=w) * np.average((ry - my) ** 2, weights=w)))


def excluded_reasons():
    """player_id -> reason for players not available for selection (config 'eligibility'): national sides that are not
    available for IPL selection, and players with no match since 'require_active_from'. They stay in every baseline
    and every method test; only the final lists leave them out."""
    el = CFG.get("eligibility", {})
    pe = load("playing_eleven_clean", columns=["match_id", "team_id", "player_id", "is_played"])
    out = {}
    ids = el.get("exclude_national_team_ids", [])
    if ids:
        for p in pe.loc[pe.team_id.isin(ids), "player_id"].unique():
            out[p] = el.get("exclude_reason", "not available")
    yr = el.get("require_active_from")
    if yr:
        m = load("matches_clean", columns=["match_id", "match_date"])
        played = pe[pe.is_played.astype(bool)].merge(m, on="match_id")
        last = played.groupby("player_id").match_date.max().pipe(pd.to_datetime).dt.year
        for p in last.index[last < yr]:
            out.setdefault(p, el.get("inactive_reason", "inactive"))
    return out


def excluded_players():
    return set(excluded_reasons())
