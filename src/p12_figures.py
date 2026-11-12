"""P12a - Report figures (PNG, light theme) built only from the saved result tables.
Palette: validated reference instance (categorical slots in fixed order, one-hue sequential blue, recessive chrome)."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from common import FIGS, load_table, load_json, get_logger

log = get_logger("p12_figures")
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
GRAY = "#c3c2b7"
SEQ = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 9.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURF, "figure.facecolor": SURF,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.axisbelow": True})


def save(fig, name):
    fig.savefig(f"{FIGS}/{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    log.info(f"figure {name}")


# 1 pressure heatmaps (chase and first innings): over x wickets, mean pressure percentile
for nm, title in (("pressure_heatmap_chase", "Chasing: how much pressure (average percentile)"),
                  ("pressure_heatmap_first", "Setting a total: how much pressure (average percentile)")):
    H = load_table(nm).set_index("over_no")
    H = H[[c for c in H.columns if c in [str(i) for i in range(0, 10)]]]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    im = ax.imshow(H.T.values * 100, aspect="auto", cmap=SEQ, vmin=0, vmax=100, origin="lower")
    ax.set_xticks(range(len(H.index))); ax.set_xticklabels(H.index.astype(int))
    ax.set_yticks(range(H.shape[1])); ax.set_yticklabels(H.columns)
    ax.set_xlabel("Over"); ax.set_ylabel("Wickets down"); ax.set_title(title, loc="left"); ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("pressure percentile", color=INK2); cb.outline.set_visible(False)
    save(fig, nm)

# 2 pointer tests: dismissal shift per pointer, survivors highlighted
T = load_table("pressure_pointer_tests")
T = T.sort_values("rel_wkt")
fig, ax = plt.subplots(figsize=(7.2, 8.5))
col = np.where(T.verdict.str.startswith("PRESSURE"), S1, GRAY)
ax.barh(range(len(T)), 100 * T.rel_wkt, color=col, height=0.62)
ax.set_yticks(range(len(T))); ax.set_yticklabels([f"{i}  {p[:44]}" for i, p in zip(T.id, T.pointer)], fontsize=8)
ax.axvline(0, color=AXIS, lw=0.8)
ax.set_xlabel("Change in dismissal rate vs baseline (%)")
ax.set_title("Pressure pointers: does an average player get out more?", loc="left")
ax.text(0.99, 0.02, "blue = passed all required tests (used)\ngrey = dropped or control", transform=ax.transAxes,
        ha="right", va="bottom", color=INK2, fontsize=8)
save(fig, "pointer_tests")

# 3 definitions: stakes vs strain
A = load_table("pressure_definition_summary")
p4 = load_json("p04_summary")
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for _, r in A.iterrows():
    ch = r.definition == p4["chosen_definition"]
    ax.scatter(r.stakes, r.strain, s=70 if ch else 40, color=S2 if ch else S1, edgecolor=SURF, linewidth=2, zorder=3)
    ax.annotate(r.definition, (r.stakes, r.strain), xytext=(6, 3), textcoords="offset points", fontsize=8,
                color=INK if ch else INK2, fontweight="bold" if ch else "normal")
ax.set_xlabel("Stakes: win-probability movement on pressure balls (x normal)")
ax.set_ylabel("Strain: dismissals vs stage normal (x)")
ax.set_title("Pressure definitions compared (averaged over 10-30% cuts)", loc="left")
save(fig, "pressure_definitions")

# 4 reliability of value metrics
R = load_table("value_metric_reliability")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, side in zip(axes, ("bat", "bowl")):
    d = R[R.side == side]
    x = np.arange(len(d))
    ax.bar(x - 0.18, d.rel_all, 0.34, color=S1, label="all balls")
    ax.bar(x + 0.18, d.rel_pressure, 0.34, color=S2, label="pressure balls")
    ax.set_xticks(x); ax.set_xticklabels(["V-1 runs", "V-2 win prob", "V-3 components", "V-4 hybrid"], fontsize=8)
    ax.set_title("Batters" if side == "bat" else "Bowlers", loc="left"); ax.axhline(0, color=AXIS, lw=0.8)
axes[0].set_ylabel("split-half reliability"); axes[0].legend(frameon=False, fontsize=8)
save(fig, "value_reliability")

# 5 experiment results
E = load_table("experiment_results")
ch = load_json("p09_summary")["chosen"]
def short(r):
    st = r.stage[0]
    if st == "A": return f"A  value {r.value}"
    if st == "B": return f"B  {r.model}" + (f" a={r.alpha:g}" if r.model == "R2-weighted" else "")
    if st == "C": return f"C  w_d={r.w_d:g}  w_c={r.w_c:g}"
    return f"D  k x{r.k_mult:g}  min {int(r.min_n)} balls"
fig, axes = plt.subplots(1, 2, figsize=(8.6, 9.6))
for ax, side in zip(axes, ("bat", "bowl")):
    d = E[E.side == side].reset_index(drop=True)
    y = np.arange(len(d))[::-1]
    isch = (d.config == str(ch[side])) & (d.stage.str.startswith("D"))
    ax.hlines(y, 0, d.objective, color=GRID, lw=1)
    ax.scatter(d.objective, y, color=np.where(isch, S2, S1), s=np.where(isch, 46, 18), zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([short(r) for _, r in d.iterrows()], fontsize=7)
    ax.set_title("Batters" if side == "bat" else "Bowlers", loc="left")
    ax.set_xlabel("out-of-sample rank correlation")
    ax.axvline(0, color=AXIS, lw=0.8)
fig.text(0.01, -0.01, "Each row = one configuration scored on matches the method never saw (52 splits). Orange = final choice. "
         "A value metric, B rating model, C gap/consistency weights, D shrinkage & minimum sample.", fontsize=7.5, color=INK2)
fig.tight_layout()
save(fig, "experiments")

# 6 forest plots of the top-20 with bootstrap CI and P(top-10)
for side, title in (("bat", "Top 20 batters under pressure"), ("bowl", "Top 20 bowlers under pressure")):
    F = load_table(f"final_top30_{side}").head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    y = np.arange(len(F))
    top10 = F.final_rank <= 10
    ax.hlines(y, F.rating_lo90, F.rating_hi90, color=np.where(top10, S1, GRAY), lw=2)
    ax.scatter(F.rating, y, color=np.where(top10, S1, GRAY), s=28, zorder=3, edgecolor=SURF, linewidth=1.5)
    ax.set_yticks(y); ax.set_yticklabels([f"{int(r)}. {n}" for r, n in zip(F.final_rank, F.name)], fontsize=8.5)
    for yy, (_, r) in zip(y, F.iterrows()):
        ax.text(1.01, yy, f"{100 * r.p_top10:.0f}%", transform=ax.get_yaxis_transform(), va="center", fontsize=8, color=INK2)
    ax.text(1.01, len(F) - 0.1, "P(top-10)", transform=ax.get_yaxis_transform(), fontsize=8, color=MUTED)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_xlabel("pressure rating, value per ball above an average player (90% bootstrap interval)")
    ax.set_title(title, loc="left")
    save(fig, f"forest_{side}")

# 7 rise or dip in cricket units (top 30; top 10 numbered by final rank)
for side in ("bat", "bowl"):
    F = load_table(f"final_top30_{side}")
    xc, yc_, unit = (("SR_usual", "SR_pressure", "strike rate") if side == "bat" else ("econ_usual", "econ_pressure", "economy"))
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    lo, hi = min(F[xc].min(), F[yc_].min()), max(F[xc].max(), F[yc_].max())
    pad = (hi - lo) * 0.06
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=AXIS, lw=1)
    rest, top = F[F.final_rank > 10], F[F.final_rank <= 10]
    ax.scatter(rest[xc], rest[yc_], s=14, color=GRAY)
    ax.scatter(top[xc], top[yc_], s=34, color=S1, edgecolor=SURF, linewidth=1.5, zorder=3)
    for _, r in top.iterrows():
        ax.annotate(str(int(r.final_rank)), (r[xc], r[yc_]), xytext=(4, 3), textcoords="offset points", fontsize=8, color=INK)
    better = "above" if side == "bat" else "below"
    ax.set_xlabel(f"usual {unit} (non-pressure balls)"); ax.set_ylabel(f"{unit} on pressure balls")
    ax.set_title(("Top-30 batters" if side == "bat" else "Top-30 bowlers") + f": pressure vs usual {unit}", loc="left")
    ax.text(0.02, 0.97, f"{better} the line = better under pressure than usual; numbers = final rank", transform=ax.transAxes,
            va="top", fontsize=7.5, color=INK2)
    save(fig, f"rise_dip_{side}")

# 8 sensitivity: top-10 overlap per variant
S = load_table("sensitivity_matrix")
for side in ("bat", "bowl"):
    d = S[S.side == side].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.6, 0.22 * len(d) + 1.2))
    y = np.arange(len(d))[::-1]
    ax.barh(y, d.top10_overlap, color=S1, height=0.6)
    NM = {"CA": "C-A leverage (chosen)", "CB": "C-B leverage x big match", "CB2": "C-B2 leverage x all big-match flags",
          "CC": "C-C stakes x strain", "CD": "C-D PCA factor", "CE": "C-E GMM regimes",
          "CF_Bhattacharjee-Lemmer": "C-F Bhattacharjee-Lemmer (published)", "CF_Thomson": "C-F Thomson zone (published)",
          "CG": "C-G simple rules"}
    ax.set_yticks(y); ax.set_yticklabels([f"{g}: {NM.get(v_, v_)}" for g, v_ in zip(d.group, d.variant)], fontsize=7.5)
    ax.set_xlim(0, 10); ax.set_xlabel("players shared with the final top-10 (out of 10)")
    ax.set_title(("Batters" if side == "bat" else "Bowlers") + ": how stable is the top-10?", loc="left")
    save(fig, f"sensitivity_{side}")

# 9 win-probability reliability of the models actually used (chase: logistic with curves for game state; first innings: Plan B)
from common import load
cp = load("ctx_preds", columns=["wp_now", "bat_won", "innings", "irregular"])
cp = cp[cp.bat_won.isin([0, 1]) & ~cp.irregular]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
for ax, inn, title in ((axes[0], 2, "Chase (logistic, curves for game state)"), (axes[1], 1, "First innings (Plan B)")):
    d = cp[cp.innings == inn]
    b_ = (d.wp_now * 10).clip(0, 9.999).astype(int)
    r = d.groupby(b_).agg(pred=("wp_now", "mean"), obs=("bat_won", "mean"))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1)
    ax.plot(r.pred, r.obs, color=S1, lw=2, marker="o", ms=4)
    ax.set_xlabel("predicted win probability"); ax.set_title(title, loc="left")
axes[0].set_ylabel("actual win rate")
save(fig, "wp_calibration")

# 9b model test: error on training matches vs unseen matches (the over-fitting gap), and a learning curve
import os as _os
from common import TABLES as _TB
if _os.path.exists(f"{_TB}/model_zoo.csv") and _os.path.exists(f"{_TB}/wp_logistic_designs.csv"):
    REN = {"Spline logistic (CURRENT)": "Spline logistic, curves on every input (previous)",
           "Spline logistic, curves on every input (previous)": "Spline logistic, curves on every input (previous)"}
    mz = load_table("model_zoo")
    mz = mz[mz.target == "chase"][["model", "train_logloss", "test_logloss"]].copy()
    mz["model"] = mz.model.replace(REN)
    ld = load_table("wp_logistic_designs")
    chz = ld[ld.model.str.startswith("CHOSEN")].iloc[0]
    mz = pd.concat([mz, pd.DataFrame([{"model": "CHOSEN: logistic, curves for game state only",
                                        "train_logloss": chz.train_logloss, "test_logloss": chz.test_logloss}])])
    mz = mz.sort_values("test_logloss", ascending=False).reset_index(drop=True)
    has_lc = _os.path.exists(f"{_TB}/model_zoo_learning_curve.csv")
    if has_lc:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.8), gridspec_kw={"width_ratios": [1.45, 1]})
    else:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
    y = np.arange(len(mz))
    ax.hlines(y, mz.train_logloss, mz.test_logloss, color=AXIS, lw=1.2, zorder=1)
    ax.scatter(mz.train_logloss, y, s=34, facecolor=SURF, edgecolor=MUTED, linewidth=1.4, zorder=3, label="training matches")
    ax.scatter(mz.test_logloss, y, s=38, color=INK2, zorder=3, label="unseen matches", edgecolor=SURF, linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(mz.model, fontsize=8.2)
    for t in ax.get_yticklabels():
        if t.get_text().startswith("CHOSEN"):
            t.set_fontweight("bold"); t.set_color(INK)
    top = len(mz) - 1
    ax.annotate("unseen", (mz.test_logloss.iloc[top], top), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
    ax.annotate("training", (mz.train_logloss.iloc[top], top), xytext=(-6, 0), textcoords="offset points", va="center", ha="right", fontsize=8, color=INK2)
    ax.set_xlabel("log-loss (lower is better)")
    ax.set_title("Chase win chance: training vs unseen matches", loc="left")
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    ax.grid(axis="y", visible=False)
    if has_lc:
        lc = load_table("model_zoo_learning_curve")
        lc = lc[lc.target == "chase"].copy()
        lc["model"] = lc.model.replace(REN).replace({"Spline logistic, curves on every input (previous)": "Spline logistic (previous)"})
        order = list(dict.fromkeys(lc.model))
        cols = [S1, S2, S3, "#eda100"]
        ends = []
        for i, mname in enumerate(order):
            d_ = lc[lc.model == mname].sort_values("train_matches")
            ax2.plot(d_.train_matches, d_.test_logloss, color=cols[i % 4], lw=2, marker="o", ms=5, label=mname,
                     markeredgecolor=SURF, markeredgewidth=1.2)
            ends.append([d_.test_logloss.iloc[-1], mname, d_.train_matches.iloc[-1]])
        ends.sort()
        ymin, ymax = ax2.get_ylim()
        gap = (ymax - ymin) * 0.06
        for j in range(1, len(ends)):
            if ends[j][0] - ends[j - 1][0] < gap:
                ends[j][0] = ends[j - 1][0] + gap
        xmax = max(e[2] for e in ends)
        true_end = {mname: lc[lc.model == mname].sort_values("train_matches").test_logloss.iloc[-1] for mname in order}
        for yv, mname, xv in ends:
            ax2.annotate(mname, xy=(xv, true_end[mname]), xytext=(xmax * 1.12, yv), textcoords="data", va="center",
                         fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkA=2, shrinkB=4))
        ax2.set_xlim(right=xmax * 1.75)
        ax2.set_xlabel("training matches")
        ax2.set_ylabel("log-loss on unseen matches")
        ax2.set_title("Chase: error on unseen matches as training data grows", loc="left")
        ax2.legend(loc="upper right", frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, "model_zoo")

# 10 best pace bowlers (transparency: spinners dominate the main list)
tp = load_table("top_pace_bowlers").iloc[::-1]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.barh(range(len(tp)), tp.score, color=S1, height=0.6)
ax.set_yticks(range(len(tp))); ax.set_yticklabels([f"{n} (overall #{int(r)})" for n, r in zip(tp.name, tp.final_rank)], fontsize=8.5)
ax.set_xlabel("final score (same scale as the main bowling list)")
ax.set_title("Best pace bowlers under pressure", loc="left")
save(fig, "top_pace_bowlers")

# 11 independent exam on public data (same players for every method)
import os
from common import TABLES
if os.path.exists(f"{TABLES}/external_validation.csv"):
    EV = load_table("external_validation")
    EV = EV[EV.group == "all public matches"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharex=True)
    for ax, side in zip(axes, ("bat", "bowl")):
        d = EV[EV.side == side].reset_index(drop=True).iloc[::-1]
        col = np.where(d.method.str.startswith("CHOSEN"), S1, GRAY)
        ax.barh(range(len(d)), d.top10_winprob_pts_per_100_pressure_balls, color=col, height=0.6)
        ax.set_yticks(range(len(d))); ax.set_yticklabels([m.replace(" (supplied data)", "") for m in d.method], fontsize=8)
        ax.axvline(0, color=AXIS, lw=0.8)
        ax.set_title("Batters" if side == "bat" else "Bowlers", loc="left")
        ax.set_xlabel("top-10 vs average: win-% points per 100 pressure balls")
    fig.suptitle("Exam on ~4,000 public matches the method never saw", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "external_exam")

log.info("P12 figures done")
