"""P16 - The Round 1 submission as a vertical slide deck (A4 portrait). Structure follows the brief's five report
items (pressure definition, performance measurement, fair comparisons, evidence for the top 10, reliability and
limitations), plus the top picks and how each could strengthen RR (Unstop page).
Every number in cards, tables and charts is read from the final result tables or computed here from the ball data;
numbers in the text were checked against the same tables. Output: report/RR_Round1_Slides.pdf + PNG previews."""
import io, os, json, subprocess, html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from common import ROOT, load, load_table, get_logger

log = get_logger("p16_slides_pdf")
OUT = os.path.join(ROOT, "report")
A = os.path.join(OUT, "assets")
HTML_PATH, PDF_PATH = os.path.join(OUT, "RR_Round1_Slides.html"), os.path.join(OUT, "RR_Round1_Slides.pdf")
PREV = os.path.join(OUT, "slides_previews")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
URL = lambda p: "file:///" + p.replace("\\", "/")
LOGO_W = URL(os.path.join(A, "rr_logo_white.png"))
IPL = json.load(open(os.path.join(A, "players", "ipl", "ipl_photos.json")))
WIKI = json.load(open(os.path.join(A, "players", "credits.json"), encoding="utf-8"))
AUTHOR = "Bishal Roy"
EMAIL = "roybishal9989@gmail.com"
GITHUB = "github.com/roybishal362/RRSuper-Selector---Season-2-Round-1"
SECTIONS = ["Pressure definition", "Performance measurement", "Fair comparisons", "Evidence for the top 10", "Reliability and limitations"]

PINK, NAVY, GREY, INK2, MUTED, RULE = "#e2178f", "#011c45", "#a9a8a2", "#4a4a52", "#7c7c84", "#e7e2ea"
SEQ = LinearSegmentedColormap.from_list("p", ["#fdf0f7", "#f9c9e2", "#f08cc0", "#e2178f", "#a50f68", "#5c0839"])
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 9, "svg.fonttype": "none", "axes.edgecolor": RULE,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": NAVY, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "none", "axes.facecolor": "none"})
esc = html.escape


FIGS = os.path.join(ROOT, "docs", "figures")


def svg(fig, name=None):
    """inline SVG for the deck; with a name, also a PNG copy for the GitHub README (charts only, no photos or logos)"""
    if name:
        os.makedirs(FIGS, exist_ok=True)
        fig.savefig(os.path.join(FIGS, name + ".png"), dpi=200, bbox_inches="tight", facecolor="white")
    b = io.StringIO()
    fig.savefig(b, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    s = b.getvalue()
    return s[s.index("<svg"):]


T = load_table
bat, bowl = T("final_top30_bat"), T("final_top30_bowl")
bat10, bowl10 = bat.head(10).copy(), bowl.head(10).copy()
EV, BR, EXU, DEFS = T("external_validation"), T("best_by_role"), T("excluded_unavailable"), T("pressure_definition_summary")

ROLE = {"Opener": "Opener", "Top order (No. 3-4)": "No. 3–4", "Middle order (No. 5-6)": "No. 5–6", "Finisher": "Finisher",
        "Pace - new ball": "Pace · new ball", "Pace - new ball + death": "Pace · new ball & death", "Pace - death": "Pace · death",
        "Pace - middle overs": "Pace · middle overs", "Spin - middle overs": "Spin · middle overs", "Spin - powerplay": "Spin · powerplay",
        "Lower order": "Lower order"}
TEAM = {"Abhishek Sharma": "SRH · India", "Suryakumar Yadav": "MI · India", "Vaibhav Sooryavanshi": "RR · India",
        "Phil Salt": "RCB · England", "Tim David": "RCB · Australia", "Yashasvi Jaiswal": "RR · India",
        "Heinrich Klaasen": "SRH · South Africa", "Rovman Powell": "KKR · West Indies", "Josh Inglis": "LSG · Australia",
        "Priyansh Arya": "PBKS", "Jasprit Bumrah": "MI · India", "Lungi Ngidi": "DC · South Africa", "Josh Hazlewood": "RCB · Australia",
        "Benjamin Sears": "New Zealand", "Nathan Ellis": "CSK · Australia", "Rashid Khan": "GT · Afghanistan", "Sunil Narine": "KKR",
        "Brydon Carse": "England", "Mitchell Santner": "MI · New Zealand", "Jamie Overton": "CSK · England"}
# ----------------------------------------------------------------------------- RR's own pressure record, IPL 2025-26
# (for the 'For RR' line on each card: where RR were below a typical team under pressure, by job)
RR_ID = 3
_v = load("values", columns=["ball_uid", "year", "level", "batting_team_id", "bowling_team_id", "is_pressure", "phase", "is_pace",
                             "runs", "team_runs", "legal", "faced", "striker_out", "bowler_wicket", "batting_order"])
_v = _v[(_v.level == "IPL") & (_v.year >= 2025) & (_v.is_pressure == 1)]
_v = _v.merge(load("ctx_preds", columns=["ball_uid", "x_bat_runs", "p_1", "p_2", "p_3", "p_4", "p_6", "p_x"]), on="ball_uid")
_v["e_team"] = _v.p_1 + 2 * _v.p_2 + 3 * _v.p_3 + 4 * _v.p_4 + 6 * _v.p_6 + _v.p_x
_v["e_legal"] = 1 - _v.p_x
_allv = load("values", columns=["ball_uid", "is_pace", "team_runs", "legal"]).merge(
    load("ctx_preds", columns=["ball_uid", "p_1", "p_2", "p_3", "p_4", "p_6", "p_x"]), on="ball_uid")
_allv["e_team"] = _allv.p_1 + 2 * _allv.p_2 + 3 * _allv.p_3 + 4 * _allv.p_4 + 6 * _allv.p_6 + _allv.p_x
_cal = (_allv.groupby("is_pace").team_runs.sum() / _allv.groupby("is_pace").legal.sum()) / \
       (_allv.groupby("is_pace").e_team.sum() / (1 - _allv.p_x).groupby(_allv.is_pace).sum())
_v["e_team"] = _v.e_team * _v.is_pace.map(_cal).values
del _allv
_v["slot"] = pd.cut(_v.batting_order, [0, 2, 4, 6, 11], labels=["open", "3-4", "5-6", "7+"])
ORD = lambda r: f"{int(r)}{'th' if 10 <= int(r) % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(int(r) % 10, 'th')}"


def _team_bat(key):
    g = _v[_v.faced == 1].groupby(["batting_team_id", key], observed=True)
    t = pd.DataFrame({"sr": 100 * g.runs.mean(), "typ": 100 * g.x_bat_runs.mean()}).reset_index()
    t["rank"] = (t.sr - t.typ).groupby(t[key]).rank(ascending=False)
    return t[t.batting_team_id == RR_ID].set_index(key)


def _team_bowl(keys):
    g = _v.groupby(["bowling_team_id"] + keys, observed=True)
    t = pd.DataFrame({"econ": 6 * g.team_runs.sum() / g.legal.sum(), "typ": 6 * g.e_team.sum() / g.e_legal.sum(),
                      "bpw": g.legal.sum() / g.bowler_wicket.sum().clip(lower=1)}).reset_index()
    t["rank"] = (t.typ - t.econ).groupby([t[k] for k in keys]).rank(ascending=False)
    t["rank_bpw"] = t.bpw.groupby([t[k] for k in keys]).rank()
    return t[t.bowling_team_id == RR_ID].set_index(keys)


RB_slot, RB_phase = _team_bat("slot"), _team_bat("phase")
RW_phase, RW_type = _team_bowl(["phase"]), _team_bowl(["phase", "is_pace"])
R = dict(open_rank=ORD(RB_slot.loc["open", "rank"]), s34=RB_slot.loc["3-4", "sr"], t34=RB_slot.loc["3-4", "typ"],
         r34=ORD(RB_slot.loc["3-4", "rank"]), rdb=ORD(RB_phase.loc[2, "rank"]),
         pp_bpw=RW_type.loc[(0, 1), "bpw"], pp_bpw_rank=ORD(RW_type.loc[(0, 1), "rank_bpw"]),
         death_pace=RW_type.loc[(2, 1), "econ"], spin_gap=RW_type.loc[(1, 0), "econ"] - RW_type.loc[(1, 0), "typ"],
         mid_rank=ORD(RW_phase.loc[1, "rank"]))
log.info(f"RR 2025-26 pressure record used on cards: { {k: (round(v, 2) if isinstance(v, float) else v) for k, v in R.items()} }")
RRFIT = {
    "Abhishek Sharma": "RR's openers were already the IPL's best under pressure. His middle-overs strike rate of 192 would suit No. 3, RR's weakest slot.",
    "Suryakumar Yadav": f"RR's No. 3–4 batters scored at {R['s34']:.0f} under pressure, against {R['t34']:.0f} typical: {R['r34']} of 10 teams. He fills that exact slot.",
    "Vaibhav Sooryavanshi": "Already an RR player. With Jaiswal he made RR's openers the best of the 10 IPL teams under pressure. Keep him.",
    "Phil Salt": "Opening is covered, but he gets out only once every 21 pressure balls, which would steady No. 3, RR's weakest slot.",
    "Tim David": f"RR's death-overs hitting under pressure ranked {R['rdb']} of 10. He needs no time to settle, which is what a No. 6 or 7 needs.",
    "Yashasvi Jaiswal": "Already an RR player. Keeping him keeps the IPL's best opening pair under pressure together.",
    "Heinrich Klaasen": f"Moved up to No. 4, he would cover the slot where RR's batters ranked {R['r34']} of 10 under pressure.",
    "Rovman Powell": f"A second finisher, for a death-overs batting unit that ranked {R['rdb']} of 10 under pressure. He played for RR in 2024.",
    "Josh Inglis": f"A No. 3–4 who keeps a chase moving, for the slot where RR ranked {R['r34']} of 10 under pressure.",
    "Priyansh Arya": "Opening cover rather than a need. On the fewest pressure balls of the ten (138), a lower-priority target.",
    "Jasprit Bumrah": f"RR's pacers went at {R['death_pace']:.1f} an over at the death under pressure, no better than typical. He saves 1.8 an over under pressure.",
    "Lungi Ngidi": f"RR's new-ball pace was tight but took a wicket only every {R['pp_bpw']:.0f} pressure balls ({R['pp_bpw_rank']} of 10). He takes one every 18.",
    "Josh Hazlewood": f"The best fix for RR's new-ball problem: a wicket every 11 pressure balls, where RR's pacers needed {R['pp_bpw']:.0f}.",
    "Benjamin Sears": f"Middle overs were RR's weakest bowling phase under pressure ({R['mid_rank']} of 10 for runs saved). He brings wickets and dots there.",
    "Nathan Ellis": f"A second fix at the death: 8.4 an over under pressure, where a typical pacer goes at 10.0 and RR's went at {R['death_pace']:.1f}.",
    "Rashid Khan": f"RR's spinners went {R['spin_gap']:.1f} an over above typical in pressure middle overs. Rashid goes 1.1 below it.",
    "Sunil Narine": "The same middle-overs gap. He saves more runs an over under pressure than any other spinner in the list.",
    "Brydon Carse": "Middle-overs pace that takes wickets, for RR's weakest bowling phase. A back-up target.",
    "Mitchell Santner": "Fills the middle-overs spin gap and takes wickets: one every 12 pressure balls.",
    "Jamie Overton": "Middle-overs pace for RR's weakest bowling phase. A back-up target rather than a first choice."}

HEAD = {"Abhishek Sharma": "Attacks from the first ball", "Suryakumar Yadav": "Scores fast, rarely gets out",
        "Vaibhav Sooryavanshi": "The fastest scorer under pressure", "Phil Salt": "Quick and hard to remove",
        "Tim David": "Fast from his first ball", "Yashasvi Jaiswal": "The hardest opener to dismiss",
        "Heinrich Klaasen": "Big hitting, rarely out", "Rovman Powell": "A boundary hitter", "Josh Inglis": "Keeps a chase moving",
        "Priyansh Arya": "Explosive at the top", "Jasprit Bumrah": "The surest pick on either list",
        "Lungi Ngidi": "Saves a run an over, new ball and death", "Josh Hazlewood": "Wickets when they matter most",
        "Benjamin Sears": "A middle-overs wicket-taker", "Nathan Ellis": "Control at the death",
        "Rashid Khan": "The most pressure overs, still cheap", "Sunil Narine": "The hardest bowler to score off",
        "Brydon Carse": "Wickets over economy", "Mitchell Santner": "Cheap, and takes wickets", "Jamie Overton": "Big numbers, small sample"}
WHY = {"Abhishek Sharma": "Powerplay strike rate 185, where a typical opener manages 142.",
       "Suryakumar Yadav": "Most pressure balls of any batter here. Scores off 73% of balls in a chase's middle overs.",
       "Vaibhav Sooryavanshi": "Strike rate 236 under pressure in just two seasons. Held back a little for his smaller sample.",
       "Phil Salt": "Out only once every 21 pressure balls while scoring at 176. Best at the death.",
       "Tim David": "Strike rate 164 in his first 10 balls. A typical finisher manages 135.",
       "Yashasvi Jaiswal": "Out once every 25 pressure balls, the best of any opener here.",
       "Heinrich Klaasen": "Strike rate 200 under pressure. 52 off 27 in T20 World Cup semis and finals.",
       "Rovman Powell": "One in four pressure balls goes for four or six. Typical: under one in five.",
       "Josh Inglis": "Scores off 72% of balls in the middle overs of a chase.",
       "Priyansh Arya": "Powerplay strike rate 188, where a typical opener manages 142.",
       "Jasprit Bumrah": "One in six death balls is a yorker. 9 wickets at 5.45 in World Cup semis and finals.",
       "Lungi Ngidi": "A wicket every 17–18 pressure balls, without relying on yorkers.",
       "Josh Hazlewood": "The best wicket rate here, and almost half his powerplay balls are dots.",
       "Benjamin Sears": "Economy only average, but a wicket every 16 balls and far more dots than usual.",
       "Nathan Ellis": "Also good with the new ball: powerplay economy 7.7, where the league goes at 8.5.",
       "Rashid Khan": "491 pressure balls, more than anyone here, and still saves a run an over.",
       "Sunil Narine": "Economy 7.1 under pressure against 8.8 typical, all of it in the IPL.",
       "Brydon Carse": "A wicket every 16 pressure balls. Only 84 balls, so a closer call.",
       "Mitchell Santner": "Economy 7.3 against 8.5 typical, and a wicket every 12 pressure balls.",
       "Jamie Overton": "Economy 7.5 against 9.4 typical, but on only 64 pressure balls. The closest call of the ten."}
WPOS = {"Brydon Carse": "50% 18%", "Benjamin Sears": "50% 12%"}
for n in list(bat10.name) + list(bowl10.name):
    if n not in WHY:
        log.warning(f"no card text for {n}")


def photo_style(name, face=False):
    if name in IPL:
        u = URL(os.path.join(A, "players", "ipl", IPL[name]))
        return f"background-image:url('{u}');background-size:{'175%' if face else 'cover'};background-position:{'50% 7%' if face else '50% 0%'};"
    if name in WIKI:
        u = URL(os.path.join(A, "players", WIKI[name]["file"]))
        return f"background-image:url('{u}');background-size:cover;background-position:{WPOS.get(name, '50% 20%')};"
    return ""


# ----------------------------------------------------------------------------- charts
# 1. the last over of the 2023 IPL final (chasing side's win chance before each ball, and after the last)
f_ = load("features", columns=["ball_uid", "match_id", "innings", "over_no", "runs", "team_runs"])
c_ = load("ctx_preds", columns=["ball_uid", "wp_now", "wp_next"])
fo = f_[(f_.match_id == 21586) & (f_.innings == 2) & (f_.over_no == 15)].merge(c_, on="ball_uid")
wp = list(fo.wp_now * 100) + [100 * fo.wp_next.iloc[-1]]
runs = list(fo.team_runs)
fig, ax = plt.subplots(figsize=(7.0, 2.75))
x = np.arange(len(wp))
ax.fill_between(x[:5], wp[:5], color="#dfe6f2", alpha=0.6, lw=0)
ax.fill_between(x[4:], wp[4:], color="#fbd6ea", alpha=0.7, lw=0)
ax.plot(x[:5], wp[:5], color=NAVY, lw=2.6, marker="o", ms=6, mec="white", mew=1.4)
ax.plot(x[4:], wp[4:], color=PINK, lw=2.6, marker="o", ms=6, mec="white", mew=1.4)
for i, v in enumerate(wp):
    ax.text(i, v + 6, f"{v:.0f}%" if v >= 10 else f"{v:.1f}%", ha="center", fontsize=8.4, color=NAVY if i < 4 else PINK, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["Start of\nlast over"] + [f"Ball {i + 1}\n{('dot' if r == 0 else f'{r} run' if r == 1 else f'{r} runs')}" for i, r in enumerate(runs)], fontsize=7.8)
ax.set_ylim(0, 118); ax.set_yticks([0, 50, 100]); ax.set_yticklabels(["0%", "50%", "100%"], fontsize=7.6)
ax.set_ylabel("Chennai's chance of winning", fontsize=8)
ax.spines["left"].set_visible(False); ax.yaxis.grid(True, color=RULE, lw=0.6); ax.set_axisbelow(True)
ax.text(1.6, 92, "Mohit Sharma: four balls,\nchance falls to 6.5%", fontsize=8.2, color=NAVY, ha="center")
ax.text(4.72, 62, "Jadeja: six,\nthen four", fontsize=8.6, color=PINK, ha="center", fontweight="bold")
CH_FINAL = svg(fig, "final_over_2023")


def dumbbell(df, xcol, ecol, xlabel, lower_better, fmt, unit, name=None):
    d = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.9, 3.35))
    y = np.arange(len(d))
    ax.hlines(y, d[ecol], d[xcol], color="#f6c6df", lw=3, zorder=1)
    ax.scatter(d[ecol], y, s=44, color=GREY, zorder=3, edgecolor="white", lw=1.2)
    ax.scatter(d[xcol], y, s=60, color=PINK, zorder=4, edgecolor="white", lw=1.2)
    ax.set_yticks(y); ax.set_yticklabels([f"{int(r)}  {n}" for r, n in zip(d.final_rank, d.name)], fontsize=8.6)
    ax.tick_params(axis="y", length=0)
    gap = (d[xcol] - d[ecol]) * (-1 if lower_better else 1)
    xmax = max(d[xcol].max(), d[ecol].max())
    for yy, g in zip(y, gap):
        ax.text(xmax * 1.03, yy, fmt(g), va="center", fontsize=8.2, color=PINK if g > 0.05 else MUTED)
    ax.set_xlim(right=xmax * 1.12); ax.set_xlabel(xlabel, fontsize=8.5)
    ax.spines["left"].set_visible(False); ax.xaxis.grid(True, color=RULE, lw=0.6); ax.set_axisbelow(True)
    ax.text(xmax * 1.12, len(d) - 0.35, unit, ha="right", va="bottom", fontsize=7.6, color=MUTED)
    return svg(fig, name)


CH_BAT = dumbbell(bat10, "SR_pressure", "SR_expected_pressure", "Strike rate on pressure balls", False, lambda g: f"+{g:.0f}", "above average", "top10_batters")
CH_BOWL = dumbbell(bowl10, "econ_pressure", "econ_expected_pressure", "Economy on pressure balls (lower is better)", True,
                   lambda g: (f"{g:.1f} saved" if g >= 0.05 else "level"), "runs saved per over", "top10_bowlers")
H = T("pressure_heatmap_chase").set_index("over_no")
H = H[[c for c in H.columns if c in [str(i) for i in range(0, 8)]]]
fig, ax = plt.subplots(figsize=(6.9, 2.15))
im = ax.imshow(H.T.values * 100, aspect="auto", cmap=SEQ, vmin=0, vmax=100, origin="lower")
ax.set_xticks(range(len(H.index))); ax.set_xticklabels(H.index.astype(int), fontsize=7.8)
ax.set_yticks(range(H.shape[1])); ax.set_yticklabels(H.columns, fontsize=7.8)
ax.set_xlabel("Over of the chase", fontsize=8.5); ax.set_ylabel("Wickets down", fontsize=8.5)
for s_ in ax.spines.values():
    s_.set_visible(False)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015); cb.outline.set_visible(False)
cb.set_label("Pressure (0–100)", fontsize=7.8, color=INK2); cb.ax.tick_params(labelsize=7, length=0)
CH_HEAT = svg(fig, "pressure_map_chase")
ev = EV[EV.method.str.startswith("CHOSEN")]
grp = ["all public matches", "T20 World Cups", "franchise leagues", "full-member T20Is (incl. World Cups)"]
lab = {"all public matches": "All 3,993 matches", "T20 World Cups": "T20 World Cups", "franchise leagues": "Franchise leagues",
       "full-member T20Is (incl. World Cups)": "T20Is, top nations"}
fig, axes = plt.subplots(1, 2, figsize=(6.9, 1.85), sharey=True)
for ax, side, title in ((axes[0], "bat", "Top 10 batters: extra runs"), (axes[1], "bowl", "Top 10 bowlers: runs saved")):
    e = ev[ev.side == side].set_index("group").reindex(grp)
    vals = e.top10_runs_per_100_pressure_balls.values
    yy = np.arange(len(grp))[::-1]
    ax.barh(yy, vals, color=PINK, height=0.56)
    for a, v_ in zip(yy, vals):
        ax.text(v_ + 0.3, a, f"+{v_:.1f}", va="center", fontsize=8.2, color=NAVY)
    ax.set_yticks(yy); ax.set_yticklabels([lab[g] for g in grp], fontsize=8.2); ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=8.6, loc="left", color=NAVY, pad=6)
    ax.set_xlim(0, max(vals) * 1.35); ax.set_xticks([])
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_visible(False)
CH_PUBLIC = svg(fig, "outside_matches_test")

# ----------------------------------------------------------------------------- icons (simple line drawings)
IC = {
    "users": '<circle cx="9" cy="8" r="3.2"/><path d="M3 19c0-3.3 2.7-5.5 6-5.5s6 2.2 6 5.5"/><circle cx="17" cy="9" r="2.4"/><path d="M15.5 13.6c2.9-.3 5.5 1.6 5.5 4.6"/>',
    "filter": '<path d="M4 5h16l-6 7.5V19l-4-2v-4.5z"/>',
    "scale": '<path d="M12 4v16M6 20h12M5 8h14"/><path d="M5 8l-3 6h6zM19 8l-3 6h6z"/>',
    "pin": '<path d="M12 21s-6-5.6-6-11a6 6 0 0 1 12 0c0 5.4-6 11-6 11z"/><circle cx="12" cy="10" r="2.2"/>',
    "globe": '<circle cx="12" cy="12" r="8.5"/><path d="M3.5 12h17M12 3.5c2.6 2.6 2.6 14.4 0 17M12 3.5c-2.6 2.6-2.6 14.4 0 17"/>',
    "ball": '<circle cx="12" cy="12" r="8.5"/><path d="M7 5.5c2.2 3.6 2.2 9.4 0 13M17 5.5c-2.2 3.6-2.2 9.4 0 13"/>',
    "bat": '<path d="M14.5 3.5l6 6-8.5 8.5-3-3z"/><path d="M9 15l-5 5"/>',
    "target": '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r="1"/>',
    "code": '<path d="M9 7l-5 5 5 5M15 7l5 5-5 5"/>',
    "check": '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
}


def icon(k):
    return f'<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">{IC[k]}</svg>'


# ----------------------------------------------------------------------------- slide frame
N = [0]


def slide(body, sec=None, title="", cls="", kicker=""):
    """sec: index into SECTIONS (the brief's five report items), shown as a tracker at the top of the slide."""
    N[0] += 1
    if cls == "cover":
        return f'<section class="s cover">{body}</section>'
    trk = "".join(f'<div class="{"on" if i == sec else ""}"><i></i><span>{i + 1} · {t}</span></div>' for i, t in enumerate(SECTIONS))
    head = (f'<div class="trk">{trk}</div>' + (f'<div class="kick">{kicker}</div>' if kicker else "") + f'<h1>{title}</h1>') if title else ""
    return (f'<section class="s {cls}"><div class="inner">{head}{body}</div>'
            f'<div class="band"><span>Who delivers when it matters? · SupeRR Selector Season 2 · Round 1 · {AUTHOR}</span>'
            f'<span class="bnd-r"><span class="no">{N[0]:02d}</span><img src="{LOGO_W}" alt="Rajasthan Royals"></span></div></section>')


def bar(p):
    return (f'<div class="conf"><div class="conf-bar"><span style="width:{max(2, round(100 * p))}%"></span></div>'
            f'<div class="conf-t">{round(100 * p)} in 100</div></div>')


def mini(r, side):
    n = r["name"]
    st = (f'{r.SR_pressure:.0f}<span> v {r.SR_expected_pressure:.0f}</span>' if side == "bat"
          else f'{r.econ_pressure:.2f}<span> v {r.econ_expected_pressure:.2f}</span>')
    return (f'<div class="mini"><div class="mph" style="{photo_style(n, face=True)}"><span class="mrk">{int(r.final_rank)}</span></div>'
            f'<div class="mnm">{esc(n)}</div><div class="mrl">{ROLE.get(r.role, r.role)}</div><div class="mst">{st}</div></div>')


def vs(label, us, typ, top, fmt):
    """pick (pink) against a typical player in the same situations (grey), bars from zero"""
    w = lambda x: f"{34 * x / top:.1f}mm"
    return (f'<div class="vl">{label}</div>'
            f'<div class="vr"><span class="vb us" style="width:{w(us)}"></span><b>{fmt(us)}</b></div>'
            f'<div class="vr"><span class="vb ty" style="width:{w(typ)}"></span><em>{fmt(typ)} typical</em></div>')


def card(r, side):
    n = r["name"]
    if side == "bat":
        v_ = vs("Strike rate under pressure", r.SR_pressure, r.SR_expected_pressure, 250, lambda x: f"{x:.0f}")
        ft = f"{r.balls_per_out_pressure:.0f} pressure balls per dismissal · {int(r.pressure_balls)} faced"
    else:
        v_ = vs("Economy under pressure (shorter is better)", r.econ_pressure, r.econ_expected_pressure, 12, lambda x: f"{x:.2f}")
        ft = f"{r.balls_per_wkt_pressure:.0f} pressure balls per wicket · {int(r.pressure_balls)} bowled"
    return (f'<div class="card"><div class="ph" style="{photo_style(n)}"><span class="crk">{int(r.final_rank)}</span></div>'
            f'<div class="cb"><div class="ctop"><div><span class="cnm">{esc(n)}</span><span class="csub">{ROLE.get(r.role, r.role)} · {esc(TEAM.get(n, ""))}</span></div>'
            f'<div class="cf"><span>Stays top 10</span>{bar(r.p_top10)}</div></div>'
            f'<div class="cmid"><div class="ctext"><div class="chl">{esc(HEAD.get(n, ""))}</div><p class="cwhy">{esc(WHY.get(n, ""))}</p></div>'
            f'<div class="cvs">{v_}<div class="vf">{ft}</div></div></div>'
            f'<div class="crr"><b>For RR</b><span>{esc(RRFIT.get(n, ""))}</span></div></div></div>')


def trace_svg(vals, w=170, h=46):
    """the signature line: a win-chance trace (white), used on the cover"""
    xs = [w * i / (len(vals) - 1) for i in range(len(vals))]
    ys = [h - h * v / 100 for v in vals]
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.25"/>' for x, y in zip(xs, ys))
    pct = lambda v: f"{v:.0f}%" if v >= 10 else f"{v:.1f}%"
    lab = "".join(f'<text x="{x - (6 if v >= 90 else 0):.1f}" y="{y - 3.2 if v < 90 else y + 1.2:.1f}">{pct(v)}</text>' for x, y, v in zip(xs, ys, vals))
    return (f'<svg viewBox="-4 -6 {w + 8} {h + 12}" class="trace"><line x1="0" y1="{h / 2}" x2="{w}" y2="{h / 2}" class="mid"/>'
            f'<polyline points="{pts}"/>{dots}{lab}</svg>')


S = []


def row(big, small, head, text, ic=None, cls=""):
    """one idea per row: a big number or label on the left, a plain explanation on the right (the 'like with like' look)"""
    return (f'<div class="fr {cls}"><div class="frn"><b>{big}</b><span>{small}</span></div><div>{icon(ic) if ic else ""}'
            f'<h4>{head}</h4><p>{text}</p></div></div>')


# 01 cover
S.append(slide(f"""
<img class="cv-wm" src="{LOGO_W}" alt="">
<div class="cv-top">SupeRR Selector · Season 2 · Round 1</div>
<div class="cv-mid"><h1 class="cv-title">Who delivers<br>when it matters?</h1><div class="cv-rule"></div>
<p class="cv-sub">The ten most reliable T20 batters and the ten most reliable bowlers under pressure, from five seasons of IPL
and international cricket, and how each of them could strengthen Rajasthan Royals.</p></div>
<div class="cv-tr">{trace_svg(wp)}<div class="cv-trc">Chennai's chance of winning before each ball of the last over of the 2023 IPL final.
This report is about balls like these.</div></div>
<div class="cv-bot"><div><div class="cv-k">Prepared by</div><div class="cv-name">{AUTHOR}</div><div class="cv-d">{EMAIL}</div></div>
<img class="cv-logo" src="{LOGO_W}" alt="Rajasthan Royals"></div>""", cls="cover"))

# 02 the answer
S.append(slide(f"""
<p class="lede">These are the 10 batters and 10 bowlers who did best when the match was on the line, in the IPL and T20
internationals from 2022 to 2026. Under each name: <b class="pk">his number on pressure balls</b> v <span class="them">what an average player managed in the same situations</span>.</p>
<div class="grid-h"><span class="pill">Batters</span><span class="ght">Strike rate under pressure <em>v an average batter in the same situations</em></span></div>
<div class="minis">{''.join(mini(r, 'bat') for _, r in bat10.iterrows())}</div>
<div class="grid-h"><span class="pill">Bowlers</span><span class="ght">Economy under pressure <em>v an average bowler of the same type (lower is better)</em></span></div>
<div class="minis">{''.join(mini(r, 'bowl') for _, r in bowl10.iterrows())}</div>
<div class="stats4">
 <div><b>200,961</b><span>balls studied, ball by ball</span></div><div><b>867</b><span>IPL and T20I matches</span></div>
 <div><b>5</b><span>seasons, 2022 to 2026, all counted equally</span></div><div><b>3,993</b><span>extra matches (World Cups, leagues) used only to test my picks</span></div>
</div>""", None, 'My <span class="pk">top 10s</span>', kicker="The answer"))

# 03 pressure: what it is
S.append(slide(f"""
<div class="defbox"><h3>Pressure, in plain words</h3>
<p>A ball is under pressure when its result can change who wins the match. A dot ball in the first over hardly matters. A dot
ball when 12 are needed off the last over can decide the game. I give every ball a pressure score, and the 20% of balls with the
highest score are my <b>pressure balls</b>. The players in this report are the ones who did best on those balls.</p></div>
<div class="chartcard"><h4>A real example: the 2023 IPL final</h4>
<p class="cap">Rain cut the match to 15 overs. Chennai needed 13 off the last over against Gujarat. The line is Chennai's chance of
winning before each ball. Watch how far single balls move it: that movement is what I call pressure.</p>
<figure class="chart">{CH_FINAL}</figure></div>
<h3>How I give every ball a pressure score</h3>
<div class="st3">
 <div><span class="fn">1</span>{icon('target')}<h4>Chance of winning</h4><p>Before every ball, a machine-learning model works out each team's
 chance of winning from the score, balls left, wickets and target. Like the win predictor on TV.</p></div>
 <div><span class="fn">2</span>{icon('ball')}<h4>How much can this ball change it?</h4><p>I check the chance after a dot, a single, a four, a
 six or a wicket. The bigger the possible swing, the more the ball matters. I call this the ball's <b>impact</b>.</p></div>
 <div><span class="fn">3</span>{icon('check')}<h4>How big is the match?</h4><p>A final or knockout counts 1.5 times, a must-win game 1.25 times,
 a normal game once, and a dead rubber 0.75 times.</p></div>
</div>
<div class="formula">Pressure = impact × how big the match is</div>
<p class="note"><b>To give you a feel:</b> in the first over of a chase, one ball moves the chance of winning by only about 1.5%. In the
last over of a close chase, an average ball moves it by about 24% (say from 50% to 74%), and 1 ball in 10 moves it by 43% or more.</p>""",
                 0, 'What is <span class="pk">pressure?</span>'))

# 04 pressure: how it was chosen
D = DEFS.set_index("definition")
DN = [("C-A leverage", "Impact (how much a ball can change the result)", True),
      ("C-C stakes x strain", "Impact × how hard the situation is", False),
      ("C-B leverage x big match", "Impact × knockout and must-win games", False),
      ("C-E GMM regimes", "Machine-learning groups of similar situations", False),
      ("C-G simple rules", "Simple rules: last 5 overs, rate 9+, close game", False),
      ("C-F Bhattacharjee-Lemmer", "A published pressure index (Bhattacharjee & Lemmer)", False),
      ("C-F Thomson zone", "A published 'clutch zone' (Thomson and others)", False),
      ("C-D PCA factor", "One combined score from many signals (PCA)", False)]
drows = "".join(f'<tr class="{"win" if w else ""}"><td>{lbl}</td><td><div class="sbar"><span style="width:{100 * D.loc[k, "score"]:.0f}%"></span></div></td>'
                f'<td class="n">{100 * D.loc[k, "score"]:.0f}</td></tr>' for k, lbl, w in DN if k in D.index)
S.append(slide(f"""
<p class="lede">Pressure can be measured in many ways: the required rate, wickets lost, dot balls in a row, a big match. I didn't want to
pick one on gut feeling, so I tested them all and let the data decide.</p>
<div class="funnel">
 <div class="fs"><b>38</b><span>ideas</span><p>Signs of pressure, like a rate above 10, three dots in a row, a new batter or a knockout</p></div>
 <div class="arrow">→</div>
 <div class="fs"><b>3</b><span>tests for each</span><p>Do players perform differently? Do these balls decide matches? Is it true every season?</p></div>
 <div class="arrow">→</div>
 <div class="fs"><b>8</b><span>full definitions</span><p>Built from the ideas that passed, including two from published research</p></div>
 <div class="arrow">→</div>
 <div class="fs win"><b>1</b><span>winner</span><p>Impact, with how big the match is added on top</p></div>
</div>
<h3>Proof that pressure balls really are different</h3>
<div class="fair tight">
{row("2.2×", "more movement", "The result really swings more",
     "On my pressure balls, the chance of winning moves 2.2 times as much as on an average ball. These are the balls that decide matches.")}
{row("+8%", "more wickets", "Batters get out more often",
     "On pressure balls, batters get out 8% more often than on other balls at the same stage of an innings (same innings, same overs, same wickets down).")}
</div>
<h3>The 8 definitions, scored out of 100</h3>
<p class="cap">Each definition got a score for how much the result really swings on its pressure balls and how much more often batters
get out on them. Higher is better.</p>
<table class="defs"><tbody>{drows}</tbody></table>
<p class="cap">Impact alone scored 100. Adding "how big the match is" scored 99, almost the same, so I kept it: a final should count more
than a dead rubber. I also tried other cut-offs from the top 10% to the top 30% of balls: all 10 batters and 6 to 10 of the bowlers stayed.</p>""",
                 0, 'How I <span class="pk">chose</span> this definition'))

# 05 pressure: where it happens
S.append(slide(f"""
<div class="chartcard"><h4>Where pressure is highest in a chase</h4>
<p class="cap">Each square is one situation, for example "over 17, 4 wickets down". The darker the pink, the more pressure. It is highest
along the band where the chase can still go either way, and fades when the chasing team has collapsed (top left) or is cruising (bottom right).</p>
<figure class="chart">{CH_HEAT}</figure></div>
<div class="fair tight dense">
{row("9 in 10", "pressure balls<br>come in chases", "Why chases?",
     "In a chase both teams know the target, so every run and wicket changes the result directly. Batting first, nobody knows yet what a "
     "winning score is, so one ball changes less. Real results agree: in the middle overs, a four instead of a dot changed the chasing team's "
     "win rate by 7%, and the first-innings team's by only 3.4%.")}
{row("41%", "of the rating<br>is first innings", "The first innings still counts",
     "I don't throw any ball away. Every ball counts, and a ball with more pressure counts more. Added up, 41% of what goes into the ratings "
     "comes from first-innings balls, so rebuilding at 30 for 3 still matters, for the batter and the bowler.")}
{row("80–94%", "in every<br>league", "It's how T20 works",
     "I checked 3,993 other matches from leagues like the Big Bash, PSL and SA20. In every league, 80% to 94% of pressure balls came in chases. "
     "So this is how T20 cricket is, not a quirk of my model.")}
</div>
<h3>Pressure from both ends</h3>
<table class="jobs"><thead><tr><th>Situation</th><th>What the batter must do</th><th>What the bowler must do</th></tr></thead><tbody>
<tr><td>Chasing, last overs, a lot needed</td><td>Find boundaries, waste no balls</td><td>Dots and yorkers; remove the set batter</td></tr>
<tr><td>Chasing, middle overs, rate climbing</td><td>Score off almost every ball without losing wickets</td><td>Build dots and force a risky shot</td></tr>
<tr><td>Top order gone early</td><td>Rebuild without stalling</td><td>Strike again before the new batters settle</td></tr>
<tr><td>Final or must-win game</td><td colspan="2">The same jobs, but every ball counts 1.5 or 1.25 times as much</td></tr>
</tbody></table>""", 0, 'Where pressure <span class="pk">happens</span>'))

# 06 the method, step by step (machine learning named in plain words)
ML = lambda t: f'<span class="mlt">{t}</span>'
S.append(slide(f"""
<p class="lede">Here is the whole system from start to finish. Machine learning sets the yardstick (what is normal in each situation).
It never picks a player directly: every player is judged on whether he beat that yardstick on the balls that mattered.</p>
<div class="fair steps">
{row("1", "clean", "Rebuild every ball",
     "I fixed how extras are recorded and rebuilt the score, wickets, target and balls left before each of the 200,961 balls. "
     + ML("Data cleaning and checks in Python"), "filter")}
{row("2", "predict", "Predict who wins, before every ball",
     "A model gives each team's chance of winning. I tested 11 machine-learning models for this job. The simplest, logistic regression, "
     "predicted new matches best; complex ones like XGBoost and Random Forest memorised old matches instead. "
     + ML("Logistic regression, beat 10 other models"), "target")}
{row("3", "pressure", "Score the pressure on every ball",
     "Impact × how big the match is. The top 20% are the pressure balls. " + ML("Chosen from 8 definitions by testing"), "scale")}
{row("4", "average", "Learn what an average player does",
     "A second model learns, from all 200,961 balls, what an average player does in each situation: how often a dot, a four, a six or a "
     "wicket. It never sees player names. " + ML("LightGBM (gradient boosting), 41 inputs"), "users")}
{row("5", "rate", "Rate every player",
     "On every ball: the player's result minus the average player's. Pressure balls count most. A player with few balls is pulled "
     "towards average, so one lucky game can't make a star. " + ML("Empirical Bayes shrinkage"), "bat")}
{row("6", "test", "Choose the settings and test them",
     "Settings were chosen by hiding matches and checking the picks on them, 52 times. The ranking was rebuilt 1,000 times on reshuffled "
     "matches, and the final picks were tested on 3,993 matches the model never saw. " + ML("Cross-validation, bootstrap, outside test"), "check")}
</div>""", 1, 'My method in <span class="pk">6 steps</span>'))

# 07 performance measurement
S.append(slide(f"""
<p class="lede">For every ball, I ask one question: did this player do better or worse than an average player would have, in exactly
the same situation?</p>
<div class="flow">
 <div class="fl"><span class="fn">1</span><b>What usually happens</b><p>The model's answer for that over, score, wickets, target, ground,
 season and competition, and how long the batter has been in.</p></div>
 <div class="fl"><span class="fn">2</span><b>What actually happened</b><p>The runs, the dot, the boundary or the wicket on this ball.</p></div>
 <div class="fl hi"><span class="fn">3</span><b>The difference</b><p>His score for that ball. Better than usual and he gains; worse and he loses.</p></div>
</div>
<div class="example"><b>Example:</b> on pressure balls, Abhishek Sharma scored at a strike rate of 184. An average batter in the same
situations would have scored at 156. That gap, ball by ball, minus what his dismissals cost the team, is his rating.</div>
<div class="fair tight">
{row("+ / −", "runs add,<br>getting out costs", "What counts for a batter",
     "Runs count for him and getting out counts against him. A wicket costs more when the team can least afford it: out at 30 for 3 in the "
     "eighth over costs far more than out off the last ball.", "bat")}
{row("+ / −", "runs saved and<br>wickets add", "What counts for a bowler",
     "Runs saved, dot balls and the wickets he took himself (run-outs don't count). Wides and no-balls count against him. Pace bowlers are "
     "compared with pace bowlers, spinners with spinners.", "ball")}
{row("Luck?", "rising above your<br>own level", "Why not compare a player with his own normal game?",
     "The brief talks about usual levels, so I tested this first: pressure performance minus normal performance, for every player. Then I "
     "split each player's matches into two halves. Players who rose under pressure in one half did not do it again in the other half. Even "
     "fake, randomly chosen 'pressure balls' showed the same pattern. It looked like luck, so I rank players on how good they are under "
     "pressure compared with everyone else in the same moments. That did repeat.", "scale")}
</div>
<div class="rule wide"><div class="wts"><div><span style="width:95%"></span><em>highest pressure</em></div><div><span style="width:25%"></span><em>ordinary ball</em></div><div><span style="width:4%"></span><em>match already decided</em></div></div>
<p><b>Adding it up: pressure balls count most.</b> The highest-pressure balls count fully, an ordinary ball about a quarter, and a ball in a
match that is already decided almost nothing. The total is the player's rating.</p></div>""",
                 1, 'How every ball <span class="pk">is scored</span>'))

# 08 fair comparisons
S.append(slide(f"""
<p class="lede">The brief lists four things a fair comparison has to handle. This is how I handled each one.</p>
<div class="fair">
{row("60", "pressure balls<br>at least", "Sample sizes",
     "A player with only a few pressure balls might just have been lucky. So the fewer balls he has, the more his rating is pulled towards "
     "average, and nobody with fewer than 60 pressure balls is ranked. Without this rule, a bowler who had bowled a single pressure ball came sixth.", "scale")}
{row("2025+", "last match in<br>2025 or 2026", "Player eligibility",
     "Every player in the data was rated, using all five seasons, each season counting the same. RR are buying for the 2027 auction, so the "
     "final lists only include players who played in 2025 or 2026. That left out Dinesh Karthik, who retired in 2024 and would have been "
     "10th among batters. Players left out still count when working out the average player.", "users")}
{row("9 → 3", "spinners in the<br>bowling top 10", "Different match contexts",
     "The average player is worked out for the same over, score, wickets, target, ground, season and stage of the tournament, so nobody gains "
     "just from a small ground or a high-scoring year (IPL scoring rose about 16% from 2022 to 2026). Bowlers are compared only with their own "
     "type. Without that rule, 9 of my top 10 bowlers were spinners, because spin is naturally cheaper. Now there are 3.", "pin")}
{row("4", "competition<br>levels", "IPL and T20I data",
     "I used both, because more matches mean fairer judgements. But a run in the IPL and a run against a weaker T20I side are not the same, so "
     "each level has its own average player: the IPL, T20Is between the top 12 nations, other T20Is, and A-team games.", "globe")}
</div>""", 2, 'Comparing <span class="pk">like with like</span>'))

# 09-12 picks
READ = ("<b>How to read a card.</b> The <b class='pk'>pink bar</b> is the player, the grey bar is an average player in the same situations. "
        "<b>Stays top 10</b> shows how sure I am: I rebuilt the ranking 1,000 times on reshuffled matches, and this is how often he stayed "
        "in the top 10. <b>For RR</b> shows where he would help, based on RR's own record under pressure in IPL 2025 and 2026.")
for side, df, word in (("bat", bat10, "batters"), ("bowl", bowl10, "bowlers")):
    for part, sl in ((1, slice(0, 5)), (2, slice(5, 10))):
        lede = f'<p class="lede sm">{READ}</p>' if part == 1 else ""
        S.append(slide(lede + '<div class="cards">' + "".join(card(r, side) for _, r in df.iloc[sl].iterrows()) + "</div>",
                       3, f'Top 10 <span class="pk">{word}</span> · {"1 to 5" if part == 1 else "6 to 10"}'))

# 13 evidence chart
S.append(slide(f"""
<p class="lede">Each row is one of my picks. The <b class="pk">pink dot</b> is what he did on pressure balls. The <span class="them">grey dot</span>
is what an average player did in the same situations. The number on the right is the gap.</p>
<div class="chartcard"><h4>Batters: strike rate under pressure (higher is better)</h4><figure class="chart">{CH_BAT}</figure></div>
<div class="chartcard"><h4>Bowlers: economy under pressure (lower is better)</h4><figure class="chart">{CH_BOWL}</figure></div>
<p class="note">Sears and Carse give away runs at about the average rate. They are in the list for wickets: each takes one every 16 pressure
balls. The order comes from the full rating (runs, wickets and how much each ball mattered), not from this chart alone.</p>""",
                 3, 'Every pick beat <span class="pk">an average player</span>'))

# 14 reliability: tests
S.append(slide(f"""
<div class="fair tight">
{row("51 / 52", "tests passed<br>(batters)", "Test 1: hide some matches, then check them",
     "I hid part of the matches, built the ranking on the rest, and checked how the picks did in the hidden matches. I did this 52 times. "
     "My top 10 batters beat an average batter in 51 of the 52 tests, by about 12 runs per 100 pressure balls. My top 10 bowlers saved runs "
     "in 50 of 52, about 8 runs per 100 pressure balls. And players ranked higher did better than players ranked lower in 51 of 52.", "check")}
{row("3,993", "matches the model<br>never saw", "Test 2: matches from outside RR's data",
     "Cricsheet.org gives free ball-by-ball records. I took every men's T20 match from 2022 to 2026 in these competitions and checked how my "
     "picks did on pressure balls there. Matches that were also in RR's data were removed, so nothing counts twice.", "globe")}
</div>
<div class="leagues"><span>T20 internationals incl. 3 World Cups <b>2,030</b></span><span>Vitality Blast <b>635</b></span><span>Big Bash <b>221</b></span>
<span>Bangladesh PL <b>203</b></span><span>Pakistan SL <b>177</b></span><span>Caribbean PL <b>167</b></span><span>ILT20 <b>134</b></span><span>SA20 <b>130</b></span>
<span>Major League Cricket <b>109</b></span><span>Lanka PL <b>96</b></span><span>Syed Mushtaq Ali <b>91</b></span></div>
<div class="chartcard"><h4>How my top 10s did in those 3,993 matches</h4>
<p class="cap"><b>How to read it:</b> +10.8 means that over 100 pressure balls, my top 10 bowlers gave away about 11 fewer runs than an average
bowler would have in the same situations. +4.2 means my top 10 batters scored about 4 more runs per 100 pressure balls than an average batter.</p>
<figure class="chart">{CH_PUBLIC}</figure></div>
<div class="two">
 <div class="box hl"><h4>World Cup semi-finals and finals</h4><p>Bumrah took 9 wickets at 5.45 an over, and Klaasen made 52 off 27 balls.</p></div>
 <div class="box warn"><h4>Where a simpler method did better</h4><p>On these outside matches, picking batters by raw strike rate under
 pressure gave a better top 10 than mine. For bowlers, a list built from all balls did slightly better at the very top (12.7 runs saved
 against my 10.8). But inside RR's data, over the 52 tests, my batting ratings did about 5 times better than raw strike rate.</p></div>
</div>""", 4, 'Did the picks hold up on <span class="pk">new matches?</span>'))

# 15 limits, deep dive, how to repeat
S.append(slide(f"""
<div class="fair tight">
{row("6%", "skill in one<br>season", "Why one season can't judge a batter",
     "In a single season, only about 6% of the difference between batters under pressure is skill. The rest is luck. A batter needs about "
     "800 pressure balls before skill is half of what you see. That's why I used all five seasons, and why my batting list did not hold up "
     "when I tested it on 2026 alone.", "bat")}
{row("8 / 10", "names stay", "If I change my choices",
     "I changed one choice at a time (the pressure definition, the cut-off, the rating method, the data used). Usually 8 of the 10 names "
     "stayed. The bowling list depends more on using T20Is too: on IPL matches alone, only 4 of the 10 bowlers would stay.", "scale")}
</div>
<div class="box"><h4>Where I am less sure</h4><ul class="cols2">
 <li>Batters are much harder to judge than bowlers</li><li>First-innings pressure is measured less sharply</li>
 <li>No fielding, dropped catches, pitch or weather in the data</li><li>Arya, Carse and Overton have fewer balls</li>
 <li>I looked at 2026 more than once while building, so it is not a perfectly clean test</li></ul></div>
<div class="gh">{icon('code')}<div><h4>Want to check my work?</h4><p>All the code, the settings and a technical report (including why the simple
model beat 10 machine-learning models) are on GitHub. RR's data is not shared.</p><div class="ghl">{GITHUB}</div></div></div>
<h3>How anyone can repeat this</h3>
<div class="steps8">{''.join(f'<div><b>{i}</b><span>{t}</span></div>' for i, t in enumerate(['Clean the data', 'Rebuild every ball', 'Learn the average player', 'Measure pressure', 'Score every ball', 'Choose settings on 52 tests', 'Rank and rerun 1,000 times', 'Test on 3,993 matches'], 1))}</div>
<p class="cap">One settings file and a fixed random seed. With RR's data, running the scripts in order rebuilds every number in this report.</p>""",
                 4, 'Where I could be <span class="pk">wrong</span>'))

# 16 appendix
dd = "".join(f"<tr><td class='rk2'>{i + 1}</td><td>{esc(a['name'])}</td><td class='n them'>{int(a.player_id)}</td><td class='gap'></td>"
             f"<td class='rk2'>{i + 1}</td><td>{esc(b['name'])}</td><td class='n them'>{int(b.player_id)}</td></tr>"
             for i, ((_, a), (_, b)) in enumerate(zip(bat10.iterrows(), bowl10.iterrows())))
nxt_b = " · ".join(f"{int(r.final_rank)} {esc(r['name'])}" for _, r in bat.iloc[10:20].iterrows())
nxt_w = " · ".join(f"{int(r.final_rank)} {esc(r['name'])}" for _, r in bowl.iloc[10:20].iterrows())
S.append(slide(f"""
<h3>A · For the Unstop dropdowns</h3>
<table class="dd"><thead><tr><th>#</th><th>Batter</th><th class='n'>ID</th><th class='gap'></th><th>#</th><th>Bowler</th><th class='n'>ID</th></tr></thead><tbody>{dd}</tbody></table>
<p class="cap">Two players in the data are called Rashid Khan. The one ranked here is ID 4722 (Afghanistan, Gujarat Titans).</p>
<h3>B · The next ten</h3>
<p class="sm"><b>Batters.</b> {nxt_b}</p><p class="sm"><b>Bowlers.</b> {nxt_w}</p>
<h3>C · Words used</h3>
<div class="gloss">
<div><b>Pressure ball</b><span>One of the 20% of balls whose result can change the match the most (impact × how big the match is).</span></div>
<div><b>Impact</b><span>How much one ball can change a team's chance of winning, on average.</span></div>
<div><b>Average player, same situations</b><span>What a typical player does in the same over, score, wickets, target, ground, season and competition. For bowlers, the same type (pace or spin).</span></div>
<div><b>Pulled towards average</b><span>A record built on few balls gets less trust, so it is moved closer to average (shrinkage).</span></div>
<div><b>Stays top 10</b><span>How often he stayed in the top 10 when the ranking was rebuilt 1,000 times on reshuffled matches (bootstrap).</span></div>
<div><b>Strike rate / economy</b><span>Runs per 100 balls faced / runs given away per over.</span></div>
</div>
<p class="cap">Data: ball-by-ball data supplied for SupeRR Selector Season 2; outside matches from Cricsheet (cricsheet.org).</p>""",
                 None, 'Dropdown IDs and <span class="pk">the next ten</span>', kicker="Appendix"))

CSS = """
@page { size: A4; margin: 0; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.4pt; line-height: 1.45; color: #1f2230;
       -webkit-print-color-adjust: exact; print-color-adjust: exact; font-variant-numeric: tabular-nums; }
.s { width: 210mm; height: 297mm; position: relative; overflow: hidden; page-break-after: always;
     background: linear-gradient(180deg, #ffffff 0%, #faf7fc 55%, #f3eef8 100%); }
.s:last-child { page-break-after: auto; }
.inner { position: absolute; left: 15mm; right: 15mm; top: 13mm; bottom: 19mm; }
.band { position: absolute; left: 0; right: 0; bottom: 0; height: 12mm; background: #011c45; color: #c9d1e3; font-size: 7.4pt;
        display: flex; align-items: center; justify-content: space-between; padding: 0 9mm 0 15mm; border-top: 1.4mm solid #e2178f; }
.bnd-r { display: flex; align-items: center; gap: 4mm; } .bnd-r img { height: 10mm; }
.band .no { color: #fff; font-weight: 700; font-size: 9pt; }
.trk { display: grid; grid-template-columns: repeat(5, 1fr); gap: 1.6mm; margin: 0 0 4mm; }
.trk div i { display: block; height: 1.1mm; background: #e6dfe9; border-radius: 0.6mm; margin-bottom: 1.1mm; }
.trk div span { font-size: 6.6pt; font-weight: 600; color: #a29fa8; letter-spacing: 0.2px; }
.trk div.on i { background: #e2178f; } .trk div.on span { color: #011c45; font-weight: 700; }
.kick { font-size: 8pt; font-weight: 700; letter-spacing: 1.6px; text-transform: uppercase; color: #e2178f; margin-bottom: 1mm; }
h1 { font-family: Georgia, serif; font-weight: 700; font-size: 26pt; line-height: 1.12; margin: 0 0 3mm; color: #011c45; letter-spacing: -0.3px; }
.pk { color: #e2178f; }
h3 { font-size: 12pt; font-weight: 700; margin: 4mm 0 1.8mm; color: #011c45; }
h4 { font-size: 11pt; font-weight: 700; margin: 0 0 1mm; color: #011c45; }
p { margin: 0 0 2mm; }
b { font-weight: 650; }
.lede { font-size: 12pt; line-height: 1.45; color: #3c3c46; margin-bottom: 3.5mm; }
.lede.sm { font-size: 9.4pt; margin-bottom: 2.5mm; }
.cap { font-size: 8.8pt; color: #6c6c76; margin: 1mm 0 2.5mm; }
.note { font-size: 9.6pt; color: #3c3c46; }
.tl { display: block; font-size: 8.4pt; color: #7c7c84; margin-top: 1.2mm; }
.sm { font-size: 8.8pt; }
.them { color: #8a8a92; }
.ic { width: 7mm; height: 7mm; color: #e2178f; margin-bottom: 1mm; }
/* cover */
.cover { background: #e2178f; color: #fff; }
.cv-wm { position: absolute; right: -40mm; top: 30mm; height: 250mm; opacity: 0.13; }
.cv-top { position: absolute; left: 20mm; top: 22mm; font-size: 9pt; letter-spacing: 2px; text-transform: uppercase; font-weight: 700; }
.cv-mid { position: absolute; left: 20mm; right: 20mm; top: 92mm; }
.cv-title { font-size: 48pt; line-height: 1.03; letter-spacing: -1px; margin: 0; color: #fff; }
.cv-rule { width: 30mm; height: 2.4px; background: #fff; margin: 9mm 0 7mm; }
.cv-sub { font-size: 14pt; line-height: 1.45; max-width: 135mm; }
.cv-bot { position: absolute; left: 0; right: 0; bottom: 0; height: 46mm; background: #011c45; display: flex; align-items: center;
          justify-content: space-between; padding: 0 20mm; }
.cv-k { font-size: 8.5pt; letter-spacing: 1.6px; text-transform: uppercase; color: #f4a6d0; font-weight: 700; }
.cv-name { font-family: Georgia, serif; font-size: 20pt; font-weight: 700; margin: 1mm 0; } .cv-d { font-size: 9pt; color: #c9d1e3; }
.cv-logo { height: 30mm; }
.cv-tr { position: absolute; left: 20mm; right: 26mm; top: 186mm; }
.trace { width: 100%; height: auto; display: block; overflow: visible; }
.trace polyline { fill: none; stroke: #fff; stroke-width: 1.1; stroke-linejoin: round; }
.trace circle { fill: #e2178f; stroke: #fff; stroke-width: 0.7; }
.trace .mid { stroke: #ffffff; stroke-opacity: 0.35; stroke-width: 0.35; stroke-dasharray: 1.2 1.2; }
.trace text { fill: #fff; font-size: 3.4px; font-weight: 700; text-anchor: middle; font-family: "Segoe UI", sans-serif; }
.cv-trc { font-size: 9pt; color: #ffe3f1; margin-top: 2mm; max-width: 150mm; line-height: 1.4; }
/* answer page */
.grid-h { margin: 2mm 0 2mm; } .pill { background: #011c45; color: #fff; font-size: 8pt; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase; padding: 1mm 3.5mm; border-radius: 6mm; }
.minis { display: grid; grid-template-columns: repeat(5, 1fr); gap: 3mm 2.5mm; margin-bottom: 3mm; }
.mini { text-align: center; }
.mph { width: 25mm; height: 25mm; margin: 0 auto 1.3mm; border-radius: 50%; position: relative;
       background-color: #f6dcea; box-shadow: inset 0 0 0 2px #fff, 0 0 0 1px #f0c3dc; }
.mrk { position: absolute; left: -1mm; top: -1mm; width: 7mm; height: 7mm; border-radius: 50%; background: #e2178f; color: #fff;
       font-weight: 700; font-size: 8.5pt; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 0 1.5px #fff; }
.mnm { font-weight: 700; font-size: 8.4pt; color: #011c45; line-height: 1.2; } .mrl { font-size: 7pt; color: #7c7c84; }
.mst { font-family: Georgia, serif; font-weight: 700; font-size: 10pt; color: #e2178f; margin-top: 0.6mm; } .mst span { color: #9a98a0; font-weight: 400; font-size: 8.4pt; }
.ght { font-size: 8.4pt; color: #011c45; font-weight: 600; margin-left: 2.5mm; } .ght em { font-style: normal; color: #8a8a92; font-weight: 400; }
.stats4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 3mm; margin-top: 2mm; }
.stats4 div, .stats3 div { background: #fff; border: 1px solid #efe6f0; border-radius: 2.5mm; padding: 2.5mm 3mm; }
.stats4 b, .stats3 b { display: block; font-family: Georgia, serif; font-size: 18pt; color: #e2178f; line-height: 1.1; }
.stats4 span, .stats3 span { font-size: 8.8pt; color: #4a4a52; line-height: 1.3; display: block; }
/* charts and blocks */
.chart { margin: 0 0 1mm; } .chart svg { width: 100%; height: auto; display: block; }
.def { display: grid; grid-template-columns: 1.15fr 1fr; gap: 5mm; margin: 2mm 0 2.5mm; }
.def-r { display: flex; flex-direction: column; gap: 1.6mm; }
.kv { display: grid; grid-template-columns: 14mm 1fr; align-items: center; background: #fff; border: 1px solid #efe6f0; border-radius: 2mm; padding: 1.3mm 2.5mm; }
.kv b { font-family: Georgia, serif; font-size: 15pt; color: #e2178f; } .kv span { font-size: 9pt; color: #3c3c46; line-height: 1.3; }
.chips { display: flex; gap: 2.4mm; flex-wrap: wrap; margin: 1mm 0 2.5mm; }
.chips span { background: #fff; border: 1px solid #f0c3dc; border-radius: 10mm; padding: 1.4mm 3.8mm; font-size: 9.8pt; } .chips b { color: #e2178f; }
.formula { background: #011c45; color: #fff; font-family: Georgia, serif; font-size: 14pt; text-align: center; padding: 3.4mm; border-radius: 2mm; margin: 1mm 0 2.5mm; }
.funnel { display: flex; align-items: stretch; gap: 1.5mm; margin: 1mm 0 3mm; }
.fs { flex: 1; background: #fff; border: 1px solid #efe6f0; border-radius: 2.5mm; padding: 3mm 2.6mm; }
.fs b { font-family: Georgia, serif; font-size: 24pt; color: #011c45; line-height: 1; } .fs span { display: block; font-weight: 700; font-size: 8.5pt; color: #e2178f; margin: 0.6mm 0 1.2mm; }
.fs p { font-size: 9pt; color: #4a4a52; line-height: 1.35; margin: 0; } .fs.win { background: #011c45; } .fs.win b, .fs.win p { color: #fff; } .fs.win span { color: #f4a6d0; }
.arrow { display: flex; align-items: center; color: #e2178f; font-size: 15pt; font-weight: 700; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 4mm; margin-top: 2.5mm; }
.box { background: #fff; border: 1px solid #efe6f0; border-radius: 2.5mm; padding: 3.6mm 4mm; font-size: 10pt; }
.box p { margin: 0 0 1.2mm; } .box ul { margin: 0 0 1.5mm; padding-left: 4mm; } .box li { margin-bottom: 0.6mm; }
.box.hl { border-color: #f0c3dc; background: #fff7fb; } .box.warn { border-color: #ecd4a8; background: #fffaf0; } .box.warn h4 { color: #8a5a12; }
.facts3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 3.5mm; margin: 1mm 0 2mm; }
.facts3 div { border-top: 2.4px solid #e2178f; padding-top: 2mm; } .facts3 b { font-family: Georgia, serif; font-size: 18pt; color: #e2178f; line-height: 1.1; }
.facts3 p { font-size: 9.6pt; color: #3c3c46; margin-top: 1mm; }
.flow { display: grid; grid-template-columns: repeat(3, 1fr); gap: 3mm; margin: 1mm 0 3mm; }
.fl { background: #fff; border: 1px solid #efe6f0; border-radius: 2.5mm; padding: 3mm; position: relative; }
.fl b { display: block; color: #011c45; font-size: 10.6pt; margin: 1mm 0; } .fl p { font-size: 9.4pt; color: #4a4a52; margin: 0; }
.fl.hi { background: #011c45; } .fl.hi b, .fl.hi p { color: #fff; }
.fn { display: inline-flex; width: 6.5mm; height: 6.5mm; border-radius: 50%; background: #e2178f; color: #fff; font-weight: 700; font-size: 8.5pt; align-items: center; justify-content: center; }
.rules { display: grid; grid-template-columns: 1fr 1fr; gap: 4mm; }
.rule { display: grid; grid-template-columns: 22mm 1fr; gap: 3mm; align-items: center; background: #fff; border: 1px solid #efe6f0; border-radius: 2.5mm; padding: 3.4mm; font-size: 9.8pt; }
.rule p { margin: 0; } .rule .big { font-family: Georgia, serif; font-size: 26pt; color: #e2178f; text-align: center; }
.wbars { height: 16mm; display: flex; align-items: flex-end; gap: 2mm; justify-content: center; } .wbars span { width: 4mm; background: #e2178f; border-radius: 1mm 1mm 0 0; }
.callout { background: #011c45; color: #fff; border-radius: 2.5mm; padding: 4mm 5mm 2mm; margin-bottom: 4mm; }
.callout h3 { color: #f4a6d0; margin-top: 0; } .callout p { font-size: 10.4pt; line-height: 1.55; }
.tiles { display: grid; grid-template-columns: 1fr 1fr; gap: 3.5mm; }
.tile { background: #fff; border: 1px solid #efe6f0; border-radius: 2.5mm; padding: 4mm 4.2mm; font-size: 10pt; }
.tile p { margin: 0; color: #3c3c46; }
.proof { display: grid; grid-template-columns: 34mm 1fr 1fr; gap: 4mm; align-items: center; background: #011c45; color: #fff; border-radius: 2.5mm; padding: 3.5mm 4.5mm; margin: 0 0 1mm; }
.proof .pt { font-family: Georgia, serif; font-size: 12pt; font-weight: 700; color: #f4a6d0; line-height: 1.25; }
.proof b { display: block; font-family: Georgia, serif; font-size: 22pt; color: #fff; line-height: 1.05; } .proof span { font-size: 9pt; color: #dfe4ef; }
.fair { display: flex; flex-direction: column; gap: 5mm; margin-top: 3mm; }
.fr { display: grid; grid-template-columns: 38mm 1fr; gap: 5mm; background: #fff; border: 1px solid #efe6f0; border-radius: 2.5mm; padding: 7.5mm 6mm; }
.fr .ic { float: right; margin: 0 0 0 3mm; width: 8mm; height: 8mm; } .fr h4 { font-size: 13pt; margin-bottom: 1.6mm; } .fr p { margin: 0; font-size: 10.4pt; line-height: 1.5; color: #2f2f38; }
.frn { border-right: 2px solid #f6c6df; padding-right: 4mm; display: flex; flex-direction: column; justify-content: center; }
.frn b { font-family: Georgia, serif; font-size: 27pt; color: #e2178f; line-height: 1.05; } .frn span { font-size: 8.6pt; color: #4a4a52; line-height: 1.3; margin-top: 1mm; }
.rule.wide { grid-template-columns: 40mm 1fr; margin-top: 3mm; gap: 5mm; }
.wts div { display: flex; flex-direction: column; margin-bottom: 1.4mm; } .wts span { display: block; height: 2.6mm; background: #e2178f; border-radius: 0 1mm 1mm 0; min-width: 1mm; }
.wts em { font-style: normal; font-size: 7.4pt; color: #6a6a72; margin-top: 0.4mm; }
.defbox { background: #011c45; color: #fff; border-radius: 2.5mm; padding: 4mm 5mm 2.5mm; margin-bottom: 3.5mm; }
.defbox h3 { color: #f4a6d0; margin: 0 0 1.5mm; font-size: 13pt; } .defbox p { font-size: 10.8pt; line-height: 1.55; } .defbox b { color: #fff; }
.chartcard { background: #fff; border: 1px solid #efe6f0; border-radius: 2.5mm; padding: 3.5mm 4.5mm 2mm; margin-bottom: 3mm; }
.chartcard h4 { font-size: 11.5pt; margin-bottom: 0.8mm; } .chartcard .cap { margin: 0 0 1.5mm; font-size: 9.2pt; color: #4a4a52; }
.st3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 3mm; margin-bottom: 3mm; }
.st3 > div { background: #fff; border: 1px solid #efe6f0; border-radius: 2.5mm; padding: 3.2mm 3.6mm; position: relative; }
.st3 .fn { position: absolute; right: 3mm; top: 3mm; } .st3 h4 { font-size: 10.6pt; } .st3 p { font-size: 9.4pt; line-height: 1.45; color: #34343c; margin: 0; }
.fair.tight { gap: 3mm; margin-top: 1mm; margin-bottom: 3mm; } .fair.tight .fr { padding: 4mm 5mm; } .fair.tight .fr p { font-size: 9.8pt; line-height: 1.45; }
.fair.tight .frn b { font-size: 22pt; } .fair.tight h4 { font-size: 11.5pt; }
.fair.tight.dense { gap: 2.2mm; } .fair.tight.dense .fr { padding: 2.8mm 5mm; } .fair.tight.dense .fr p { font-size: 9.4pt; line-height: 1.4; }
.fair.steps { gap: 2.6mm; } .fair.steps .fr { padding: 3.6mm 5mm; grid-template-columns: 30mm 1fr; } .fair.steps .fr p { font-size: 9.9pt; line-height: 1.5; }
.fair.steps .frn b { font-size: 26pt; } .fair.steps .frn span { text-transform: uppercase; letter-spacing: 1px; font-weight: 700; color: #e2178f; font-size: 7.6pt; }
.mlt { display: block; width: fit-content; margin-top: 1.6mm; background: #f4f6fb; border-left: 2.4px solid #011c45; padding: 0.6mm 2.4mm; font-size: 8.4pt; font-weight: 600; color: #011c45; border-radius: 0 1.4mm 1.4mm 0; }
ul.cols2 { columns: 2; column-gap: 8mm; } ul.cols2 li { break-inside: avoid; }
.example { background: #fff7fb; border-left: 3px solid #e2178f; padding: 3mm 4mm; font-size: 10pt; margin: 3mm 0 0; border-radius: 0 2mm 2mm 0; }
.closing { margin-top: 6mm; background: #e2178f; color: #fff; border-radius: 2.5mm; padding: 5mm 6mm; font-family: Georgia, serif; font-size: 14pt; line-height: 1.4; }
.closing span { display: block; font-family: "Segoe UI", sans-serif; font-size: 8pt; font-weight: 700; letter-spacing: 1.6px; text-transform: uppercase; color: #ffd6ec; margin-bottom: 1.5mm; }
/* picks */
.cards { display: flex; flex-direction: column; gap: 2.2mm; }
.card { display: grid; grid-template-columns: 33mm 1fr; gap: 3.6mm; background: #fff; border: 1px solid #efe6f0; border-radius: 3mm;
        padding: 2.2mm; height: 42mm; }
.ph { width: 33mm; height: 37.6mm; border-radius: 2.4mm; position: relative; background-color: #f6dcea; background-repeat: no-repeat;
      background-image: radial-gradient(circle at 50% 30%, #fde9f3, #efd3e4); }
.crk { position: absolute; left: 1.6mm; top: 1.6mm; width: 8mm; height: 8mm; border-radius: 50%; background: #e2178f; color: #fff;
       font-weight: 700; font-size: 10pt; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 0 1.5px #fff; }
.cb { display: flex; flex-direction: column; min-width: 0; }
.ctop { display: flex; justify-content: space-between; align-items: baseline; gap: 3mm; }
.cnm { font-family: Georgia, serif; font-weight: 700; font-size: 13.5pt; color: #011c45; line-height: 1.15; margin-right: 2.4mm; }
.csub { font-size: 7.6pt; color: #7c7c84; white-space: nowrap; }
.cmid { display: grid; grid-template-columns: 1fr 58mm; gap: 4mm; margin-top: 0.6mm; flex: 1; }
.chl { font-size: 9.6pt; font-weight: 700; color: #e2178f; margin: 0 0 0.6mm; }
.cwhy { font-size: 8.4pt; line-height: 1.35; color: #34343c; margin: 0; }
.cvs { border-left: 1px solid #f2ebf3; padding-left: 3.4mm; }
.vl { font-size: 7pt; color: #6a6a72; margin-bottom: 0.8mm; }
.vr { display: flex; align-items: center; gap: 1.6mm; height: 4.2mm; }
.vb { display: block; height: 3.2mm; border-radius: 0 1mm 1mm 0; flex: none; }
.vb.us { background: #e2178f; } .vb.ty { background: #c9c7cc; height: 2.4mm; }
.vr b { font-family: Georgia, serif; font-size: 11pt; color: #e2178f; } .vr em { font-style: normal; font-size: 7.6pt; color: #7c7c84; white-space: nowrap; }
.vf { font-size: 7pt; color: #6a6a72; margin-top: 0.8mm; }
.crr { display: flex; gap: 2.4mm; align-items: baseline; background: #f4f6fb; border-left: 2.4px solid #011c45; border-radius: 0 1.6mm 1.6mm 0;
       padding: 0.9mm 2.2mm; margin-top: 0.8mm; }
.crr b { flex: none; font-size: 7.2pt; letter-spacing: 1px; text-transform: uppercase; color: #011c45; }
.crr span { font-size: 8pt; color: #2a2f45; line-height: 1.3; }
.cf { display: flex; gap: 2mm; align-items: center; font-size: 7pt; color: #8a8a92; flex: none; }
.conf { display: flex; align-items: center; gap: 1.6mm; } .conf-bar { width: 14mm; height: 2mm; background: #f1e6ef; border-radius: 1mm; overflow: hidden; }
.conf-bar span { display: block; height: 100%; background: #e2178f; } .conf-t { font-size: 7.2pt; color: #4a4a52; white-space: nowrap; }
/* tables */
table { border-collapse: collapse; width: 100%; background: #fff; }
th { font-size: 8.4pt; font-weight: 600; color: #7c7c84; text-align: left; padding: 1.4mm 2mm; border-bottom: 1.2px solid #011c45; line-height: 1.25; background: #fff; }
td { padding: 1.8mm 2mm; border-bottom: 0.6px solid #ece6ef; vertical-align: middle; font-size: 9.8pt; }
th.n, td.n { text-align: right; }
table.defs tr.win td { font-weight: 700; color: #011c45; background: #fff2f9; }
table.defs td:first-child { width: 92mm; } table.defs td { padding: 1.5mm 2mm; }
.sbar { width: 62mm; height: 2.4mm; background: #f1e6ef; border-radius: 1.2mm; overflow: hidden; } .sbar span { display: block; height: 100%; background: #e2178f; }
table.jobs td:first-child { font-weight: 600; width: 34%; } table.jobs td { padding: 1.2mm 2mm; font-size: 9.2pt; line-height: 1.3; }
table.roles td.rl { width: 28%; font-weight: 650; color: #011c45; }
table.dd td { padding: 1.15mm 2mm; } td.gap, th.gap { width: 8mm; border: none; background: transparent; } td.rk2 { width: 7mm; color: #e2178f; font-weight: 700; }
.legend { display: flex; gap: 6mm; font-size: 9.6pt; color: #4a4a52; margin: 0 0 1mm; }
.legend i { display: inline-block; width: 2.8mm; height: 2.8mm; border-radius: 50%; margin-right: 1.6mm; }
.fit { display: grid; grid-template-columns: 1fr 1fr; gap: 3.5mm; margin-top: 2mm; }
.ft { background: #fff; border: 1px solid #efe6f0; border-radius: 2.5mm; padding: 3.8mm 4.2mm; font-size: 9.9pt; } .ft p { margin: 0; color: #3c3c46; }
.ft:last-child { grid-column: span 2; }
.stats3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 3mm; margin-bottom: 1mm; }
.leagues { display: flex; flex-wrap: wrap; gap: 1.6mm; margin: 1mm 0 2mm; }
.leagues span { background: #fff; border: 1px solid #efe6f0; border-radius: 10mm; padding: 1.1mm 3mm; font-size: 8.8pt; color: #3c3c46; } .leagues b { color: #e2178f; }
.mini3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 2mm; margin: 1.5mm 0 2mm; }
.mini3 b { display: block; font-family: Georgia, serif; font-size: 17pt; color: #e2178f; line-height: 1.1; } .mini3 span { font-size: 8.3pt; color: #4a4a52; line-height: 1.25; display: block; }
.gh { display: grid; grid-template-columns: 12mm 1fr; gap: 3mm; background: #011c45; color: #fff; border-radius: 2.5mm; padding: 3.5mm 4mm; margin-top: 3.5mm; }
.gh .ic { width: 10mm; height: 10mm; color: #f4a6d0; } .gh h4 { color: #fff; } .gh p { font-size: 10pt; color: #dfe4ef; margin: 0 0 1.5mm; }
.ghl { display: inline-block; background: #e2178f; color: #fff; font-weight: 700; font-size: 8.8pt; padding: 1mm 3mm; border-radius: 1.5mm; }
.steps8 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 2mm; }
.steps8 div { background: #fff; border: 1px solid #efe6f0; border-radius: 2mm; padding: 2mm 2.4mm; display: flex; gap: 2mm; align-items: center; font-size: 9.4pt; color: #011c45; }
.steps8 b { flex: none; width: 6mm; height: 6mm; border-radius: 50%; background: #e2178f; color: #fff; font-size: 8pt; display: flex; align-items: center; justify-content: center; }
.gloss { display: grid; grid-template-columns: 1fr 1fr; gap: 2mm 4mm; } .gloss div { background: #fff; border: 1px solid #efe6f0; border-radius: 2mm; padding: 2mm 2.6mm; }
.gloss b { display: block; color: #011c45; font-size: 9.8pt; } .gloss span { font-size: 9pt; color: #4a4a52; }
"""
doc = f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Who delivers when it matters?</title><style>{CSS}</style></head><body>{"".join(S)}</body></html>'
open(HTML_PATH, "w", encoding="utf-8").write(doc)
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--no-margins", "--allow-file-access-from-files",
                f"--print-to-pdf={PDF_PATH}", URL(HTML_PATH)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=240)
import fitz
os.makedirs(PREV, exist_ok=True)
for f__ in os.listdir(PREV):
    os.remove(os.path.join(PREV, f__))
d = fitz.open(PDF_PATH)
for i, pg in enumerate(d):
    pg.get_pixmap(dpi=80).save(os.path.join(PREV, f"slide_{i + 1:02d}.png"))
log.info(f"{len(d)} slides; PDF {PDF_PATH}")
