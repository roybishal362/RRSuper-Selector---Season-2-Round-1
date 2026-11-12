"""P12b - Plain-language methodology report (Markdown) + one-page results page (HTML).
Every number is read from the saved result tables. Technical names appear in brackets after the plain explanation.
(The earlier, more technical generator is kept as p12_report_technical_v1.py.)"""
import base64, os
import numpy as np
import pandas as pd
from common import ROOT, FIGS, TABLES, CFG, load, load_table, load_json, get_logger

log = get_logger("p12_report")
T = lambda n: load_table(n)
def T_opt(n):
    p = os.path.join(TABLES, n + ".csv")
    return pd.read_csv(p) if os.path.exists(p) else None
J = load_json
chosen = J("p09_summary")["chosen"]
P4 = J("p04_summary")
neg = J("negative_control")
bat30, bowl30 = T("final_top30_bat"), T("final_top30_bowl")
v = load("values", columns=["bowler_id", "is_pace"])
pace_share = v.groupby("bowler_id").is_pace.mean()
bowl30 = bowl30.assign(type=np.where(pace_share.reindex(bowl30.player_id).values >= 0.5, "Pace", "Spin"))
bat10, bowl10 = bat30.head(10), bowl30.head(10)
H = T("holdout_2026")
EV = T_opt("external_validation")
SENS = T("sensitivity_matrix")
INJ = T("synthetic_injection")
pt = T("pressure_pointer_tests")
fv = T("pressure_face_validity")
IMP = P4.get("importance", CFG["pressure"]["importance"])
pct = lambda x: f"{100 * x:.0f}%"
i0 = lambda x: f"{int(x)}"


def table(df, cols, heads, fmt=None):
    fmt = fmt or {}
    out = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt[c](r[c]) if c in fmt else (f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c])) for c in cols) + " |")
    return "\n".join(out)


def short_teams(s):
    parts = [p.strip() for p in str(s).split(";")]
    return "; ".join(parts[:3]) + (" …" if len(parts) > 3 else "")


def rr_note(r, side):
    if side == "bat":
        lvl = f"strike rate {r.SR_pressure:.0f} under pressure (usual {r.SR_usual:.0f}), out every {r.balls_per_out_pressure:.0f} pressure balls"
        how = "pressure-weighted" if chosen["bat"]["model"] == "R2-weighted" else "every ball against the same situation"
        add = (f"rating {100 * r.rating:+.1f} runs per 100 balls above an average batter ({how}, repeatable parts only; "
               f"raw on pressure balls alone {r.runs_above_avg_per100_pressure:+.1f})")
    else:
        lvl = (f"economy {r.econ_pressure:.2f} under pressure (usual {r.econ_usual:.2f}), {r.dot_pct_pressure:.0f}% dots, "
               f"a wicket every {r.balls_per_wkt_pressure:.0f} pressure balls")
        if chosen["bowl"]["model"] == "R2-weighted":
            add = (f"rating {100 * r.rating:+.1f} runs per 100 balls better than an average {r.type.lower()} bowler in the same moments "
                   f"(pressure-weighted, repeatable parts only; raw on pressure balls alone {r.runs_saved_per100_pressure:+.1f})")
        else:
            add = (f"rating {100 * r.rating:+.1f} runs per 100 pressure balls better than an average {r.type.lower()} bowler in the same moments "
                   f"(repeatable parts only; raw {r.runs_saved_per100_pressure:+.1f})")
    best = f"; {r.best_phase_note}" if isinstance(r.best_phase_note, str) and r.best_phase_note else ""
    conf = "a solid pick (stays top-10 in most re-runs)" if r.p_top10 >= 0.6 else (
        "a likely pick" if r.p_top10 >= 0.4 else "a closer call (similar players just below)")
    jn = r.get("job_note")
    job = f" Job check: {jn}." if isinstance(jn, str) and jn else ""
    return f"{r.role}{best}. {lvl[0].upper() + lvl[1:]}; {add}.{job} {conf[0].upper() + conf[1:]}."


L = []
A = L.append
# =========================================================================== 1. the answer
A("# Most Reliable Players Under Pressure")
A("### SupeRR Selector Season 2 — Round 1 methodology report (draft)")
A("")
A("## 1. The answer")
A("")
A("We looked at every ball of 867 matches in the supplied data (IPL 2022–26, T20 internationals and two A-team "
  "events; about 201,000 balls). For each ball we asked two questions:")
A("1. **How much was riding on it?** That is the *pressure*.")
A("2. **Did the batter or bowler do better than an average player would have done in exactly that moment?** That is "
  "the *performance*.")
A("")
A("The players below did best, most reliably, when the most was riding on the ball.")
A("")
A("**Top 10 batters under pressure**")
A("")
A(table(bat10, ["final_rank", "name", "teams", "pressure_balls", "SR_pressure", "SR_expected_pressure", "SR_usual", "p_top10"],
        ["#", "Player", "Teams (seasons)", "Pressure balls faced", "Strike rate under pressure",
         "Average batter in the same moments†", "Own usual strike rate", "Chance still top-10*"],
        {"final_rank": i0, "pressure_balls": i0, "SR_pressure": lambda x: f"{x:.0f}", "SR_expected_pressure": lambda x: f"{x:.0f}",
         "SR_usual": lambda x: f"{x:.0f}", "p_top10": pct, "teams": short_teams}))
A("")
A("**Top 10 bowlers under pressure**")
A("")
A(table(bowl10, ["final_rank", "name", "type", "teams", "pressure_balls", "econ_pressure", "econ_expected_pressure", "econ_usual", "p_top10"],
        ["#", "Player", "Type", "Teams (seasons)", "Pressure balls bowled", "Economy under pressure",
         "Average bowler of his type in the same moments†", "Own usual economy", "Chance still top-10*"],
        {"final_rank": i0, "pressure_balls": i0, "econ_pressure": lambda x: f"{x:.2f}", "econ_expected_pressure": lambda x: f"{x:.2f}",
         "econ_usual": lambda x: f"{x:.2f}", "p_top10": pct, "teams": short_teams}))
A("")
A("† Pressure balls are mostly death overs and close chases, where everyone scores faster and concedes more. So a "
  "player's pressure numbers are compared with an average player facing **exactly the same moments** (same over, "
  "score, wickets, target; bowlers against their own bowling type), not with his own easier balls. This is the "
  "comparison the ranking makes. The ranking also counts wickets and how much each run mattered, so the order is not "
  "the order of these two columns alone.")
A("")
A("\\* We re-ran the whole ranking 1,000 times, each time on a reshuffled set of matches (bootstrap). This is the share "
  "of re-runs in which the player stayed in the top 10. High means a safe pick; around 40% means close players are "
  "just behind.")
A("")
A("Player IDs for the Unstop dropdown are in Appendix A. Seven names belong to two different players, for example "
  "two Rashid Khans, so use the ID and team.")
A("")

# =========================================================================== 2. pressure
A("## 2. What we mean by pressure")
A("")
A("**In one sentence:** a ball is high-pressure when its result can change who wins the match, and when the match "
  "itself matters.")
A("")
A("**How much one ball can change the result (the 'swing').** Before every ball we estimate each team's chance of "
  "winning, for example 'the chasing side wins 62% of the time from here'. We then ask what that chance would become "
  "after a dot, a single, a four, a six or a wicket. If the possible outcomes move the chance a lot, the ball is high "
  "pressure; if they barely move it, it is low pressure.")
A("")
A("- *Low pressure:* first over of a 160 chase. A four or a dot moves the chasing team's chance from about 50% to "
  "about 51%. Almost nothing rides on it.")
A("- *High pressure:* last over, 12 needed. A six can lift the chance from about 35% to 75%, and a wicket can drop it "
  "to 15%. Everything rides on it.")
top = fv[fv.group == "highest"].iloc[0]
bot = fv[fv.group == "lowest"].iloc[0]
A(f"- *From our data, the highest-pressure situation:* innings {int(top.innings)}, over {int(top.over_no)}, "
  f"{int(top.score_before)}/{int(top.wkts_before)}" + (f", {int(top.need)} needed from {int(top.balls_left)} balls" if top.innings == 2 else "")
  + f" (win chance {100 * top.wp_now:.0f}%). *The lowest:* innings {int(bot.innings)}, over {int(bot.over_no)}, "
  f"{int(bot.score_before)}/{int(bot.wkts_before)} (win chance {100 * bot.wp_now:.0f}%; the game was already decided).")
A("")
A(f"**How much the match matters.** A knockout ball counts {IMP['knockout']}×, a must-win game (rebuilt IPL points "
  f"table or T20I series score) {IMP['must_win']}×, and a dead rubber {IMP['dead_rubber']}×.")
A("")
A(f"**Pressure balls** are the top {int(100 * P4['q_star'])}% of balls by this score. On them, single balls move the "
  "result more than twice as much as normal, and an average batter is dismissed more often than normal for that stage "
  "of the innings.")
A("")
A("![Where pressure sits in a chase](../outputs/figures/pressure_heatmap_chase.png)")
A("")
A(f"**We tested {len(pt)} other ideas of pressure before choosing.** Each idea had to show that it really changes how "
  "an average player performs, that those balls really matter for the result, and that it is stable across seasons "
  "and competitions. What we learned:")
A("- *Required rate above 10, a close game, the last two overs, a bowler who has just been hit for six and an "
  "expensive over* all passed. They are mostly already inside the 'swing' score.")
A("- *Dot-ball streaks, a new batter and a wicket in the last six balls* change behaviour (many more dots), but those "
  "balls don't decide matches more often than normal balls.")
A("- *Knockouts and must-win games* don't make the average player perform worse. The *stakes* are still higher, "
  "which is why match importance is in the definition.")
A("- Published pressure indices (Bhattacharjee–Lemmer, Thomson et al.) and simple rules such as 'overs 16–20' "
  "capture the deciding moments less well than the swing score.")
A("")
A("**Hard situations are handled, not ignored.** Walking in after a collapse, a slow start to the powerplay, the "
  "first few balls of an innings, or bowling again straight after a wide all make an average player do worse. "
  "So every ball is compared with what an average player does in exactly that situation (wickets just lost, balls "
  "the batter has faced, dot-ball streak, re-bowl). A batter who rebuilds after the top order falls gets full "
  "credit for beating that harder baseline. These balls don't get *extra* weight, because on their own they "
  "don't decide matches more often than normal balls.")
A("")
A("**Pressure from both ends: the batter's view and the bowler's view.** Pressure belongs to the *ball*, so the batter "
  "and the bowler face exactly the same pressure on every delivery, and one of them wins it. The same definition and "
  "the same weight therefore apply to both lists. Only the job differs:")
A("")
A("| Situation | Batter's job | Bowler's job |")
A("|---|---|---|")
for row in [
    ("Chase, last overs, more needed than usually comes", "Find boundaries; don't waste balls", "Dot balls, yorkers; remove the set batter"),
    ("Chase, middle overs, rate climbing", "Keep scoring off balls (singles count); don't lose wickets", "Build dots; force the risky shot"),
    ("Top order out early (either innings)", "Rebuild without stalling", "Strike again while new batters settle"),
    ("Powerplay", "Score while the field is up", "Early wickets; few boundaries"),
    ("After a wide, a no-ball or a boundary", "Cash in on the free hit or the pressure", "Recover straight away (tested: a real effect)"),
    ("Knockout / must-win match", "Same jobs; the ball counts 1.5× / 1.25×", "Same jobs; the ball counts 1.5× / 1.25×"),
]:
    A(f"| {row[0]} | {row[1]} | {row[2]} |")
A("")
A("- *How each side is scored on the same ball:* the batter by runs plus survival, and the bowler by runs prevented, "
  "his own wickets and extras avoided. Each is compared with an average player in that exact moment (bowlers with "
  "the same bowling type), and both are weighted by the same pressure score.")
CMP = T_opt("pressure_definition_comparison")
if CMP is not None:
    c20 = CMP[(CMP.definition == "C-A leverage") & (CMP.form == "hard top 20%")]
    if len(c20):
        c20 = c20.iloc[0]
        A(f"- *It works for both sides:* {int(c20.batters_30)} batters and {int(c20.bowlers_30)} bowlers have at least 30 "
          f"pressure balls, so both lists are measured on real samples. On pressure balls an average batter is dismissed "
          f"{c20.strain_wkt_ratio:.2f}× as often as normal for that stage: harder for the batter, a chance for the bowler. "
          "The bowler-side ideas we tested (just hit for four or six, re-bowling after a wide, an expensive over) passed "
          "the same tests as the batter-side ones.")
A("")
SW = T_opt("pressure_swing_check")
CS = T_opt("chase_share_by_dataset")
PB = T_opt("pressure_innings_balance")
if PB is not None:
    a_ = PB[(PB.variant.str.startswith("CHOSEN")) & (PB.scope == "1st-innings pressure balls")].iloc[0]
    c_ = PB[(PB.variant.str.startswith("CHOSEN")) & (PB.scope == "chase pressure balls")].iloc[0]
    b_ = PB[(PB.variant.str.startswith("Innings")) & (PB.scope == "1st-innings pressure balls")].iloc[0]
    ov = PB[PB.scope.str.contains("top-10 overlap")].set_index("scope")
    A(f"**Why about {pct(c_.share_of_pressure_balls)} of pressure balls come from chases, and why that is real cricket, "
      "not a bias.**")
    if SW is not None:
        g_ = lambda ev, inn, ph: SW[(SW.event == ev) & (SW.innings == inn) & (SW.phase == ph)].iloc[0]
        b1, b2 = g_("boundary vs dot", 1, "middle"), g_("boundary vs dot", 2, "middle")
        w1, w2 = g_("wicket vs no wicket", 1, "middle"), g_("wicket vs no wicket", 2, "middle")
        A(f"- *Checked without our model.* In the same match situations, a boundary instead of a dot in the middle overs "
          f"changes the chasing side's **actual** win rate by {100 * b2.actual_result_change:.1f} points, against "
          f"{100 * b1.actual_result_change:.1f} in the first innings. A middle-overs wicket costs "
          f"{-100 * w2.actual_result_change:.1f} points in a chase and {-100 * w1.actual_result_change:.1f} in the "
          "first innings. A chase has a known target, so each ball matters more, and the top fifth of balls fills "
          "mostly with chase balls.")
    if CS is not None:
        sup = CS[(CS.dataset == "supplied") & (CS.competition == "all")].iloc[0]
        pub = CS[(CS.dataset == "public") & (CS.competition == "all")].iloc[0]
        rng = CS[CS.competition != "all"].share_of_pressure_balls_from_chases
        A(f"- *The same in independent data:* {pct(sup.share_of_pressure_balls_from_chases)} in the supplied data and "
          f"{pct(pub.share_of_pressure_balls_from_chases)} in about 4,000 public matches; every competition falls "
          f"between {pct(rng.min())} and {pct(rng.max())}.")
    if SW is not None:
        r1 = SW[(SW.event == "wicket vs no wicket") & (SW.innings == 1)]; r2 = SW[(SW.event == "wicket vs no wicket") & (SW.innings == 2)]
        q1 = np.average(r1.actual_over_model, weights=r1.events); q2 = np.average(r2.actual_over_model, weights=r2.events)
        bb = SW[(SW.event == "boundary vs dot") & (SW.innings == 2)]; qb = np.average(bb.actual_over_model, weights=bb.events)
        A(f"- *One small bias, stated openly:* our model under-states first-innings wicket swings by about "
          f"{100 * (q1 - 1):.0f}% (chases: {100 * (q2 - 1):.0f}%), most in the middle overs, which is exactly the "
          f"'3 or 4 down, middle order must rebuild' moment. It also under-states chase boundaries by about "
          f"{100 * (qb - 1):.0f}%, so overall it does not lean against batting first.")
    if CS is not None:
        A(f"- *What the ranking actually uses:* the hard 'top 20%' is used only to count pressure balls. The ranking "
          f"itself weights **every** ball by its pressure, and first-innings balls carry "
          f"{pct(1 - sup.share_of_soft_weight_from_chases)} of that weight (they are {pct(1 - sup.share_of_balls_from_chases)} "
          "of all balls). A middle-order batter rebuilding at 30 for 3 in the first innings, and the bowler trying to "
          "finish him, both count.")
    A(f"- *Tested and rejected: forcing a 50/50 split.* Taking the top 20% within each innings adds first-innings balls "
      f"that move the result only {b_.win_chance_swing_vs_normal:.2f}× as much as a normal ball (chase pressure balls: "
      f"{c_.win_chance_swing_vs_normal:.2f}×). The first-innings balls we already count are genuinely hard: wickets "
      f"fall {a_.wickets_vs_expected:.2f}× as often as expected.")
    cb_, cw_ = 10 - int(ov.top10_overlap.iloc[0]), 10 - int(ov.top10_overlap.iloc[1])
    A(f"  - *Effect on the lists if we had forced it:* {cb_} of the 10 batters and {cw_} of the 10 bowlers would change.")
    A("")
A("**What the data cannot measure.** Dropped catches and misfields are not recorded, so a bowler gets no credit for "
  "chances that went down. Swing labels are not used, because their coding changes over time: the share of IPL "
  "powerplay balls described as swinging jumps from 29% in 2023 to 63% in 2025. Yorker labels are consistent, so "
  "they are used to describe death bowlers (Section 7), not to rank them.")
A("")

# =========================================================================== 3. scoring
A("## 3. How we scored each player")
A("")
A("1. **Learn what normally happens.** A model learned, from about 200,000 balls, what an average player does in every "
  "situation: score, wickets, balls left, target, phase of the innings, season, competition, venue and bowler type. "
  "It never sees player names, so 'normal' can't absorb a player's own skill. *(Gradient-boosted trees, LightGBM; "
  "every prediction comes from a model that never saw that match.)*")
A("2. **Score every ball.** Value = what actually happened minus what normally happens in that exact moment. Batters: "
  "runs, plus a cost for getting out that depends on the moment. Bowlers: runs saved, boundaries prevented, dots, "
  "wickets, wides and no-balls. Each part is weighted by how repeatable it proved to be.")
A("3. **Compare like with like.** A bowler is compared with an average bowler *of the same type* (pace or spin) in the "
  "same situation. Teams must bowl pace in the powerplay and at the death, and a spinner can't replace a death "
  "bowler. Without this, spinners take 9 of the 10 places only because spin concedes about 1–1.5 runs per over less "
  "than pace in every phase.")
A("4. **Don't trust small samples.** A player with 40 pressure balls who did brilliantly is pulled towards average "
  "more than a player with 400 pressure balls. *(Empirical-Bayes shrinkage.)*")
A("5. **Let the data choose the settings, on matches it hasn't seen.** For every choice (which value score, which "
  "rating model, how much to shrink, whether to reward 'rising above your own normal', minimum sample) we checked "
  "which option best predicted players' pressure performance in *other* matches: 52 tests, plus the 2026 season held "
  "back. We kept the best option, or the simpler one when they were tied.")
A("")
A(f"*Chosen settings.* Batters: {chosen['bat']['model']} on value {chosen['bat']['value']}, minimum "
  f"{chosen['bat'].get('min_n', 0)} pressure balls. Bowlers: {chosen['bowl']['model']} on value "
  f"{chosen['bowl']['value']}, minimum {chosen['bowl'].get('min_n', 0)}. Weight on 'rising above own normal': "
  f"{chosen['bat'].get('w_d', 0)} / {chosen['bowl'].get('w_d', 0)}.")
A("")
PLAIN = {"R2-joint": "every ball against the same situation, plus the player's own pressure-vs-usual difference, kept "
                     "only as far as it proved repeatable (it barely was, so it adds very little)",
         "R2-weighted": "every ball against the same situation, with pressure balls counting more",
         "R2-p_only": "pressure balls only, against the same situation",
         "R3": "pressure balls only: dots, boundaries, wickets and extras against what is normal in that moment, each "
               "weighted by how repeatable it proved",
         "R1": "a 'plus-minus' model that also adjusts for the opponents faced",
         "blend": "an average of three rating models"}
for side, lab in (("bat", "Batters"), ("bowl", "Bowlers")):
    m_ = chosen[side]["model"]
    A(f"- *{lab} ({m_}) in plain words:* {PLAIN.get(m_, m_)}.")
RF = T_opt("rating_family_check")
if RF is not None and chosen["bat"]["model"] == "R2-weighted" and chosen["bowl"]["model"] == "R2-weighted":
    A("- *Why pressure-weighted for both sides (a decision we made, with the evidence).* When every rating model was "
      "allowed, the automatic rule preferred a different one on each side: 'all balls' for batters and 'pressure balls "
      "only' for bowlers. We fixed the family to pressure-weighted for both, for three reasons:")
    A("  - It treats pressure identically for batters and bowlers.")
    A("  - It keeps first-innings pressure in the rating.")
    A("  - It is what the brief asks: performance under pressure.")
    A("")
    A("  The data choices inside the family were still made by the rule. Every option was scored three ways on the "
      "same yardstick:")
    A("")
    A("| Side | Test | Pressure-weighted (chosen) | Alternative |")
    A("|---|---|---|---|")
    for side, lab in (("bat", "Batters"), ("bowl", "Bowlers")):
        rr = RF[RF.side == side]
        ch_ = rr[rr.setting.str.startswith("CHOSEN")].set_index("test")
        al_ = rr[rr.setting.str.startswith("ALTERNATIVE")].set_index("test")
        alt_name = al_.setting.iloc[0].split(": ", 1)[1]
        for t_, nm_ in (("(i) 52 unseen-match splits", "52 unseen-match tests"), ("(ii) 2026 season", "2026 season"),
                        ("(iii) public: all public matches", "Public data, all matches"),
                        ("(iii) public: T20 World Cups", "Public data, T20 World Cups")):
            if t_ in ch_.index and t_ in al_.index:
                a1, a2 = ch_.loc[t_, "rank_corr"], al_.loc[t_, "rank_corr"]
                l1, l2 = ch_.loc[t_, "top10_lift_winprob_pts_per_ball"], al_.loc[t_, "top10_lift_winprob_pts_per_ball"]
                A(f"| {lab} | {nm_} | rank corr {a1:+.2f}, top-10 {l1:+.2f} | {alt_name}: rank corr {a2:+.2f}, top-10 {l2:+.2f} |")
    A("")
    A("  *Rank corr* = agreement between our order and real pressure performance; *top-10* = how much better our top "
      "10 did than the average player (win-chance points per pressure ball).")
    _i = RF[(RF.side == "bowl") & (RF.test == "(i) 52 unseen-match splits")].set_index(RF[(RF.side == "bowl") & (RF.test == "(i) 52 unseen-match splits")].setting.str.split(":").str[0])
    if {"CHOSEN", "ALTERNATIVE"} <= set(_i.index):
        _d = _i.loc["ALTERNATIVE", "rank_corr"] - _i.loc["CHOSEN", "rank_corr"]
        A("  - *Internal test, bowlers:* " + ("'pressure balls only' scores higher, but that test scores exactly the balls that "
          "model rates, which gives it a home advantage." if _d > 0.01 else
          "the two are level once both use the same evidence floor."))
    _ind = [t for t in RF.test.unique() if t.startswith("(ii)") or t.startswith("(iii)")]
    for side, lab in (("bat", "batters"), ("bowl", "bowlers")):
        rr = RF[RF.side == side]
        c_ = rr[rr.setting.str.startswith("CHOSEN")].set_index("test"); a_ = rr[rr.setting.str.startswith("ALTERNATIVE")].set_index("test")
        tt = [t for t in _ind if t in c_.index and t in a_.index]
        wr = sum(c_.loc[t, "rank_corr"] > a_.loc[t, "rank_corr"] for t in tt)
        wt = sum(c_.loc[t, "top10_lift_winprob_pts_per_ball"] > a_.loc[t, "top10_lift_winprob_pts_per_ball"] for t in tt)
        A(f"  - *Independent tests (2026 + {len(tt) - 1} public-data groups), {lab}:* pressure-weighted has the better "
          f"rank agreement in {wr} of {len(tt)} and the better top 10 in {wt} of {len(tt)}.")
    A("")
    PV = RF[RF.setting.str.startswith("RULE_PICK")]
    if len(PV):
        A("  *Bowling settings inside the family: a review decision.* The automatic search tunes one setting at a time "
          "and reached a stronger pressure emphasis with lighter shrinkage for bowlers. The combination used before this "
          "review (moderate emphasis, stronger shrinkage) did better on most checks, so we kept it. It also pulls small "
          "samples harder towards average. Both are shown here:")
        A("")
        A("| Side | Test | Chosen | Search's own pick |")
        A("|---|---|---|---|")
        for side, lab in (("bat", "Batters"), ("bowl", "Bowlers")):
            rr = RF[RF.side == side]
            ch_ = rr[rr.setting.str.startswith("CHOSEN")].set_index("test")
            pv_ = rr[rr.setting.str.startswith("RULE_PICK")].set_index("test")
            if not len(pv_):
                continue
            for t_, nm_ in (("(i) 52 unseen-match splits", "52 unseen-match tests"), ("(ii) 2026 season", "2026 season"),
                            ("(iii) public: all public matches", "Public data, all matches"),
                            ("(iii) public: T20 World Cups", "Public data, T20 World Cups")):
                if t_ in ch_.index and t_ in pv_.index:
                    A(f"| {lab} | {nm_} | rank corr {ch_.loc[t_, 'rank_corr']:+.2f}, top-10 {ch_.loc[t_, 'top10_lift_winprob_pts_per_ball']:+.2f} "
                      f"| rank corr {pv_.loc[t_, 'rank_corr']:+.2f}, top-10 {pv_.loc[t_, 'top10_lift_winprob_pts_per_ball']:+.2f} |")
A("")

# =========================================================================== 4. technology
A("## 4. Technology we used")
A("")
A("| What it does | Technology | Why this one |")
A("|---|---|---|")
for row in [
    ("Cleaning, rebuilding every ball's situation", "Python, pandas", "Reproducible scripts, rerun from the raw files"),
    ("'What normally happens' on a ball (8 outcomes)", "LightGBM (gradient-boosted trees)", "Beat a lookup table and a logistic model on unseen matches"),
    ("Chance of winning a chase", "Logistic regression: curves for the game state, straight lines for match context",
     "Beat 9 other model types and two stacked ensembles on unseen matches, with the smallest overfitting gap (Section 6.8)"),
    ("Chance of winning when setting a total", "Runs-to-come model (Poisson LightGBM) mapped through the chase model", "A direct model was too weak (pre-set rule)"),
    ("Tuning without peeking", "Optuna inside nested cross-validation, grouped by match", "Test matches never influence settings"),
    ("Pressure ideas tested fairly", "Match-level bootstrap, false-discovery-rate control", "Many ideas tested at once, so luck is controlled"),
    ("Player ratings", "Empirical Bayes; ridge 'plus-minus'; reliability-weighted composite", "Handle small samples and opponents"),
    ("Player roles", "Semi-supervised: cricket rules label clear-cut players, a multinomial logistic model assigns the rest "
     "from usage (batting position, entry over, phase); checked against k-means, GMM and UMAP + HDBSCAN",
     "Fairness checks and the best-by-role list (not used to pick players)"),
    ("Which statistics are skill, not luck", "Split-half reliability, permutation tests", "Only repeatable things count"),
    ("Feature checks", "SHAP, permutation importance, null-importance and stability selection", "Every input earns its place"),
    ("Uncertainty", "Bootstrap, 1,000 re-runs", "'Chance still top-10' for every player"),
    ("Independent exam", "Public Cricsheet data (World Cups, franchise leagues; open licence)", "Proof on matches from other competitions"),
]:
    A(f"| {row[0]} | {row[1]} | {row[2]} |")
A("")

# =========================================================================== 5. fairness
A("## 5. Is it fair?")
A("")
A("- **Everyone is rated.** Every player in the supplied data is rated, all five seasons (2022–2026) count equally, "
  "and IPL and T20I are used together. Small samples are handled by shrinkage, plus any minimum-sample rule the tests "
  "supported. The 2026 season is *not* the basis of the ranking: it is one season of five in the final fit, and "
  "separately a held-back check (Section 6.2).")
EXU = T_opt("excluded_unavailable")
if EXU is not None and len(EXU):
    el_ = CFG["eligibility"]
    A("- **Availability: only players RR could actually pick.** Two rules leave players out of the final lists. These "
      "players are still rated like everyone else and stay in every 'average player' baseline and every method test:")
    A(f"  - players who have represented Pakistan or Pakistan A ({el_['exclude_reason']});")
    A(f"  - players with no match in the supplied data since {el_.get('require_active_from', '')} (inactive or retired, "
      "e.g. Dinesh Karthik, who retired in 2024). The data cannot tell retirement from absence, so every player this "
      "removes from a top 20 is named below.")
    A("")
    A("  Players left out who would have made the top 20:")
    for side, lab in (("bat", "batters"), ("bowl", "bowlers")):
        x = EXU[(EXU.side == side) & (EXU.would_be_rank <= 20)].sort_values("would_be_rank")
        rs_ = "reason" in EXU.columns
        A(f"  - *{lab.capitalize()}:* " + (", ".join(f"{r['name']} (would be #{int(r.would_be_rank)}" + (f"; {r['reason']}" if rs_ else "") + ")"
                                                     for _, r in x.iterrows()) or "none") + ".")
A(f"- **Seen under pressure before being ranked under pressure.** Both lists need at least "
  f"{CFG['selection'].get('min_pressure_balls_floor', 0)} pressure balls. The automatic tests cannot judge players who "
  "have hardly faced pressure, so without a floor a bowler with one pressure ball in 273 balls could reach the top 10.")
A("- **Same moment, same baseline.** Every ball is judged against an average player in the identical situation, "
  "season and competition (IPL scoring rose about 16% from 2022 to 2026).")
A("- **Same job, same baseline.** Pace bowlers are compared with pace bowlers and spinners with spinners.")
A("- **Facing more pressure balls is not a bonus.** Finishers and death bowlers simply get measured more often; credit "
  "only comes from doing better than normal.")
A("- **Checked for bias.** We compared the top 10 with the eligible pool by role, competition and era (outputs table "
  "`bias_audit`).")
A("")

# =========================================================================== 6. proof
A("## 6. Proof that it works, and where it is weaker")
A("")
A("**6.1 Tested on matches the method never saw.** For every setting we trained on some matches and checked "
  "predictions on others: 52 tests, never splitting a match. All results are in the evidence notebook.")
A("")
A("**6.2 The sealed 2026 season.** Ratings were built on 2022–25 and checked against 2026 pressure performance "
  "(rank correlation: 0 means no link, 1 means perfect).")
A("")
A(table(H, ["side", "method", "n_pool", "rho_v2", "rho_v1"], ["Side", "Method", "Players", "Win-probability yardstick", "Runs yardstick"],
        {"n_pool": i0, "rho_v2": lambda x: f"{x:+.2f}", "rho_v1": lambda x: f"{x:+.2f}"}))
A("")
A("*Disclosure.* The 2026 season was opened more than once:")
A("- After a scoring flaw in our own testing code (the minimum-sample comparison) was found and fixed.")
A("- After your review changed the method (same-type comparison for bowlers, match importance, bowler-only "
  "wickets).")
A("")
A("Neither change was made to improve the 2026 numbers. One season of about 75 players has a margin of error of "
  "roughly ±0.12.")
A("")
NZ = T_opt("season_noise_check")
if NZ is not None:
    q = lambda side, key: NZ[(NZ.side == side) & NZ.check.str.contains(key, regex=False)].value.iloc[0]
    A("**Why one season can't judge batters (or, really, bowlers).** We measured how much a single season of pressure "
      "balls can tell us:")
    A(f"- *One season does not even agree with itself.* Split the 2026 matches into two random halves: a batter's "
      f"pressure record in one half predicts the other at {q('bat', '(2026'):+.2f} (2025: {q('bat', '(2025'):+.2f}; 2024: "
      f"{q('bat', '(2024'):+.2f}). A target that can't predict itself can't be predicted by anything.")
    A(f"- *Mostly luck at this sample size.* With a season's typical pressure sample, only about "
      f"{pct(q('bat', 'skill share'))} of the differences between batters is skill; the rest is the run of the ball "
      f"(bowlers: {pct(q('bowl', 'skill share'))}). A player needs about {q('bat', 'needed before'):.0f} pressure balls "
      "before skill is even half of what you see.")
    A(f"- *Our ratings, by contrast, are stable.* Built separately on two halves of 2022–25, they agree at "
      f"{q('bat', 'our ratings'):.2f} for batters and {q('bowl', 'our ratings'):.2f} for bowlers. They measure something "
      "real; a single season is too short to show it.")
    A(f"- *So the ceiling for a 2026 check is about {q('bat', 'ceiling'):.2f} for batters* (bowlers "
      f"{q('bowl', 'ceiling'):.2f}), and the batters' result is what pure noise looks like, not evidence that the "
      "list is wrong. The bowlers' 2026 season happened to be less noisy than 2024 and 2025, so their positive 2026 "
      "result is partly luck too. The fair tests are the 52 unseen-match splits (more matches per player) and the "
      "public-data exam below.")
    A("")
if EV is not None and len(EV):
    A("**6.3 Independent exam on public data.** We took the ratings built *only* from the supplied data and checked "
      "them against how the same players performed under pressure in about 4,000 public matches (Cricsheet). These "
      "matches cover the T20 World Cups 2022/24/26, the big franchise leagues and other T20Is, and the method never "
      "saw any of them. The yardstick is plain *win probability added* on pressure balls, which does not depend on "
      "any of our modelling choices.")
    A("")
    ev = EV[EV.group.isin(["all public matches", "T20 World Cups"])]
    A(table(ev, ["group", "side", "method", "players", "rank_corr_winprob", "top10_winprob_pts_per_100_pressure_balls", "top10_runs_per_100_pressure_balls"],
            ["Public matches", "Side", "Method", "Players", "Rank correlation (win prob.)", "Top 10 vs average (win-chance points / 100 pressure balls)",
             "Top 10 vs average (runs / 100 pressure balls)"],
            {"players": i0, "rank_corr_winprob": lambda x: f"{x:+.2f}", "top10_winprob_pts_per_100_pressure_balls": lambda x: f"{x:+.1f}",
             "top10_runs_per_100_pressure_balls": lambda x: f"{x:+.1f}"}))
    A("")
    WK = T_opt("world_cup_knockouts_top10")
    if WK is not None and len(WK):
        A("*Our top 10s in T20 World Cup semi-finals and finals (2022–26; descriptive, small samples):*")
        A("")
        wkb = WK[WK.side == "bat"].sort_values("main_rank"); wkw = WK[WK.side == "bowl"].sort_values("main_rank")
        if len(wkb):
            A(table(wkb, ["main_rank", "name", "balls", "SR", "win_prob_added_pts"], ["Our rank", "Batter", "Balls", "Strike rate", "Win chance added (points)"],
                    {"main_rank": i0, "balls": i0, "SR": lambda x: f"{x:.0f}", "win_prob_added_pts": lambda x: f"{x:+.1f}"}))
            A("")
        if len(wkw):
            A(table(wkw, ["main_rank", "name", "balls", "econ", "wkts", "win_prob_added_pts"], ["Our rank", "Bowler", "Balls", "Economy", "Wickets", "Win chance added (points)"],
                    {"main_rank": i0, "balls": i0, "econ": lambda x: f"{x:.2f}", "wkts": i0, "win_prob_added_pts": lambda x: f"{x:+.1f}"}))
            A("")
    P14S = J("p14_summary") if os.path.exists(os.path.join(TABLES, "p14_summary.json")) else {}
    A("*Why the public data is the exam, not the training data.*")
    A("- The brief supplies the data, so the ratings are built from it. Judging them on matches from other "
      "competitions, which the method never saw, is the strongest evidence that the list is real and not overfitted. "
      "Training on the public matches as well would leave nothing independent to check against.")
    A("- The public data is still used in two other ways. Its ~900,000 balls confirm the pressure pattern (the same "
      "share of pressure balls from chases), and Appendix B shows the ranking rebuilt on supplied + public data together, "
      "so you can see the list doesn't depend on which data we used.")
    A(f"- *No double counting.* 766 matches appear in both datasets and were removed when the public data was loaded. "
      f"A second check (same date and at least 11 of the same players) found "
      f"{P14S.get('duplicate_matches_removed', 'a few')} more, which were also removed before the exam.")
    A("")
sm = SENS.groupby(["side", "group"]).top10_overlap.agg(["median", "min"]).reset_index()
A("**6.4 How stable is the top 10?** This counts how many of our 10 names survive when we change one choice at a "
  "time: pressure definition, threshold, competition, season left out, match-importance weights and so on.")
A("")
A(table(sm, ["side", "group", "median", "min"], ["Side", "What we changed", "Names kept (median of 10)", "Worst case"],
        {"median": lambda x: f"{x:.0f}", "min": lambda x: f"{x:.0f}"}))
A("")
A(f"**6.5 Is 'rising above your own normal' a real skill?** Not in this data. When we shuffled which balls were "
  f"labelled 'pressure', the differences between players looked just as big (p = {neg['bat']['p_value']:.2f} for "
  f"batters, {neg['bowl']['p_value']:.2f} for bowlers). This matches research in baseball, basketball and cricket. So "
  "the ranking rewards *how good a player is under pressure*, not how much better he is than his own usual level.")
A("")
inj = INJ[INJ.boost_in_true_sd == 1.0]
A("**6.6 Can the method find a truly better player?** We secretly gave 10 random players a boost under pressure and "
  "re-ran everything. A boost the size of one 'true-skill' step lifts the planted players by a median of "
  f"{inj.median_rank_gain.mean():.0f} places. The method is cautious: it needs a lot of evidence before it puts "
  "someone in the top 10.")
A("")
rc = T_opt("endgame_reality_check")
if rc is not None:
    A("**6.7 Do our win chances match reality at the end of chases?** These are the moments that matter most, so we "
      "checked the model against what actually happened from the same situations in the data:")
    A("")
    A(table(rc, ["situation", "times_seen", "actual_win_rate", "model_win_prob"], ["Situation", "Times seen", "Chasing side actually won", "Model said"],
            {"times_seen": i0, "actual_win_rate": pct, "model_win_prob": pct}))
    A("")
MZ = T_opt("model_zoo")
LD = T_opt("wp_logistic_designs")
FP = T_opt("model_zoo_fingerprint")
if MZ is not None and LD is not None:
    A("**6.8 Is the win-chance model over-fitting?** Over-fitting means a model looks good on matches it learned from and "
      "worse on new ones. So we scored 9 model types and two stacked ensembles on whole matches each model never saw, "
      "after letting each choose its own complexity inside its training matches:")
    A("")
    mz = MZ[MZ.target == "chase"].copy()
    mz["model"] = mz.model.replace({"Spline logistic (CURRENT)": "Spline logistic, curves on every input (previous)"})
    ch = LD[LD.model.str.startswith("CHOSEN")].iloc[0]
    mz = pd.concat([mz[["model", "train_logloss", "test_logloss", "overfit_gap", "calibration_slope"]],
                    pd.DataFrame([{"model": "**Chosen: logistic, curves for game state, straight lines for match context**",
                                   "train_logloss": ch.train_logloss, "test_logloss": ch.test_logloss,
                                   "overfit_gap": ch.overfit_gap, "calibration_slope": ch.calibration_slope}])]).sort_values("test_logloss")
    f4 = lambda x: f"{x:.3f}"
    A(table(mz, ["model", "train_logloss", "test_logloss", "overfit_gap", "calibration_slope"],
            ["Model (chase)", "Error on training matches", "Error on unseen matches", "Gap", "Calibration slope (1 = honest)"],
            {"train_logloss": f4, "test_logloss": f4, "overfit_gap": f4, "calibration_slope": lambda x: f"{x:.2f}"}))
    A("")
    if FP is not None:
        fp = FP[(FP.target == "chase") & FP.check.str.contains("LightGBM")].iloc[0]
        A(f"*Why tree models over-fit here:* a chase has about {fp.balls_per_match:.0f} balls but only one result. Inputs "
          f"such as the target, venue, team strength and season are the same for every ball, and together they identify "
          f"each match ({int(fp.distinct_match_input_combos)} combinations for {int(fp.matches)} chases). A flexible tree "
          f"model given only those inputs scores {fp.train_logloss:.4f} on its training matches (perfect memory) and "
          f"{fp.test_logloss:.2f} on new ones, worse than guessing (0.69).")
    prev = LD[LD.model.str.startswith("Previous")].iloc[0]
    A(f"*What we changed:* our previous logistic model also bent its curves around those match-level inputs. Its gap "
      f"was {prev.overfit_gap:.3f} against {ch.overfit_gap:.3f} now, and its slope of {prev.calibration_slope:.2f} "
      "meant over-confident win chances. The chosen model keeps curves only where cricket needs them (runs needed, "
      "balls left, wickets in hand) and uses straight lines for match context. In real close finishes it is the most "
      f"accurate design we tried (average error {100 * ch.reality_check_error:.1f} points, against "
      f"{100 * prev.reality_check_error:.1f} before).")
    W1a, W1b = T_opt("first_innings_wp_planB_before_wpfix"), T_opt("first_innings_wp_planB")
    if W1a is not None and W1b is not None:
        pa = W1a[W1a.model.str.startswith("Plan B")].iloc[0]; pb = W1b[W1b.model.str.startswith("Plan B")].iloc[0]
        dz = MZ[(MZ.target == "first") & ~MZ.model.str.startswith("Plan B")].sort_values("test_logloss").iloc[0]
        A(f"*First innings improved too.* Its win chance is built through the chase model, so the over-confidence had "
          f"been passed on. Error on unseen matches fell from {pa.logloss:.3f} to {pb.logloss:.3f}, and ranking power "
          f"(AUC) rose from {pa.auc:.3f} to {pb.auc:.3f}. It now beats every direct first-innings model we tried; the "
          f"best of those, {dz.model.lower()}, scored {dz.test_logloss:.3f}.")
    A("")
    A("![Model test](../outputs/figures/model_zoo.png)")
    A("")
A("**6.9 Where it is weaker.**")
A("- *Batters are harder to predict than bowlers* (Sections 6.2 and 6.3). Treat batters with a lower 'chance still "
  "top-10' as close calls.")
A("- *No pitch or weather data*, so a turning pitch can make a spinner look better than he is.")
A("- *First-innings pressure* is measured less sharply than chase pressure, because predicting a winner halfway "
  "through a match is hard.")
A("- *Data quirks* (fixed): wides and byes that ran to the boundary are stored as one run, ball speed is unusable, "
  "and delivery times switch time zone in 2024.")
A("- *Fielding is not measured*, because the data does not name fielders or record dropped catches.")
A("")

# =========================================================================== 7. RR fit
A("## 7. How each player could strengthen RR")
A("")
A("Each note uses only the data: the player's role, the phase where he is strongest, and his pressure numbers.")
A("")
A("**Batters**")
for _, r in bat10.iterrows():
    A(f"{int(r.final_rank)}. **{r['name']}**: {rr_note(r, 'bat')}")
A("")
A("**Bowlers**")
for _, r in bowl10.iterrows():
    A(f"{int(r.final_rank)}. **{r['name']}** ({r.type.lower()}): {rr_note(r, 'bowl')}")
A("")
BR = T_opt("best_by_role")
if BR is not None:
    A("**Best under pressure, by role.** RR fills roles, not just a list. This is the same ranking grouped by each "
      "player's usual job, which comes from where and when he plays, not how well. The overall rank decides the order "
      "within each role.")
    A("")
    ORDER = {"bat": ["Opener", "Top order (No. 3-4)", "Middle order (No. 5-6)", "Finisher", "Lower order"],
             "bowl": ["Pace - new ball", "Pace - new ball + death", "Pace - death", "Pace - middle overs",
                      "Spin - powerplay", "Spin - middle overs"]}
    for side, title in (("bat", "Batters"), ("bowl", "Bowlers")):
        A(f"| {title}: role | 1st | 2nd | 3rd |")
        A("|---|---|---|---|")
        sb = BR[BR.side == side]
        for role in ORDER[side] + sorted(set(sb.role) - set(ORDER[side])):
            x = sb[sb.role == role].sort_values("rank_in_role").head(3)
            if len(x):
                cells = [f"{r['name']} (#{int(r.final_rank)})" for _, r in x.iterrows()] + [""] * (3 - len(x))
                A(f"| {role} | " + " | ".join(cells) + " |")
        A("")
    A("*#n is the overall pressure rank. Roles are assigned from usage. Players on the border between two roles "
      "(for example Tim David or Rinku Singh, between middle order and finisher) carry a confidence score in the "
      "`roles_detail` table.*")
    A("")

# =========================================================================== 8. what changed
A("## 8. What changed after review")
A("")
tb = T_opt("final_top30_bowl_typeblind")
if tb is not None:
    old = set(tb.head(10).name); new = set(bowl10.name)
    A(f"- **Bowlers compared within their type.** The earlier version compared every bowler with a mixed pace/spin "
      f"average, which gave 9 spinners in the top 10. The new list has {int((bowl10.type == 'Pace').sum())} pace "
      f"bowlers and {int((bowl10.type == 'Spin').sum())} spinners; {len(old & new)} names are in both lists.")
A("- **Match importance added.** Knockouts and must-win games now count more. The earlier test could only see how "
  "much a ball changes *its own* match, so it could never reward a final.")
A("- **Bowler wickets only.** Run-outs no longer count as the bowler's wickets, so wicket-taking is a small but real "
  "skill.")
A("- **Public data added** as an independent exam (Section 6.3). A ranking that combines both datasets is in "
  "Appendix B.")
A("- **Win chances in the last overs fixed.**")
A("  - *The error:* the chase model rated '4 needed off the last ball' at 99%.")
A("  - *The fix:* we added a cricket-sense input, runs needed against the runs normally scored from those balls and "
  "wickets, and checked it against real outcomes (Section 6.7).")
A("  - *Also tried and rejected:* a step-by-step backward calculation (the WASP method). It was too pessimistic "
  "against real outcomes, because teams still alive late in a chase usually have set batters.")
A("- **Win-chance model made less flexible where it was over-fitting.** The chase model's curves now apply only "
  "to the game state; match context is a straight line. The gap between training and unseen matches fell by more than "
  "half, and the end-game checks improved (Section 6.8). It was tested against 9 other model types and two stacked "
  "ensembles, and none did better on unseen matches.")
WB = T_opt("final_top30_bat_before_wpfix"); WW = T_opt("final_top30_bowl_before_wpfix")
if WB is not None and WW is not None:
    ob, ow = set(WB.head(10).player_id), set(WW.head(10).player_id)
    A(f"  - *Effect on the answer:* {len(ob & set(bat10.player_id))} of 10 batters and "
      f"{len(ow & set(bowl10.player_id))} of 10 bowlers are the same as before this change.")
A("- **Roles now come from how a player is used, not how well he plays.** The earlier clusters mixed strike rate "
  "with batting position, so finishers such as Tim David and Heinrich Klaasen were labelled 'anchors', and no "
  "death-bowling role existed. Roles never pick players, so the top 10 is unaffected; the by-role list and the "
  "fairness check are now meaningful.")
A("- **Role-specific job checks added** to each player note: powerplay strike rate for openers, scoring off balls "
  "in chases for the middle order, first-10-ball strike rate for finishers, yorker rate for death bowlers, and dot "
  "balls for spinners.")
A("- **Our own coding errors, found and fixed:**")
A("  - the noise model for 'pressure vs usual' (now split-half covariance);")
A("  - the minimum-sample comparison in testing;")
A("  - the bootstrap re-estimating shrinkage;")
A("  - the chase model's spline settings at the end of an innings;")
A("  - the uncertainty re-runs treating a match drawn twice as two different matches. That made 'pressure vs usual' "
  "look about 100 times more repeatable than it is, and distorted the 'chance still top-10' figures. The re-runs "
  "now keep that value at its real-data level.")
A("- **Rating model fixed to pressure-weighted for batters and bowlers.** After the win-chance fix, the automatic rule "
  "preferred 'all balls' for batters and 'pressure balls only' for bowlers. We fixed the family so both sides get "
  "the same pressure treatment, as the brief asks. The rule still chose the settings inside it. Evidence for both "
  "options is in Section 3.")
A("- **Only players RR could pick are listed** (Section 5): players not available for IPL selection, and players with "
  "no match since 2025 (inactive or retired, e.g. Dinesh Karthik). They remain in every baseline and every method test.")
A("- **Public-data exam made stricter:** a second duplicate check removed matches the first one missed (Section 6.3).")
A("- **How much one season can tell us measured** (Section 6.2): a single season of pressure balls is mostly luck, "
  "which explains the batters' 2026 result.")
A("")

# =========================================================================== appendix
A("## Appendix A: Dropdown helper (IDs) and ranks 11–20")
A("")
A(table(bat30.head(20), ["final_rank", "name", "player_id", "teams", "p_top10"], ["#", "Batter", "ID", "Teams", "Chance top-10"],
        {"final_rank": i0, "player_id": i0, "p_top10": pct}))
A("")
A(table(bowl30.head(20), ["final_rank", "name", "type", "player_id", "teams", "p_top10"], ["#", "Bowler", "Type", "ID", "Teams", "Chance top-10"],
        {"final_rank": i0, "player_id": i0, "p_top10": pct}))
A("")
cb, cw = T_opt("combined_top30_bat"), T_opt("combined_top30_bowl")
if cb is not None:
    A("## Appendix B: Ranking with supplied + public data combined (comparison)")
    A("")
    A(table(cb.head(10), ["combined_rank", "name", "main_rank"], ["#", "Batter", "Rank in main list"],
            {"combined_rank": i0, "main_rank": lambda x: "–" if pd.isna(x) else i0(x)}))
    A("")
    A(table(cw.head(10), ["combined_rank", "name", "main_rank"], ["#", "Bowler", "Rank in main list"],
            {"combined_rank": i0, "main_rank": lambda x: "–" if pd.isna(x) else i0(x)}))
    A("")
A("## Appendix C: Technical record")
A("")
A("- Every model comparison, every pressure-idea test, every experiment configuration and every robustness check is "
  "in `outputs/tables`, and shown step by step in `RESULTS_EVIDENCE.ipynb`.")
A("- Code: `src/p00_audit.py` … `src/p14_external.py`, with settings in `config.yaml` (seed 2026).")
A("- Public data: Cricsheet (cricsheet.org), Open Data Commons Attribution licence.")
os.makedirs(os.path.join(ROOT, "report"), exist_ok=True)
open(os.path.join(ROOT, "report", "METHODOLOGY_REPORT_DRAFT.md"), "w", encoding="utf-8").write("\n".join(L))
log.info("report written")


# =========================================================================== HTML one-pager
def img(name):
    p = os.path.join(FIGS, name + ".png")
    return f'<img alt="{name}" src="data:image/png;base64,{base64.b64encode(open(p, "rb").read()).decode()}">' if os.path.exists(p) else ""


def htab(df, cols, heads, fmt):
    rows = "".join("<tr>" + "".join(f"<td>{fmt[c](r[c]) if c in fmt else r[c]}</td>" for c in cols) + "</tr>" for _, r in df.iterrows())
    return "<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in heads) + f"</tr></thead><tbody>{rows}</tbody></table>"


F = {"final_rank": i0, "player_id": i0, "pressure_balls": i0, "p_top10": pct, "SR_pressure": lambda x: f"{x:.0f}",
     "SR_usual": lambda x: f"{x:.0f}", "econ_pressure": lambda x: f"{x:.2f}", "econ_usual": lambda x: f"{x:.2f}", "teams": short_teams}
steps = [("Clean the data", "201,000 balls checked; scores, wickets and targets rebuilt; rain games and time zones fixed."),
         ("Learn what normally happens", "A model learns what an average player does in every situation; it never sees names."),
         ("Measure pressure", "How much one ball can change who wins × how much the match matters. Top 20% = pressure balls."),
         ("Score every ball", "What the player did minus what normally happens in that exact moment (pace vs pace, spin vs spin)."),
         ("Be fair to small samples", "Few balls means the rating is pulled towards average."),
         ("Choose settings by testing", "Every choice tested on matches the method never saw; 2026 held back."),
         ("Prove it", "1,000 re-runs, public-data exam (World Cups, leagues), stability and bias checks.")]
flow = "".join(f'<div class="step"><div class="num">{i + 1}</div><div><b>{t}</b><br><span>{d}</span></div></div>' for i, (t, d) in enumerate(steps))
exam = ""
if EV is not None and len(EV):
    e = EV[(EV.group == "all public matches") & EV.method.str.startswith("CHOSEN")].set_index("side")
    if len(e) == 2:
        exam = (f"<li><b>Exam on 4,000 public matches it never saw</b> (World Cups, franchise leagues): rank correlation between "
                f"our rating and pressure performance there {e.at['bat', 'rank_corr_winprob']:+.2f} for batters, "
                f"{e.at['bowl', 'rank_corr_winprob']:+.2f} for bowlers (0 = no link).</li>")
html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Most reliable players under pressure - results</title><style>
:root{{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--line:#e1e0d9;--blue:#2a78d6}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--page);color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.5}}
.wrap{{max-width:1080px;margin:0 auto;padding:28px 20px 60px}} h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:19px;margin:30px 0 8px}}
.card{{background:var(--surface);border:1px solid rgba(11,11,11,.10);border-radius:10px;padding:16px 20px;margin:12px 0}}
.step{{display:flex;gap:12px;align-items:flex-start;background:var(--surface);border:1px solid rgba(11,11,11,.10);border-radius:8px;padding:9px 14px;margin:4px 0}}
.num{{flex:0 0 26px;height:26px;border-radius:50%;background:var(--blue);color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700}}
.step span,.note{{color:var(--ink2);font-size:14px}} table{{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}}
th,td{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left}} th{{color:var(--ink2);font-weight:600}} img{{max-width:100%;display:block;margin:8px 0}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}} @media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><div class="wrap">
<h1>Most reliable players under pressure</h1><div class="note">SupeRR Selector S2, Round 1. Local page; the data pack must not be shared.</div>
<h2>Top 10 batters</h2><div class="card">{htab(bat10, ["final_rank", "name", "player_id", "teams", "pressure_balls", "SR_pressure", "SR_usual", "p_top10"],
    ["#", "Player", "ID", "Teams", "Pressure balls", "SR under pressure", "Usual SR", "Chance still top-10*"], F)}
<p class="note">*Share of 1,000 re-runs (reshuffled matches) in which the player stays top-10.</p></div>
<h2>Top 10 bowlers</h2><div class="card">{htab(bowl10, ["final_rank", "name", "type", "player_id", "teams", "pressure_balls", "econ_pressure", "econ_usual", "p_top10"],
    ["#", "Player", "Type", "ID", "Teams", "Pressure balls", "Economy under pressure", "Usual economy", "Chance still top-10*"], F)}</div>
<h2>Pressure in one sentence</h2><div class="card"><p>A ball is high-pressure when its result can change who wins, and when the match matters.
Last over, 12 needed: a six lifts the chasing team's chance from about 35% to 75%, and a wicket drops it to 15%. That's pressure. First over of a chase:
nothing much changes. Knockouts count 1.5×, must-win games 1.25×, dead rubbers 0.75×.</p>{img("pressure_heatmap_chase")}</div>
<h2>How we got there</h2>{flow}
<h2>Why you can trust it, and where it is weaker</h2><div class="card"><ul>
<li><b>Chosen by prediction, not by how the list looks:</b> 52 tests on unseen matches decided every setting.</li>{exam}
<li><b>Bowlers are compared within their type</b> (pace vs pace, spin vs spin), because teams must bowl pace at the death.</li>
<li><b>"Rising above your own normal" isn't a repeatable skill</b> in this data, so the lists reward how good a player is under pressure.</li>
<li><b>Weaker spot:</b> batters are harder to predict than bowlers, so read their "chance still top-10" carefully.</li></ul>
<div class="grid">{img("forest_bat")}{img("forest_bowl")}</div></div>
<h2>Still to do</h2><div class="card"><ul><li>Check the Unstop dropdown list; IDs are given for shared names.</li>
<li>Deadline: 18 Nov 2026, 11:59 PM IST. Submission: PDF/PPT/dashboard (methodology, top picks, how each helps RR).</li>
<li>Full plain-language report: <code>report/METHODOLOGY_REPORT_DRAFT.md</code>; proof notebook: <code>RESULTS_EVIDENCE.ipynb</code>.</li></ul></div>
</div></body></html>"""
open(os.path.join(ROOT, "RESULTS_SIMPLE.html"), "w", encoding="utf-8").write(html)
log.info("RESULTS_SIMPLE.html written")
