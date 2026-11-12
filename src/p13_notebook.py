"""Builds and executes RESULTS_EVIDENCE.ipynb: one section per phase, showing the saved result tables and figures
(the proof behind every choice). The notebook only reads outputs/ - rerunning it never changes a result."""
import os
import nbformat as nbf
from nbclient import NotebookClient
from common import ROOT

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# SupeRR Selector S2 — Round 1: evidence notebook
Every table and chart below is read from `outputs/` (written by `src/p00_audit.py` … `src/p12_report.py`).
Nothing here is typed in by hand. Re-run the scripts to regenerate, then re-run this notebook.

**Order:** P0 audit → P1 ball state → P2 features → P3 context models → P4 pressure → P5 values → P6 ratings →
P7 unsupervised → P8 feature selection → P9 experiments → P10 final ranking → P11 robustness.""")
code("""import json, os, pandas as pd, numpy as np
from IPython.display import display, Markdown, Image
pd.set_option('display.max_columns', 40); pd.set_option('display.width', 220); pd.set_option('display.max_colwidth', 70)
OUT = '%s'
T = lambda n: pd.read_csv(os.path.join(OUT, 'tables', n + '.csv'))
J = lambda n: json.load(open(os.path.join(OUT, 'tables', n + '.json'), encoding='utf-8'))
F = lambda n: display(Image(filename=os.path.join(OUT, 'figures', n + '.png')))""" % os.path.join(ROOT, "outputs").replace("\\", "/"))

md("## P0 — Data audit\nCleaning rules and every data problem found (the report's limitations section draws on this).")
code("print(open(os.path.join(OUT, 'audit_report.md'), encoding='utf-8').read())")
md("## P1 — Ball-state reconstruction: validation gates\nGate 1: ≥ 99.9% of complete overs have 6 legal balls. Gate 2: rebuilt margins have median error 0, IQR within ±3.")
code("J('p01_summary')")
md("## P2 — Feature catalogue (ball level, pre-ball information only)")
code("cat = T('feature_catalogue_ball'); print(cat.family.value_counts().to_string()); cat")

md("""## P3 — Context models (player-blind, out-of-fold by match)
**M1 ball outcome.** Lower multi-log-loss is better. Baselines: lookup table and spline logistic.""")
code("display(T('m1_model_comparison')); display(T('m1_calibration'))")
md("""**M2 chase / M3 first-innings win probability.** Plan A was monotone LightGBM with nested Optuna tuning. The pre-set gates were: beat the logistic baseline with calibration error ≤ 0.03 (M2), and AUC ≥ 0.72 (M3). Both failed, so the Plan B models were used.""")
code("display(T('wp_model_comparison')); display(T('first_innings_wp_planB')); J('p03_summary')")
code("display(T('m4_model_comparison')); F('wp_calibration')")
md("**End of chases: does the model match reality?** The last two overs, plus a rejected alternative (the Plan C backward calculation).")
code("display(T('endgame_reality_check').round(3)); display(T('wp_endgame_variants').round(3)); display(T('wp_endgame_calibration').round(3))")
md("""**Is the win-probability model over-fitting? (P3b, `src/p03b_model_zoo.py`)** 9 model types + 2 stacked ensembles, same match-grouped
folds; each model chooses its complexity on inner folds of its training matches only. Over-fitting = low error on training matches,
higher on unseen ones (`overfit_gap`); calibration slope < 1 = over-confident. Rows labelled CURRENT/previous are the
curves-on-every-input logistic used before this check.""")
code("""import os
_t = lambda n: os.path.exists(os.path.join(OUT, 'tables', n + '.csv'))
if _t('model_zoo'): display(T('model_zoo').round(4))
if _t('wp_logistic_designs'): display(T('wp_logistic_designs').round(4))
if _t('wp_logistic_designs_situations'): display(T('wp_logistic_designs_situations').pivot_table(index='situation', columns='model', values='model_win_prob', sort=False).round(3))
if _t('model_zoo_fingerprint'): display(T('model_zoo_fingerprint').round(4))
if _t('model_zoo_learning_curve'): display(T('model_zoo_learning_curve').round(4))
if os.path.exists(os.path.join(OUT, 'figures', 'model_zoo.png')): F('model_zoo')""")

md("""## P4 — Pressure: 38 pointers, 5 pre-registered tests, 9 combined definitions
T1 real effect (FDR-controlled), T2 consequential, T3 enough data, T4 separates players, T5 stable.""")
code("""t = T('pressure_pointer_tests')
display(t[['id','pointer','family','share_balls','t1_wkt','q_wkt','t1_dot','q_dot','t1_bdry','q_bdry','t2_lift','T1','T2','T3','T4','T5','verdict']].round(4))
print(t.verdict.value_counts().to_string())""")
code("F('pointer_tests')")
md("**Combined definitions.** Judged on stakes and strain only, never on the ranking they produce.")
code("display(T('pressure_definition_summary').sort_values('score', ascending=False)); J('p04_summary')")
code("e = T('pressure_definition_comparison'); e[e.definition.str.startswith('C-A')]")
code("F('pressure_definitions'); F('pressure_heatmap_chase'); F('pressure_heatmap_first')")
md("**Face validity.** The 20 highest- and 20 lowest-pressure situations (one per match).")
code("T('pressure_face_validity')")

md("## P5 — Value per ball: reliability of the four candidates")
code("display(T('value_metric_reliability')); J('p05_components')")
code("F('value_reliability')")

md("""## P6 — Ratings and the clutch question
`tau_clutch` is the true spread of players' pressure-vs-usual gaps. It is the split-half covariance (noise cancels), so 0 means not repeatable.""")
code("display(T('clutch_spread')); display(T('r1_penalty_cv'))")
code("display(T('rating_agreement_bat')); display(T('rating_agreement_bowl'))")

md("""## P7 — Roles, pressure regimes, anomalies (used for fairness checks and the by-role list, never to pick players)
**Roles by usage (semi-supervised).** Cricket rules label the clear-cut players (seeds); a multinomial logistic model on usage
inputs only (batting position mix, entry over, phase shares; pace and spin separately) assigns everyone else with a confidence.
`cv_agreement_on_seeds` = can usage reproduce the rules; `bootstrap_ari` = stability; `ari_with_roles` = agreement with purely
unsupervised views (k-means, GMM, UMAP + HDBSCAN) on the same inputs.""")
code("display(T('role_method_check')); display(T('role_centroids_batters').round(2)); display(T('role_centroids_bowlers').round(2))")
code("""rd = T('roles_detail'); print('lowest-confidence assignments (players between two roles):')
display(rd.sort_values('confidence').groupby('side').head(8)[['side','name','role','confidence']].round(2))""")
code("display(T('clustering_comparison_batters').sort_values('silhouette', ascending=False).head(8)); display(T('clustering_comparison_bowlers').sort_values('silhouette', ascending=False).head(8))")
code("display(T('pressure_regimes')); o = T('one_match_wonders'); print('one-match wonders:', o.groupby('side').one_match_wonder.sum().to_dict())")

md("## P8 — Feature selection scorecard")
code("""sc = T('feature_scorecard'); print(sc.groupby('track').decision.value_counts().to_string())
display(T('fs_family_ablation'))
display(sc[sc.feature.astype(str).str.contains('_delta')][['feature','reliability','spearman_future_pressure','decision']].round(3))""")

md("""## P9 — Experiments: every configuration, scored on unseen matches
The objective is the rank correlation between a player's training-period score and his win-probability contribution on pressure balls in matches the method never saw. It averages 2 rolling-origin splits and 50 random match splits. A simpler option within one standard error wins.""")
code("""E = T('experiment_results'); display(E[['stage','side','config','objective','rho_v2_random','rho_v2_rolling','rho_v1_random','top10_lift_v2_x100','complexity']].round(4))
J('p09_summary')""")
code("F('experiments')")
md("**Sealed 2026 season, opened once.** The chosen method is compared with simple baselines.")
code("display(T('holdout_2026').round(4)); print('First opening (before the Stage D pool fix, disclosed):'); display(T('holdout_2026_first_opening_buggy_pool').round(4))")

md("## P10 — Final rankings (fitted on all seasons; 1,000 match-bootstrap re-runs)")
code("""b = T('final_top30_bat'); display(b.head(10)); display(b.iloc[10:30][['final_rank','name','player_id','teams','score','p_top10','rank_lo90','rank_hi90']])""")
code("F('forest_bat'); F('rise_dip_bat')")
code("""w = T('final_top30_bowl'); display(w.head(10)); display(w.iloc[10:30][['final_rank','name','player_id','teams','score','p_top10','rank_lo90','rank_hi90']])""")
code("F('forest_bowl'); F('rise_dip_bowl')")
md("**Best pace bowlers under the method**:")
code("display(T('top_pace_bowlers')); F('top_pace_bowlers')")

md("## P11 — Robustness")
code("s = T('sensitivity_matrix'); display(s); display(s.groupby(['side','group']).top10_overlap.agg(['median','min']))")
code("F('sensitivity_bat'); F('sensitivity_bowl')")
code("display(J('negative_control')); display(T('synthetic_injection')); display(T('bias_audit').round(3))")
md("""**Where pressure falls, and an innings-balanced alternative.** Share of balls vs share of pressure balls vs share of the
ranking weight, by innings, phase and batting position. Then the same pressure score ranked *within each innings*, judged
with P4's own yardsticks and by how much the top 10 would change.""")
code("display(T('pressure_where_it_falls').round(3)); display(T('pressure_innings_balance').round(3))")
md("**Best under pressure, by role** (overall rank decides the order inside each role):")
code("display(T('best_by_role'))")
md("""**Is ~90% of pressure balls from chases real?** Model-free swing check: in the same match situation, how much the
*actual* final result moves after a wicket (or a boundary instead of a dot), against what the model says
(`actual_over_model` > 1 = the model under-states it). Then the chase share in the supplied vs public data.""")
code("display(T('pressure_swing_check').round(3)); display(T('chase_share_by_dataset').round(3))")
md("""**Why one season can't judge batters.** Split-half reliability of one season's pressure record, reliability of our
ratings, the skill share of the spread at a season's sample size, season-to-season agreement with no model, and the
resulting ceiling on any 2026 correlation.""")
code("display(T('season_noise_check').round(3))")
md("""**Rating-model family decision (P14b).** Pressure-weighted (chosen, both sides) vs what the rule picked when every
family was allowed, scored three ways on one yardstick: 52 unseen-match splits, the 2026 season, the public-data exam.""")
code("display(T('rating_family_check').drop(columns='config').round(4))")
md("**Players left out of the lists** (not available for IPL selection: Pakistan / Pakistan A; or no match since 2025: inactive or retired). Rated like everyone else, with the rank they would have had:")
code("""x = T('excluded_unavailable'); cols = [c for c in ['side','name','role','reason','would_be_rank','score'] if c in x.columns]
display(x[x.would_be_rank <= 30][cols].round(3))""")

md("""## P14 — Independent exam on public data (Cricsheet, open licence)
The context models trained on the supplied data are applied unchanged to about 4,000 public matches they never saw:
the T20 World Cups 2022/24/26, franchise leagues, Syed Mushtaq Ali and other T20Is. Players are linked to pack IDs
through 766 shared matches, by agreement on the same deliveries.

All methods are scored on the same players. Anyone a method excludes is ranked last.""")
code("display(J('p00x_summary')); l = T('external_player_links'); print(len(l), 'linked players; median shared deliveries', l.n.median()); l.head(10)")
code("ev = T('external_validation'); display(ev.round(3)); F('external_exam')")
code("print('Combined (supplied + public) top 10s vs main list:'); display(T('combined_top30_bat').head(15)); display(T('combined_top30_bowl').head(15))")
code("print('T20 World Cup semi-finals / finals, main top-10 players (descriptive):'); T('world_cup_knockouts_top10').round(2)")
md("## Before vs after the review (type-blind vs same-type bowler baseline)")
code("""old = T('final_top30_bowl_typeblind').head(10)[['final_rank','name']].rename(columns={'name':'previous (type-blind)'})
new = T('final_top30_bowl').head(10)[['final_rank','name']].rename(columns={'name':'now (same-type)'})
display(old.merge(new, on='final_rank')); display(T('top_pace_bowlers'))""")
md("## Before vs after the win-probability fix (curves on every input → curves for game state only)")
code("""for s in ('bat', 'bowl'):
    p_ = os.path.join(OUT, 'tables', f'final_top30_{s}_before_wpfix.csv')
    if os.path.exists(p_):
        old = pd.read_csv(p_).head(10)[['final_rank','name']].rename(columns={'name':'before'})
        new = T(f'final_top30_{s}').head(10)[['final_rank','name','role']].rename(columns={'name':'after'})
        display(old.merge(new, on='final_rank'))""")

md("""## Leakage and overfitting checklist
- Folds are grouped by `match_id` everywhere; balls are never split at random.
- Rolling-origin season validation. The 2026 holdout has been opened several times as fixes were made after review
  (disclosed in the report), so it is a confirmation check, not a pristine test.
- Model complexity checked against over-fitting with train-vs-unseen gaps for 9 model types + stacking (P3b).
- Optuna tuning only inside inner folds (nested); calibration from out-of-fold predictions only.
- Pre-ball features only; `stroke` and delivery outcome fields are never model inputs.
- Context models are player-blind.
- In P9, ratings come from the training split only.
- Staged search with a fixed objective, all variants reported (P9 table).
- Every player estimate is shrunk; negative control and synthetic injection were run (P11).
- Fixed seed 2026; one `config.yaml`; scripts rerun end-to-end from the raw CSVs.""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
path = os.path.join(ROOT, "RESULTS_EVIDENCE.ipynb")
NotebookClient(nb, timeout=600, kernel_name="python3", allow_errors=True, resources={"metadata": {"path": ROOT}}).execute()
nbf.write(nb, path)
print("notebook written:", path)
