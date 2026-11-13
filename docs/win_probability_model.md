# Why a simple logistic regression beat the tree models

**Technical note: the win-probability model behind "impact" and pressure**

SupeRR Selector Season 2, Round 1 · Bishal Roy · November 2026

---

## 1. Summary

Pressure in this project is built on one quantity: before every ball, how far could that ball move the batting side's chance of winning? Getting that chance right matters more than anything else in the pipeline, because every pressure weight, and so every rating, inherits its errors.

I tested ten alternatives against a logistic regression on the same match-grouped folds. The alternatives were seven machine-learning models (decision tree, random forest, extra trees, XGBoost, LightGBM, CatBoost, k-nearest neighbours) and three combinations (two stacks and a plain average). The final model is a logistic regression with smooth curves on the game state and straight lines for match context. It had the best score on unseen matches for chases, and it was the best calibrated.

The reason is structural, not luck:

1. **The real sample is about 820 matches, not 92,000 balls.** Every ball in a chase shares one outcome, so the data holds far less independent information than its row count suggests.
2. **Match-level inputs act as a fingerprint.** The combination of target, venue par, team-strength gap, season and competition is unique for every one of the 820 chases. A flexible tree model can learn each match's result from its fingerprint. Given only those match-level inputs, LightGBM reached a training log-loss of 0.0001 and a test log-loss of 1.93, nearly three times worse than a coin toss (0.69).
3. **Win chance is a smooth, nearly additive function of a few game-state variables.** A model that assumes that shape spends its limited information well. Trees approximate the same surface with steps and need far more matches to do it.

## 2. What the model has to do

The chase model predicts whether the chasing side wins from the state before each ball. The first-innings model does the same for the side batting first.

| Group | Inputs | Treatment in the final model |
|---|---|---|
| Game state | runs needed, balls left, wickets in hand, required run rate, `z_chase` | smooth curves (natural cubic splines, 4 knots at quantiles) |
| Match context | target, innings length, competition level, season, venue par, team-strength (Elo) gap | straight lines (standardised) |

`z_chase` is the runs still needed compared with what a typical side scores from that many balls and wickets. Without it, the model put "4 needed off the last ball" at 99%. In the data, sides in that spot with 4 or more wickets in hand won once in four tries.

**Impact** for a ball is then the expected absolute change in win chance over the possible outcomes (dot, 1, 2, 3, 4, 6, wicket, extra), each weighted by its probability from a separate ball-outcome model:

    impact = Σ P(outcome) × |WP(after outcome) − WP(now)|

Because impact is built from differences between predicted probabilities, calibration matters more than ranking ability. A model that is a little overconfident everywhere inflates every impact value.

## 3. How the models were compared

- **Folds.** 5-fold cross-validation grouped by match. No match is ever split between training and testing.
- **Tuning.** Each model's complexity setting (tree depth, leaf size, number of neighbours, penalty) was chosen on three inner match-grouped folds of the training part only. Boosted models also early-stopped on a 15% match-level slice of their own training data.
- **Metrics on unseen matches.** Log-loss (the main score), Brier score, AUC, expected calibration error (ECE), calibration slope (1.0 is ideal: below 1 means too extreme, above 1 too timid), calibration in the last overs, and the **overfitting gap** (test log-loss minus training log-loss).

> **Reading log-loss.** It is not a percentage. For each ball it charges −ln(the chance the model gave to what actually happened): 50% costs 0.69, 30% costs 1.20, 10% costs 2.30. The score is the average, lower is better, and 0 (always 100% sure and right) is impossible in cricket. For a win-or-lose prediction a coin toss scores 0.693, so the final chase model's 0.393 means it is far better than guessing.

Notebook: [`notebooks/05_why_logistic_regression_won.ipynb`](../notebooks/05_why_logistic_regression_won.ipynb). Tables: `docs/results/model_zoo.csv`, `model_zoo_fingerprint.csv`, `model_zoo_learning_curve.csv`, `wp_logistic_designs.csv`.

## 4. Results: chases

| Model | Test log-loss | Brier | AUC | ECE | Calibration slope | Overfitting gap |
|---|---:|---:|---:|---:|---:|---:|
| **Final: logistic, curves on game state, lines on context** | **0.3935** | 0.1278 | **0.9013** | **0.0093** | 0.946 | 0.017 |
| Logistic, straight lines only | 0.3939 | 0.1280 | 0.9011 | 0.0120 | 0.950 | 0.014 |
| Stacking (meta: extra trees) | 0.3996 | 0.1275 | 0.8999 | 0.0150 | 0.952 | 0.038 |
| Average of spline + LightGBM + CatBoost | 0.4004 | 0.1279 | 0.9003 | 0.0183 | 1.101 | 0.064 |
| Stacking (meta: logistic) | 0.4004 | 0.1279 | 0.8999 | 0.0122 | 0.942 | 0.011 |
| Random forest | 0.4014 | **0.1272** | 0.9002 | 0.0166 | 0.974 | 0.109 |
| XGBoost (monotone) | 0.4019 | 0.1288 | 0.8986 | 0.0139 | 1.067 | 0.084 |
| Extra trees | 0.4056 | 0.1296 | 0.8981 | 0.0173 | 1.105 | 0.084 |
| Logistic, curves on every input (earlier version) | 0.4076 | 0.1318 | 0.8954 | 0.0198 | 0.860 | 0.045 |
| LightGBM (monotone) | 0.4081 | 0.1303 | 0.8958 | 0.0182 | 1.028 | 0.106 |
| CatBoost | 0.4101 | 0.1296 | 0.8983 | 0.0295 | 1.218 | 0.044 |
| Decision tree | 0.4221 | 0.1351 | 0.8875 | 0.0433 | 0.851 | 0.044 |
| k-nearest neighbours | 0.4413 | 0.1397 | 0.8849 | 0.0483 | 1.208 | 0.057 |

Two honest notes:

- Random forest has a slightly lower Brier score (0.1272 against 0.1278). The difference is in the third decimal place. On log-loss, calibration and overfitting it is clearly worse, so it does not change the decision.
- The plain straight-line logistic is almost as good overall. The curves earn their place in the end-game, which is where pressure lives (section 7).

## 5. Why the tree models lose

### 5.1 The match fingerprint

Inputs such as target, venue par, season and team-strength gap are constant within a match. Taken together, they identify the match.

| Check (match-level inputs only) | Matches | Distinct input combinations | Train log-loss | Test log-loss |
|---|---:|---:|---:|---:|
| Chase: spline logistic | 820 | 820 | 0.546 | 0.588 |
| Chase: flexible LightGBM | 820 | 820 | 0.0001 | 1.927 |
| First innings: spline logistic | 823 | 823 | 0.659 | 0.706 |
| First innings: flexible LightGBM | 823 | 823 | 0.001 | 2.198 |

With 820 combinations for 820 chases, a tree can isolate each match in a leaf and read off its result. That is pure memorisation, and it fails completely on new matches. Tuning, early stopping and monotone constraints reduce the damage in the full models, but the gaps in section 4 (0.08 to 0.11 for random forest, LightGBM, XGBoost and extra trees, against 0.017 for the final model) show it is never removed.

### 5.2 Few independent examples

Each chase contributes about 113 balls that share one result. The effective sample for learning "who wins from here" is closer to 820 than to 92,000. Flexible models need many independent examples to learn a smooth surface from steps. A logistic regression with a handful of curves needs very few.

### 5.3 Learning curves

| Training matches | Spline logistic (earlier version) | LightGBM | CatBoost | Random forest |
|---:|---:|---:|---:|---:|
| 65 | 0.680 | 0.536 | 0.495 | 0.511 |
| 163 | 0.517 | 0.439 | 0.437 | 0.460 |
| 328 | 0.441 | 0.422 | 0.427 | 0.426 |
| 656 | **0.408** | 0.409 | 0.418 | 0.421 |

(Test log-loss on unseen matches, chases.) The trees learn faster when data is tiny, but they flatten out. With all the data the logistic catches up and passes them. There is no extra structure left for the flexible models to find.

### 5.4 Stacking cannot fix a shared blind spot

Stacking learns how to combine base models from out-of-fold predictions. Here the base models share the same weakness (memorised match context), and the meta-learner is itself trained on about 820 independent matches. The best stack (meta: extra trees) still lost to the single logistic, and it overfitted more (gap 0.038 against 0.017).

## 6. Choosing the logistic design

| Logistic design (chase) | Test log-loss | ECE | Calibration slope | Overfitting gap | Last-overs calibration error | Real end-game error |
|---|---:|---:|---:|---:|---:|---:|
| Plain (straight lines) | 0.3939 | 0.0120 | 0.950 | 0.014 | 0.033 | 0.082 |
| Curves on every input, 6 knots (earlier version) | 0.4076 | 0.0198 | 0.860 | 0.045 | 0.028 | 0.035 |
| **Curves on game state, lines on context, 4 knots (final)** | **0.3935** | **0.0093** | 0.946 | 0.017 | 0.029 | **0.028** |
| Curves on game state (6 knots), lines on context | 0.3951 | 0.0094 | 0.939 | 0.019 | 0.031 | 0.040 |
| Same, stronger penalty (C = 0.1) | 0.3954 | 0.0105 | 0.976 | 0.017 | 0.023 | 0.039 |
| Curves on game state only, no match context | 0.4073 | 0.0099 | 0.956 | 0.013 | 0.023 | 0.050 |

The earlier version put curves on every input, including the match-level ones. That let it start memorising matches too: its gap was three times larger and its calibration slope fell to 0.86, meaning its probabilities were too extreme. Keeping the curves for the game state only, where the real shape is curved, fixed both problems.

## 7. Where it matters: the end of a chase

Win rate in real matches compared with each model's prediction (chases with 4 or more wickets in hand):

| Situation | Times seen | Real win rate | Plain logistic | Earlier version | Final model |
|---|---:|---:|---:|---:|---:|
| 8 to 12 needed off 6 balls | 45 | 80% | 73% | 85% | 86% |
| 13 to 16 needed off 6 balls | 32 | 66% | 46% | 64% | 62% |
| 15 to 20 needed off 12 balls | 60 | 83% | 78% | 88% | 88% |
| 21 to 26 needed off 12 balls | 50 | 64% | 55% | 70% | 66% |
| 25 to 32 needed off 18 balls | 87 | 82% | 75% | 82% | 82% |

The plain logistic is too pessimistic for the chasing side in exactly the overs that decide matches, so it would understate pressure there. The final model tracks the real outcomes closely. Other checks:

- **Monotonicity.** Win chance never rises when more runs are needed or a wicket falls. It rises with balls left in all but 0.04% of cases.
- **WASP-style backward calculation.** Tested and rejected. It was too pessimistic against real end-game outcomes, because teams still alive late in a chase usually have set batters.

## 8. First innings

The first innings is harder for every model, because the target does not exist yet.

| First-innings model | Test log-loss | AUC | ECE |
|---|---:|---:|---:|
| **Plug-in (final): predicted final total, mapped through the chase model** | **0.6182** | **0.713** | 0.027 |
| Logistic (curves on game state) | 0.6194 | 0.713 | 0.021 |
| LightGBM (calibrated) | 0.6507 | 0.672 | 0.031 |
| XGBoost | 0.6992 | 0.654 | 0.109 |

No first-innings model reached the AUC target of 0.72 that I had set in advance. This is a stated limitation. First-innings pressure is measured less sharply, and the model understates the real effect of first-innings wickets by about 23% (8% in chases), measured without any model in `pressure_swing_check`. Two design choices limit the impact on the rankings:

- every ball counts in proportion to its pressure, so first-innings balls still carry 41% of the rating weight;
- player values are measured in runs against a typical player, not in win chance.

## 9. Reproducing

Run [`04_win_chance_and_average_player.ipynb`](../notebooks/04_win_chance_and_average_player.ipynb) (the final models), then [`05_why_logistic_regression_won.ipynb`](../notebooks/05_why_logistic_regression_won.ipynb) (the full comparison; slow, because every model is tuned on match-grouped folds).

Results are written to `outputs/tables/` (the copies used here are in `docs/results/`). The figure `docs/figures/model_zoo.png` shows training against unseen-match log-loss for every model, and the learning curves.
