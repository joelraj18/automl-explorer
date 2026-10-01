# AutoML Explorer: decision tree specification

This page documents **exactly** how the engine decides what to do with an uploaded file. Every threshold below is a constant in the code:

- `automl/profiling.py`: column roles, target suggestion, split dates, duplicates
- `automl/decision.py`: quality gates, task, rule-of-thumb model, sampling, model constructors (`MODELS`)
- `automl/codegen.py`: the notebook cells each branch generates, tuning search spaces (`SEARCH_SPACES`)
- `automl/narrative.py`: wording thresholds used in the plain-English summary (`interpret`)

If you change a constant in the code, update this page too.

## Stage P: Profiling (runs once per file, cached)

### P1. Column roles (`_profile_column`), checked in this order, first match wins
| # | Condition | Role | What happens to it |
|---|---|---|---|
| 1 | distinct non-null values ≤ 1 | `constant` | dropped |
| 2 | dtype is datetime64 | `datetime` | expanded to `_year`, `_month`, `_weekday` |
| 3 | dtype is bool | `categorical` | one-hot |
| 4 | numeric **and** integer **and** unique ratio > 0.95 **and** rows ≥ 30 **and** (values are consecutive *or* the name matches the ID pattern¹) | `id` | dropped |
| 5 | any other numeric | `numeric` | median-imputed and standardised |
| 6 | text: ≥ 90% of a 200-row sample parses as dates, and ≤ 50% parses as plain numbers | `datetime` | expanded |
| 7 | text: average length > 30 characters **and** > 50% of values contain a space (2,000-row sample) | `text` | dropped |
| 8 | text: unique ratio > 0.95 **and** rows ≥ 30 | `id` | dropped |
| 9 | text: distinct values > 50 | `high_card_categorical` | one-hot, capped at 20 columns |
| 10 | otherwise | `categorical` | one-hot, capped at 20 columns |

¹ ID name pattern: `…_id`, `…_uuid`, `…_guid`, `…_key`, `…_index`, `id_…`, `id`, `index`, `Unnamed: 0` (case-insensitive).
The unique ratio is distinct values ÷ non-null values.

### P2. Dataset-level facts
- **Exact duplicates:** rows identical in every column.
- **Look-alike rows:** rows identical once the `id` columns are removed, minus the exact duplicates.
- **Missing fraction:** the share of all cells that are empty.

### P3. Target suggestion (`suggest_target`), pre-selected in the UI
1. Usable columns are those with role `categorical` or `numeric` **and** 2–15 distinct values.
2. If any usable column's name matches `status | target | label | class | churn | outcome | default | fraud | cancel | survived | result | response | y`,
   the **last** such column is chosen. Reason: "its name looks like an outcome…".
3. Otherwise, if the **last column** is usable and categorical, it is chosen.
4. Otherwise no suggestion is made, and the default is clustering.

### P4. Split dates (`find_date_parts`)
- Applies to numeric columns named `<prefix>[_]year` that also have `<prefix>[_]month` and `<prefix>[_]date` or `<prefix>[_]day`.
- They are combined into a real date, and impossible combinations (e.g. 29 Feb 2018) are counted.

---

## Stage 0: Quality gates (any failure → **HALT**, with an explanation)
| Gate | Threshold |
|---|---|
| Too few rows | rows < **30** |
| Too few columns | columns < **2** |
| Too empty | > **60%** of all cells missing |
| Bad target | target role is `text`, `datetime` or `constant` |
| No usable inputs (checked after clean-up) | 0 features, or **< 2** features when there is no target |
| Single class (classification, after removing rare classes) | fewer than 2 classes left |

## Stage 0b: Clean-up (applied to every non-target column, in this order)
1. When **no target** is chosen, the suggested target column is **excluded** from the clustering inputs.
2. Role `id` → drop.
3. Role `constant` → drop.
4. Role `text` → drop.
5. Missing in **> 40%** of rows → drop.
6. Role `datetime` → `_year`, `_month`, `_weekday` (0 = Monday).
7. Role `numeric` → numeric feature.
8. Role `categorical` or `high_card_categorical` → categorical feature.

Then:
- **Exact duplicates** are dropped.
- **Look-alike rows** are kept, but flagged (supervised only), and an extra score on unseen rows is reported.
- **Split dates** (P4) add `<prefix>_weekday`.
- Remaining gaps are filled with the **median** (numbers) or a **"missing"** category.

---

## Stage 1: Task
```
target chosen?
├─ NO  → CLUSTERING (Branch A)
└─ YES → drop rows where the target is empty
         target numeric (not bool) AND distinct values > 15 ?
         ├─ YES → REGRESSION (Branch B1)
         └─ NO  → CLASSIFICATION (Branch B2)
```

### Branch A: Clustering
| Setting | Rule |
|---|---|
| Algorithm | rows > 10,000 → `MiniBatchKMeans(n_init=3, batch_size=2048)`; otherwise `KMeans(n_init=10)`; both use k-means++ and Euclidean distance |
| k range | 2 … max(2, min(8, rows ÷ 10)) |
| Sampling | rows > 100,000 → random sample of 100,000 |
| Choosing k | highest **silhouette** on a 5,000-row sample; inertia (elbow) is shown alongside |

Generated cells: setup → prepare → EDA → scale/one-hot → **PCA** (components for 80% of the variance, plus loadings) → **choose k** (elbow and
silhouette) → **fit + profiles** (PCA map with centroids; per-cluster means as z-scores) → **hierarchical**.

The hierarchical cell uses ward linkage on 1,500 rows and draws a dendrogram cut at k. It compares `AgglomerativeClustering` with K-Means by
silhouette and adjusted Rand index: ARI > 0.6 means "robust".

### Branch B1: Regression
| Decision | Rule |
|---|---|
| Log target | \|skew\| > 1 **and** min ≥ 0 → `TransformedTargetRegressor(log1p / expm1)` for every candidate; OLS on log1p(y) |
| Rule-of-thumb model, < 1,000 rows | inputs > 30 → `LassoCV(cv=5)`; otherwise max \|r\| between numeric inputs > 0.9 (5k-row sample, first 50 columns) → `RidgeCV(α = 10⁻³…10³, 13 values)`; otherwise `LinearRegression` |
| Rule-of-thumb model, 1,000–100,000 rows | `RandomForestRegressor` |
| Rule-of-thumb model, > 100,000 rows | `HistGradientBoostingRegressor` |
| Candidates actually compared | [the linear model above, or `LinearRegression`], `DecisionTreeRegressor`, `RandomForestRegressor`, `AdaBoostRegressor`, `HistGradientBoostingRegressor` |
| Metric | R² (and MAE reported) |

### Branch B2: Classification
| Decision | Rule |
|---|---|
| Rare classes | classes with **< 2** rows are dropped |
| Subtype | 2 classes → binary; otherwise multiclass |
| Stratified split | if ⌊rows × 0.2⌋ ≥ number of classes |
| Imbalance | smallest class share ÷ largest < **0.25** → `class_weight="balanced"` (on LR, tree, RF and HGB; AdaBoost has none), and the metric becomes **Macro F1**; otherwise **Accuracy** |
| Positive class (binary only) | the **rarer** class. It drives the insights, logit odds ratios, ROC and threshold tuning |
| Rule-of-thumb model | < 1,000 rows → `LogisticRegression`; 1,000–100,000 → `RandomForestClassifier`; > 100,000 → `HistGradientBoostingClassifier` |
| Candidates actually compared | `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `AdaBoostClassifier`, `HistGradientBoostingClassifier` |

> The rule-of-thumb model is the engine's *first guess*. It is explained in the decision trace and sets the sampling limit, but the
> **model that is actually used is the cross-validation winner** among the five candidates.

---

## Stage 3: Speed (supervised)
The row cap depends on the family of the rule-of-thumb model:
- tree (random forest) → **50,000**
- linear, boosting or HGB → **100,000**

Classification samples are stratified.

---

## Fixed model specifications (`MODELS` in `automl/decision.py`)
| Model | Constructor |
|---|---|
| LogisticRegression | `max_iter=1000` (+ `class_weight` if imbalanced) |
| DecisionTree (Clf/Reg) | `max_depth=8, min_samples_leaf=20, random_state=42` |
| RandomForest (Clf/Reg) | `n_estimators=200, min_samples_leaf=2, oob_score=True, n_jobs=-1, random_state=42` |
| AdaBoost (Clf/Reg) | `n_estimators=100, random_state=42` |
| HistGradientBoosting (Clf/Reg) | `random_state=42` (defaults otherwise) |
| LinearRegression / RidgeCV / LassoCV | see B1 |
| KMeans / MiniBatchKMeans | see Branch A |

## Preprocessing (inside every pipeline, so nothing leaks)
- **Numbers:** `SimpleImputer(median)` → `StandardScaler`.
- **Categories:** `OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=20, sparse_output=False)`.
- The readable-rules tree uses the same steps **without** scaling, so its thresholds are in real units.

## Supervised evaluation pipeline (generated cells, in order)
| # | Cell | Specification |
|---|---|---|
| 1 | Setup | `metrics` dictionary; `note()` |
| 2 | Prepare | the clean-up above, plus the sample |
| 3 | EDA | 5,000-row sample; histograms (≤ 12 columns), correlation heatmap (≤ 15), target plot |
| 4 | Insights (regression, or binary classification) | outcome by group: categories as-is, numbers with > 10 distinct values cut into 5 quantile bands; groups under 30 rows hidden; top 4 columns by spread |
| 5 | Split | `test_size=0.2, random_state=42`, stratified if allowed |
| 6 | Preprocess | fitted on training rows |
| 7 | Baseline | `DummyRegressor(mean)` or `DummyClassifier(most_frequent)` |
| 8 (regression) | VIF | 20,000-row statsmodels sample; reference category = mode; top-10 categories (rest = "other"); VIF on 5,000 rows; drop the max while VIF > 10 |
| 9 (regression) | OLS | `sm.OLS` (log target if skewed); R², adjusted R², F p-value, coefficients with 95% CI and p-values; test R² |
| 10 (regression) | Assumptions | RESET (power 2, F), Breusch-Pagan, Jarque-Bera: p ≥ 0.05 = OK; Durbin-Watson in 1.5–2.5 = OK; residual, histogram and Q-Q plots |
| 8 (binary) | Logit | `sm.Logit(maxiter=300)` on 20,000 rows; odds ratios with 95% CI and p-values; warns if it doesn't converge (perfect separation) |
| – | Compare | 10,000-row training sample, `cross_val_score(cv=3)` per candidate; winner = highest mean |
| – | Tune | `RandomizedSearchCV(n_iter=min(8, grid size), cv=3, n_jobs=-1)` on the same sample; spaces below |
| – | Train | refit the tuned winner on **all** training rows; train/test score (gap > 0.15 → overfitting warning); OOB score for forests; unseen-row score when look-alike rows exist; MAE and predicted-vs-actual (regression); report, confusion matrix and ROC-AUC (classification) |
| – | Threshold (binary) | out-of-fold probabilities (`cv=3`, ≤ 20,000 training rows); threshold = argmax F1 on the precision-recall curve; test table at 0.50 vs tuned |
| – | Rules | `DecisionTree(max_depth=3)` on unscaled inputs; `plot_tree`; root split; score vs the winner (within 0.05 → "most") |
| – | Explain | `permutation_importance(n_repeats=3)` on ≤ 2,000 test rows, original columns |

### Tuning search spaces (`SEARCH_SPACES` in `automl/codegen.py`)
| Model | Space |
|---|---|
| LogisticRegression | C ∈ {0.01, 0.1, 1, 10, 100} |
| DecisionTree | max_depth ∈ {3, 5, 8, 12, None}, min_samples_leaf ∈ {1, 5, 20, 50}, ccp_alpha ∈ {0, 1e-4, 1e-3} |
| RandomForest | n_estimators ∈ {100, 200, 400}, max_features ∈ {sqrt, 0.5, 1.0}, min_samples_leaf ∈ {1, 2, 5} |
| AdaBoost | n_estimators ∈ {50, 100, 200}, learning_rate ∈ {0.05, 0.1, 0.5, 1.0} |
| HistGradientBoosting | learning_rate ∈ {0.03, 0.05, 0.1, 0.2}, max_leaf_nodes ∈ {15, 31, 63}, l2_regularization ∈ {0, 0.1, 1} |
| LinearRegression / RidgeCV / LassoCV | not tuned (they fit their own α, or have none) |

## Interpretation thresholds used in the plain-English text
| Measure | Threshold and meaning |
|---|---|
| Model vs baseline | gain ≤ 0.01: "not better than guessing"; < 0.10: "small but real"; otherwise "clearly learning" |
| Silhouette | ≥ 0.5 strong; ≥ 0.25 reasonable; otherwise weak |
| ROC-AUC | ≥ 0.9 "very well"; ≥ 0.75 "reasonably"; otherwise "weakly" |
| Unseen-row gap | > 0.02 → "duplicates flatter the score" |
| VIF | < 5 safe; 5–10 "read with care"; > 10 dropped |
| p-values | < 0.05 significant |

## Worked trace: INN Hotels (`booking_status`)
- **Profile:** 36,275 rows; `Booking_ID` is an ID (100% unique text); 10,275 look-alike rows; arrival year/month/date detected as a split date, with 37 impossible dates.
- **Target:** `booking_status` was suggested by name.
- **Task:** 2 classes → binary; share ratio 0.33 ÷ 0.67 = 0.49 is not < 0.25, so the metric is Accuracy; the positive class is `Canceled`.
- **Model:** the rule-of-thumb pick is RandomForest (1,000–100,000 rows), so the cap is 50,000 and no sampling is needed.
- **Cells:** logit odds → compare 5 → RandomForest wins (CV 0.878) → tuned → test accuracy 0.91, ROC-AUC 0.96, unseen rows 0.876 → threshold 0.44.

## Known limits / quirks worth knowing
- `n_features > 30` (the Lasso rule) counts input columns *before* one-hot encoding.
- The collinearity check (Ridge rule) only looks at original numeric columns, not derived date features.
- Multiclass targets get no insights, logit or threshold cells; only the comparison, report and confusion matrix.
- Hierarchical clustering is limited to 1,500 rows, and the statsmodels cells to 20,000 rows (sampled).
