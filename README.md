# AutoML Explorer

Upload a CSV or Excel file. AutoML Explorer then does four things:

1. It profiles every column: number, category, date, ID, or free text?
2. It decides the machine-learning task and model with a transparent rule engine.
3. It generates notebook-style Python cells and runs them live.
4. It explains every step and every result in plain English.

The app is built for beginners. Every decision comes with the same three-part explanation:

> 🔍 **What we found:** `customer_id` has a different value in 100% of rows - it looks like an ID.
> 💡 **Why it matters:** An ID only names a row; it says nothing about the outcome. A model would just memorise it.
> 🎯 **What we're doing now:** We drop `customer_id`, so that the model learns real patterns instead of row numbers.

A **Beginner / Expert** switch in the sidebar shows or hides these explanations. The whole notebook can also be downloaded as a `.ipynb` file, with the explanations included as markdown cells.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project layout

| File | What it does |
|---|---|
| `app.py` | Streamlit page layout only |
| `automl/profiling.py` | Gives each column a *role* (`numeric`, `categorical`, `high_card_categorical`, `datetime`, `id`, `constant`, `text`) and a reason for it |
| `automl/decision.py` | The decision engine: quality gates → task → model → sampling. Every choice is recorded as a `Step` |
| `automl/narrative.py` | All the beginner wording: `TEMPLATES` (found / why / action), per-cell stories, `GLOSSARY`, and `interpret()`, which turns metrics into sentences |
| `automl/codegen.py` | Turns a decision into code cells, and exports them to a Jupyter notebook |
| `automl/runner.py` | Runs the cells in one shared namespace and captures printed output, figures and errors |

## The decision tree

```
Upload
  │
  ├─ Quality gates ── < 30 rows / < 2 columns / > 60% empty ──▶ stop and explain why
  │
  ├─ Target suggestion ── a column named like an outcome (status, label, churn…) is pre-selected
  │
  ├─ Column clean-up
  │    ID-like, constant, free-text, > 40% empty  ──▶ drop
  │    dates                                       ──▶ year / month / weekday
  │    year + month + day columns                  ──▶ combined, weekday added, impossible dates flagged
  │    exact duplicate rows                        ──▶ drop
  │    rows identical apart from the ID            ──▶ keep, but also score on "unseen" test rows
  │    remaining gaps                              ──▶ median / 'missing' category
  │
  ├─ No target ─▶ CLUSTERING (any outcome-like column is left out of the inputs)
  │                 ≤ 10k rows: KMeans        > 10k rows: MiniBatchKMeans
  │                 k chosen by best silhouette score (k = 2…8, on a sample)
  │
  └─ Target chosen
       ├─ numeric and > 15 distinct values ─▶ REGRESSION
       │     skewed and ≥ 0  ─▶ learn log(target)
       │     < 1k rows       ─▶ LinearRegression (LassoCV if > 30 features, RidgeCV if two features correlate > 0.9)
       │     1k–100k rows    ─▶ RandomForestRegressor
       │     > 100k rows     ─▶ HistGradientBoostingRegressor
       │
       └─ otherwise ─▶ CLASSIFICATION (binary / multiclass)
             classes seen once        ─▶ dropped
             smallest/largest < 0.25  ─▶ class_weight='balanced', judged by macro F1
             < 1k rows                ─▶ LogisticRegression
             1k–100k rows             ─▶ RandomForestClassifier
             > 100k rows              ─▶ HistGradientBoostingClassifier
```

Every supervised run also does the following:
- a **key drivers** table: the outcome rate or average for each group of each column, e.g. "cancel rate by segment"
- a stratified train/test split, and a **baseline** model to beat
- a **model comparison**: the rule-based pick above becomes one of three candidates (linear, decision tree, forest or boosting), scored by 3-fold cross-validation on the training rows, and the winner is used
- train-vs-test scores, so overfitting shows up
- for binary targets, a **threshold tuned for the rarer class**, using out-of-fold predictions, never the test set
- permutation feature importance on the original column names

Large files are sampled so that the app stays fast: 50k rows for random forests, 200k for linear and boosting models, and 100k for clustering.

## Worked example

[`examples/inn_hotels/`](examples/inn_hotels/) is a full project on 36k hotel bookings. It includes a hand-checked, fully explained notebook (EDA, statsmodels, pruned trees, ensembles, business recommendations) and a comparison with what AutoML produces.

## Adding a new explanation or rule

1. Add a template to `TEMPLATES` in `automl/narrative.py`, with `found`, `why` and `action`.
2. In `automl/decision.py`, append `story("your_key", ...)` to `d.steps` where the rule fires.
3. If the rule changes the code, read the new `Decision` field in `automl/codegen.py`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The tests check column roles and every decision branch. They also **execute the generated code** on synthetic data that includes IDs, dates, gaps and mixed types, and they assert that the model beats the baseline.
