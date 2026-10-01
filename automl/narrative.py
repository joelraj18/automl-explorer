"""Beginner-friendly explanations.

Every decision the engine makes is recorded as a ``Step`` with three parts:

* found  - the evidence we saw in the data (with real numbers / column names)
* why    - why that evidence matters, in plain English
* action - what we do about it, phrased as "We ..., so that ..."

All the wording lives in ``TEMPLATES`` so the voice stays consistent and new
explanations are added in one place.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    key: str
    found: str
    why: str
    action: str

    def markdown(self) -> str:
        return (
            f"🔍 **What we found:** {self.found}  \n"
            f"💡 **Why it matters:** {self.why}  \n"
            f"🎯 **What we're doing now:** {self.action}"
        )

    def short(self) -> str:
        """One-line version for the sidebar decision trace."""
        return f"{self.found} → {self.action}"


TEMPLATES: dict[str, dict[str, str]] = {
    # ── Data quality ──────────────────────────────────────────────────────
    "overview": {
        "found": "Your file has {rows:,} rows and {cols} columns ({n_num} numeric, {n_cat} text/category).",
        "why": "The size of the data and the type of each column decide which tools are safe to use.",
        "action": "We profile every column first, so that each later decision is based on evidence instead of guesses.",
    },
    "id_column": {
        "found": "`{col}` has a different value in {pct:.0f}% of rows - it looks like an ID.",
        "why": "An ID only names a row; it says nothing about the outcome. A model would just memorise it.",
        "action": "We drop `{col}`, so that the model learns real patterns instead of row numbers.",
    },
    "constant_column": {
        "found": "`{col}` has the same value in every row.",
        "why": "A column that never changes can't help tell rows apart.",
        "action": "We drop `{col}`, so that it doesn't waste time or memory.",
    },
    "text_column": {
        "found": "`{col}` contains long free text (about {avg_len:.0f} characters per row).",
        "why": "Free text needs special language tools; treating each sentence as a category would create thousands of useless columns.",
        "action": "We set `{col}` aside for now, so that the pipeline stays fast and simple.",
    },
    "missing_column": {
        "found": "`{col}` is empty in {pct:.0f}% of rows.",
        "why": "When most of a column is missing, any value we fill in is mostly invented.",
        "action": "We drop `{col}`, so that the model isn't trained on made-up numbers.",
    },
    "datetime_column": {
        "found": "`{col}` holds dates.",
        "why": "Models can't read a date directly, but the year, month and weekday often carry real signal (e.g. seasons, weekends).",
        "action": "We split `{col}` into year / month / weekday numbers, so that the model can use the time information.",
    },
    "high_card_column": {
        "found": "`{col}` is a category column with {n_unique} different values.",
        "why": "Turning every rare value into its own column would make the data huge and noisy.",
        "action": "We keep only its most common values and group the rest as 'other', so that the data stays compact.",
    },
    "duplicates": {
        "found": "{n:,} row(s) are exact copies of other rows.",
        "why": "Duplicates can land in both the training and test sets, which makes scores look better than they really are.",
        "action": "We remove duplicate rows, so that the test score is honest.",
    },
    "lookalike_rows": {
        "found": "{n:,} rows ({pct:.0f}%) are identical in every column except the ID {ids}.",
        "why": "They may be genuine repeat records, so deleting them would be wrong. But a test row with an identical twin in training is an 'easy question', which can make the test score look better than it really is.",
        "action": "We keep them, and also report the score on test rows that have no exact twin in training, so that you get an honest number.",
    },
    "date_parts": {
        "found": "`{year}`, `{month}` and `{day}` together form one date ({invalid} rows are impossible dates, like 29 February in a non-leap year).",
        "why": "Stored separately, they can't tell the model the day of the week - and weekend vs weekday often matters.",
        "action": "We combine them into a real date and add `{prefix}_weekday`; impossible dates are left blank and filled like any other gap.",
    },
    "target_like_excluded": {
        "found": "`{col}` looks like an outcome column ({reason}).",
        "why": "Using the answer as a clustering input would just group rows by that answer.",
        "action": "We leave `{col}` out of the clustering, so that the groups come from the other columns. Pick it as the target if you want to predict it instead.",
    },
    "suggested_target": {
        "found": "`{col}` looks like the outcome column: {reason}.",
        "why": "Most tables are collected to predict one thing. Predicting it gives a model with a clear, checkable score.",
        "action": "We pre-selected `{col}` as the target, so that we predict it. Choose 'Nothing' if you'd rather look for groups.",
    },
    "compare_models": {
        "found": "Rules of thumb point to **{rule_model}** for data of this size and shape.",
        "why": "Rules of thumb can be wrong for a particular dataset.",
        "action": "We test three different kinds of model ({names}) with cross-validation, so that the data - not a guess - picks the winner.",
    },
    "positive_class": {
        "found": "The rarer class is `{pos}` ({pct:.1f}% of rows).",
        "why": "In most business problems the rare outcome (a cancellation, a fraud, a churn) is the one worth catching.",
        "action": "We treat `{pos}` as the class to catch and tune the model's decision threshold for it, so that it finds more of them.",
    },
    "missing_values": {
        "found": "{n_cols} feature column(s) still have some missing values (e.g. {examples}).",
        "why": "Most models crash or behave badly when they see an empty cell.",
        "action": "We fill gaps with the median (numbers) or a 'missing' category (categories), so that no rows have to be thrown away.",
    },
    # ── Task choice ───────────────────────────────────────────────────────
    "task_clustering": {
        "found": "No target column was chosen.",
        "why": "Without a 'right answer' column we can't train a model to predict something - but we can still look for natural groups.",
        "action": "We run clustering, so that rows that look alike are grouped together.",
    },
    "task_regression": {
        "found": "`{target}` is numeric with {n_unique} different values.",
        "why": "A target with many possible numbers is a quantity (like a price), not a label.",
        "action": "We treat this as **regression**, so that the model predicts a number.",
    },
    "task_classification": {
        "found": "`{target}` has {n_unique} distinct values{numeric_note}.",
        "why": "A target with a small set of values is a label (like yes/no or a type), not a quantity.",
        "action": "We treat this as **{subtype} classification**, so that the model predicts which group each row belongs to.",
    },
    "target_missing": {
        "found": "{n:,} row(s) have no value for the target `{target}`.",
        "why": "A row without an answer can't teach the model anything.",
        "action": "We drop those rows, so that every training example has a known answer.",
    },
    "rare_classes": {
        "found": "{n_classes} class(es) in `{target}` appear only once: {examples}.",
        "why": "A class seen once can't appear in both the training and the test set.",
        "action": "We drop those {n_rows} row(s), so that the train/test split works.",
    },
    "imbalanced": {
        "found": "Classes are uneven: the smallest class is {min_pct:.1f}% of rows vs {max_pct:.1f}% for the largest.",
        "why": "A model can score high accuracy by always guessing the big class while ignoring the small one.",
        "action": "We weight the rare classes more heavily and judge the model by **macro F1**, so that every class counts equally.",
    },
    "skewed_target": {
        "found": "`{target}` is skewed (skew = {skew:.2f}): a few very large values and many small ones.",
        "why": "Models chase the big values and do worse on typical rows.",
        "action": "We train on log(`{target}`) and convert predictions back, so that errors on small and large values are treated fairly.",
    },
    # ── Model choice ──────────────────────────────────────────────────────
    "model_small": {
        "found": "The data is small ({rows:,} rows).",
        "why": "Complex models tend to memorise small datasets (overfitting). Simple linear models are more reliable here and easy to interpret.",
        "action": "We use **{model}**, so that the result is stable and explainable.",
    },
    "model_medium": {
        "found": "The data is medium-sized ({rows:,} rows).",
        "why": "There is enough data for a model that captures curves and interactions between columns.",
        "action": "We use **{model}** (many decision trees voting together), so that non-linear patterns are captured without much tuning.",
    },
    "model_large": {
        "found": "The data is large ({rows:,} rows).",
        "why": "Random forests get slow on big data; histogram-based gradient boosting is built for speed at this scale.",
        "action": "We use **{model}**, so that training stays fast without losing accuracy.",
    },
    "many_features": {
        "found": "There are {n_features} feature columns for only {rows:,} rows.",
        "why": "With many columns and few rows, a linear model can fit noise.",
        "action": "We use **LassoCV**, which switches off unhelpful columns, so that only useful features are kept.",
    },
    "collinear": {
        "found": "`{a}` and `{b}` move almost together (correlation {r:.2f}).",
        "why": "When two inputs say the same thing, a plain linear model gets confused about how much credit to give each.",
        "action": "We use **RidgeCV**, which shares credit between similar columns, so that the coefficients stay stable.",
    },
    "sampled": {
        "found": "The data has {rows:,} rows.",
        "why": "Training on every row would be slow, and beyond a point extra rows add very little accuracy.",
        "action": "We train on a random sample of {n:,} rows, so that results arrive in seconds.",
    },
    "clustering_model": {
        "found": "There are {rows:,} rows and {n_features} usable feature columns.",
        "why": "{reason}",
        "action": "We use **{model}**, so that grouping is {benefit}.",
    },
    # ── Halts ─────────────────────────────────────────────────────────────
    "halt_small": {
        "found": "The file has only {rows} rows.",
        "why": "With fewer than 30 rows any model result would mostly be luck.",
        "action": "We stop here. Please upload a larger dataset, so that the results mean something.",
    },
    "halt_columns": {
        "found": "The file has only {cols} column(s).",
        "why": "We need at least one input column to learn from.",
        "action": "We stop here. Please upload data with more columns.",
    },
    "halt_missing": {
        "found": "{pct:.0f}% of all cells are empty.",
        "why": "With most of the data missing, anything we learn would be built on guesses.",
        "action": "We stop here. Please clean the data first, so that there is real information to learn from.",
    },
    "halt_target_type": {
        "found": "`{target}` is a {role} column.",
        "why": "{reason}",
        "action": "We stop here. Please pick a different target column.",
    },
    "halt_no_features": {
        "found": "After cleaning, no usable feature columns are left.",
        "why": "A model needs at least one input column to learn from.",
        "action": "We stop here. Please check the column profile to see why each column was dropped.",
    },
}


def story(key: str, **values) -> Step:
    t = TEMPLATES[key]
    return Step(key, t["found"].format(**values), t["why"].format(**values), t["action"].format(**values))


# Fixed explanations for the notebook cells (these don't depend on the data).
CELL_STORIES: dict[str, Step] = {
    "setup": Step(
        "setup",
        "Your file is already loaded into a table called `df`.",
        "Every cell below works on this table, step by step - just like a Jupyter notebook.",
        "We import the libraries we need, so that the later cells can use them.",
    ),
    "eda": Step(
        "eda",
        "Before modelling we know the column types, but not what the values look like.",
        "Plots reveal outliers, skew and relationships that summary numbers hide.",
        "We draw distributions and correlations, so that you can sanity-check the data with your own eyes.",
    ),
    "split": Step(
        "split",
        "We need a fair way to measure how good the model is.",
        "Testing a model on the same rows it learned from is like grading a student on questions they've already seen.",
        "We hide 20% of rows as a **test set**, so that the final score reflects performance on new, unseen data.",
    ),
    "split_stratified": Step(
        "split_stratified",
        "We need a fair way to measure how good the model is, and the target is a set of classes.",
        "Testing on training rows is like grading a student on questions they've already seen. A random split could also leave a rare class out of the test set.",
        "We hide 20% of rows as a **test set**, keeping the same class mix in both parts (stratified split), so that the score is fair for every class.",
    ),
    "preprocess": Step(
        "preprocess",
        "The columns mix numbers and categories, and some may have gaps.",
        "Models only understand complete tables of numbers.",
        "We build one preprocessing recipe (fill gaps → {scale}turn categories into 0/1 columns), so that exactly the same steps run on training and test data.",
    ),
    "baseline": Step(
        "baseline",
        "A score like 0.80 means nothing on its own.",
        "We need a reference point: what would a 'dumb' strategy that ignores every input score?",
        "We score a baseline that always guesses {guess}, so that we can tell whether the real model actually learned something.",
    ),
    "insights": Step(
        "insights",
        "We know the target, but not yet which columns move it.",
        "Tables like 'cancellation rate by market segment' are insights a business can act on - and they show what a model should be able to learn.",
        "We compare the {outcome} across groups of every column and chart the biggest differences, so that the key drivers are visible before any modelling.",
    ),
    "compare": Step(
        "compare",
        "Models learn in different ways: straight lines (linear), if/else rules (decision tree) or many trees voting (forest / boosting).",
        "No single kind of model is best for every dataset.",
        "We score each candidate with 3-fold cross-validation on (a sample of) the training rows, so that the winner is chosen without ever touching the test set.",
    ),
    "train": Step(
        "train",
        "The comparison picked a winner and we have a baseline to beat.",
        "This is the step where the model learns patterns from all training rows.",
        "We retrain the winning model on every training row and score it on training and test rows, so that we can also spot overfitting (great on train, poor on test).",
    ),
    "threshold": Step(
        "threshold",
        "By default a model only says `{pos}` when it is more than 50% sure.",
        "50% is arbitrary. A lower threshold catches more `{pos}` cases (higher recall) but raises more false alarms (lower precision).",
        "We pick the threshold with the best F1 for `{pos}` from cross-validated predictions on training rows, so that the test set stays untouched for the final check.",
    ),
    "explain": Step(
        "explain",
        "We have a trained model, but not yet *why* it predicts what it does.",
        "Knowing which columns matter builds trust and often teaches you something about the problem.",
        "We shuffle one column at a time and measure how much the test score drops (permutation importance), so that we see which inputs the model relies on.",
    ),
    "cluster_prep": Step(
        "cluster_prep",
        "Clustering measures how 'far apart' rows are.",
        "If one column is in thousands and another in fractions, the big one would dominate the distance.",
        "We fill gaps, scale every column to the same range and turn categories into 0/1 columns, so that every feature has a fair say.",
    ),
    "choose_k": Step(
        "choose_k",
        "We don't know in advance how many groups the data contains.",
        "Picking the number of clusters (k) by hand is guesswork.",
        "We try k = {k_min}…{k_max} and keep the one with the best **silhouette score**, so that the groups are as distinct as possible.",
    ),
    "cluster_fit": Step(
        "cluster_fit",
        "We now know a good number of clusters.",
        "Groups are only useful if we can see and describe them.",
        "We fit the final model, draw the clusters in 2-D (PCA) and compare the average of each column per cluster, so that you can tell what makes each group different.",
    ),
}


GLOSSARY: dict[str, str] = {
    "Feature": "An input column the model uses to make a prediction.",
    "Target": "The column we want to predict (the 'answer').",
    "Classification": "Predicting a label or group, e.g. spam / not spam.",
    "Regression": "Predicting a number, e.g. a house price.",
    "Clustering": "Grouping similar rows together when there is no answer column.",
    "Train / test split": "Learning from one part of the data and checking on another part the model has never seen.",
    "Overfitting": "When a model memorises the training data and does poorly on new data.",
    "Baseline": "The score of a trivial strategy (e.g. always guess the most common class). A real model must beat it.",
    "Accuracy": "The share of predictions that are exactly right.",
    "Macro F1": "The average F1 score over all classes, so small classes count as much as big ones. 1.0 is perfect.",
    "R²": "How much of the variation in the target the model explains. 1.0 is perfect, 0 is no better than guessing the average.",
    "MAE": "Mean absolute error: on average, how far predictions are from the truth, in the target's own units.",
    "Silhouette score": "How well separated clusters are, from -1 to 1. Above 0.5 is strong, 0.25-0.5 is reasonable, below 0.25 is weak.",
    "One-hot encoding": "Turning a category column into several 0/1 columns, one per category.",
    "Scaling": "Rescaling numeric columns to a similar range so no column dominates just because its numbers are bigger.",
    "Imputation": "Filling in missing values with a sensible guess, e.g. the median.",
    "Permutation importance": "How much the score drops when one column is randomly shuffled. A big drop means the model relies on that column.",
    "PCA": "A way to squash many columns into 2 so the data can be drawn on a flat chart.",
}


def interpret(metrics: dict, target: str | None) -> list[str]:
    """Turn the numbers produced by the generated code into plain sentences."""
    task = metrics.get("task")
    out: list[str] = []

    if task == "clustering":
        k, sil = metrics.get("k"), metrics.get("silhouette")
        if k is None or sil is None:
            return out
        strength = "strong" if sil >= 0.5 else "reasonable" if sil >= 0.25 else "weak"
        out.append(f"The data splits best into **{k} groups**, with a silhouette score of **{sil:.2f}** - that is **{strength}** separation.")
        if strength == "weak":
            out.append("Weak separation means the groups overlap a lot; treat them as rough tendencies rather than clear-cut types.")
        out.append("Look at the per-cluster averages table to describe each group in your own words.")
        return out

    name = metrics.get("metric_name")
    test, train, base = metrics.get("test_score"), metrics.get("train_score"), metrics.get("baseline_score")
    if test is None or name is None:
        return out

    cv = metrics.get("cv_scores")
    if cv and len(cv) > 1:
        ranked = sorted(cv.items(), key=lambda kv: kv[1], reverse=True)
        others = ", ".join(f"{n} {v:.2f}" for n, v in ranked[1:])
        out.append(f"We compared {len(cv)} kinds of model; **{ranked[0][0]}** won cross-validation with {ranked[0][1]:.2f} (vs {others}).")

    if name == "R²":
        out.append(f"**R² = {test:.2f}** on the test set: the model explains about **{max(test, 0) * 100:.0f}%** of the variation in `{target}`.")
        if "mae" in metrics:
            out.append(f"On average its predictions are off by **{metrics['mae']:,.3g}** (MAE, in the units of `{target}`).")
    else:
        out.append(f"**{name} = {test:.2f}** on the test set (1.0 would be perfect).")

    if base is not None:
        gain = test - base
        if gain <= 0.01:
            out.append(f"⚠️ The baseline scores {base:.2f}, so the model is **not doing better than guessing**. The inputs may not contain enough signal about `{target}`.")
        elif gain < 0.1:
            out.append(f"The baseline scores {base:.2f}, so the model adds a **small but real** improvement.")
        else:
            out.append(f"The baseline scores {base:.2f}, so the model is **clearly learning** something useful.")

    if train is not None and train - test > 0.15:
        out.append(f"⚠️ Training score ({train:.2f}) is much higher than test score ({test:.2f}) - a sign of **overfitting**. More data or a simpler model would help.")
    elif train is not None:
        out.append(f"Training ({train:.2f}) and test ({test:.2f}) scores are close, so the model **generalises** well to new rows.")

    unseen = metrics.get("test_score_unseen")
    if unseen is not None:
        share = metrics.get("unseen_share", 0)
        gap = test - unseen
        verdict = ("so the duplicates make the headline score look a little better than it really is"
                   if gap > 0.02 else "so the duplicates are not inflating the score")
        out.append(f"On the {share:.0%} of test rows with no identical twin in training, {name} is **{unseen:.2f}** "
                   f"(vs {test:.2f} overall), {verdict}.")

    if "threshold" in metrics:
        pos, t = metrics["positive"], metrics["threshold"]
        out.append(f"For `{pos}`, moving the decision threshold from 0.50 to **{t:.2f}** changes recall "
                   f"{metrics['recall_default']:.2f} → {metrics['recall_tuned']:.2f}, precision "
                   f"{metrics['precision_default']:.2f} → {metrics['precision_tuned']:.2f} and F1 "
                   f"{metrics['f1_default']:.2f} → **{metrics['f1_tuned']:.2f}**. Pick the threshold that matches the "
                   f"business cost: a missed `{pos}` vs a false alarm.")

    drivers = metrics.get("drivers")
    if drivers:
        pct = "overall_outcome" in metrics and metrics.get("task") == "classification"
        show = (lambda v: f"{v:.0%}") if pct else (lambda v: f"{v:,.3g}")
        for col, info in list(drivers.items())[:2]:
            out.append(f"Key driver `{col}`: the outcome ranges from {show(info['low_value'])} (`{info['low']}`) "
                       f"to {show(info['high_value'])} (`{info['high']}`).")

    top = metrics.get("top_features")
    if top:
        out.append("The most influential columns are " + ", ".join(f"`{c}`" for c in top[:3]) + ".")
    return out
