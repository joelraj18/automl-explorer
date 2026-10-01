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
    "train": Step(
        "train",
        "The data is prepared and we have a baseline to beat.",
        "This is the step where the model learns patterns from the training rows.",
        "We train **{model}** and score it on both training and test rows, so that we can also spot overfitting (great on train, poor on test).",
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

    top = metrics.get("top_features")
    if top:
        out.append("The most influential columns are " + ", ".join(f"`{c}`" for c in top[:3]) + ".")
    return out
