"""Turn a ``Decision`` into notebook-style code cells.

Each cell is small and does one thing, and carries its own story (what we
found / why it matters / what we're doing) so a beginner can follow along.
The generated code is plain, copy-pasteable Python - it is exactly what runs
in the app, and it can be downloaded as a Jupyter notebook.
"""
from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass

from .decision import TEST_SIZE, Decision
from .narrative import CELL_STORIES, Step

EDA_SAMPLE = 5_000
MAX_HIST_COLS = 12
MAX_CORR_COLS = 15


@dataclass(frozen=True)
class Cell:
    id: str
    title: str
    story: Step
    code: str


def _story(key: str, **values) -> Step:
    s = CELL_STORIES[key]
    return Step(key, s.found.format(**values), s.why.format(**values), s.action.format(**values))


def _cell(cell_id: str, title: str, story: Step, body: str) -> Cell:
    why = re.sub(r"[*`]", "", story.action)
    comment = textwrap.fill(why, width=88, initial_indent="# Why: ", subsequent_indent="#      ")
    return Cell(cell_id, title, story, f"{comment}\n{textwrap.dedent(body).strip()}\n")


# ── Shared cells ──────────────────────────────────────────────────────────
def _setup_cell(d: Decision) -> Cell:
    return _cell("setup", "Setup", CELL_STORIES["setup"], f"""
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        metrics = {{"task": {d.task!r}}}  # the cells below store their key numbers here
    """)


def _prepare_cell(d: Decision) -> Cell:
    lines = ["data = df.copy()"]
    if d.drop_cols:
        lines.append(f"\n# Columns that can't help the model (IDs, constants, free text, mostly empty)\ndata = data.drop(columns={d.drop_cols!r})")
    for col in d.date_cols:
        lines.append(textwrap.dedent(f"""
            # Split the date column {col!r} into numbers a model can use
            when = pd.to_datetime(data[{col!r}], errors="coerce", format="mixed")
            data[{col + '_year'!r}] = when.dt.year
            data[{col + '_month'!r}] = when.dt.month
            data[{col + '_weekday'!r}] = when.dt.dayofweek
            data = data.drop(columns={col!r})"""))
    for prefix, y, m, day in d.date_parts:
        lines.append(textwrap.dedent(f"""
            # {y} / {m} / {day} are one date split in three: combine them to get the weekday
            when = pd.to_datetime(pd.DataFrame({{"year": data[{y!r}], "month": data[{m!r}], "day": data[{day!r}]}}), errors="coerce")
            data[{prefix + '_weekday'!r}] = when.dt.dayofweek  # 0 = Monday; impossible dates stay blank"""))
    if d.drop_duplicates:
        lines.append("\ndata = data.drop_duplicates()")
    if d.target:
        lines.append(f"\ntarget = {d.target!r}\ndata = data.dropna(subset=[target])  # rows without an answer can't teach anything")
    if d.rare_classes:
        lines.append(f"data = data[~data[target].isin({d.rare_classes!r})]  # classes seen only once")
    if d.sample_rows:
        if d.task == "classification":
            lines.append(textwrap.dedent(f"""
                # Keep a smaller sample with the same class mix, so training stays fast
                if len(data) > {d.sample_rows}:
                    from sklearn.model_selection import train_test_split
                    data, _ = train_test_split(data, train_size={d.sample_rows}, stratify=data[target], random_state=42)"""))
        else:
            lines.append(textwrap.dedent(f"""
                # Keep a random sample, so training stays fast
                if len(data) > {d.sample_rows}:
                    data = data.sample(n={d.sample_rows}, random_state=42)"""))
    lines.append(f"\nnum_features = {d.num_features!r}\ncat_features = {d.cat_features!r}")
    if d.cat_features:
        lines.append("# Treat blanks in category columns as their own 'missing' category, and store categories as text\n"
                     "data[cat_features] = data[cat_features].fillna(\"missing\").astype(str)")
    lines.append('\nprint(f"Working with {len(data):,} rows and {len(num_features) + len(cat_features)} feature columns.")')
    if d.target:
        lines.append("print(f\"Target: {target!r}\")")

    dropped = len(d.drop_cols) + len(d.date_cols)
    found = (f"The engine flagged {dropped} column(s) to drop or convert." if dropped
             else "All columns are usable as they are.")
    story = Step(
        "prepare", found,
        "Messy inputs (IDs, dates, blanks) either crash models or quietly make them worse.",
        "We apply the clean-up decisions from the engine (see the decision trace), so that the model only sees useful, well-formed columns.",
    )
    return _cell("prepare", "Prepare the data", story, "\n".join(lines))


def _eda_cell(d: Decision) -> Cell:
    hist_cols = d.num_features[:MAX_HIST_COLS]
    corr_cols = d.num_features[:MAX_CORR_COLS]
    body = [f"sample = data.sample(min(len(data), {EDA_SAMPLE}), random_state=42)  # plotting a sample is much faster"]
    if hist_cols:
        body.append(textwrap.dedent(f"""
            # 1) How is each numeric column distributed? Look for skew and outliers.
            sample[{hist_cols!r}].hist(bins=30, figsize=(12, {2.5 * ((len(hist_cols) + 3) // 4):.1f}), layout=({(len(hist_cols) + 3) // 4}, 4), color="#0071e3", edgecolor="white")
            plt.suptitle("Distribution of numeric columns")
            plt.tight_layout(); plt.show()"""))
    if len(corr_cols) >= 2:
        body.append(textwrap.dedent(f"""
            # 2) Which numeric columns move together? (+1 = together, -1 = opposite, 0 = unrelated)
            plt.figure(figsize=(8, 6))
            sns.heatmap(sample[{corr_cols!r}].corr(), annot={len(corr_cols) <= 10}, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
            plt.title("Correlation between numeric columns")
            plt.tight_layout(); plt.show()"""))
    if d.task == "classification":
        body.append(textwrap.dedent("""
            # 3) How many rows are in each class?
            sample[target].value_counts().head(20).plot.bar(color="#0071e3", figsize=(8, 4))
            plt.title(f"Rows per class of {target!r}"); plt.ylabel("rows")
            plt.tight_layout(); plt.show()"""))
    elif d.task == "regression":
        body.append(textwrap.dedent("""
            # 3) What does the target look like?
            sample[target].plot.hist(bins=40, color="#0071e3", edgecolor="white", figsize=(8, 4))
            plt.title(f"Distribution of {target!r}"); plt.xlabel(target)
            plt.tight_layout(); plt.show()"""))
    elif d.cat_features:
        body.append(textwrap.dedent(f"""
            # 3) Most common values of a category column
            sample[{d.cat_features[0]!r}].value_counts().head(15).plot.bar(color="#0071e3", figsize=(8, 4))
            plt.title("Most common values of {d.cat_features[0]}")
            plt.tight_layout(); plt.show()"""))
    return _cell("eda", "Explore the data", CELL_STORIES["eda"], "\n".join(body))


def _preprocess_body(d: Decision) -> str:
    if d.scale_numeric:
        numeric = ('numeric_steps = Pipeline([\n'
                   '    ("fill_gaps", SimpleImputer(strategy="median")),\n'
                   '    ("scale", StandardScaler()),  # put every number on the same scale\n'
                   '])')
    else:
        numeric = 'numeric_steps = SimpleImputer(strategy="median")  # tree models don\'t need scaling'
    return f"""
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        {textwrap.indent(numeric, ' ' * 8).strip()}
        # One 0/1 column per category; rare categories are grouped into one 'infrequent' column
        category_steps = OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=20, sparse_output=False)

        preprocessor = ColumnTransformer([
            ("numbers", numeric_steps, num_features),
            ("categories", category_steps, cat_features),
        ])
    """


# ── Supervised cells ──────────────────────────────────────────────────────
def _insights_cell(d: Decision) -> Cell:
    if d.task == "regression":
        outcome_code, label, as_pct = "data[target]", f"average `{d.target}`", False
    else:
        outcome_code, label, as_pct = f"data[target] == {d.positive_class!r}", f"rate of `{d.positive_class}`", True
    fmt = "{:.1%}" if as_pct else "{:,.3g}"
    body = f"""
        outcome = {outcome_code}
        overall = outcome.mean()
        print(f"Overall {label.replace('`', '')}: {{overall:{fmt[2:-1]}}}")

        def group_outcome(col):
            values = data[col]
            if col in num_features and values.nunique() > 10:
                lowest = values.min()
                values = pd.qcut(values, q=5, duplicates="drop")  # 5 equal-sized bands, labelled like "12-39"
                values = values.cat.rename_categories(lambda band: f"{{max(band.left, lowest):.4g}}-{{band.right:.4g}}")
            table = outcome.groupby(values, observed=True).agg(["mean", "size"])
            return table[table["size"] >= 30]  # ignore tiny groups - they are too noisy

        groups = {{col: group_outcome(col) for col in num_features + cat_features}}
        groups = {{col: t for col, t in groups.items() if len(t) >= 2}}
        spread = pd.Series({{col: t["mean"].max() - t["mean"].min() for col, t in groups.items()}}).sort_values(ascending=False)
        top = spread.index[:4].tolist()
        metrics["overall_outcome"] = float(overall)
        metrics["drivers"] = {{
            col: {{"low": str(groups[col]["mean"].idxmin()), "low_value": float(groups[col]["mean"].min()),
                   "high": str(groups[col]["mean"].idxmax()), "high_value": float(groups[col]["mean"].max())}}
            for col in top
        }}

        print("Columns where the {label.replace('`', '')} differs most between groups:")
        for col in top:
            print(f"\\n{{col}}")
            print(groups[col]["mean"].rename_axis(None).map({fmt!r}.format).to_string())

        fig, axes = plt.subplots(1, len(top), figsize=(4 * len(top), 4), squeeze=False)
        for ax, col in zip(axes[0], top):
            groups[col]["mean"].plot.bar(ax=ax, color="#0071e3")
            ax.axhline(overall, color="grey", linestyle="--", linewidth=1)  # overall level for comparison
            ax.set_title(col); ax.set_xlabel(""); ax.tick_params(axis="x", labelrotation=45)
        axes[0][0].set_ylabel({label.replace('`', '')!r})
        plt.tight_layout(); plt.show()
    """
    return _cell("insights", "Find the key drivers", _story("insights", outcome=label), body)


def _compare_cell(d: Decision) -> Cell:
    scoring = {"R²": "r2", "Macro F1": "f1_macro", "Accuracy": "accuracy"}[d.primary_metric]
    imports = sorted({d.model_import(n) for n in d.candidates})
    entries = "\n".join(f"    {n!r}: {d.model_constructor(n)}," for n in d.candidates)
    wrap = ""
    if d.log_target:
        wrap = """
            # Every candidate learns log(target) and converts its predictions back
            from sklearn.compose import TransformedTargetRegressor
            candidates = {name: TransformedTargetRegressor(regressor=m, func=np.log1p, inverse_func=np.expm1)
                          for name, m in candidates.items()}
        """
    body = textwrap.dedent("""
        from sklearn.model_selection import cross_val_score
        {imports}

        candidates = {{
        {entries}
        }}
    """).format(imports="\n".join(imports), entries=entries) + textwrap.dedent(wrap) + textwrap.dedent(f"""
        check = X_train.sample(min(len(X_train), 10000), random_state=42)  # a sample keeps this quick
        cv_scores = {{}}
        for name, candidate in candidates.items():
            candidate_pipe = Pipeline([("prep", preprocessor), ("model", candidate)])
            cv_scores[name] = float(cross_val_score(candidate_pipe, check, y_train.loc[check.index], cv=3, scoring={scoring!r}).mean())
            print(f"{{name:32s}} {{metrics['metric_name']}} = {{cv_scores[name]:.3f}}")

        best_name = max(cv_scores, key=cv_scores.get)
        metrics["cv_scores"] = cv_scores
        metrics["best_model"] = best_name
        print(f"\\nWinner: {{best_name}}")

        pd.Series(cv_scores).sort_values().plot.barh(color="#0071e3", figsize=(7, 3))
        plt.axvline(metrics["baseline_score"], color="grey", linestyle="--", label="baseline")
        plt.xlabel(f"cross-validated {{metrics['metric_name']}} (higher = better)"); plt.legend()
        plt.title("Which kind of model fits this data best?")
        plt.tight_layout(); plt.show()
    """)
    return _cell("compare", "Compare candidate models", CELL_STORIES["compare"], body)


def _threshold_cell(d: Decision) -> Cell:
    body = f"""
        from sklearn.base import clone
        from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
        from sklearn.model_selection import cross_val_predict

        positive = {d.positive_class!r}
        pos_col = list(pipe.classes_).index(positive)

        # Out-of-fold probabilities: each training row is scored by a model that never saw it
        check = X_train.sample(min(len(X_train), 20000), random_state=42)
        y_check = y_train.loc[check.index] == positive
        oof = cross_val_predict(clone(pipe), check, y_train.loc[check.index], cv=3, method="predict_proba")[:, pos_col]
        precision, recall, thresholds = precision_recall_curve(y_check, oof)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        threshold = float(thresholds[int(np.argmax(f1[:-1]))])

        # Final check on the untouched test set
        test_proba = pipe.predict_proba(X_test)[:, pos_col]
        is_pos = y_test == positive
        table = {{}}
        for label, t in [("default 0.50", 0.5), (f"tuned {{threshold:.2f}}", threshold)]:
            pred = test_proba >= t
            table[label] = {{"precision": precision_score(is_pos, pred, zero_division=0),
                            "recall": recall_score(is_pos, pred), "F1": f1_score(is_pos, pred)}}
        table = pd.DataFrame(table).T
        print(f"Test-set results for {{positive!r}}:")
        print(table.round(3).to_string())
        metrics.update(positive=positive, threshold=threshold,
                       f1_default=float(table["F1"].iloc[0]), f1_tuned=float(table["F1"].iloc[1]),
                       recall_default=float(table["recall"].iloc[0]), recall_tuned=float(table["recall"].iloc[1]),
                       precision_default=float(table["precision"].iloc[0]), precision_tuned=float(table["precision"].iloc[1]))

        plt.figure(figsize=(7, 4))
        plt.plot(thresholds, precision[:-1], label="precision")
        plt.plot(thresholds, recall[:-1], label="recall")
        plt.plot(thresholds, f1[:-1], label="F1")
        plt.axvline(threshold, color="grey", linestyle="--", label=f"chosen {{threshold:.2f}}")
        plt.xlabel(f"threshold: say {{positive!r}} when the model is at least this sure"); plt.legend()
        plt.title("The precision / recall trade-off")
        plt.tight_layout(); plt.show()
    """
    return _cell("threshold", "Tune the decision threshold", _story("threshold", pos=d.positive_class), body)



def _split_cell(d: Decision) -> Cell:
    strat = ", stratify=y" if d.stratify else ""
    body = f"""
        from sklearn.model_selection import train_test_split

        X = data[num_features + cat_features]
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={TEST_SIZE}, random_state=42{strat})
        print(f"Training rows: {{len(X_train):,}}   Test rows: {{len(X_test):,}}")
    """
    return _cell("split", "Split into training and test sets",
                 CELL_STORIES["split_stratified" if d.stratify else "split"], body)


def _preprocess_cell(d: Decision) -> Cell:
    story = _story("preprocess", scale="scale numbers → " if d.scale_numeric else "")
    return _cell("preprocess", "Build the preprocessing recipe", story, _preprocess_body(d))


def _baseline_cell(d: Decision) -> Cell:
    if d.task == "regression":
        guess = "the average value"
        body = """
            from sklearn.dummy import DummyRegressor
            from sklearn.metrics import r2_score

            def score(y_true, y_pred):
                return r2_score(y_true, y_pred)

            metrics["metric_name"] = "R²"
            baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
        """
    else:
        guess = "the most common class"
        if d.primary_metric == "Macro F1":
            fn = 'f1_score(y_true, y_pred, average="macro")'
            imp = "from sklearn.metrics import f1_score"
        else:
            fn = "accuracy_score(y_true, y_pred)"
            imp = "from sklearn.metrics import accuracy_score"
        body = f"""
            from sklearn.dummy import DummyClassifier
            {imp}

            def score(y_true, y_pred):
                return {fn}

            metrics["metric_name"] = {d.primary_metric!r}
            baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
        """
    body += """
            metrics["baseline_score"] = score(y_test, baseline.predict(X_test))
            print(f"Baseline {metrics['metric_name']}: {metrics['baseline_score']:.3f}  <- the score to beat")
    """
    return _cell("baseline", "Set a baseline to beat", _story("baseline", guess=guess), body)


def _train_cell(d: Decision) -> Cell:
    lines = [
        "from sklearn.base import clone",
        "",
        "model = clone(candidates[best_name])  # a fresh copy of the winning model",
        'pipe = Pipeline([("prep", preprocessor), ("model", model)])',
        "pipe.fit(X_train, y_train)",
        "y_pred = pipe.predict(X_test)",
        "",
        'metrics["train_score"] = score(y_train, pipe.predict(X_train))',
        'metrics["test_score"] = score(y_test, y_pred)',
        "print(f\"{best_name}  {metrics['metric_name']}  train: {metrics['train_score']:.3f}   test: {metrics['test_score']:.3f}\")",
    ]
    if d.lookalikes:
        lines += [
            "",
            "# Honest check: test rows with an identical twin in training are 'easy'. Score the rest on their own.",
            "row_key = pd.util.hash_pandas_object(X, index=False)",
            "unseen = ~row_key.loc[X_test.index].isin(set(row_key.loc[X_train.index]))",
            'metrics["unseen_share"] = float(unseen.mean())',
            'metrics["test_score_unseen"] = score(y_test[unseen], y_pred[unseen.to_numpy()])',
            "print(f\"Test rows with no twin in training: {unseen.mean():.0%}  ->  {metrics['metric_name']} on them: {metrics['test_score_unseen']:.3f}\")",
        ]
    if d.task == "regression":
        lines += [
            "",
            "from sklearn.metrics import mean_absolute_error",
            'metrics["mae"] = mean_absolute_error(y_test, y_pred)',
            "print(f\"Average error (MAE): {metrics['mae']:,.3f}\")",
            "",
            "# Points close to the dashed line are good predictions",
            "plt.figure(figsize=(6, 6))",
            'plt.scatter(y_test, y_pred, alpha=0.4, color="#0071e3")',
            "lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]",
            'plt.plot(lims, lims, "k--", linewidth=1)',
            'plt.xlabel("actual"); plt.ylabel("predicted"); plt.title("Predicted vs actual")',
            "plt.tight_layout(); plt.show()",
        ]
    else:
        lines += [
            "",
            "from sklearn.metrics import ConfusionMatrixDisplay, classification_report",
            "print(classification_report(y_test, y_pred, zero_division=0))",
            "",
            "# Diagonal = correct predictions; off-diagonal = which classes get confused",
            "if y.nunique() <= 20:",
            "    fig, ax = plt.subplots(figsize=(6, 5))",
            '    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)',
            '    ax.set_title("Confusion matrix (test set)")',
            "    plt.tight_layout(); plt.show()",
        ]
    return _cell("train", "Train and evaluate the winner", CELL_STORIES["train"], "\n".join(lines))


def _explain_cell(d: Decision) -> Cell:
    scoring = {"R²": "r2", "Macro F1": "f1_macro", "Accuracy": "accuracy"}[d.primary_metric]
    body = f"""
        from sklearn.inspection import permutation_importance

        check = X_test.sample(min(len(X_test), 2000), random_state=42)  # a sample keeps this quick
        result = permutation_importance(pipe, check, y_test.loc[check.index], scoring={scoring!r}, n_repeats=3, random_state=42)
        importance = pd.Series(result.importances_mean, index=check.columns).sort_values(ascending=False)
        metrics["top_features"] = importance.index[:5].tolist()

        importance.head(15).sort_values().plot.barh(color="#0071e3", figsize=(8, 5))
        plt.title("Which columns matter most?"); plt.xlabel("drop in score when the column is shuffled")
        plt.tight_layout(); plt.show()
    """
    return _cell("explain", "Explain the model", CELL_STORIES["explain"], body)


# ── Clustering cells ──────────────────────────────────────────────────────
def _cluster_prep_cell(d: Decision) -> Cell:
    body = _preprocess_body(d) + """
        X = preprocessor.fit_transform(data[num_features + cat_features])
        print(f"Clustering {X.shape[0]:,} rows using {X.shape[1]} prepared columns.")
    """
    return _cell("preprocess", "Prepare features for clustering", CELL_STORIES["cluster_prep"], body)


def _choose_k_cell(d: Decision) -> Cell:
    k_min, k_max = d.k_range
    body = f"""
        {d.model_import()}
        from sklearn.metrics import silhouette_score

        rng = np.random.default_rng(42)
        X_small = X[rng.choice(len(X), size=min(len(X), {EDA_SAMPLE}), replace=False)]  # a sample keeps the search fast

        scores = {{}}
        for k in range({k_min}, {k_max + 1}):
            labels = {d.model_constructor()}.fit_predict(X_small)
            scores[k] = silhouette_score(X_small, labels)
            print(f"k={{k}}: silhouette = {{scores[k]:.3f}}")
        best_k = max(scores, key=scores.get)
        print(f"Best number of clusters: {{best_k}}")

        plt.figure(figsize=(7, 4))
        plt.plot(list(scores), list(scores.values()), "o-", color="#0071e3")
        plt.axvline(best_k, color="grey", linestyle="--")
        plt.xlabel("number of clusters (k)"); plt.ylabel("silhouette score (higher = better)")
        plt.title("Choosing the number of clusters")
        plt.tight_layout(); plt.show()
    """
    return _cell("choose_k", "Choose the number of clusters", _story("choose_k", k_min=k_min, k_max=k_max), body)


def _cluster_fit_cell(d: Decision) -> Cell:
    body = f"""
        from sklearn.decomposition import PCA

        k = best_k
        model = {d.model_constructor()}
        data["cluster"] = model.fit_predict(X)
        metrics["k"] = int(k)
        metrics["silhouette"] = float(scores[k])

        # Squash all columns into 2 so the clusters can be drawn
        points = PCA(n_components=2, random_state=42).fit_transform(X_small)
        plt.figure(figsize=(7, 6))
        plt.scatter(points[:, 0], points[:, 1], c=model.predict(X_small), cmap="tab10", alpha=0.6, s=12)
        plt.xlabel("PCA 1"); plt.ylabel("PCA 2"); plt.title(f"{{k}} clusters (2-D view)")
        plt.tight_layout(); plt.show()

        print("Rows per cluster:")
        print(data["cluster"].value_counts().sort_index().to_string())
        if num_features:
            print("\\nAverage of each numeric column per cluster (compare across rows to describe each group):")
            print(data.groupby("cluster")[num_features].mean().round(2).to_string())
    """
    return _cell("cluster_fit", "Fit clusters and describe them", CELL_STORIES["cluster_fit"], body)


# ── Public API ────────────────────────────────────────────────────────────
def build_cells(d: Decision) -> list[Cell]:
    if d.halted:
        return []
    cells = [_setup_cell(d), _prepare_cell(d), _eda_cell(d)]
    if d.task == "clustering":
        return cells + [_cluster_prep_cell(d), _choose_k_cell(d), _cluster_fit_cell(d)]
    if d.task == "regression" or d.positive_class is not None:
        cells.append(_insights_cell(d))
    cells += [_split_cell(d), _preprocess_cell(d), _baseline_cell(d), _compare_cell(d), _train_cell(d)]
    if d.positive_class is not None:
        cells.append(_threshold_cell(d))
    return cells + [_explain_cell(d)]


def to_notebook(cells: list[Cell], decision: Decision, filename: str) -> str:
    """Export the cells (with their stories) as a .ipynb JSON string."""
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

    reader = "read_csv" if filename.lower().endswith(".csv") else "read_excel"
    nb_cells = [new_markdown_cell(
        "# AutoML Explorer notebook\n\n"
        f"Generated from `{filename}`. Each step explains **what we found**, **why it matters** "
        "and **what we're doing**.\n\n## Decision trace\n\n"
        + "\n".join(f"- {s.short()}" for s in decision.steps)
    )]
    for cell in cells:
        nb_cells.append(new_markdown_cell(f"## {cell.title}\n\n{cell.story.markdown()}"))
        code = cell.code
        if cell.id == "setup":
            code += f'\ndf = pd.{reader}("{filename}")  # make sure the file is next to this notebook\n'
        nb_cells.append(new_code_cell(code))
    return nbformat.writes(new_notebook(cells=nb_cells))
