"""Generated code must actually run and beat the baseline."""
import json

import numpy as np
import pandas as pd
import pytest

from automl import build_cells, decide, interpret, profile_dataset, run_cells, to_notebook


def execute(df, target):
    d = decide(df, profile_dataset(df), target)
    cells = build_cells(d)
    results, metrics = run_cells(cells, df)
    errors = [(r.cell_id, r.stdout) for r in results if r.error]
    assert not errors, errors
    return d, cells, results, metrics


@pytest.mark.parametrize("fixture,target", [("classification_df", "churn"), ("regression_df", "price")])
def test_supervised_pipeline_runs_and_beats_baseline(request, fixture, target):
    df = request.getfixturevalue(fixture)
    _, cells, results, m = execute(df, target)
    assert m["test_score"] > m["baseline_score"] + 0.1
    assert m["top_features"]
    assert all(r.figures for r in results if r.cell_id in ("eda", "train", "explain"))
    assert all(c.code.startswith("# Why:") for c in cells)
    assert interpret(m, target)


def test_clustering_pipeline_runs(blobs_df):
    _, _, _, m = execute(blobs_df, None)  # messy columns: must still run
    assert m["k"] >= 2

    clean = blobs_df[["f1", "f2", "f3"]]  # clean blobs: must find the 3 groups
    _, _, _, m = execute(clean, None)
    assert m["k"] == 3 and m["silhouette"] > 0.5


def test_mixed_type_categories_and_unseen_values():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        "x": rng.normal(size=n),
        "mixed": rng.choice([1, "two", 3.0, None], n).tolist(),  # mixed types used to crash OneHotEncoder
        "rare": [f"r{i % 60}" for i in range(n)],
    })
    df["y"] = df["x"] * 2 + rng.normal(size=n) * 0.1
    execute(df, "y")


def test_error_skips_later_cells(classification_df):
    d = decide(classification_df, profile_dataset(classification_df), "churn")
    cells = build_cells(d)
    broken = cells[:2] + [cells[2].__class__("boom", "Boom", cells[2].story, "raise ValueError('bad')")] + cells[3:]
    results, _ = run_cells(broken, classification_df)
    assert results[2].error == "ValueError: bad"
    assert all(r.skipped for r in results[3:])


def test_notebook_export_is_valid_json(classification_df):
    d = decide(classification_df, profile_dataset(classification_df), "churn")
    nb = json.loads(to_notebook(build_cells(d), d, "data.csv"))
    sources = ["".join(c["source"]) for c in nb["cells"]]
    assert any('pd.read_csv("data.csv")' in s for s in sources)
    assert any("What we found" in s for s in sources)


def test_each_plt_show_is_its_own_figure(classification_df):
    """Plots must not draw on top of each other (histograms, heatmap, class bars = 3 images)."""
    d = decide(classification_df, profile_dataset(classification_df), "churn")
    results, _ = run_cells(build_cells(d), classification_df)
    eda = next(r for r in results if r.cell_id == "eda")
    assert len(eda.figures) == 3
