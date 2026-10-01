import numpy as np
import pandas as pd

from automl.decision import decide
from automl.profiling import profile_dataset


def run(df, target):
    return decide(df, profile_dataset(df), target)


def keys(d):
    return [s.key for s in d.steps]


def test_messy_columns_are_dropped_or_converted(classification_df):
    d = run(classification_df, "churn")
    assert set(d.drop_cols) == {"customer_id", "source"}
    assert d.date_cols == ["signup"]
    assert "signup_year" in d.num_features and "city" in d.cat_features
    assert {"id_column", "constant_column", "datetime_column", "missing_values"} <= set(keys(d))


def test_imbalanced_binary_classification(classification_df):
    d = run(classification_df, "churn")
    assert (d.task, d.subtype, d.model) == ("classification", "binary", "RandomForestClassifier")
    assert d.class_weight and d.stratify and d.primary_metric == "Macro F1"
    assert "class_weight='balanced'" in d.model_constructor()
    assert "imbalanced" in keys(d)


def test_numeric_target_with_few_values_is_classification():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.random(300), "grade": rng.integers(1, 6, 300)})
    d = run(df, "grade")
    assert d.task == "classification" and d.subtype == "multiclass"
    assert d.model == "LogisticRegression"  # small data -> simple model


def test_skewed_regression_uses_log_target(regression_df):
    d = run(regression_df, "price")
    assert d.task == "regression" and d.log_target
    assert d.model == "LinearRegression" and d.scale_numeric


def test_collinear_small_regression_uses_ridge():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    df = pd.DataFrame({"a": a, "a_copy": a + rng.normal(scale=0.01, size=200), "y": a * 3 + rng.normal(size=200)})
    assert run(df, "y").model == "RidgeCV"


def test_clustering_without_target(blobs_df):
    d = run(blobs_df, None)
    assert d.task == "clustering" and d.model == "KMeans" and d.scale_numeric


def test_large_data_is_sampled_and_uses_fast_model():
    rng = np.random.default_rng(0)
    n = 120_000
    df = pd.DataFrame({"x": rng.random(n), "y": rng.random(n) * 100})
    d = run(df, "y")
    assert d.model == "HistGradientBoostingRegressor"
    assert "sampled" not in keys(d)  # 120k < 200k limit for gradient boosting
    d = run(pd.DataFrame({"x": rng.random(60_000), "y": rng.random(60_000) * 100}), "y")
    assert d.model == "RandomForestRegressor" and d.sample_rows == 50_000


def test_rare_classes_are_removed():
    rng = np.random.default_rng(0)
    labels = ["a"] * 100 + ["b"] * 100 + ["only_once"]
    df = pd.DataFrame({"x": rng.random(201), "label": labels})
    d = run(df, "label")
    assert d.rare_classes == ["only_once"] and d.subtype == "binary"


def test_quality_gates():
    assert keys(run(pd.DataFrame({"a": range(10), "b": range(10)}), None))[-1] == "halt_small"
    assert run(pd.DataFrame({"a": range(100)}), None).halted
    empty = pd.DataFrame({"a": [np.nan] * 99 + [1.0], "b": [np.nan] * 99 + [2.0]})
    assert keys(run(empty, None))[-1] == "halt_missing"
    dates = pd.DataFrame({"x": range(50), "when": pd.date_range("2024-01-01", periods=50)})
    assert keys(run(dates, "when"))[-1] == "halt_target_type"


def test_every_step_reads_as_a_sentence(classification_df):
    for s in run(classification_df, "churn").steps:
        assert s.found and s.why and s.action
        assert "{" not in s.markdown()  # no unfilled template placeholders
