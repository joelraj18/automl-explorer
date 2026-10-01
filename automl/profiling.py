"""Column profiling: work out the *role* of every column, not just its dtype.

A dtype says how a value is stored; a role says how a model should use it.
An integer column can be a real quantity (``age``) or just a row number
(``customer_id``); a string column can be a category, a date or free text.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_integer_dtype, is_numeric_dtype

ROLES = ("numeric", "categorical", "high_card_categorical", "datetime", "id", "constant", "text")

MAX_CATEGORIES = 50       # more distinct values than this -> high-cardinality
ID_UNIQUE_RATIO = 0.95    # unique in at least 95% of rows -> candidate ID
TEXT_MIN_AVG_LEN = 30     # average string length above this -> free text
SAMPLE_ROWS = 2000        # rows inspected for the (slower) string checks

_ID_NAME = re.compile(r"(^|[_\s])(id|uuid|guid|key|index)$|^id[_\s]|^(id|index|unnamed: 0)$", re.IGNORECASE)


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    dtype: str
    n_unique: int
    missing_frac: float
    role: str
    reason: str
    avg_len: float = 0.0


@dataclass(frozen=True)
class DatasetProfile:
    n_rows: int
    n_cols: int
    missing_frac: float
    duplicate_rows: int
    columns: tuple[ColumnProfile, ...] = field(default_factory=tuple)

    def get(self, name: str) -> ColumnProfile:
        return next(c for c in self.columns if c.name == name)

    def by_role(self, *roles: str) -> list[str]:
        return [c.name for c in self.columns if c.role in roles]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "column": c.name,
                "type": c.dtype,
                "unique values": c.n_unique,
                "missing %": round(c.missing_frac * 100, 1),
                "role": c.role,
                "why this role": c.reason,
            }
            for c in self.columns
        )


def _looks_like_dates(s: pd.Series) -> bool:
    sample = s.dropna().astype(str).head(200)
    if sample.empty or pd.to_numeric(sample, errors="coerce").notna().mean() > 0.5:
        return False  # plain numbers stored as text are not dates
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() >= 0.9


def _profile_column(name: str, s: pd.Series, n_rows: int) -> ColumnProfile:
    n_unique = int(s.nunique(dropna=True))
    missing = float(s.isna().mean()) if n_rows else 0.0
    unique_ratio = n_unique / max(int(s.notna().sum()), 1)
    dtype = str(s.dtype)

    def make(role: str, reason: str, avg_len: float = 0.0) -> ColumnProfile:
        return ColumnProfile(name, dtype, n_unique, missing, role, reason, avg_len)

    if n_unique <= 1:
        return make("constant", "only one distinct value")
    if is_datetime64_any_dtype(s):
        return make("datetime", "stored as a date/time")
    if is_bool_dtype(s):
        return make("categorical", "true/false values")
    if is_numeric_dtype(s):
        if is_integer_dtype(s) and unique_ratio > ID_UNIQUE_RATIO and n_rows >= 30:
            consecutive = int(s.max()) - int(s.min()) + 1 == n_unique
            if consecutive or _ID_NAME.search(str(name)):
                return make("id", f"unique in {unique_ratio:.0%} of rows and {'counts up 1, 2, 3…' if consecutive else 'named like an ID'}")
        return make("numeric", "numbers")

    # Strings / objects / categoricals: inspect a sample to keep this fast.
    sample = s.dropna().head(SAMPLE_ROWS).astype(str)
    avg_len = float(sample.str.len().mean()) if not sample.empty else 0.0
    if _looks_like_dates(s):
        return make("datetime", "text that parses as dates")
    if avg_len > TEXT_MIN_AVG_LEN and sample.str.contains(" ").mean() > 0.5:
        return make("text", f"long free text (~{avg_len:.0f} characters)", avg_len)
    if unique_ratio > ID_UNIQUE_RATIO and n_rows >= 30:
        return make("id", f"unique in {unique_ratio:.0%} of rows")
    if n_unique > MAX_CATEGORIES:
        return make("high_card_categorical", f"{n_unique} different categories")
    return make("categorical", f"{n_unique} categories")


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    n_rows, n_cols = df.shape
    cols = tuple(_profile_column(str(c), df[c], n_rows) for c in df.columns)
    return DatasetProfile(
        n_rows=n_rows,
        n_cols=n_cols,
        missing_frac=float(df.isna().to_numpy().mean()) if df.size else 0.0,
        duplicate_rows=int(df.duplicated().sum()),
        columns=cols,
    )
