"""
ESG gap evaluation and Random Forest classification on structured E/S/G scores.

Loads data.csv, computes overall ESG score (from dataset), derives Safe/Risk labels
from a threshold vs a benchmark, trains a Random Forest classifier on pillar scores,
and writes scores, labels, gaps, and metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# Paths (same folder as this script)
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data.csv"
OUTPUT_CSV = ROOT / "esg_model_output.csv"
METRICS_JSON = ROOT / "esg_model_metrics.json"

# Feature columns (structured ESG inputs)
FEATURE_COLS = ["environment_score", "social_score", "governance_score"]
SCORE_COL = "total_score"
ID_COLS = ["ticker", "name"]

# Model
TEST_SIZE = 0.2
RANDOM_STATE = 42
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = None

# Labels
SAFE = "Safe"
RISK = "Risk"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLS + [SCORE_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df = df.dropna(subset=FEATURE_COLS + [SCORE_COL])
    return df


def compute_benchmark(train_scores: pd.Series, method: str = "median") -> float:
    """Benchmark = reference sustainability level (e.g. median of training totals)."""
    if method == "median":
        return float(train_scores.median())
    if method == "mean":
        return float(train_scores.mean())
    raise ValueError("method must be 'median' or 'mean'")


def assign_labels(scores: pd.Series, threshold: float) -> pd.Series:
    """Safe if score >= threshold, else Risk."""
    labels = np.where(scores >= threshold, SAFE, RISK)
    return pd.Series(labels, index=scores.index)


def run_pipeline(
    data_path: Path = DATA_PATH,
    benchmark_method: str = "median",
    threshold_mode: str = "benchmark",
) -> tuple[pd.DataFrame, dict]:
    """
    threshold_mode:
      - 'benchmark': classification threshold equals benchmark (gap sign aligns with label).
      - 'median': threshold = median(train total_score) independently (can match benchmark if median).
    """
    df = load_data(data_path)
    n = len(df)
    if n < 10:
        raise ValueError("Not enough rows after dropping missing values.")

    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True

    train_scores = df.loc[train_mask, SCORE_COL]
    benchmark = compute_benchmark(train_scores, benchmark_method)

    if threshold_mode == "benchmark":
        threshold = benchmark
    elif threshold_mode == "median":
        threshold = float(train_scores.median())
    else:
        raise ValueError("threshold_mode must be 'benchmark' or 'median'")

    y = assign_labels(df[SCORE_COL], threshold)
    X = df[FEATURE_COLS].values

    X_train, X_test = X[train_idx], X[test_idx]
    y_train = y.iloc[train_idx].values
    y_test = y.iloc[test_idx].values

    clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X)
    proba = clf.predict_proba(X)
    classes = list(clf.classes_)
    # column for P(Safe) if both classes exist
    if SAFE in classes:
        safe_i = classes.index(SAFE)
        p_safe = proba[:, safe_i]
    else:
        p_safe = np.ones(len(df))

    gap = df[SCORE_COL].astype(float) - benchmark

    out = df[[c for c in ID_COLS if c in df.columns]].copy()
    out["environment_score"] = df["environment_score"]
    out["social_score"] = df["social_score"]
    out["governance_score"] = df["governance_score"]
    out["esg_score"] = df[SCORE_COL]
    out["benchmark"] = benchmark
    out["esg_gap"] = gap
    out["true_label"] = y.to_numpy()
    out["predicted_label"] = y_pred
    out["p_safe"] = p_safe

    y_test_pred = clf.predict(X_test)
    metrics = {
        "n_samples": int(n),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "benchmark_method": benchmark_method,
        "benchmark_value": benchmark,
        "classification_threshold": threshold,
        "threshold_mode": threshold_mode,
        "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
        "test_precision_safe": float(
            precision_score(
                y_test, y_test_pred, pos_label=SAFE, average="binary", zero_division=0
            )
        ),
        "test_recall_safe": float(
            recall_score(
                y_test, y_test_pred, pos_label=SAFE, average="binary", zero_division=0
            )
        ),
        "test_f1_safe": float(
            f1_score(y_test, y_test_pred, pos_label=SAFE, average="binary", zero_division=0)
        ),
        "test_f1_macro": float(f1_score(y_test, y_test_pred, average="macro", zero_division=0)),
        "feature_importances": {
            FEATURE_COLS[i]: float(clf.feature_importances_[i])
            for i in range(len(FEATURE_COLS))
        },
        "classification_report_test": classification_report(
            y_test, y_test_pred, zero_division=0
        ),
    }

    return out, metrics


def main() -> None:
    results, metrics = run_pipeline()
    results.to_csv(OUTPUT_CSV, index=False)
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {METRICS_JSON}")
    print(metrics["classification_report_test"])


if __name__ == "__main__":
    main()
