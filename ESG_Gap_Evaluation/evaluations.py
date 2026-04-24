"""
Additional evaluations for the ESG study: CV, ROC/PR, confusion matrix, PCA,
clustering quality, and non-parametric tests across industries.

Run: python evaluations.py
Outputs: evaluation_plots/*.png, evaluation_results.json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ESG import (
    DATA_PATH,
    FEATURE_COLS,
    RISK,
    SAFE,
    SCORE_COL,
    RF_MAX_DEPTH,
    RF_N_ESTIMATORS,
    RANDOM_STATE,
    TEST_SIZE,
    assign_labels,
    compute_benchmark,
    load_data,
)

ROOT = Path(__file__).resolve().parent
PLOTS_DIR = ROOT / "evaluation_plots"
RESULTS_JSON = ROOT / "evaluation_results.json"

CV_SPLITS = 5
KMEANS_K_MAX = 8


def prepare_xy(df: pd.DataFrame):
    """Same labeling as ESG.run_pipeline (benchmark = train median)."""
    n = len(df)
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    train_scores = df.loc[train_mask, SCORE_COL]
    benchmark = compute_benchmark(train_scores, "median")
    threshold = benchmark
    y = assign_labels(df[SCORE_COL], threshold)
    X = df[FEATURE_COLS].values
    return X, y, train_idx, test_idx, benchmark, threshold


def train_rf():
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )


def classification_diagnostics(X, y, train_idx, test_idx, clf) -> dict:
    X_train, X_test = X[train_idx], X[test_idx]
    y_train = y.iloc[train_idx].values
    y_test = y.iloc[test_idx].values
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    classes = list(clf.classes_)
    if SAFE not in classes:
        return {"error": "Only one class in training data"}
    safe_i = classes.index(SAFE)
    p_safe = clf.predict_proba(X_test)[:, safe_i]
    y_bin = (y_test == SAFE).astype(int)

    roc_auc = float(roc_auc_score(y_bin, p_safe))
    ap = float(average_precision_score(y_bin, p_safe))
    mcc = float(matthews_corrcoef(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_bin, p_safe, ax=ax, name="Safe vs Risk")
    ax.set_title("ROC curve (positive class: Safe)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    prec, rec, _ = precision_recall_curve(y_bin, p_safe)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color="#2980b9", linewidth=2)
    ax.set_xlabel("Recall (Safe)")
    ax.set_ylabel("Precision (Safe)")
    ax.set_title(f"Precision–Recall curve (AP = {ap:.3f})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_precision_recall.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    cm = confusion_matrix(y_test, y_pred, labels=[RISK, SAFE])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[RISK, SAFE])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    imp = clf.feature_importances_
    ax.barh(FEATURE_COLS, imp, color=["#27ae60", "#2980b9", "#8e44ad"])
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest feature importances")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "test_roc_auc": roc_auc,
        "test_average_precision_safe": ap,
        "test_matthews_corrcoef": mcc,
        "test_balanced_accuracy": bal_acc,
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision_safe": float(
            precision_score(y_test, y_pred, pos_label=SAFE, zero_division=0)
        ),
        "test_recall_safe": float(recall_score(y_test, y_pred, pos_label=SAFE, zero_division=0)),
        "test_f1_safe": float(f1_score(y_test, y_pred, pos_label=SAFE, zero_division=0)),
        "classification_report_test": classification_report(y_test, y_pred, zero_division=0),
    }


def cross_validation_report(X, y_text: pd.Series) -> dict:
    # Binary 0/1 for sklearn metrics that require numeric targets (e.g. roc_auc)
    y_bin = (y_text == SAFE).astype(int).values
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    clf = train_rf()
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "roc_auc": "roc_auc",
        "balanced_accuracy": "balanced_accuracy",
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = cross_validate(
            clf,
            X,
            y_bin,
            cv=skf,
            scoring=scoring,
            n_jobs=-1,
        )
    return {
        f"cv_{k}_mean": float(np.mean(cv[f"test_{k}"]))
        for k in scoring
    } | {
        f"cv_{k}_std": float(np.std(cv[f"test_{k}"]))
        for k in scoring
    }


def pca_plot(X, y: pd.Series) -> dict:
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=2, random_state=RANDOM_STATE)),
        ]
    )
    Z = pipe.fit_transform(X)
    pca = pipe.named_steps["pca"]
    ev = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, color in [(SAFE, "#27ae60"), (RISK, "#c0392b")]:
        m = y.values == label
        ax.scatter(Z[m, 0], Z[m, 1], alpha=0.45, s=28, label=label, c=color)
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var.)")
    ax.set_title("PCA of E/S/G pillar scores (scaled)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "05_pca_safe_risk.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {
        "pca_explained_variance_ratio_pc1": float(ev[0]),
        "pca_explained_variance_ratio_pc2": float(ev[1]),
    }


def clustering_silhouette(X) -> dict:
    from sklearn.metrics import silhouette_score

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    ks = range(2, KMEANS_K_MAX + 1)
    scores = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(Xs)
        scores.append(float(silhouette_score(Xs, labels)))
    best_k = int(ks[int(np.argmax(scores))])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(ks), scores, marker="o", color="#8e44ad")
    ax.set_xlabel("k (clusters)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("K-means silhouette vs k (E/S/G scaled)")
    ax.axvline(best_k, color="gray", linestyle="--", label=f"best k={best_k}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "06_kmeans_silhouette.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = km.fit_predict(Xs)
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(Xs[:, 0], Xs[:, 1], c=cluster_labels, cmap="tab10", alpha=0.6, s=25)
    ax.set_xlabel("Environment (scaled)")
    ax.set_ylabel("Social (scaled)")
    ax.set_title(f"K-means clusters (k={best_k}) in first two scaled dimensions")
    fig.colorbar(scatter, ax=ax, label="cluster")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "07_kmeans_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "kmeans_silhouette_by_k": {int(k): s for k, s in zip(ks, scores)},
        "kmeans_best_k": best_k,
    }


def industry_kruskal(df: pd.DataFrame) -> dict | None:
    col = "industry"
    if col not in df.columns:
        return None
    top = df[col].value_counts().head(5).index
    groups = [df.loc[df[col] == ind, SCORE_COL].dropna().values for ind in top]
    if any(len(g) < 3 for g in groups):
        return {"note": "Too few rows per top industry for Kruskal–Wallis"}
    h_stat, p_val = stats.kruskal(*groups)
    return {
        "kruskal_wallis_top5_industries_total_score": {
            "H_statistic": float(h_stat),
            "p_value": float(p_val),
            "industries": [str(x)[:50] for x in top.tolist()],
        }
    }


def pillar_t_test_safe_vs_risk(df: pd.DataFrame, y: pd.Series) -> dict:
    """Compare pillar means between Safe and Risk (Welch t-test; large n)."""
    out = {}
    for pillar in FEATURE_COLS:
        a = df.loc[y == SAFE, pillar].dropna()
        b = df.loc[y == RISK, pillar].dropna()
        t_stat, p_two = stats.ttest_ind(a, b, equal_var=False)
        out[pillar] = {
            "mean_safe": float(a.mean()),
            "mean_risk": float(b.mean()),
            "welch_t_stat": float(t_stat),
            "p_value_two_sided": float(p_two),
        }
    return {"pillars_safe_vs_risk_welch_ttest": out}


def run_all():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    df = load_data(DATA_PATH)
    X, y, train_idx, test_idx, benchmark, threshold = prepare_xy(df)
    clf = train_rf()

    results: dict = {
        "benchmark": benchmark,
        "classification_threshold": threshold,
        "classification_diagnostics": classification_diagnostics(
            X, y, train_idx, test_idx, clf
        ),
        "stratified_kfold_cv": cross_validation_report(X, y),
        "pca": pca_plot(X, y),
        "clustering": clustering_silhouette(X),
        "pillars_tests": pillar_t_test_safe_vs_risk(df, y),
    }
    ind_k = industry_kruskal(df)
    if ind_k:
        results["industry_tests"] = ind_k

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    results = run_all()
    print(f"Saved plots under {PLOTS_DIR}")
    print(f"Saved {RESULTS_JSON}")
    cd = results["classification_diagnostics"]
    if "test_roc_auc" in cd:
        print(f"Test ROC-AUC: {cd['test_roc_auc']:.4f}")
        print(f"Test average precision (Safe): {cd['test_average_precision_safe']:.4f}")
    cv = results["stratified_kfold_cv"]
    print(f"CV accuracy: {cv['cv_accuracy_mean']:.4f} ± {cv['cv_accuracy_std']:.4f}")


if __name__ == "__main__":
    main()
