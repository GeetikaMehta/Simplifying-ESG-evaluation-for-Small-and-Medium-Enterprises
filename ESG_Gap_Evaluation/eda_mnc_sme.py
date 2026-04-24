"""
EDA: large-cap / MNC proxy vs small-company proxy — ESG gap and score comparisons.

This project CSV has no market-cap column. Default rule (transparent, research caveat):
  - "Large / MNC (proxy)": listings on NYSE (exchange contains NEW YORK STOCK EXCHANGE).
  - "Small / other (proxy)": all other exchanges (mostly NASDAQ, OTC, etc.).

Override with a two-column CSV: ticker,segment  (segment = MNC or SME).

Inputs:  data.csv + esg_model_output.csv (for esg_gap). Run ESG.py first if outputs are missing.

Outputs: eda_plots/mnc_vs_sme/*.png, eda_mnc_sme_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data.csv"
MODEL_OUT_PATH = ROOT / "esg_model_output.csv"
PLOTS_DIR = ROOT / "eda_plots" / "mnc_vs_sme"
SUMMARY_JSON = ROOT / "eda_mnc_sme_summary.json"

PILLARS = ("environment_score", "social_score", "governance_score")
SEG_MNC = "Large / MNC (proxy)"
SEG_SME = "Small / other (proxy)"


def is_nyse(exchange: str) -> bool:
    if pd.isna(exchange):
        return False
    s = str(exchange).upper()
    return "NEW YORK STOCK EXCHANGE" in s or s.strip() == "NYSE"


def load_merged() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing {DATA_PATH}")
    if not MODEL_OUT_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {MODEL_OUT_PATH}. Run ESG.py first to generate model outputs."
        )
    raw = pd.read_csv(DATA_PATH)
    out = pd.read_csv(MODEL_OUT_PATH)
    merge_cols = ["ticker", "esg_gap", "benchmark"]
    for c in ("true_label", "predicted_label", "p_safe"):
        if c in out.columns:
            merge_cols.append(c)
    use = [c for c in merge_cols if c in out.columns]
    return raw.merge(out[use], on="ticker", how="inner")


def assign_segments(
    df: pd.DataFrame, segments_path: Path | None
) -> pd.DataFrame:
    d = df.copy()
    if segments_path and segments_path.is_file():
        seg = pd.read_csv(segments_path)
        tcol = next(c for c in seg.columns if c.lower() == "ticker")
        scol = next(c for c in seg.columns if c.lower() in ("segment", "group", "size"))
        seg = seg[[tcol, scol]].rename(columns={tcol: "ticker", scol: "seg_raw"})

        def norm_seg(x) -> str | float:
            u = str(x).upper().strip()
            if u in ("MNC", "LARGE", "BIG"):
                return SEG_MNC
            if u in ("SME", "SMALL", "SMALLCAP"):
                return SEG_SME
            return np.nan

        seg["segment"] = seg["seg_raw"].map(norm_seg)
        seg = seg.drop_duplicates(subset=["ticker"], keep="first")
        d = d.merge(seg[["ticker", "segment"]], on="ticker", how="left")
    if "segment" not in d.columns:
        d["segment"] = np.nan
    fallback = pd.Series(
        np.where(d["exchange"].map(is_nyse), SEG_MNC, SEG_SME),
        index=d.index,
    )
    d["segment"] = d["segment"].fillna(fallback)
    return d


def plot_gap_box_violin(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    order = [SEG_MNC, SEG_SME]
    sns.boxplot(
        data=df,
        x="segment",
        y="esg_gap",
        order=order,
        ax=axes[0],
        hue="segment",
        hue_order=order,
        palette="Set2",
        legend=False,
    )
    axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("ESG gap (score − benchmark) by segment")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=12)

    sns.violinplot(
        data=df,
        x="segment",
        y="esg_gap",
        order=order,
        ax=axes[1],
        hue="segment",
        hue_order=order,
        palette="pastel",
        legend=False,
    )
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title("Gap distribution (violin)")
    axes[1].set_xlabel("")
    fig.suptitle("ESG gap: large-listing proxy vs other listings", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_gap_box_violin.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_total_and_pillars(df: pd.DataFrame) -> None:
    order = [SEG_MNC, SEG_SME]
    cols_titles = [
        ("total_score", "Total ESG score"),
        ("environment_score", "Environment"),
        ("social_score", "Social"),
        ("governance_score", "Governance"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (col, title) in zip(axes.ravel(), cols_titles):
        sns.boxplot(
            data=df,
            x="segment",
            y=col,
            order=order,
            ax=ax,
            hue="segment",
            hue_order=order,
            palette="Set2",
            legend=False,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=10)
    fig.suptitle("ESG scores by segment", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_scores_by_segment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_kde_gap(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for seg, color in [(SEG_MNC, "#2980b9"), (SEG_SME, "#c0392b")]:
        sub = df.loc[df["segment"] == seg, "esg_gap"].dropna()
        if len(sub) > 2:
            sns.kdeplot(sub, ax=ax, label=seg, color=color, fill=True, alpha=0.25)
    ax.axvline(0, color="black", linestyle="--")
    ax.set_title("KDE of ESG gap by segment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_kde_esg_gap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mean_ci_bar(df: pd.DataFrame) -> None:
    order = [SEG_MNC, SEG_SME]
    stats_rows = []
    for seg in order:
        x = df.loc[df["segment"] == seg, "esg_gap"].dropna().values
        stats_rows.append(
            {
                "segment": seg,
                "mean": float(np.mean(x)),
                "sem": float(stats.sem(x)) if len(x) > 1 else 0.0,
                "n": int(len(x)),
            }
        )
    s = pd.DataFrame(stats_rows)
    fig, ax = plt.subplots(figsize=(6, 5))
    xpos = np.arange(len(s))
    ax.bar(xpos, s["mean"], yerr=1.96 * s["sem"], capsize=6, color=["#2980b9", "#c0392b"], alpha=0.85)
    ax.set_xticks(xpos)
    ax.set_xticklabels([x.replace(" (proxy)", "") for x in s["segment"]], rotation=10)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Mean ESG gap (± 95% CI approx.)")
    ax.set_title("Average gap vs benchmark by segment")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_mean_gap_ci.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_tests(df: pd.DataFrame) -> dict:
    g0 = df.loc[df["segment"] == SEG_MNC, "esg_gap"].dropna().values
    g1 = df.loc[df["segment"] == SEG_SME, "esg_gap"].dropna().values
    out: dict = {
        "n_mnc_proxy": int(len(g0)),
        "n_sme_proxy": int(len(g1)),
        "gap_mean_mnc_proxy": float(np.mean(g0)) if len(g0) else None,
        "gap_mean_sme_proxy": float(np.mean(g1)) if len(g1) else None,
        "gap_median_mnc_proxy": float(np.median(g0)) if len(g0) else None,
        "gap_median_sme_proxy": float(np.median(g1)) if len(g1) else None,
    }
    if len(g0) > 2 and len(g1) > 2:
        u_stat, p_mw = stats.mannwhitneyu(g0, g1, alternative="two-sided")
        t_stat, p_t = stats.ttest_ind(g0, g1, equal_var=False)
        out["mannwhitney_gap_U"] = float(u_stat)
        out["mannwhitney_gap_pvalue"] = float(p_mw)
        out["welch_ttest_gap_pvalue"] = float(p_t)
    return out


def run(segments_path: Path | None = None) -> dict:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    df = load_merged()
    df = assign_segments(df, segments_path)
    plot_gap_box_violin(df)
    plot_total_and_pillars(df)
    plot_kde_gap(df)
    plot_mean_ci_bar(df)
    tests = compute_tests(df)
    summary = {
        "method_note": (
            "Default segments use NYSE vs non-NYSE as a listing-size proxy, not true revenue-based SME/MNC. "
            "Provide --segments CSV with ticker + segment (MNC/SME) for your own split."
        ),
        "segments_file_used": str(segments_path) if segments_path else None,
        "group_counts": df["segment"].value_counts().to_dict(),
        "tests_on_esg_gap": tests,
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="MNC vs SME proxy EDA (ESG gap)")
    p.add_argument(
        "--segments",
        type=Path,
        default=None,
        help="CSV with columns ticker and segment (MNC or SME) to override exchange rule",
    )
    args = p.parse_args()
    summary = run(args.segments)
    print(f"Saved plots under {PLOTS_DIR}")
    print(f"Saved {SUMMARY_JSON}")
    print(summary["method_note"])
    print("Group sizes:", summary["group_counts"])


if __name__ == "__main__":
    main()
