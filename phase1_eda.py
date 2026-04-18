"""
phase1_eda.py
-------------
Phase 1: Exploratory Data Analysis, Preprocessing, and Data Splitting.

What this script does
---------------------
1.  Loads the 3-class dataset (Normal / Depression / Suicidal).
2.  Computes basic text statistics (word count, character count, token length).
3.  Generates four EDA figures:
      - Fig 1 : Class distribution bar chart.
      - Fig 2 : Approximate token-length ECDF + bucket breakdown.
      - Fig 3 : Top informative word frequency heatmap per class.
4.  Applies two preprocessing pipelines:
      - Classical ML : lowercase → remove noise → stopwords → Snowball stem.
      - Transformer  : encoding fix + URL removal only.
5.  Creates a stratified 70 / 15 / 15 train / val / test split.
6.  Saves all figures and split arrays to the paths defined in config.py.

Usage
-----
    python phase1_eda.py

Outputs (written to OUT_DIR and SPLITS_DIR from config.py)
----------------------------------------------------------
    outputs/fig1_class_distribution.png
    outputs/fig2_token_length.png
    outputs/fig3_top_words_heatmap.png
    outputs/eda_summary.csv
    outputs/splits/X_train.npy   outputs/splits/X_val.npy   outputs/splits/X_test.npy
    outputs/splits/X_train_raw.npy  outputs/splits/X_val_raw.npy  outputs/splits/X_test_raw.npy
    outputs/splits/y_train.npy   outputs/splits/y_val.npy   outputs/splits/y_test.npy
"""

import os
import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split

from config import (
    DATA_PATH, OOD_PATH, OUT_DIR, SPLITS_DIR,
    LABEL_ORDER, COLORS, PALETTE,
    TEST_SIZE, VAL_RATIO, SEED,
)
from utils import set_seeds, preprocess_classical, preprocess_transformer

warnings.filterwarnings("ignore")
set_seeds(SEED)

os.makedirs(OUT_DIR,    exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)


# ── 1. Load data ──────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV, rename columns, filter to 3 target classes, and save OOD file.

    The full dataset contains more than three classes. Posts that do not belong
    to Normal, Depression, or Suicidal are extracted and saved as ood.csv for
    use in Phase 4 out-of-distribution robustness testing.
    """
    df_full = pd.read_csv(path)
    df_full = df_full.rename(columns={"statement": "text", "status": "label"})
    df_full = df_full.dropna(subset=["text"]).reset_index(drop=True)
    df_full["text"]  = df_full["text"].astype(str)
    df_full["label"] = df_full["label"].str.strip()

    # ── Save OOD file (all posts NOT in the 3 target classes) ────────────────
    df_ood = df_full[~df_full["label"].isin(LABEL_ORDER)].copy()
    df_ood = df_ood.rename(columns={"text": "text", "label": "status"})
    ood_path = OOD_PATH
    os.makedirs(os.path.dirname(ood_path) if os.path.dirname(ood_path) else ".", exist_ok=True)
    df_ood.to_csv(ood_path, index=False)
    print(f"\nOOD file saved: {ood_path}")
    print(f"  {len(df_ood):,} posts across {df_ood['status'].nunique()} OOD classes:")
    for cls, cnt in df_ood["status"].value_counts().items():
        print(f"    {cls:30s}: {cnt:,}")

    # ── Filter to 3 target classes ───────────────────────────────────────────
    df = df_full[df_full["label"].isin(LABEL_ORDER)].reset_index(drop=True)

    print(f"\nFull dataset : {df_full.shape[0]:,} rows")
    print(f"3-class subset: {df.shape[0]:,} rows")
    vc = df["label"].value_counts()
    for cls in LABEL_ORDER:
        print(f"  {cls:12s}: {vc[cls]:6,}  ({vc[cls]/len(df)*100:.1f}%)")
    return df


# ── 2. Feature computation ────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add word_count, char_count, avg_word_len, and approx_tokens columns."""
    df = df.copy()
    df["word_count"]   = df["text"].str.split().str.len()
    df["char_count"]   = df["text"].str.len()
    df["avg_word_len"] = df["char_count"] / df["word_count"].replace(0, np.nan)
    df["approx_tokens"] = (df["word_count"] * 1.3).astype(int)

    print("\nWord count statistics per class:")
    print(
        df.groupby("label")["word_count"]
          .describe()[["count", "mean", "50%", "std", "min", "max"]]
          .rename(columns={"50%": "median"})
          .astype({"count": int})
          .round(1)
          .loc[LABEL_ORDER]
    )
    df.groupby("label")["word_count"].describe().to_csv(
        os.path.join(OUT_DIR, "eda_summary.csv")
    )
    return df


# ── 3. EDA figures ────────────────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame) -> None:
    """Figure 1 — Class distribution bar chart."""
    vc     = df["label"].value_counts()
    counts = [vc[l] for l in LABEL_ORDER]
    pcts   = [c / sum(counts) * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(LABEL_ORDER, counts, color=PALETTE, edgecolor="white", linewidth=0.8)

    ax.set_ylabel("Number of Posts", fontsize=14, labelpad=14)
    ax.set_ylim(0, max(counts) * 1.20)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=14)

    for bar, cnt, pct in zip(bars, counts, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 150,
            f"{cnt:,} ({pct:.1f}%)",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig1_class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_token_length(df: pd.DataFrame) -> None:
    """Figure 2 — Token-length ECDF and bucket breakdown."""
    pct_over_512 = (df["approx_tokens"] > 512).mean() * 100
    print(f"\nPosts exceeding 512 tokens: {pct_over_512:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"wspace": 0.32})

    # Panel A — ECDF
    ax = axes[0]
    for lbl, col in zip(LABEL_ORDER, PALETTE):
        vals = np.sort(df.loc[df["label"] == lbl, "approx_tokens"].values)
        ecdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, ecdf, lw=2.3, color=col, label=lbl)

    for thresh, label in [(128, "128 tokens"), (256, "256 tokens"), (512, "512 tokens")]:
        ax.axvline(thresh, color="dimgray", linestyle=":", linewidth=1.6)
        ax.text(thresh + 20, 0.18, label, rotation=90,
                color="dimgray", fontsize=12, ha="center", va="bottom")

    ax.set_xlim(0, 700); ax.set_ylim(0, 1.01)
    ax.set_xlabel("Approximate Token Count", fontsize=14, labelpad=12)
    ax.set_ylabel("Cumulative Proportion",   fontsize=14, labelpad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(frameon=False, fontsize=12, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=3)

    # Panel B — stacked bucket bars
    ax = axes[1]
    bins_labels = ["≤128", "129–256", "257–512", ">512"]
    bin_colors  = ["#A8DADC", "#457B9D", "#E76F51", "#6C757D"]

    def bin_percents(series):
        t = series.values
        return np.array([
            (t <= 128).sum(),
            ((t > 128) & (t <= 256)).sum(),
            ((t > 256) & (t <= 512)).sum(),
            (t > 512).sum()
        ]) / len(t) * 100

    all_bins = np.stack([
        bin_percents(df.loc[df["label"] == l, "approx_tokens"])
        for l in LABEL_ORDER
    ])

    x = np.arange(len(LABEL_ORDER)); bottom = np.zeros(len(LABEL_ORDER))
    for i, (lab, col) in enumerate(zip(bins_labels, bin_colors)):
        vals = all_bins[:, i]
        ax.bar(x, vals, width=0.58, bottom=bottom,
               color=col, edgecolor="white", linewidth=0.8, label=lab)
        for xi, v, btm in zip(x, vals, bottom):
            if v >= 6:
                ax.text(xi, btm + v / 2, f"{v:.0f}%", ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="white" if i >= 1 else "black")
        bottom += vals

    ax.set_xticks(x); ax.set_xticklabels(LABEL_ORDER, fontsize=10)
    ax.set_ylabel("Percentage of Posts", fontsize=14)
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.30)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(frameon=False, fontsize=12, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=4)

    plt.tight_layout(rect=[0, 0.10, 1, 0.95])
    path = os.path.join(OUT_DIR, "fig2_token_length.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_top_words_heatmap(df: pd.DataFrame) -> None:
    """Figure 3 — Top informative word frequency heatmap per class."""
    CUSTOM_SW = ENGLISH_STOP_WORDS.union({
        "just", "like", "want", "really", "know", "time", "day", "today",
        "did", "don", "got", "make", "going", "think", "people", "life",
        "things", "thing", "way", "ve", "ll", "im", "doesn", "didn",
        "isn", "wasn", "aren", "weren", "can", "could", "would", "should",
        "say", "said", "feel", "feeling",
    })

    def quick_clean(text):
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\d+", " ", text)
        return [t for t in text.split() if t not in CUSTOM_SW and len(t) > 2]

    TOP_N = 12
    class_counters = {}
    for lbl in LABEL_ORDER:
        tokens = []
        for text in df.loc[df["label"] == lbl, "text"]:
            tokens.extend(quick_clean(text))
        class_counters[lbl] = Counter(tokens)

    top_words = set()
    for lbl in LABEL_ORDER:
        top_words.update([w for w, _ in class_counters[lbl].most_common(TOP_N)])

    mat = pd.DataFrame(index=sorted(top_words), columns=LABEL_ORDER).fillna(0)
    for lbl in LABEL_ORDER:
        for w in top_words:
            mat.loc[w, lbl] = class_counters[lbl][w]

    mat = mat.astype(int)   # fix: ensure numeric dtype for seaborn heatmap
    mat["total"] = mat.sum(axis=1)
    mat = mat.sort_values("total", ascending=False).drop(columns="total").head(20)

    highlight = {"depression": "#F4A261", "anxiety": "#F4A261", "help": "#F4A261",
                 "die": "#E76F51", "kill": "#E76F51", "end": "#E76F51", "anymore": "#E76F51"}

    fig, ax = plt.subplots(figsize=(13, 9))
    hm = sns.heatmap(mat, annot=True, fmt="d", cmap="Blues",
                     linewidths=0.5, linecolor="white",
                     cbar_kws={"label": "Frequency"},
                     annot_kws={"fontsize": 12}, ax=ax)
    ax.set_xlabel("Class", fontsize=16, labelpad=14)
    ax.set_ylabel("Word",  fontsize=16)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label("Frequency", fontsize=15)
    for tick in ax.get_yticklabels():
        if tick.get_text() in highlight:
            tick.set_bbox(dict(facecolor="none", edgecolor=highlight[tick.get_text()],
                               linewidth=2, boxstyle="round,pad=0.22"))

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig3_top_words_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── 4. Preprocessing ──────────────────────────────────────────────────────────

def apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Apply both preprocessing pipelines and return augmented DataFrame."""
    import time
    print("\nApplying classical ML pipeline …")
    t0 = time.time()
    df["text_clean"] = df["text"].apply(preprocess_classical)
    print(f"  Done in {time.time() - t0:.1f}s")

    print("Applying transformer pipeline …")
    t0 = time.time()
    df["text_transformer"] = df["text"].apply(preprocess_transformer)
    print(f"  Done in {time.time() - t0:.1f}s")
    return df


# ── 5. Train / Val / Test split ───────────────────────────────────────────────

def split_and_save(df: pd.DataFrame) -> None:
    """
    Create a stratified 70/15/15 train/val/test split and save all arrays.

    Classical-preprocessed text → X_train / X_val / X_test.
    Transformer-preprocessed text → X_train_raw / X_val_raw / X_test_raw.
    Labels → y_train / y_val / y_test.
    """
    X_clean = np.array(df["text_clean"].tolist())
    X_raw   = np.array(df["text_transformer"].tolist())
    y       = np.array(df["label"].tolist())

    # First split: 85% train+val, 15% test
    X_c_tv, X_c_test, X_r_tv, X_r_test, y_tv, y_test = train_test_split(
        X_clean, X_raw, y,
        test_size=TEST_SIZE, stratify=y, random_state=SEED,
    )
    # Second split: ~15% of full → val
    X_c_train, X_c_val, X_r_train, X_r_val, y_train, y_val = train_test_split(
        X_c_tv, X_r_tv, y_tv,
        test_size=VAL_RATIO, stratify=y_tv, random_state=SEED,
    )

    print(f"\nSplit sizes:")
    print(f"  Train : {len(y_train):,} | Val : {len(y_val):,} | Test : {len(y_test):,}")
    for split_name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        counts = {l: (y_split == l).sum() for l in LABEL_ORDER}
        print("  " + split_name + ": " + "  |  ".join(
            f"{l} {counts[l]:,} ({counts[l]/len(y_split)*100:.1f}%)"
            for l in LABEL_ORDER
        ))

    splits = {
        "X_train": X_c_train, "X_val": X_c_val,   "X_test": X_c_test,
        "X_train_raw": X_r_train, "X_val_raw": X_r_val, "X_test_raw": X_r_test,
        "y_train": y_train,   "y_val": y_val,     "y_test": y_test,
    }
    for name, arr in splits.items():
        path = os.path.join(SPLITS_DIR, f"{name}.npy")
        np.save(path, arr)
    print(f"\nSplits saved to: {SPLITS_DIR}/")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    df = compute_features(df)

    print("\nGenerating EDA figures …")
    plot_class_distribution(df)
    plot_token_length(df)
    plot_top_words_heatmap(df)

    df = apply_preprocessing(df)
    split_and_save(df)

    print("\nPhase 1 complete.")
