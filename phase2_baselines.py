"""
phase2_baselines.py
-------------------
Phase 2: Baseline Models (9 models benchmarked against WellBERT).

Models trained and evaluated
-----------------------------
1.  VADER               — rule-based sentiment baseline (no training)
2.  LR (BoW)            — logistic regression on bag-of-words features
3.  LR (TF-IDF)         — logistic regression on TF-IDF features
4.  LinearSVM (TF-IDF)  — linear SVM on TF-IDF features
5.  Naive Bayes (TF-IDF)— multinomial Naive Bayes on TF-IDF features
6.  LSA + LR            — logistic regression on 300-dim LSA embeddings
7.  Decision Tree       — decision tree on TF-IDF features
8.  Random Forest       — random forest on TF-IDF features
9.  Zero-Shot LLM       — BART-large-MNLI (run separately, GPU required)

Note: The Zero-Shot LLM (model 9) is implemented in phase2_llm.py because
      it requires a GPU runtime and a separate session in Google Colab.

Usage
-----
    python phase2_baselines.py

Outputs
-------
    outputs/splits/preds_vader.npy
    outputs/splits/preds_lr_bow.npy
    outputs/splits/preds_lr_tfidf.npy
    outputs/splits/preds_svm.npy
    outputs/splits/preds_nb.npy
    outputs/splits/preds_dt.npy
    outputs/splits/preds_rf.npy
    outputs/splits/preds_lsa.npy
    outputs/fig4_confusion_matrices.png
    outputs/fig5_f1_comparison.png
    outputs/fig6_dep_sui_confusion.png
    outputs/phase2_results_summary.csv
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from config import LABEL_ORDER, COLORS, PALETTE, OUT_DIR, SPLITS_DIR, SEED
from utils import set_seeds, evaluate, ALL_RESULTS, ALL_CMS

warnings.filterwarnings("ignore")
set_seeds(SEED)


# ── Load saved splits ─────────────────────────────────────────────────────────

def load_splits():
    """Load all train/val/test splits from SPLITS_DIR."""
    def npy(name): return np.load(os.path.join(SPLITS_DIR, f"{name}.npy"),
                                  allow_pickle=True)
    X_train     = npy("X_train")
    X_val       = npy("X_val")
    X_test      = npy("X_test")
    X_val_raw   = npy("X_val_raw")
    X_test_raw  = npy("X_test_raw")
    y_train     = npy("y_train")
    y_val       = npy("y_val")
    y_test      = npy("y_test")
    print(f"Splits loaded  Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}")
    return X_train, X_val, X_test, X_val_raw, X_test_raw, y_train, y_val, y_test


# ── Model 1: VADER ────────────────────────────────────────────────────────────

def run_vader(X_val_raw, X_test_raw, y_val, y_test):
    """
    Rule-based VADER sentiment classifier.
    Threshold combination is tuned on the validation set via grid search.
    """
    try:
        from vader_sentiment.vader_sentiment import SentimentIntensityAnalyzer
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "vader-sentiment", "-q"])
        from vader_sentiment.vader_sentiment import SentimentIntensityAnalyzer

    print("=" * 55)
    print("  Model 1: VADER Sentiment Baseline")
    print("=" * 55)

    sia = SentimentIntensityAnalyzer()

    # Grid-search thresholds on validation set
    best_f1, best_t = -1, (-0.60, -0.15)
    for t_sui in [-0.75, -0.65, -0.55, -0.45]:
        for t_dep in [-0.25, -0.15, -0.05]:
            preds_v = []
            for text in X_val_raw:
                s = sia.polarity_scores(str(text))
                if s["compound"] <= t_sui and s["neg"] >= 0.25:
                    preds_v.append("Suicidal")
                elif s["compound"] <= t_dep:
                    preds_v.append("Depression")
                else:
                    preds_v.append("Normal")
            f1 = f1_score(y_val, preds_v, average="macro")
            if f1 > best_f1:
                best_f1, best_t = f1, (t_sui, t_dep)

    t_sui, t_dep = best_t
    print(f"Best thresholds (val): suicidal ≤ {t_sui}, depression ≤ {t_dep}")
    print(f"Val macro F1: {best_f1:.3f}")

    preds_vader = []
    for text in X_test_raw:
        s = sia.polarity_scores(str(text))
        if s["compound"] <= t_sui and s["neg"] >= 0.25:
            preds_vader.append("Suicidal")
        elif s["compound"] <= t_dep:
            preds_vader.append("Depression")
        else:
            preds_vader.append("Normal")

    preds_vader = np.array(preds_vader)
    evaluate("VADER", y_test, preds_vader)
    return preds_vader


# ── Models 2–8: Supervised Classical ─────────────────────────────────────────

def run_classical_models(X_train, X_test, y_train, y_test):
    """Train and evaluate all 7 supervised classical ML models."""
    results = {}

    # ── LR (BoW)
    print("=" * 55)
    print("  Model 2: Logistic Regression — Bag-of-Words")
    print("=" * 55)
    pipe_lr_bow = Pipeline([
        ("vec", CountVectorizer(max_features=50000, ngram_range=(1, 2),
                                min_df=2, max_df=0.95)),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                                   solver="saga", n_jobs=-1, random_state=SEED)),
    ])
    t0 = time.time()
    pipe_lr_bow.fit(X_train, y_train)
    print(f"Train time: {time.time() - t0:.1f}s")
    preds = pipe_lr_bow.predict(X_test)
    evaluate("LR (BoW)", y_test, preds)
    results["preds_lr_bow"] = preds

    # ── LR (TF-IDF)
    print("=" * 55)
    print("  Model 3: Logistic Regression — TF-IDF")
    print("=" * 55)
    pipe_lr_tfidf = Pipeline([
        ("vec", TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                                min_df=2, max_df=0.95, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                                   solver="saga", n_jobs=-1, random_state=SEED)),
    ])
    t0 = time.time()
    pipe_lr_tfidf.fit(X_train, y_train)
    print(f"Train time: {time.time() - t0:.1f}s")
    preds = pipe_lr_tfidf.predict(X_test)
    evaluate("LR (TF-IDF)", y_test, preds)
    results["preds_lr_tfidf"] = preds

    # ── LinearSVM (TF-IDF)
    print("=" * 55)
    print("  Model 4: Linear SVM — TF-IDF")
    print("=" * 55)
    pipe_svm = Pipeline([
        ("vec", TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                                min_df=2, max_df=0.95, sublinear_tf=True)),
        ("clf", LinearSVC(C=0.5, max_iter=2000, class_weight="balanced",
                          random_state=SEED)),
    ])
    t0 = time.time()
    pipe_svm.fit(X_train, y_train)
    print(f"Train time: {time.time() - t0:.1f}s")
    preds = pipe_svm.predict(X_test)
    evaluate("LinearSVM (TF-IDF)", y_test, preds)
    results["preds_svm"] = preds

    # ── Naive Bayes (TF-IDF)
    print("=" * 55)
    print("  Model 5: Multinomial Naive Bayes — TF-IDF")
    print("=" * 55)
    pipe_nb = Pipeline([
        ("vec", TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                                min_df=2, max_df=0.95)),
        ("clf", MultinomialNB(alpha=0.1)),
    ])
    t0 = time.time()
    pipe_nb.fit(X_train, y_train)
    print(f"Train time: {time.time() - t0:.1f}s")
    preds = pipe_nb.predict(X_test)
    evaluate("Naive Bayes (TF-IDF)", y_test, preds)
    results["preds_nb"] = preds

    # ── LSA + LR
    print("=" * 55)
    print("  Model 6: Dense Embeddings (LSA 300d) + LR")
    print("=" * 55)
    pipe_lsa = Pipeline([
        ("vec",  TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                                 min_df=2, max_df=0.95, sublinear_tf=True)),
        ("svd",  TruncatedSVD(n_components=300, random_state=SEED)),
        ("norm", Normalizer(copy=False)),
        ("clf",  LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                                    solver="saga", random_state=SEED)),
    ])
    t0 = time.time()
    pipe_lsa.fit(X_train, y_train)
    print(f"Train time: {time.time() - t0:.1f}s")
    preds = pipe_lsa.predict(X_test)
    evaluate("Embeddings+LR (LSA)", y_test, preds)
    results["preds_lsa"] = preds

    # ── Decision Tree (TF-IDF)
    print("=" * 55)
    print("  Model 7: Decision Tree — TF-IDF")
    print("=" * 55)
    pipe_dt = Pipeline([
        ("vec", TfidfVectorizer(max_features=30000, ngram_range=(1, 2),
                                min_df=2, max_df=0.95, sublinear_tf=True)),
        ("clf", DecisionTreeClassifier(max_depth=50, class_weight="balanced",
                                       random_state=SEED)),
    ])
    t0 = time.time()
    pipe_dt.fit(X_train, y_train)
    print(f"Train time: {time.time() - t0:.1f}s")
    preds = pipe_dt.predict(X_test)
    evaluate("Decision Tree (TF-IDF)", y_test, preds)
    results["preds_dt"] = preds

    # ── Random Forest (TF-IDF)
    print("=" * 55)
    print("  Model 8: Random Forest — TF-IDF")
    print("=" * 55)
    pipe_rf = Pipeline([
        ("vec", TfidfVectorizer(max_features=30000, ngram_range=(1, 2),
                                min_df=2, max_df=0.95, sublinear_tf=True)),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=50,
                                       max_features="sqrt",
                                       class_weight="balanced",
                                       n_jobs=-1, random_state=SEED)),
    ])
    t0 = time.time()
    pipe_rf.fit(X_train, y_train)
    print(f"Train time: {time.time() - t0:.1f}s")
    preds = pipe_rf.predict(X_test)
    evaluate("Random Forest (TF-IDF)", y_test, preds)
    results["preds_rf"] = preds

    return results


# ── Save predictions ──────────────────────────────────────────────────────────

def save_predictions(preds_dict: dict) -> None:
    """Save each prediction array as a .npy file in SPLITS_DIR."""
    for name, arr in preds_dict.items():
        path = os.path.join(SPLITS_DIR, f"{name}.npy")
        np.save(path, arr)
    print(f"\nPredictions saved to: {SPLITS_DIR}/")


# ── Comparison figures ────────────────────────────────────────────────────────

MODEL_ORDER  = ["VADER", "LR (BoW)", "LR (TF-IDF)", "LinearSVM (TF-IDF)",
                "Naive Bayes (TF-IDF)", "Decision Tree (TF-IDF)",
                "Random Forest (TF-IDF)", "Embeddings+LR (LSA)"]
MODEL_LABELS = ["VADER", "LR (BoW)", "LR (TF-IDF)", "SVM (TF-IDF)",
                "NB (TF-IDF)", "DT (TF-IDF)", "RF (TF-IDF)", "LSA+LR"]


def plot_confusion_matrices() -> None:
    """Figure 4 — 3×3 grid of row-normalised confusion matrices."""

    def plot_cm(cm, name, ax, show_xlabel=False, show_ylabel=False):
        rs      = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(rs > 0, cm.astype(float) / rs, 0)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER,
                    ax=ax, cbar=False, linewidths=0.5, vmin=0, vmax=1,
                    annot_kws={"size": 14})
        for i in range(3):
            for j in range(3):
                ax.text(j+0.5, i+0.75, f"({cm[i,j]})",
                        ha="center", va="center", fontsize=12, color="gray")
        ax.set_title(name, fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted" if show_xlabel else "", fontsize=14, labelpad=15)
        ax.set_ylabel("True"      if show_ylabel else "", fontsize=14, labelpad=15)
        if not show_xlabel: ax.set_xticklabels([])
        if not show_ylabel: ax.set_yticklabels([])
        ax.tick_params(labelsize=14)

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for i, (k, lbl) in enumerate(zip(MODEL_ORDER, MODEL_LABELS)):
        plot_cm(ALL_CMS[k], lbl, axes.flatten()[i],
                show_xlabel=(i // 3 == 2), show_ylabel=(i % 3 == 0))
    for j in range(len(MODEL_ORDER), 9):
        axes.flatten()[j].set_visible(False)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    path = os.path.join(OUT_DIR, "fig4_confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_f1_comparison() -> None:
    """Figure 5 — Per-class and macro F1 grouped bar chart."""
    macro_f1s  = [ALL_RESULTS[k]["macro_f1"]      for k in MODEL_ORDER]
    normal_f1s = [ALL_RESULTS[k]["Normal_f1"]     for k in MODEL_ORDER]
    dep_f1s    = [ALL_RESULTS[k]["Depression_f1"] for k in MODEL_ORDER]
    sui_f1s    = [ALL_RESULTS[k]["Suicidal_f1"]   for k in MODEL_ORDER]

    x = np.arange(len(MODEL_ORDER)); w = 0.18
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.bar(x-1.5*w, normal_f1s, w, label="Normal",     color=COLORS["Normal"],     alpha=0.9, edgecolor="white")
    ax.bar(x-0.5*w, dep_f1s,    w, label="Depression", color=COLORS["Depression"], alpha=0.9, edgecolor="white")
    ax.bar(x+0.5*w, sui_f1s,    w, label="Suicidal",   color=COLORS["Suicidal"],   alpha=0.9, edgecolor="white")
    ax.bar(x+1.5*w, macro_f1s,  w, label="Macro F1",  color="#6C757D",             alpha=0.9, edgecolor="white")

    ax.set_xticks(x); ax.set_xticklabels(MODEL_LABELS, fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1-Score", fontsize=14, labelpad=15)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(fontsize=12, loc="upper left")
    ax.axhline(0.33, color="gray", linestyle=":", lw=1.2, alpha=0.8)
    ax.spines[["top","right"]].set_visible(False)

    for i, vals in enumerate(zip(normal_f1s, dep_f1s, sui_f1s, macro_f1s)):
        for j, (val, off) in enumerate(zip(vals, [-1.5, -0.5, 0.5, 1.5])):
            ax.text(i+off*w, val+0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig5_f1_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_dep_sui_confusion() -> None:
    """Figure 6 — Depression ↔ Suicidal critical miss and false alarm rates."""
    sui_as_dep, dep_as_sui = [], []
    for k in MODEL_ORDER:
        cm = ALL_CMS[k]
        ts = cm[2].sum(); td = cm[1].sum()
        sui_as_dep.append(cm[2,1] / ts * 100 if ts > 0 else 0)
        dep_as_sui.append(cm[1,2] / td * 100 if td > 0 else 0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    y_pos = np.arange(len(MODEL_ORDER))

    axes[0].barh(y_pos, sui_as_dep, color=COLORS["Depression"], alpha=0.85)
    axes[0].set_yticks(y_pos); axes[0].set_yticklabels(MODEL_LABELS, fontsize=10)
    axes[0].set_xlabel("% of True Suicidal Posts Predicted as Depression", fontsize=14, labelpad=14)
    axes[0].set_title("Critical Miss Rate\n(Suicidal → Depression)", fontsize=14, fontweight="bold")
    axes[0].tick_params(axis="both", labelsize=14)
    axes[0].spines[["top","right"]].set_visible(False)
    for i, v in enumerate(sui_as_dep):
        axes[0].text(v+0.4, i, f"{v:.1f}%", va="center", fontsize=12)

    axes[1].barh(y_pos, dep_as_sui, color=COLORS["Suicidal"], alpha=0.85)
    axes[1].set_yticks(y_pos); axes[1].set_yticklabels(MODEL_LABELS, fontsize=10)
    axes[1].set_xlabel("% of True Depression Posts Predicted as Suicidal", fontsize=14, labelpad=14)
    axes[1].set_title("False Alarm Rate\n(Depression → Suicidal)", fontsize=14, fontweight="bold")
    axes[1].spines[["top","right"]].set_visible(False)
    axes[1].tick_params(axis="both", labelsize=14)
    for i, v in enumerate(dep_as_sui):
        axes[1].text(v+0.4, i, f"{v:.1f}%", va="center", fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)
    path = os.path.join(OUT_DIR, "fig6_dep_sui_confusion.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def save_results_table() -> None:
    """Save results summary CSV."""
    sui_as_dep, dep_as_sui = [], []
    for k in MODEL_ORDER:
        cm = ALL_CMS[k]
        ts = cm[2].sum(); td = cm[1].sum()
        sui_as_dep.append(cm[2,1] / ts * 100 if ts > 0 else 0)
        dep_as_sui.append(cm[1,2] / td * 100 if td > 0 else 0)

    rows = []
    for k, lbl in zip(MODEL_ORDER, MODEL_LABELS):
        r = ALL_RESULTS[k]
        rows.append({
            "Model":         lbl,
            "Accuracy":      round(r["acc"],           3),
            "Macro F1":      round(r["macro_f1"],      3),
            "Normal F1":     round(r["Normal_f1"],     3),
            "Depression F1": round(r["Depression_f1"], 3),
            "Suicidal F1":   round(r["Suicidal_f1"],   3),
            "Sui→Dep (%)":   round(sui_as_dep[MODEL_ORDER.index(k)], 1),
            "Dep→Sui (%)":   round(dep_as_sui[MODEL_ORDER.index(k)], 1),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    path = os.path.join(OUT_DIR, "phase2_results_summary.csv")
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    (X_train, X_val, X_test,
     X_val_raw, X_test_raw,
     y_train, y_val, y_test) = load_splits()

    # VADER
    preds_vader = run_vader(X_val_raw, X_test_raw, y_val, y_test)

    # Classical supervised models
    classical_preds = run_classical_models(X_train, X_test, y_train, y_test)

    # Save all predictions
    all_preds = {"preds_vader": preds_vader, **classical_preds}
    save_predictions(all_preds)

    # Figures and summary table
    plot_confusion_matrices()
    plot_f1_comparison()
    plot_dep_sui_confusion()
    save_results_table()

    print("\nPhase 2 complete.")
    print("Note: Zero-Shot LLM (Model 9) is in phase2_llm.py — run on GPU runtime.")
