"""
phase4_evaluation.py
--------------------
Phase 4: Evaluation and Analysis.

What this script does
---------------------
1.  Computational efficiency benchmarking (WellBERT vs LR TF-IDF).
2.  Performance by text length (median split at 47 words).
3.  Out-of-distribution (OOD) robustness testing on 4 unseen classes:
      Anxiety, Bipolar, Stress, Personality disorder.
4.  Qualitative error analysis — collects and displays misclassified examples
    for the two critical confusion pairs (Suicidal→Depression, Depression→Suicidal).

Requirements
------------
    GPU runtime — NVIDIA T4 or better.
    Phase 3 must be completed first (bert_best512.pt must exist).

Usage
-----
    python phase4_evaluation.py

Outputs
-------
    outputs/fig_length_analysis512.png
    outputs/fig_ood_robustness512.png
    outputs/phase4_efficiency512.csv
    outputs/phase4_ood_results512.csv
    outputs/phase4_error_analysis512.csv
"""

import os
import re
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast

from transformers import BertTokenizerFast, BertForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import (
    LABEL_ORDER, LABEL2ID, ID2LABEL,
    COLORS, PALETTE,
    OUT_DIR, SPLITS_DIR, CKPT_PATH, OOD_PATH,
    MODEL_NAME, MAX_LEN, INFER_BATCH_SIZE,
    OOD_CLASSES, N_ERROR_EXAMPLES, SEED,
    KEEP_WORDS,
)
from utils import set_seeds, preprocess_classical

warnings.filterwarnings("ignore")
set_seeds(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")


# ── Load model for inference ──────────────────────────────────────────────────

def load_model():
    """Load WellBERT checkpoint and return model + tokenizer."""
    print("Loading tokenizer and model …")
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model     = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_ORDER),
        id2label=ID2LABEL, label2id=LABEL2ID,
    ).to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()
    print("WellBERT loaded ✓")
    return model, tokenizer


# ── Inference helper ──────────────────────────────────────────────────────────

class _TextDS(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts; self.tok = tokenizer
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tok(str(self.texts[idx]), max_length=MAX_LEN,
                       padding="max_length", truncation=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0)}


@torch.no_grad()
def predict(model, tokenizer, texts, batch_size: int = INFER_BATCH_SIZE):
    """Run inference and return string label predictions."""
    model.eval()
    loader = DataLoader(_TextDS(texts, tokenizer), batch_size=batch_size,
                        shuffle=False, num_workers=2, pin_memory=True)
    preds = []
    for batch in loader:
        with autocast(enabled=DEVICE.type == "cuda"):
            logits = model(
                input_ids      = batch["input_ids"].to(DEVICE),
                attention_mask = batch["attention_mask"].to(DEVICE),
            ).logits
        preds.extend(logits.argmax(dim=-1).cpu().numpy())
    return np.array([ID2LABEL[p] for p in preds])


def preprocess_transformer(text: str) -> str:
    """Minimal transformer preprocessing (encoding fix + URL removal)."""
    text = str(text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"http\S+|www\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ── Computational Efficiency Benchmarking ──────────────────────────────────

def benchmark_efficiency(model, tokenizer, X_test, y_train):
    """Compare inference speed of WellBERT vs LR (TF-IDF)."""
    print("=" * 55)
    print("  Phase 4 — Computational Efficiency")
    print("=" * 55)

    # Warmup
    _ = predict(model, tokenizer, X_test[:64])

    t0 = time.time()
    _  = predict(model, tokenizer, X_test)
    elapsed = time.time() - t0
    bert_throughput = len(X_test) / elapsed
    bert_ms_per     = elapsed / len(X_test) * 1000
    print(f"\nWellBERT Inference (n={len(X_test):,}, max_len={MAX_LEN})")
    print(f"  Total time    : {elapsed:.1f}s")
    print(f"  Throughput    : {bert_throughput:.0f} posts/sec")
    print(f"  Latency       : {bert_ms_per:.2f} ms/post")

    # LR (TF-IDF) — refit for timing
    X_train_clean = np.load(os.path.join(SPLITS_DIR, "X_train.npy"), allow_pickle=True)
    X_test_clean  = np.load(os.path.join(SPLITS_DIR, "X_test.npy"),  allow_pickle=True)

    pipe_lr = Pipeline([
        ("vec", TfidfVectorizer(max_features=50000, ngram_range=(1,2),
                                min_df=2, max_df=0.95, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                                   solver="saga", n_jobs=-1, random_state=SEED)),
    ])
    print("\nRetraining LR (TF-IDF) for timing …")
    t0 = time.time()
    pipe_lr.fit(X_train_clean, y_train)
    lr_train_t = time.time() - t0

    t0 = time.time()
    _  = pipe_lr.predict(X_test_clean)
    lr_infer_t = time.time() - t0
    lr_throughput = len(X_test_clean) / lr_infer_t
    lr_ms_per     = lr_infer_t / len(X_test_clean) * 1000

    print(f"\nLR (TF-IDF) (n={len(X_test_clean):,})")
    print(f"  Train time    : {lr_train_t:.1f}s")
    print(f"  Inference time: {lr_infer_t:.2f}s")
    print(f"  Throughput    : {lr_throughput:.0f} posts/sec")
    print(f"  Latency       : {lr_ms_per:.4f} ms/post")

    eff_df = pd.DataFrame([
        {"Model": "LR (TF-IDF)", "Train Time (s)": round(lr_train_t, 1),
         "Inference (s)": round(lr_infer_t, 2),
         "Throughput (posts/s)": round(lr_throughput, 0),
         "Latency (ms/post)": round(lr_ms_per, 4), "Macro F1": 0.799},
        {"Model": "WellBERT", "Train Time (s)": "~1800 (GPU)",
         "Inference (s)": round(elapsed, 1),
         "Throughput (posts/s)": round(bert_throughput, 0),
         "Latency (ms/post)": round(bert_ms_per, 2), "Macro F1": 0.849},
    ])
    path = os.path.join(OUT_DIR, "phase4_efficiency512.csv")
    eff_df.to_csv(path, index=False)
    print(f"\nSaved: {path}")


# ── Performance by Text Length ─────────────────────────────────────────────

def performance_by_length(model, tokenizer, X_test, y_test):
    """Evaluate WellBERT on short (≤median) and long (>median) posts."""
    print("\n" + "=" * 55)
    print("  Phase 4 — Performance by Text Length")
    print("=" * 55)

    word_counts = np.array([len(str(t).split()) for t in X_test])
    median_wc   = int(np.median(word_counts))
    short_mask  = word_counts <= median_wc
    long_mask   = ~short_mask
    print(f"Median word count: {median_wc}")
    print(f"Short posts (≤{median_wc} words): {short_mask.sum():,}")
    print(f"Long  posts (>{median_wc} words): {long_mask.sum():,}")

    print("\nRunning inference on short posts …")
    preds_short = predict(model, tokenizer, X_test[short_mask])
    print("Running inference on long posts …")
    preds_long  = predict(model, tokenizer, X_test[long_mask])

    rep_short = classification_report(y_test[short_mask], preds_short,
                                      target_names=LABEL_ORDER, output_dict=True)
    rep_long  = classification_report(y_test[long_mask],  preds_long,
                                      target_names=LABEL_ORDER, output_dict=True)

    print(f"\nShort (≤{median_wc} words, n={short_mask.sum():,}):")
    print(classification_report(y_test[short_mask], preds_short,
                                 target_names=LABEL_ORDER, digits=3))
    print(f"Long (>{median_wc} words, n={long_mask.sum():,}):")
    print(classification_report(y_test[long_mask], preds_long,
                                 target_names=LABEL_ORDER, digits=3))

    groups    = [f"Short (≤{median_wc} words; n={short_mask.sum():,})",
                 f"Long (>{median_wc} words); n={long_mask.sum():,})"]
    reps      = [rep_short, rep_long]
    macro_f1s = [r["macro avg"]["f1-score"]  for r in reps]
    norm_f1s  = [r["Normal"]["f1-score"]     for r in reps]
    dep_f1s   = [r["Depression"]["f1-score"] for r in reps]
    sui_f1s   = [r["Suicidal"]["f1-score"]   for r in reps]

    x = np.arange(2); w = 0.18
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x-1.5*w, norm_f1s,  w, label="Normal",     color=COLORS["Normal"],     alpha=0.9, edgecolor="white")
    ax.bar(x-0.5*w, dep_f1s,   w, label="Depression", color=COLORS["Depression"], alpha=0.9, edgecolor="white")
    ax.bar(x+0.5*w, sui_f1s,   w, label="Suicidal",   color=COLORS["Suicidal"],   alpha=0.9, edgecolor="white")
    ax.bar(x+1.5*w, macro_f1s, w, label="Macro F1",   color="#6C757D",            alpha=0.9, edgecolor="white")

    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel("F1-Score", fontsize=16, labelpad=14)
    ax.set_ylim(0, 1.08)
    ax.spines[["top","right"]].set_visible(False)
    ax.axhline(0.33, color="gray", linestyle=":", lw=1.2, alpha=0.5)
    ax.legend(fontsize=16, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=4, frameon=False)
    for i, vals in enumerate(zip(norm_f1s, dep_f1s, sui_f1s, macro_f1s)):
        for val, off in zip(vals, [-1.5, -0.5, 0.5, 1.5]):
            ax.text(i + off*w, val + 0.01, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=12)
    ax.tick_params(axis="both", labelsize=15)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    path = os.path.join(OUT_DIR, "fig_length_analysis512.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── OOD Robustness Testing ─────────────────────────────────────────────────

def ood_robustness(model, tokenizer):
    """Evaluate WellBERT on 4 out-of-distribution mental health classes."""
    print("\n" + "=" * 55)
    print("  OOD Robustness Testing")
    print("=" * 55)

    df_ood = pd.read_csv(OOD_PATH)
    df_ood = df_ood.rename(columns={"statement": "text", "status": "label"})
    df_ood = df_ood.dropna(subset=["text"]).reset_index(drop=True)
    df_ood["text"]  = df_ood["text"].astype(str)
    df_ood["label"] = df_ood["label"].str.strip()

    print(f"OOD posts: {len(df_ood):,}")
    for cls in OOD_CLASSES:
        n = (df_ood["label"] == cls).sum()
        print(f"  {cls:25s}: {n:,}")

    df_ood["text_clean"] = df_ood["text"].apply(preprocess_transformer)
    preds_ood            = predict(model, tokenizer, df_ood["text_clean"].values)
    df_ood["bert_pred"]  = preds_ood

    ood_results = {}
    for cls in OOD_CLASSES:
        mask  = df_ood["label"] == cls
        total = mask.sum()
        cp    = preds_ood[mask.values]
        rates = {l: (cp == l).sum() / total * 100 for l in LABEL_ORDER}
        ood_results[cls] = rates
        print(f"\n  {cls} (n={total:,}):")
        for l, r in rates.items():
            print(f"    → {l:12s}: {r:.1f}%")


    rows = [{"OOD Class": cls,
             **{f"→ {l} (%)": round(ood_results[cls][l], 1) for l in LABEL_ORDER}}
            for cls in OOD_CLASSES]
    ood_df = pd.DataFrame(rows)
    path   = os.path.join(OUT_DIR, "phase4_ood_results512.csv")
    ood_df.to_csv(path, index=False)
    print(f"\nSaved: {path}")

    norm_rates = [ood_results[c]["Normal"]     for c in OOD_CLASSES]
    dep_rates  = [ood_results[c]["Depression"] for c in OOD_CLASSES]
    sui_rates  = [ood_results[c]["Suicidal"]   for c in OOD_CLASSES]
    x = np.arange(len(OOD_CLASSES))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].bar(x, norm_rates, color=COLORS["Normal"],     alpha=0.9, edgecolor="white", label="Normal")
    axes[0].bar(x, dep_rates,  color=COLORS["Depression"], alpha=0.9, edgecolor="white", label="Depression",
                bottom=norm_rates)
    axes[0].bar(x, sui_rates,  color=COLORS["Suicidal"],   alpha=0.9, edgecolor="white", label="Suicidal",
                bottom=np.array(norm_rates) + np.array(dep_rates))
    axes[0].set_xticks(x); axes[0].set_xticklabels(OOD_CLASSES, fontsize=12)
    axes[0].set_ylabel("% of Posts", fontsize=14, labelpad=12)
    axes[0].set_ylim(0, 115)
    axes[0].set_title("Full Prediction Breakdown\n(stacked)", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=12, loc="upper center", ncol=3, frameon=False)
    axes[0].spines[["top","right"]].set_visible(False)
    axes[0].tick_params(axis="both", labelsize=12)
    for i, (n, d, s) in enumerate(zip(norm_rates, dep_rates, sui_rates)):
        if n > 4: axes[0].text(i, n/2,   f"{n:.0f}%", ha="center", fontsize=12, color="white", fontweight="bold")
        if d > 4: axes[0].text(i, n+d/2, f"{d:.0f}%", ha="center", fontsize=12, color="white", fontweight="bold")
        if s > 4: axes[0].text(i, n+d+s/2, f"{s:.0f}%", ha="center", fontsize=12, color="white", fontweight="bold")

    axes[1].bar(x, sui_rates, color=COLORS["Suicidal"], alpha=0.85, edgecolor="white")
    axes[1].set_xticks(x); axes[1].set_xticklabels(OOD_CLASSES, fontsize=12)
    axes[1].set_ylabel("% Classified as Suicidal", fontsize=14, labelpad=12)
    axes[1].set_ylim(0, max(sui_rates) * 1.3)
    axes[1].set_title("Suicidal False Alarm Rate\nper OOD Class", fontsize=14, fontweight="bold")
    axes[1].spines[["top","right"]].set_visible(False)
    axes[1].tick_params(axis="both", labelsize=12)
    for i, v in enumerate(sui_rates):
        axes[1].text(i, v+0.05, f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig_ood_robustness512.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Qualitative Error Analysis ─────────────────────────────────────────────

def error_analysis(X_test, y_test, preds_bert):
    """Collect and display misclassified examples for key confusion pairs."""
    print("\n" + "=" * 55)
    print("  Phase 4 — Qualitative Error Analysis")
    print("=" * 55)

    error_rows = []
    for true_cls in LABEL_ORDER:
        for pred_cls in LABEL_ORDER:
            if true_cls == pred_cls:
                continue
            mask    = (y_test == true_cls) & (preds_bert == pred_cls)
            indices = np.where(mask)[0]
            for idx in indices[:N_ERROR_EXAMPLES]:
                error_rows.append({
                    "true_label": true_cls,
                    "pred_label": pred_cls,
                    "text":       X_test[idx],
                    "word_count": len(str(X_test[idx]).split()),
                })

    errors_df = pd.DataFrame(error_rows)
    path      = os.path.join(OUT_DIR, "phase4_error_analysis512.csv")
    errors_df.to_csv(path, index=False)
    print(f"Total error examples: {len(errors_df)}")
    print(errors_df.groupby(["true_label","pred_label"]).size()
                   .reset_index(name="count").to_string(index=False))
    print(f"Saved: {path}")

    # Display critical examples
    for true_cls, pred_cls in [("Suicidal", "Depression"), ("Depression", "Suicidal")]:
        subset = errors_df[
            (errors_df["true_label"] == true_cls) &
            (errors_df["pred_label"] == pred_cls)
        ].head(5)
        print(f"\n{'═' * 60}")
        print(f"  TRUE: {true_cls}  →  PREDICTED: {pred_cls}")
        print(f"{'═' * 60}")
        for _, row in subset.iterrows():
            print(f"\n  [{row['word_count']} words]")
            print(f"  {str(row['text'])[:250]}")
            print(f"  {'─' * 50}")

    # Word count distribution figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (true_cls, pred_cls) in zip(axes, [("Suicidal", "Depression"),
                                                ("Depression", "Suicidal")]):
        subset = errors_df[
            (errors_df["true_label"] == true_cls) &
            (errors_df["pred_label"] == pred_cls)
        ]
        correct_mask = (y_test == true_cls) & (y_test == preds_bert)
        correct_wc   = np.array([len(str(t).split()) for t in X_test[correct_mask]])
        error_wc     = subset["word_count"].values

        ax.hist(correct_wc.clip(max=300), bins=30, alpha=0.55,
                color="#4C9BE8", label="Correctly classified", density=True)
        ax.hist(error_wc.clip(max=300),   bins=15, alpha=0.70,
                color="#E76F51", label="Misclassified",         density=True)
        ax.set_title(f"True {true_cls} → Predicted {pred_cls}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Word Count", fontsize=10)
        ax.set_ylabel("Density",    fontsize=10)
        ax.legend(fontsize=9)
        ax.spines[["top","right"]].set_visible(False)

    plt.suptitle("Word Count Distribution: Correct vs Misclassified Posts",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig_error_word_count.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")



if __name__ == "__main__":
    def npy(name): return np.load(os.path.join(SPLITS_DIR, f"{name}.npy"),
                                  allow_pickle=True)

    X_test     = npy("X_test_raw")
    y_test     = npy("y_test")
    y_train    = npy("y_train")
    preds_bert = npy("preds_bert512")

    model, tokenizer = load_model()

    benchmark_efficiency(model, tokenizer, X_test, y_train)
    performance_by_length(model, tokenizer, X_test, y_test)
    ood_robustness(model, tokenizer)
    error_analysis(X_test, y_test, preds_bert)

    print("\nPhase 4 complete.")
