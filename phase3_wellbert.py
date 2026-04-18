"""
phase3_wellbert.py
------------------
Phase 3: WellBERT Fine-Tuning.

What this script does
---------------------
1.  Loads the raw (transformer-preprocessed) train/val/test splits.
2.  Tokenizes using BertTokenizerFast (max_len=512).
3.  Builds a BertForSequenceClassification model with a 3-class head.
4.  Trains for 5 epochs using:
      - AdamW optimizer with differential weight decay
      - Linear warmup scheduler (10% of total steps)
      - Weighted cross-entropy loss (inverse frequency + 1.5× Suicidal boost)
      - Mixed-precision FP16 training
      - Gradient clipping at 1.0
5.  Saves the best checkpoint by validation macro F1.
6.  Evaluates on the held-out test set and saves predictions.
7.  Plots the WellBERT confusion matrix.
8.  Generates the full 10-model comparison figure (requires Phase 2 predictions).

Requirements
------------
    GPU runtime — NVIDIA T4 or better (Runtime → Change runtime type in Colab)

Usage
-----
    python phase3_wellbert.py

Outputs
-------
    outputs/bert_best512.pt                  — best model checkpoint
    outputs/splits/preds_bert512.npy         — test-set predictions
    outputs/bert_confusion512.png            — WellBERT confusion matrix
    outputs/fig_full_comparison.png          — all 10 models comparison
    outputs/phase3_results_summary512.csv    — results table
"""

import os
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

from config import (
    LABEL_ORDER, LABEL2ID, ID2LABEL,
    COLORS, PALETTE,
    OUT_DIR, SPLITS_DIR, CKPT_PATH,
    MODEL_NAME, MAX_LEN, BATCH_SIZE, EPOCHS, LR,
    WARMUP_RATIO, WEIGHT_DECAY, MAX_GRAD_NORM, SUICIDAL_BOOST,
    INFER_BATCH_SIZE, SEED,
)
from utils import set_seeds

warnings.filterwarnings("ignore")
set_seeds(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"Memory : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: No GPU detected. Training will be very slow on CPU.")


# ── Dataset ───────────────────────────────────────────────────────────────────

class MentalHealthDataset(Dataset):
    """
    PyTorch Dataset for mental health text classification.

    Parameters
    ----------
    texts     : array-like of str   Raw post texts.
    labels    : array-like of str   Class labels from LABEL_ORDER.
    tokenizer : BertTokenizerFast   Pre-loaded tokenizer.
    max_len   : int                 Maximum token sequence length.
    """

    def __init__(self, texts, labels, tokenizer, max_len: int):
        self.texts     = texts
        self.labels    = [LABEL2ID[l] for l in labels]
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── DataLoader worker seed ────────────────────────────────────────────────────

def seed_worker(worker_id: int) -> None:
    """Ensure each DataLoader worker uses a deterministic seed."""
    import random
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# ── Training and evaluation functions ────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, criterion, scaler):
    """
    Run one training epoch.

    Returns
    -------
    (float, float) — mean loss and accuracy over all batches.
    """
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0

    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        with autocast(enabled=DEVICE.type == "cuda"):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = criterion(outputs.logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        preds          = outputs.logits.argmax(dim=-1)
        total_loss    += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_n       += labels.size(0)

    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    """
    Evaluate the model on a DataLoader.

    Returns
    -------
    tuple — (mean_loss, accuracy, macro_f1, predictions, true_labels)
    """
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["label"].to(DEVICE)

        with autocast(enabled=DEVICE.type == "cuda"):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = criterion(outputs.logits, labels)

        total_loss += loss.item() * labels.size(0)
        all_preds.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n   = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro")
    return total_loss / n, acc, f1, np.array(all_preds), np.array(all_labels)


# ── Inference helper ──────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, tokenizer, texts, batch_size: int = INFER_BATCH_SIZE):
    """
    Run inference on a list of texts and return string labels.

    Parameters
    ----------
    model     : fine-tuned BertForSequenceClassification
    tokenizer : BertTokenizerFast
    texts     : array-like of str
    batch_size: int

    Returns
    -------
    np.ndarray of str — predicted class labels.
    """
    model.eval()

    class _DS(Dataset):
        def __init__(self, texts):
            self.texts = texts
        def __len__(self): return len(self.texts)
        def __getitem__(self, idx):
            enc = tokenizer(
                str(self.texts[idx]), max_length=MAX_LEN,
                padding="max_length", truncation=True, return_tensors="pt",
            )
            return {"input_ids":      enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0)}

    loader = DataLoader(_DS(texts), batch_size=batch_size,
                        shuffle=False, num_workers=2, pin_memory=True)
    preds  = []
    for batch in loader:
        with autocast(enabled=DEVICE.type == "cuda"):
            logits = model(
                input_ids      = batch["input_ids"].to(DEVICE),
                attention_mask = batch["attention_mask"].to(DEVICE),
            ).logits
        preds.extend(logits.argmax(dim=-1).cpu().numpy())
    return np.array([ID2LABEL[p] for p in preds])


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Load splits ───────────────────────────────────────────────────────────
    def npy(name): return np.load(os.path.join(SPLITS_DIR, f"{name}.npy"),
                                  allow_pickle=True)
    X_train = npy("X_train_raw"); X_val = npy("X_val_raw"); X_test = npy("X_test_raw")
    y_train = npy("y_train");     y_val = npy("y_val");     y_test = npy("y_test")
    print(f"Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}")

    # ── Class weights ─────────────────────────────────────────────────────────
    counts  = np.array([(y_train == l).sum() for l in LABEL_ORDER], dtype=float)
    weights = counts.sum() / (len(LABEL_ORDER) * counts)
    weights[LABEL_ORDER.index("Suicidal")] *= SUICIDAL_BOOST
    weights = weights / weights.sum() * len(LABEL_ORDER)
    class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)

    print("\nClass weights (inverse frequency + Suicidal boost):")
    for l, w in zip(LABEL_ORDER, weights):
        print(f"  {l:12s}: {w:.4f}")

    # ── Tokenizer & DataLoaders ───────────────────────────────────────────────
    print("\nLoading tokenizer …")
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    g = torch.Generator(); g.manual_seed(SEED)

    def make_loader(X, y, shuffle):
        ds = MentalHealthDataset(X, y, tokenizer, MAX_LEN)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                          num_workers=2, pin_memory=True,
                          worker_init_fn=seed_worker, generator=g)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   shuffle=False)
    test_loader  = make_loader(X_test,  y_test,  shuffle=False)
    print(f"Batches — Train: {len(train_loader):,}  Val: {len(val_loader):,}  Test: {len(test_loader):,}")

    # ── Model, optimizer, scheduler ───────────────────────────────────────────
    print("\nLoading BERT model …")
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_ORDER),
        id2label=ID2LABEL, label2id=LABEL2ID,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)], "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters()
                    if     any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ], lr=LR, eps=1e-8)

    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = GradScaler(enabled=DEVICE.type == "cuda")

    print(f"Total steps: {total_steps:,}  Warmup: {warmup_steps:,}")

    # ── Fine-tuning loop ──────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  WellBERT Fine-Tuning")
    print("=" * 55)

    history     = defaultdict(list)
    best_val_f1 = -1.0
    best_epoch  = -1

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler)
        val_loss, val_acc, val_f1, _, _ = eval_epoch(model, val_loader, criterion)
        elapsed = time.time() - t0

        for k, v in [("train_loss", train_loss), ("val_loss", val_loss),
                     ("train_acc", train_acc),   ("val_acc", val_acc),
                     ("val_f1", val_f1)]:
            history[k].append(v)

        print(f"\n  Epoch {epoch}/{EPOCHS}  ({elapsed:.0f}s)")
        print(f"    train loss={train_loss:.4f}  train acc={train_acc:.3f}")
        print(f"    val   loss={val_loss:.4f}  val   acc={val_acc:.3f}  "
              f"val macro F1={val_f1:.3f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"    ✓ New best — checkpoint saved")

    print(f"\nTraining complete.")
    print(f"Best val macro F1 : {best_val_f1:.3f} at epoch {best_epoch}")
    print(f"Checkpoint        : {CKPT_PATH}")

    # ── Test set evaluation ───────────────────────────────────────────────────
    print("\nLoading best checkpoint …")
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))

    _, _, _, preds_arr, labels_arr = eval_epoch(model, test_loader, criterion)
    preds_bert = np.array([ID2LABEL[p] for p in preds_arr])

    print(f"\n{'─' * 55}")
    print("  WellBERT — Test Set Results")
    print(f"{'─' * 55}")
    print(classification_report(y_test, preds_bert, target_names=LABEL_ORDER, digits=3))

    np.save(os.path.join(SPLITS_DIR, "preds_bert512.npy"), preds_bert)

    # ── Confusion matrix figure ───────────────────────────────────────────────
    cm      = confusion_matrix(y_test, preds_bert, labels=LABEL_ORDER)
    rs      = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(rs > 0, cm.astype(float) / rs, 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER,
                ax=ax, cbar=False, linewidths=0.5, vmin=0, vmax=1,
                annot_kws={"size": 14})
    for i in range(3):
        for j in range(3):
            ax.text(j+0.5, i+0.75, f"({cm[i,j]})",
                    ha="center", va="center", fontsize=12, color="gray")
    ax.set_xlabel("Predicted", fontsize=14, labelpad=14)
    ax.set_ylabel("True",      fontsize=14, labelpad=14)
    ax.tick_params(axis="both", labelsize=14)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "bert_confusion512.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # ── Full 10-model comparison (requires Phase 2 predictions) ───────────────
    PHASE2_MODELS = [
        ("VADER",           "preds_vader"),
        ("LR (BoW)",        "preds_lr_bow"),
        ("LR (TF-IDF)",     "preds_lr_tfidf"),
        ("SVM (TF-IDF)",    "preds_svm"),
        ("NB (TF-IDF)",     "preds_nb"),
        ("DT (TF-IDF)",     "preds_dt"),
        ("RF (TF-IDF)",     "preds_rf"),
        ("LSA+LR",          "preds_lsa"),
        ("Zero-Shot LLM",   "preds_llm"),
    ]

    all_names, all_macro, all_normal, all_dep, all_sui = [], [], [], [], []
    for model_name_k, fname in PHASE2_MODELS:
        fpath = os.path.join(SPLITS_DIR, f"{fname}.npy")
        if not os.path.exists(fpath):
            print(f"Skipping {model_name_k} — prediction file not found: {fpath}")
            continue
        p   = np.load(fpath, allow_pickle=True)
        rep = classification_report(y_test, p, target_names=LABEL_ORDER, output_dict=True)
        all_names.append(model_name_k)
        all_macro.append(rep["macro avg"]["f1-score"])
        all_normal.append(rep["Normal"]["f1-score"])
        all_dep.append(rep["Depression"]["f1-score"])
        all_sui.append(rep["Suicidal"]["f1-score"])

    rep_bert = classification_report(y_test, preds_bert, target_names=LABEL_ORDER,
                                     output_dict=True)
    all_names.append("WellBERT")
    all_macro.append(rep_bert["macro avg"]["f1-score"])
    all_normal.append(rep_bert["Normal"]["f1-score"])
    all_dep.append(rep_bert["Depression"]["f1-score"])
    all_sui.append(rep_bert["Suicidal"]["f1-score"])

    x = np.arange(len(all_names)); w = 0.18
    fig, ax = plt.subplots(figsize=(22, 6))
    ax.bar(x-1.5*w, all_normal, w, label="Normal",     color=COLORS["Normal"],     alpha=0.9, edgecolor="white")
    ax.bar(x-0.5*w, all_dep,    w, label="Depression", color=COLORS["Depression"], alpha=0.9, edgecolor="white")
    ax.bar(x+0.5*w, all_sui,    w, label="Suicidal",   color=COLORS["Suicidal"],   alpha=0.9, edgecolor="white")
    ax.bar(x+1.5*w, all_macro,  w, label="Macro F1",  color="#6C757D",             alpha=0.9, edgecolor="white")
    ax.axvspan(x[-1]-0.45, x[-1]+0.45, alpha=0.07, color="black", zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(all_names, fontsize=9)
    ax.set_ylim(0, 1.10); ax.set_ylabel("F1-Score", fontsize=12)
    ax.legend(fontsize=10, loc="lower right")
    ax.axhline(0.33, color="gray", linestyle=":", lw=1.2, alpha=0.5)
    ax.spines[["top","right"]].set_visible(False)
    for i, vals in enumerate(zip(all_normal, all_dep, all_sui, all_macro)):
        for j, (val, off) in enumerate(zip(vals, [-1.5,-0.5,0.5,1.5])):
            ax.text(i+off*w, val+0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=6)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig_full_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # ── Results CSV ───────────────────────────────────────────────────────────
    rows = [{"Model": n, "Macro F1": round(m, 3), "Normal F1": round(no, 3),
             "Depression F1": round(d, 3), "Suicidal F1": round(s, 3)}
            for n, m, no, d, s in zip(all_names, all_macro, all_normal, all_dep, all_sui)]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    path = os.path.join(OUT_DIR, "phase3_results_summary512.csv")
    df.to_csv(path, index=False)
    print(f"Saved: {path}")

    print("\nPhase 3 complete.")
