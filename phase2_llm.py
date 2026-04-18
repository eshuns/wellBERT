"""
phase2_llm.py
-------------
Phase 2 — Model 9: Zero-Shot LLM Baseline (BART-large-MNLI).

Run this script in a SEPARATE GPU runtime session.
It requires a GPU (NVIDIA T4 or better) and ~8 GB VRAM.

What this script does
---------------------
Classifies all test-set posts using facebook/bart-large-mnli with zero-shot
natural language inference. Each post is scored against three candidate labels:
    - "normal everyday content"
    - "depression and sadness"
    - "suicidal thoughts and self-harm"

No task-specific training is performed. This baseline establishes the
performance ceiling achievable from a large pre-trained model with zero
task-specific adaptation.

Usage
-----
    python phase2_llm.py

Outputs
-------
    outputs/splits/preds_llm.npy
"""

import os
import time
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import pipeline

from config import LABEL_ORDER, SPLITS_DIR
from utils import set_seeds, evaluate

warnings.filterwarnings("ignore")
set_seeds()


# ── Dataset wrapper ───────────────────────────────────────────────────────────

class TextDataset(Dataset):
    """Simple wrapper to feed raw text strings to the pipeline."""

    def __init__(self, texts):
        self.texts = [str(t)[:512] for t in texts]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


# ── Candidate labels and mapping ──────────────────────────────────────────────

CANDIDATE_LABELS = [
    "normal everyday content",
    "depression and sadness",
    "suicidal thoughts and self-harm",
]

LABEL_MAP = {
    "normal everyday content":         "Normal",
    "depression and sadness":          "Depression",
    "suicidal thoughts and self-harm": "Suicidal",
}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Verify GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU detected. Please switch to a GPU runtime before running "
            "this script (Runtime → Change runtime type → T4 GPU in Colab)."
        )
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    # Load test split
    X_test_raw = np.load(os.path.join(SPLITS_DIR, "X_test_raw.npy"), allow_pickle=True)
    y_test     = np.load(os.path.join(SPLITS_DIR, "y_test.npy"),     allow_pickle=True)
    print(f"Test set: {len(y_test):,} posts")

    # Load BART-large-MNLI
    print("Loading facebook/bart-large-mnli …")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0,
    )
    print("Model loaded ✓\n")

    # Run inference
    dataset = TextDataset(X_test_raw)
    total   = len(dataset)
    print(f"Classifying {total:,} posts …\n")

    all_preds = []
    t0        = time.time()

    for i, result in enumerate(
        classifier(dataset, CANDIDATE_LABELS, multi_label=False, batch_size=64)
    ):
        all_preds.append(LABEL_MAP[result["labels"][0]])

        if (i + 1) % 64 == 0 or (i + 1) == total:
            done    = i + 1
            elapsed = time.time() - t0
            eta     = (elapsed / done) * (total - done)
            pct     = done / total * 100
            bar     = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
            print(f"  [{bar}] {done:,}/{total:,}  {pct:.0f}%  "
                  f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone — {elapsed:.1f}s  ({total / elapsed:.0f} posts/sec)")

    # Evaluate and save
    preds_llm = np.array(all_preds)
    evaluate("Zero-Shot LLM", y_test, preds_llm)

    out_path = os.path.join(SPLITS_DIR, "preds_llm.npy")
    np.save(out_path, preds_llm)
    print(f"\nPredictions saved: {out_path}")
    print("\nPhase 2 LLM complete.")
