"""
config.py
---------
Central configuration file for the WellBERT project.
All paths, hyperparameters, and constants are defined here.
Import this module in every phase script to ensure consistency.
"""

import os

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
# IMPORTANT: Before running any phase script, place your data file in the
# correct location relative to this project folder.
#
# Expected folder structure:
#
#   wellBERT/
#   ├── data/
#   │   └── Combined_Data.csv    ← download from Kaggle, rename (remove space)
#   ├── config.py
#   ├── phase1_eda.py
#   └── ...
#
# NOTE: data/ood.csv is created AUTOMATICALLY when you run phase1_eda.py.
#       You do not need to create or download it separately.
#
# If your file is stored elsewhere, update DATA_PATH below with the full path:
#   DATA_PATH = "/Users/yourname/Downloads/Combined_Data.csv"

DATA_PATH  = "data/Combined_Data.csv"         # main 3-class dataset
OOD_PATH   = "data/ood.csv"                   # out-of-distribution dataset
OUT_DIR    = "outputs"                        # root output directory
SPLITS_DIR = os.path.join(OUT_DIR, "splits") # train/val/test .npy splits
CKPT_PATH  = os.path.join(OUT_DIR, "bert_best512.pt")  # WellBERT checkpoint

# ── Class labels ──────────────────────────────────────────────────────────────
LABEL_ORDER = ["Normal", "Depression", "Suicidal"]
LABEL2ID    = {l: i for i, l in enumerate(LABEL_ORDER)}
ID2LABEL    = {i: l for l, i in LABEL2ID.items()}

# ── Colour palette (consistent across all figures) ───────────────────────────
COLORS  = {"Normal": "#4C9BE8", "Depression": "#F4A261", "Suicidal": "#E76F51"}
PALETTE = [COLORS[l] for l in LABEL_ORDER]

# ── Data split ratios ─────────────────────────────────────────────────────────
TEST_SIZE    = 0.15      # 15% held-out test set
VAL_RATIO    = 0.1765    # 17.65% of train+val → ~15% of full dataset

# ── Classical ML preprocessing ───────────────────────────────────────────────
# Words kept despite appearing in standard stopword lists
KEEP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "no", "not", "nor", "never", "neither", "nobody", "nothing",
    "nowhere", "cannot", "cant", "dont", "wont", "isnt", "wasnt",
    "aint", "shouldnt", "wouldnt", "couldnt", "neednt", "hadnt", "hasnt",
}

# ── WellBERT hyperparameters ──────────────────────────────────────────────────
MODEL_NAME     = "bert-base-uncased"
MAX_LEN        = 512        # maximum token sequence length
BATCH_SIZE     = 16         # training batch size
EPOCHS         = 5          # number of fine-tuning epochs
LR             = 1e-5       # AdamW learning rate
WARMUP_RATIO   = 0.1        # fraction of steps used for linear warmup
WEIGHT_DECAY   = 0.01       # applied to all non-bias / non-LayerNorm params
MAX_GRAD_NORM  = 1.0        # gradient clipping threshold
SUICIDAL_BOOST = 1.5        # extra loss multiplier for Suicidal class

# ── Inference ─────────────────────────────────────────────────────────────────
INFER_BATCH_SIZE = 64       # larger batch for inference (no gradients stored)

# ── OOD evaluation ────────────────────────────────────────────────────────────
OOD_CLASSES = ["Anxiety", "Bipolar", "Stress", "Personality disorder"]

# ── Error analysis ────────────────────────────────────────────────────────────
N_ERROR_EXAMPLES = 8        # examples collected per confusion pair
