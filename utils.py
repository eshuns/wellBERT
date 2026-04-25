"""
utils.py
--------
Shared utility functions used across all phases of the WellBERT pipeline.

Functions
---------
set_seeds            : Fix all random seeds for reproducibility.
preprocess_classical : Classical ML text preprocessing (lowercase, stem, etc.)
preprocess_transformer : Minimal preprocessing for BERT input.
evaluate             : Compute and print classification metrics; store results.
"""

import os
import re
import random
import warnings

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from config import LABEL_ORDER, KEEP_WORDS, SEED

warnings.filterwarnings("ignore")


def set_seeds(seed: int = SEED) -> None:
    """
    Fix random seeds for Python, NumPy, and (if available) PyTorch.
    Call this at the top of every script before any model or data operations.

    Parameters
    ----------
    seed : int
        Random seed value (default: SEED from config.py).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


# ── Preprocessing ─────────────────────────────────────────────────────────────

_STOPWORDS = ENGLISH_STOP_WORDS - KEEP_WORDS
_stemmer   = SnowballStemmer("english")


def preprocess_classical(text: str) -> str:
    """
    Classical ML preprocessing pipeline.

    Steps
    -----
    1. Lowercase.
    2. Remove URLs.
    3. Remove punctuation and digits.
    4. Tokenise, filter stopwords (retaining first-person pronouns and
       negations defined in KEEP_WORDS), and apply Snowball stemming.

    Parameters
    ----------
    text : str
        Raw post text.

    Returns
    -------
    str
        Space-joined stemmed tokens.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    tokens = [
        _stemmer.stem(t)
        for t in text.split()
        if t not in _STOPWORDS and len(t) > 1
    ]
    return " ".join(tokens)


def preprocess_transformer(text: str) -> str:
    """
    Minimal preprocessing for BERT input.

    Only fixes encoding artefacts and removes bare URLs.
    All other content (casing, punctuation, numbers) is preserved so that
    BERT's pre-trained contextual representations remain intact.

    Parameters
    ----------
    text : str
        Raw post text.

    Returns
    -------
    str
        Lightly cleaned text.
    """
    text = str(text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"http\S+|www\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ── Evaluation ────────────────────────────────────────────────────────────────

# Module-level stores — populated by evaluate() and used by comparison plots
ALL_RESULTS: dict = {}
ALL_CMS:     dict = {}


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Print a full classification report and store per-class metrics.

    Results are stored in the module-level ALL_RESULTS and ALL_CMS dicts
    so that comparison plots in phase2 can access them.

    Parameters
    ----------
    name   : str        Model name used as the dict key.
    y_true : np.ndarray Ground-truth labels.
    y_pred : np.ndarray Predicted labels.

    Returns
    -------
    dict
        Metrics dict with keys: acc, macro_f1, {Class}_f1/p/r for each class.
    """
    rep = classification_report(
        y_true, y_pred, target_names=LABEL_ORDER, output_dict=True
    )
    print(f"\n{'─' * 55}")
    print(f"  {name}")
    print(f"{'─' * 55}")
    print(classification_report(y_true, y_pred, target_names=LABEL_ORDER, digits=3))

    metrics = {
        "acc":      accuracy_score(y_true, y_pred),
        "macro_f1": rep["macro avg"]["f1-score"],
        **{f"{c}_f1": rep[c]["f1-score"]  for c in LABEL_ORDER},
        **{f"{c}_p":  rep[c]["precision"] for c in LABEL_ORDER},
        **{f"{c}_r":  rep[c]["recall"]    for c in LABEL_ORDER},
    }

    ALL_RESULTS[name] = metrics
    ALL_CMS[name]     = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)

    return metrics
