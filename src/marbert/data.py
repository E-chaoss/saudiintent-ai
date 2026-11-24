"""Data helpers for Task 4 that do not need PyTorch (so they can be tested anywhere)."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]          # repo root
sys.path.insert(0, str(ROOT / "src"))
from normalize import normalize_ar                  # noqa: E402  (shared by every model)
from labels import INTENTS, LABEL2ID                # noqa: E402


def load_split(path):
    """Read train.csv or validation.csv and return (texts, label_ids).

    Every text goes through normalize_ar(), exactly like the other models and the demo.
    Stops with a clear error if a column or an intent is wrong, instead of training on bad data.
    """
    df = pd.read_csv(ROOT / path)
    missing = {"text", "intent"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s): {sorted(missing)}")
    unknown = sorted(set(df["intent"]) - set(INTENTS))
    if unknown:
        raise ValueError(f"{path} has intents that are not in src/labels.py: {unknown}")
    texts = [normalize_ar(t) for t in df["text"]]
    labels = [LABEL2ID[i] for i in df["intent"]]
    return texts, labels


def class_weights(labels, n_classes=len(INTENTS)):
    """Weight per class = n_samples / (n_classes * count), the same rule as scikit-learn's
    class_weight="balanced". Rare classes get a bigger weight in the loss."""
    counts = np.bincount(labels, minlength=n_classes)
    if (counts == 0).any():
        raise ValueError("A class has no training rows; cannot compute class weights.")
    return len(labels) / (n_classes * counts)


def search_grid(cfg):
    """All (learning_rate, epochs) pairs to try, in a fixed order."""
    lrs = [float(x) for x in cfg["learning_rates"]]
    eps = [int(x) for x in cfg["epochs"]]
    return list(itertools.product(lrs, eps))
