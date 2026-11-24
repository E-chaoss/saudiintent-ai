"""Checks that the data every task depends on is still what the team agreed on.
They run in seconds and need no GPU:   python -m pytest tests/
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "marbert"))
from normalize import normalize_ar                      # noqa: E402
from labels import INTENTS                              # noqa: E402
from data import class_weights, load_split, search_grid  # noqa: E402

TRAIN = ROOT / "data/processed/train.csv"
VAL = ROOT / "data/processed/validation.csv"
TEST = ROOT / "data/saudi_test/saudi_test_FINAL.csv"
STYLES = {"MSA", "Saudi Dialect", "Spelling Mistakes", "Arabic-English Code-switching"}


def test_labels_are_the_8_intents_in_every_file():
    assert len(INTENTS) == 8 and len(set(INTENTS)) == 8
    for path in (TRAIN, VAL, TEST):
        assert set(pd.read_csv(path)["intent"]) == set(INTENTS), path.name


def test_saudi_test_set_shape():
    t = pd.read_csv(TEST)
    assert list(t.columns) == ["id", "base_request_id", "text", "intent", "style"]
    assert len(t) == 480 and t["id"].is_unique
    assert set(t["style"]) == STYLES
    assert (t["intent"].value_counts() == 60).all()
    assert (t["style"].value_counts() == 120).all()
    assert t["base_request_id"].nunique() == 120
    assert (t.groupby("base_request_id")["style"].nunique() == 4).all()


def test_no_overlap_between_train_validation_and_test():
    norm = lambda p: set(pd.read_csv(p)["text"].map(normalize_ar))  # noqa: E731
    tr, va, te = norm(TRAIN), norm(VAL), norm(TEST)
    assert not tr & va, "train and validation share messages"
    assert not (tr | va) & te, "the Saudi test set shares messages with training data"


def test_marbert_data_helpers():
    texts, labels = load_split("data/processed/train.csv")
    assert len(texts) == len(labels) == len(pd.read_csv(TRAIN))
    w = class_weights(labels)
    assert w.argmax() == INTENTS.index("cancel_order")   # the smallest class gets the biggest weight
    assert len(search_grid({"learning_rates": [2e-5, 3e-5, 5e-5], "epochs": [3, 4]})) == 6
