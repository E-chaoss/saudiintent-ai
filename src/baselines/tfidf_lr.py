"""TF-IDF + Logistic Regression intent classifier (Task 3)."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from labels import INTENTS
from normalize import normalize_ar


def build_pipeline(c_value: float, class_weight: str) -> Pipeline:
    cw = None if class_weight == "none" else class_weight
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))),
            ("clf", LogisticRegression(C=c_value, max_iter=3000, class_weight=cw)),
        ]
    )


def train_and_evaluate(c_value: float, class_weight: str) -> None:
    train_df = pd.read_csv(REPO_ROOT / "data" / "processed" / "train.csv")
    val_df = pd.read_csv(REPO_ROOT / "data" / "processed" / "validation.csv")

    x_train = [normalize_ar(str(t)) for t in train_df["text"]]
    y_train = train_df["intent"]

    x_val = [normalize_ar(str(t)) for t in val_df["text"]]
    y_val = val_df["intent"]

    pipeline = build_pipeline(c_value=c_value, class_weight=class_weight)
    pipeline.fit(x_train, y_train)

    val_preds = pipeline.predict(x_val)
    macro_f1 = f1_score(y_val, val_preds, labels=INTENTS, average="macro", zero_division=0)
    report = classification_report(y_val, val_preds, labels=INTENTS, output_dict=True, zero_division=0)

    model_dir = REPO_ROOT / "models" / "tfidf_lr"
    results_dir = REPO_ROOT / "results" / "tfidf_lr"
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_dir / "model.joblib")

    metrics = {
        "model": "tfidf_lr_char_wb_2_5",
        "C": c_value,
        "class_weight": None if class_weight == "none" else class_weight,
        "split": "validation",
        "num_samples": len(val_df),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": {k: report[k] for k in INTENTS},
    }
    (results_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Validation macro-F1 = {macro_f1:.4f}")


def predict_saudi_test() -> None:
    model_file = REPO_ROOT / "models" / "tfidf_lr" / "model.joblib"
    if not model_file.exists():
        raise FileNotFoundError("Enable 'train' mode first to save the model.")

    pipeline = joblib.load(model_file)
    test_df = pd.read_csv(REPO_ROOT / "data" / "saudi_test" / "saudi_test_FINAL.csv")

    x_test = [normalize_ar(str(t)) for t in test_df["text"]]
    probs = pipeline.predict_proba(x_test)
    classes = list(pipeline.classes_)

    best_idx = np.argmax(probs, axis=1)
    out_df = pd.DataFrame(
        {
            "id": test_df["id"],
            "predicted_intent": [classes[i] for i in best_idx],
            "confidence": [round(float(probs[r, c]), 4) for r, c in enumerate(best_idx)],
        }
    )

    out_dir = REPO_ROOT / "results" / "tfidf_lr"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_dir / "predictions_saudi_test.csv", index=False, encoding="utf-8")
    print(f"{len(out_df)} predictions have been saved to results/tfidf_lr/predictions_saudi_test.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict"], required=True)
    parser.add_argument("--C", type=float, default=10.0)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
    args = parser.parse_args()

    if args.mode == "train":
        train_and_evaluate(c_value=args.C, class_weight=args.class_weight)
    else:
        predict_saudi_test()


if __name__ == "__main__":
    main()
