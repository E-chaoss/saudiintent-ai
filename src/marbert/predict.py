"""Task 4: use the fine-tuned MARBERTv2 on new messages.

In Python (e.g. the Task 6 demo), with src/ on the import path:
    from marbert.predict import IntentPredictor
    predictor = IntentPredictor("models/marbert/best")   # or a Hugging Face Hub id
    predictor.predict("وين وصل طلبي؟")
    # {'intent': 'track_order', 'confidence': 0.97, 'scores': {...all 8 intents...}}

From the command line (repo root):
    python src/marbert/predict.py --text "وين وصل طلبي؟"
    python src/marbert/predict.py --input data/saudi_test/saudi_test_FINAL.csv \
        --output results/marbert/predictions_saudi_test.csv

The CSV output follows docs/predictions_format.md (id, predicted_intent, confidence),
so Task 5 can evaluate every model the same way.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from normalize import normalize_ar   # noqa: E402  same normalization as in training
from labels import INTENTS           # noqa: E402

DEFAULT_MODEL = ROOT / "models" / "marbert" / "best"


class IntentPredictor:
    def __init__(self, model_path=DEFAULT_MODEL, device=None, max_length=128):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path)).to(self.device)
        self.model.eval()
        info = Path(model_path) / "training_info.json"
        if info.exists():                                  # use the same max_length as training
            max_length = json.loads(info.read_text(encoding="utf-8"))["config"]["max_length"]
        self.max_length = max_length
        # Safety check: the model's label order must match src/labels.py.
        model_labels = [self.model.config.id2label[i] for i in range(len(INTENTS))]
        if model_labels != INTENTS:
            raise ValueError(f"Model labels {model_labels} do not match src/labels.py {INTENTS}")

    @torch.no_grad()
    def predict_batch(self, texts, batch_size=32):
        results = []
        for start in range(0, len(texts), batch_size):
            chunk = [normalize_ar(t) for t in texts[start:start + batch_size]]
            enc = self.tokenizer(chunk, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
            probs = torch.softmax(self.model(**enc).logits, dim=-1).cpu()
            for p in probs:
                best = int(p.argmax())
                results.append({
                    "intent": INTENTS[best],
                    "confidence": round(float(p[best]), 4),
                    "scores": {label: round(float(p[i]), 4) for i, label in enumerate(INTENTS)},
                })
        return results

    def predict(self, text):
        return self.predict_batch([text])[0]


def main():
    import pandas as pd

    ap = argparse.ArgumentParser(description="Predict intents with the fine-tuned MARBERTv2.")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="model folder or Hugging Face Hub id")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="one message")
    group.add_argument("--input", help="CSV with columns id,text")
    ap.add_argument("--output", help="where to write predictions (with --input)")
    args = ap.parse_args()

    predictor = IntentPredictor(args.model)
    if args.text:
        print(json.dumps(predictor.predict(args.text), ensure_ascii=False, indent=2))
        return

    df = pd.read_csv(args.input)
    if not {"id", "text"} <= set(df.columns):
        sys.exit(f"{args.input} needs the columns id and text")
    preds = predictor.predict_batch(df["text"].tolist())
    out = pd.DataFrame({"id": df["id"],
                        "predicted_intent": [p["intent"] for p in preds],
                        "confidence": [p["confidence"] for p in preds]})
    output = Path(args.output or "predictions.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, encoding="utf-8")
    print(f"{len(out)} predictions -> {output}")


if __name__ == "__main__":
    main()
