"""Zero-shot LLM intent classifier (Task 3) using OpenAI."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from sklearn.metrics import classification_report, f1_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from labels import INTENTS
from normalize import normalize_ar

VALID_INTENTS = set(INTENTS)

IntentLabel = Literal[
    "track_order",
    "cancel_order",
    "change_order",
    "get_refund",
    "track_refund",
    "payment_issue",
    "complaint",
    "contact_human_agent",
]


class IntentPrediction(BaseModel):
    intent: IntentLabel


SYSTEM_PROMPT = """You are an Arabic customer-support intent classifier for a Saudi e-commerce store.
Classify each normalized customer message into EXACTLY ONE of the following 8 intents:

1. track_order: Know the status, location, or arrival time of an existing order.
2. cancel_order: Cancel an order that was placed.
3. change_order: Modify the items of an existing order (add, remove, edit a product). NOT the shipping address.
4. get_refund: Ask to get money back.
5. track_refund: Check the status of a refund already requested.
6. payment_issue: A problem while paying (declined card, failed payment, charged incorrectly).
7. complaint: File a complaint or claim about the company or service.
8. contact_human_agent: Speak to a human / live agent.

Decision rules for borderline messages:
- If the customer asks to cancel AND get money back, choose `cancel_order` (the action requested first).
- If asking "where is my money?" after a refund was already requested, choose `track_refund`. If initiating a refund request ("I want my money back"), choose `get_refund`.
- If charged twice or card declined during payment, choose `payment_issue`. If expressing general dissatisfaction or filing a formal complaint about service/company, choose `complaint`.

Return ONLY the exact intent label."""


def get_client() -> OpenAI:
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError("LLM_API_KEY is missing in .env")
    return OpenAI(api_key=api_key)


def classify_text(client: OpenAI, text: str, model: str) -> str:
    clean_text = normalize_ar(str(text))
    if not clean_text.strip():
        return "invalid"

    try:
        response = client.beta.chat.completions.parse(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": clean_text},
            ],
            response_format=IntentPrediction,
        )
        parsed = response.choices[0].message.parsed
        if parsed and parsed.intent in VALID_INTENTS:
            return parsed.intent
    except Exception as exc:
        print(f"[WARN] {exc}", file=sys.stderr)

    return "invalid"


def run_validation(client: OpenAI, model: str, limit: int | None = None) -> None:
    df = pd.read_csv(REPO_ROOT / "data" / "processed" / "validation.csv")
    if limit:
        df = df.head(limit).copy()

    df["predicted_intent"] = [classify_text(client, t, model) for t in df["text"]]
    invalid_count = int((df["predicted_intent"] == "invalid").sum())
    macro_f1 = f1_score(df["intent"], df["predicted_intent"], labels=INTENTS, average="macro", zero_division=0)
    report = classification_report(df["intent"], df["predicted_intent"], labels=INTENTS, output_dict=True, zero_division=0)

    out_dir = REPO_ROOT / "results" / "zero_shot_llm"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model": model,
        "split": "validation",
        "num_samples": len(df),
        "invalid_predictions": invalid_count,
        "macro_f1": round(float(macro_f1), 4),
        "per_class": {k: report[k] for k in INTENTS},
    }
    (out_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Validation macro-F1 = {macro_f1:.4f} | invalid = {invalid_count}")


def run_saudi_test(client: OpenAI, model: str) -> None:
    df = pd.read_csv(REPO_ROOT / "data" / "saudi_test" / "saudi_test_FINAL.csv")
    preds = [classify_text(client, t, model) for t in df["text"]]

    out_df = pd.DataFrame(
        {
            "id": df["id"],
            "predicted_intent": preds,
            "confidence": "",
        }
    )

    out_dir = REPO_ROOT / "results" / "zero_shot_llm"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "predictions_saudi_test.csv"
    out_df.to_csv(out_file, index=False, encoding="utf-8")
    print(f"{len(out_df)} predictions have been saved to {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    client = get_client()
    if args.split == "val":
        run_validation(client, args.model, limit=args.limit)
    else:
        run_saudi_test(client, args.model)


if __name__ == "__main__":
    main()
