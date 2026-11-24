# Task 3: TF-IDF + Logistic Regression, and zero-shot LLM

| | |
|---|---|
| **Owner** | ___ |
| **Status** | Not started in this repo |
| **Inputs** | `data/processed/train.csv`, `data/processed/validation.csv`, `src/normalize.py`, `src/labels.py` |
| **Outputs** | `results/tfidf_lr/` and `results/zero_shot_llm/`: validation results and `predictions_saudi_test.csv` in the [shared format](../../docs/predictions_format.md) |

## What to build

**A. TF-IDF + Logistic Regression**
- Text to numbers with TF-IDF, then Logistic Regression for the 8 intents.
- Evaluate on validation; save the model and the results.
- Starting point: `src/data_prep/baseline_tfidf.py` (Task 1 sanity check).

**B. Zero-shot LLM**
- Choose an LLM; write one fixed prompt that lists the 8 intents (definitions in `docs/intents.md`).
- The model must answer exactly one intent per message; anything else is written as `invalid`.
- API keys go in `.env` (copy `.env.example`), never in code or notebooks.

## Rules
- Call `normalize_ar()` on every text. Use the intent names from `src/labels.py`.
- Choose settings on validation only. The Saudi test set is only for the final predictions.
