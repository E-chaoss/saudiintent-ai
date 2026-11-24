# Predictions file format (Tasks 3 and 4 → Task 5)

> **Status: proposed.** To be confirmed by the Task 5 (evaluation) owner.

Every model writes its predictions on the Saudi test set in the same format, so one evaluation
script can compare all three models.

**Location:** `results/<model>/predictions_saudi_test.csv`
(for example `results/marbert/`, `results/tfidf_lr/`, `results/zero_shot_llm/`)

| Column | Type | Meaning |
|---|---|---|
| `id` | text | Message id from `data/saudi_test/saudi_test_FINAL.csv` (e.g. `SA-001`) |
| `predicted_intent` | text | One of the 8 intents in `src/labels.py`, spelled exactly the same |
| `confidence` | number 0–1 | The model's probability for the predicted intent. Leave empty if the model has none (e.g. a zero-shot LLM) |

Rules:
- One row per test message: 480 rows, every `id` once.
- No gold labels and no `style` column in this file: Task 5 joins them from the test file by `id`.
- UTF-8, comma-separated, with a header row.
- If the LLM answers something that is not one of the 8 intents, write it as `invalid`
  so Task 5 can count it as an error instead of the script crashing.
