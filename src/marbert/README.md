# Task 4: MARBERTv2 fine-tuning

| | |
|---|---|
| **Owner** | Hussam Jamal Khan |
| **Status** | Trained on Google Colab (Tesla T4, about 3 minutes); Saudi test predictions delivered to Task 5 |
| **Inputs** | `data/processed/train.csv`, `data/processed/validation.csv` (Task 1) |
| **Outputs** | Fine-tuned model, validation results, best settings, predictions on the Saudi test set for Task 5 |

## Files

| File | What it does |
|---|---|
| `config.yaml` | All settings: model, learning rates, epochs, batch size, seed. Change settings here, not in code |
| `data.py` | Reads the CSV files, applies `normalize_ar()`, turns intents into label ids. No PyTorch, so it is tested in `tests/` |
| `train.py` | Fine-tunes MARBERTv2 for every setting, scores each on validation macro-F1, keeps the best model |
| `predict.py` | Loads the model and predicts: one message, or a whole CSV (for Task 5) |
| `../../notebooks/task4_marbert_colab.ipynb` | Runs all of the above on a free Colab GPU |

## How to run

On Google Colab, open the notebook and run the cells in order. Or on any machine with a GPU, from the repo root:

```bash
pip install -r requirements-marbert.txt
python src/marbert/train.py          # 6 runs: lr 2e-5 / 3e-5 / 5e-5  x  3 / 4 epochs
python src/marbert/predict.py --text "وين وصل طلبي؟"
python src/marbert/predict.py --input data/saudi_test/saudi_test_FINAL.csv \
    --output results/marbert/predictions_saudi_test.csv
```

## How it works (short version)

1. **MARBERTv2** is a BERT model already trained on a huge amount of Arabic tweets, dialects included.
   It already "understands" Arabic; we only teach it our 8 intents.
2. `from_pretrained(..., num_labels=8)` loads MARBERTv2 and adds a small new **classification head**:
   one layer that turns the sentence representation into 8 scores, one per intent.
3. **Fine-tuning**: for each batch of 16 messages the model predicts, the loss (cross-entropy) measures
   how wrong it was, `loss.backward()` computes how to change every weight, and the optimizer (AdamW)
   changes them a little. The learning rate controls "a little".
4. **Warm-up + linear decay**: the learning rate starts at 0, rises during the first 10% of steps, then
   falls to 0. This keeps the pre-trained knowledge from being damaged at the start.
5. **Search**: each of the 6 settings trains a fresh copy from the same seed, so the only difference
   between runs is the setting. The best macro-F1 **on validation** wins; a tie keeps the earlier setting.
6. **Saudi test set**: only `predict.py` reads it, after the search. It is never used to choose anything.

## Results

From the search in `results/marbert/` (validation set, 241 messages):

| | Value |
|---|---|
| Best learning rate | 2e-5 |
| Best epochs | 3 |
| Validation macro-F1 | 0.9922 |
| Validation accuracy | 0.9917 (2 errors) |

- All 6 settings reached the same validation macro-F1 (0.9922), so the tie rule kept the first and lightest
  setting. Higher learning rates and more epochs only lowered the training loss (0.30 down to 0.02).
  The validation set comes from the same translated MSA data as training, so it cannot separate the settings;
  the Saudi test set (Task 5) is where the models differ.
- Both validation errors are `track_refund` messages predicted as `get_refund`, the pair that
  `docs/intents.md` lists as easily confused.
- Predictions on the Saudi test set: `results/marbert/predictions_saudi_test.csv` (480 rows,
  [shared format](../../docs/predictions_format.md)). Scoring them is Task 5's job.
- The model itself (about 650 MB) is not in Git. It is saved on Google Drive.

## Open decisions

- **Class weights** for `cancel_order` (fewer training rows): `class_weights: false` until the team decides.
  See [ADR 0003](../../docs/decisions/0003-marbert-training-setup.md).
- **Predictions file format** shared with Task 5: proposed in [predictions_format.md](../../docs/predictions_format.md).

## Suggestion: confidence-based fallback to a human agent

> **Status: proposed** by the Task 4 owner, for the team to discuss. Nothing here is implemented.

**Observation.** On a first try with Saudi-dialect messages, the fine-tuned model chose the right intent
but with low confidence (for example 0.35 to 0.45), while it is confident on the MSA validation set. The model
was trained on translated MSA, so dialect messages are harder for it. The Saudi test set (Task 5) will show
how often this happens.

**Suggestion.** Use the model's `confidence` (already returned by `predict.py` and written in the
predictions file) to decide what the system does:

| Confidence | What the system does |
|---|---|
| High (at or above a threshold) | Route the message to the predicted intent |
| Low (below the threshold) | Do not guess: hand the message to a human agent (`contact_human_agent`) |

**How the threshold could be chosen.** Not on the Saudi test set. Options to discuss: a fixed value
agreed by the team, or a value chosen on validation data, then reported on the test set as
"share of messages handed to a human" next to accuracy on the messages the model kept.

**Why it helps.** A wrong intent sends the customer down the wrong path; a hand-off to a person is
slower but safe. In the demo it also shows that the system knows when it is unsure.
