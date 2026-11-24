# Task 5: Model evaluation and results analysis

| | |
|---|---|
| **Owner** | ___ |
| **Status** | Not started in this repo |
| **Inputs** | `data/saudi_test/saudi_test_FINAL.csv` and each model's `results/<model>/predictions_saudi_test.csv` ([format](../../docs/predictions_format.md)) |
| **Outputs** | Evaluation report, comparison tables and charts, confusion matrices, error analysis, final model recommendation (in `results/evaluation/`) |

## What to build
- One evaluation script for all three models (join predictions to the test set by `id`).
- Accuracy, Precision, Recall and F1 (macro) overall **and for each of the 4 styles separately**.
  The style values are listed in `docs/saudi_test_set_guide.md`.
- Confusion matrices and charts; error analysis (which intents and styles fail, and why).
- Compare the models and recommend the best one for the demo (Task 6).

## Rules
- Do not change the test file. Its columns and style names are the reference.
- Count `invalid` predictions as errors.
