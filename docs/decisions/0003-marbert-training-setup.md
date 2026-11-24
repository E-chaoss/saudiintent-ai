# ADR 0003: MARBERTv2 training setup (Task 4)

Status: Accepted for the search grid and fixed settings on 24 November 2026.
**Class weights: proposed, team decision pending.**

## Decision

- Model: `UBC-NLP/MARBERTv2` with a new 8-class classification head.
- Search grid (from the task plan): learning rate 2e-5, 3e-5, 5e-5 x 3 and 4 epochs = 6 runs.
- Fixed for every run: batch size 16, max length 128, AdamW with weight decay 0.01, 10% linear warm-up
  then linear decay, gradient clipping 1.0, seed 42.
- Each run starts from the same seed and is scored after its last epoch.
- The best run is chosen by **macro-F1 on the validation set**; a tie keeps the earlier setting in the grid.
- Only the best model is saved, to `models/marbert/best/` (not committed).

## Open: class weights

`cancel_order` has 81 training rows; the other intents have 117–140. The Task 1 sanity baseline already
uses `class_weight="balanced"`. Options:

1. **No class weights** (current default, `class_weights: false`): simplest, standard setting.
2. **Balanced class weights** (`class_weights: true`): rare classes count more in the loss, matching
   what the baseline does.

The team should choose one setting for all models where it applies, and record the choice here.

## Rationale

Macro-F1 gives each intent the same importance, so a model cannot look good by being right only on the
large classes. Fixed seeds and one config file make every number in the report reproducible.
