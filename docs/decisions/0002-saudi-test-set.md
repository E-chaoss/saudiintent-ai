# ADR 0002: The Saudi test set

Status: Accepted on 24 November 2026. **The test set is complete; the second-member review is pending.**

## Decision

- `data/saudi_test/saudi_test_FINAL.csv` (480 messages: 8 intents x 15 base requests x 4 styles) is
  the only data used for the final comparison of the models.
- The file is kept exactly as the team delivered it. Its columns and style names are the reference;
  [the guide](../saudi_test_set_guide.md) was updated to match the file, not the other way round.
- The review is recorded in a separate file, `data/saudi_test/saudi_test_review.csv`, so the test file
  itself does not change.
- **After the review is recorded, the set is frozen.** Any correction found by the review is made once,
  before freezing, and listed below.
- The test set is never used for training, for choosing hyper-parameters, or for choosing between
  settings of the same model. It is only used for the final predictions (Tasks 3 and 4) and the
  evaluation (Task 5).

## Corrections after review

None yet (review pending).

## Rationale

If any decision is made by looking at test results, the test no longer measures how the models
handle Saudi messages they have never seen, and the comparison in the final report is not valid.
