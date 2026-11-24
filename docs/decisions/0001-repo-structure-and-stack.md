# ADR 0001: Repository structure and stack

Status: Accepted by Hussam Jamal Khan on 24 November 2026.

## Decision

- One repository for the whole project. Each task has its own folder (see the README map).
- Python 3.10+. Base libraries in `requirements.txt` (pandas, numpy, scikit-learn, rapidfuzz, PyYAML, pytest);
  deep-learning libraries in `requirements-marbert.txt` (PyTorch, Transformers), so tasks that do not
  need a GPU stay light.
- Shared code lives in `src/`: `normalize.py` (one Arabic normalization for every model and the demo)
  and `labels.py` (the 8 intents in one fixed order, which defines the label ids).
- Task 1 was moved into `src/data_prep/` without changes to its logic. Only the lines that find the
  repo root and `normalize.py` were changed, because the scripts moved one folder deeper.
- Data in `data/` is committed (it is small). Model weights are **not** committed: they go to the
  Hugging Face Hub or Google Drive. Raw Bitext data is downloaded, not committed.
- Code licence: MIT. Data licence: CDLA-Sharing-1.0, as required by Bitext (see `NOTICE.md`).

## Rationale

New team members and AI agents should find every task in a predictable place, and every model must
see exactly the same text and label ids, otherwise their results cannot be compared.

## Consequences

Changing `normalize.py` or `labels.py` affects every model. Such a change needs a new ADR and a re-run
of every model.
