# AGENTS.md: SaudiIntent AI

Rules for AI coding agents **and** team members. Read this before planning, changing or validating anything.

**The project:** classify Arabic customer-support messages into 8 intents, and measure how well three
models (TF-IDF + Logistic Regression, a zero-shot LLM, fine-tuned MARBERTv2) handle Saudi messages in
4 styles: MSA, Saudi dialect, spelling mistakes, Arabic-English code-switching.
Team Tech Minds, Samsung Innovation Campus × Misk, 2026.

## 1. Commands (from the repo root)

```bash
pip install -r requirements.txt            # base (data, TF-IDF, tests)
pip install -r requirements-marbert.txt    # + PyTorch / Transformers (Task 4, needs a GPU to train)
python -m pytest -q tests/                 # data checks: run before and after every change
python src/marbert/train.py                # Task 4 training (GPU)
python src/marbert/predict.py --text "وين وصل طلبي؟"
```

## 2. Where things are

| Path | What | Task |
|---|---|---|
| `src/normalize.py` | `normalize_ar()`: the ONE Arabic normalization for every model and the demo | shared |
| `src/labels.py` | The 8 intents in a fixed order (= label ids) | shared |
| `src/data_prep/` | Data preparation and translation pipeline | 1 |
| `data/processed/` | `train.csv`, `validation.csv` | 1 |
| `data/saudi_test/` | `saudi_test_FINAL.csv`, the 480-message test set | 2 |
| `src/baselines/` | TF-IDF + LR and zero-shot LLM | 3 |
| `src/marbert/` | MARBERTv2 fine-tuning and prediction | 4 |
| `src/evaluation/` | Shared evaluation of all models | 5 |
| `app/` | Web demo | 6 |
| `results/<model>/` | Small result files (committed) | 3–5 |
| `models/` | Model weights (NOT committed) | 3–4 |
| `docs/decisions/` | ADRs: decisions, with status and reasons | all |

## 3. Source of truth (higher wins)

1. Task plan agreed by the team, and the ADRs in `docs/decisions/` (accepted ones)
2. `docs/intents.md` (intent definitions and decision rules) and `src/labels.py`
3. `docs/saudi_test_set_guide.md` and `docs/predictions_format.md`
4. Task READMEs and code
5. Notebooks and experiments (lowest)

If two sources disagree, follow the higher one and **report the conflict. Do not silently resolve it.**

## 4. Rules

**Data and evaluation**
- **Never train on, tune on, or choose settings with the Saudi test set.** It is only for final predictions and Task 5.
- Do not change `data/saudi_test/saudi_test_FINAL.csv`. Its columns and style names are the reference.
  The review is recorded in `saudi_test_review.csv`. Status: review pending; frozen after review (ADR 0002).
- Do not edit `data/processed/` or `data/interim/` by hand. They are produced by `src/data_prep/`.
- Every model, the evaluation and the demo call `normalize_ar()` and use `src/labels.py`.
  Changing either one needs a new ADR and a re-run of every model.
- Choose settings by **validation macro-F1**. Report metrics for each of the 4 styles, not only overall.
- Predictions files follow `docs/predictions_format.md`.

**Scope**
- Do only the task asked. No unrelated refactors, renames or "improvements".
- Do not add a new library, model provider or service without asking and recording it in an ADR.
- Do not change status words such as "proposed", "pending", "accepted", "frozen". They have meaning here.
- When something is not decided, say so and state your assumption. Do not invent requirements.

**Security and licences**
- Never commit secrets (API keys, Hugging Face tokens). Use `.env` (see `.env.example`).
- Never commit model weights (`models/`, `*.safetensors`, `*.bin`, `*.pt`) or raw Bitext data.
- No real customer data. Test messages are written by the team.
- Data derived from Bitext stays under CDLA-Sharing-1.0 with `NOTICE.md`. Code is MIT.

## 5. Workflow

1. Read the relevant docs for the area you touch (§3).
2. Look at the current code and `git status`. Never overwrite uncommitted work you did not make.
3. State a plan before substantial changes.
4. Change only the requested scope.
5. Validate: run `python -m pytest -q tests/`, and run the code you changed. "It should work" is not validation.
6. Review the full diff: only intended files, no secrets, no weights, no scratch files.
7. Report: files changed, how it was tested, metrics before/after if a model changed, assumptions, what is left.

**Do not commit or push unless explicitly asked.** Team branches: see `CONTRIBUTING.md`.

## 6. Other agent files

`CLAUDE.md` only points here. Do not copy these rules into other files; link to this one.
