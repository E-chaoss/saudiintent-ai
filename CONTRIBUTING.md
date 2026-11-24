# Contributing (Tech Minds team)

## Setup

```bash
git clone https://github.com/<owner>/saudiintent-ai.git
cd saudiintent-ai
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q tests/                             # should pass before you start
```

Task 4 (MARBERTv2) needs a GPU: use `notebooks/task4_marbert_colab.ipynb` on Google Colab.

## Branches

Do not work directly on `main`. Each task has its own branch:

| Task | Branch | Folder |
|---|---|---|
| 1 Data preparation | `task1-data` | `src/data_prep/`, `data/processed/` |
| 2 Saudi test set (review) | `task2-test-review` | `data/saudi_test/` |
| 3 TF-IDF + LR, zero-shot LLM | `task3-baselines` | `src/baselines/`, `results/tfidf_lr/`, `results/zero_shot_llm/` |
| 4 MARBERTv2 | `task4-marbert` | `src/marbert/`, `results/marbert/` |
| 5 Evaluation | `task5-evaluation` | `src/evaluation/`, `results/evaluation/` |
| 6 Prototype | `task6-app` | `app/` |

```bash
git checkout -b task3-baselines
# ...work...
git add src/baselines results/tfidf_lr
git commit -m "Task 3: add TF-IDF + LR baseline"
git push -u origin task3-baselines
```

Then open a **Pull Request** to `main` on GitHub and ask one teammate to review it.

## Before you open a Pull Request

- [ ] `python -m pytest -q tests/` passes.
- [ ] Your code calls `normalize_ar()` and uses `src/labels.py`.
- [ ] You did not use the Saudi test set to choose settings.
- [ ] No API keys, tokens, model weights or raw data in the commit.
- [ ] Your task README is updated (status, how to run, results).
- [ ] Changes to shared files (`src/normalize.py`, `src/labels.py`, `docs/intents.md`, the test set) were agreed with the team first.

## Rules for everyone

The full rules (for people and AI agents) are in [AGENTS.md](AGENTS.md).
