# SaudiIntent AI

**Arabic customer-support intent classification, tested on how Saudis actually write.**

Customers write to support chats in Modern Standard Arabic, in Saudi dialect, with spelling mistakes,
and mixing in English words ("ابي اسوي cancel للطلب"). SaudiIntent AI trains models to recognise **8
customer intents** and measures how well three approaches handle each of these 4 styles:

| Approach | Type |
|---|---|
| TF-IDF + Logistic Regression | Traditional machine learning |
| Zero-shot LLM | Large language model, no training |
| MARBERTv2 fine-tuned | Arabic deep learning (BERT trained on dialects) |

Team G21: Samsung Innovation Campus × Misk, 2026.

## The 8 intents

`track_order`, `cancel_order`, `change_order`, `get_refund`, `track_refund`, `payment_issue`,
`complaint`, `contact_human_agent`. Definitions, examples and decision rules: [docs/intents.md](docs/intents.md).

## Data

| Set | Size | Source |
|---|---|---|
| Training / validation | 965 / 241 messages (80/20) | [Bitext](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) customer-support data, 8 intents, translated to MSA with an LLM, cleaned and reviewed ([report](docs/task1/data_preparation_report.md)) |
| Saudi test set | 480 messages | Written by the team: 120 requests × 4 styles ([guide](docs/saudi_test_set_guide.md), [distribution](docs/saudi_test_distribution.md)). Never used for training |

## Project status

| Task | Folder | Owner | Status |
|---|---|---|---|
| 1. Data preparation and translation | [`src/data_prep/`](src/data_prep/) | ___ | ✅ Done |
| 2. Saudi test set | [`data/saudi_test/`](data/saudi_test/) | ___ | ✅ Written · ⏳ review pending |
| 3. TF-IDF + LR and zero-shot LLM | [`src/baselines/`](src/baselines/) | ___ | Not started in this repo |
| 4. MARBERTv2 fine-tuning | [`src/marbert/`](src/marbert/) | Hussam Jamal Khan | Code ready · training pending |
| 5. Evaluation and analysis | [`src/evaluation/`](src/evaluation/) | ___ | Not started in this repo |
| 6. Prototype and live demo | [`app/`](app/) | ___ | Not started in this repo |

## Results

To be filled in by Task 5: macro-F1 per model, overall and for each style.

| Model | Overall | MSA | Saudi Dialect | Spelling Mistakes | Code-switching |
|---|---|---|---|---|---|
| TF-IDF + LR | — | — | — | — | — |
| Zero-shot LLM | — | — | — | — | — |
| MARBERTv2 | — | — | — | — | — |

## Quick start

```bash
git clone https://github.com/<owner>/saudiintent-ai.git
cd saudiintent-ai
pip install -r requirements.txt
python -m pytest -q tests/              # checks the data (seconds, no GPU)
python src/data_prep/baseline_tfidf.py  # quick sanity baseline
```

MARBERTv2 needs a GPU: open [`notebooks/task4_marbert_colab.ipynb`](notebooks/task4_marbert_colab.ipynb) in Google Colab.

## Repository map

```text
saudiintent-ai/
├── AGENTS.md, CLAUDE.md      rules for AI agents and the team
├── CONTRIBUTING.md           branches, pull requests, checklist
├── src/
│   ├── normalize.py          shared Arabic normalization (every model uses it)
│   ├── labels.py             the 8 intents, fixed order
│   ├── data_prep/            Task 1
│   ├── baselines/            Task 3
│   ├── marbert/              Task 4
│   └── evaluation/           Task 5
├── app/                      Task 6
├── data/
│   ├── raw/                  Bitext original (download, not committed)
│   ├── interim/              Task 1 working files
│   ├── processed/            train.csv, validation.csv
│   └── saudi_test/           saudi_test_FINAL.csv
├── notebooks/                Colab notebooks
├── results/                  metrics and predictions per model
├── models/                   model weights (not committed)
├── tests/                    data checks
└── docs/                     intents, guides, reports, decisions (ADRs)
```

## Licence

- **Code:** [MIT](LICENSE).
- **Data:** derived from the Bitext dataset, shared under **CDLA-Sharing-1.0**. See [NOTICE.md](NOTICE.md) and [LICENSE-DATA.txt](LICENSE-DATA.txt).
