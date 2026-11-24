# SaudiIntent AI

Arabic customer-support intent classification (8 intents), tested on MSA, Saudi dialect,
spelling mistakes and Arabic-English code-switching.

Team G21: Samsung Innovation Campus × Misk, 2026.

## Requirements

- Python 3.10 or newer
- Git
- A GPU only for fine-tuning deep-learning models (Google Colab works)

## Setup

```bash
git clone https://github.com/E-chaoss/saudiintent-ai.git
cd saudiintent-ai

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

For deep-learning work (PyTorch and Transformers):

```bash
pip install -r requirements-marbert.txt
```

## Check the environment

```bash
python -m pytest -q tests/
```

All tests should pass. They only read the data files and need no GPU.

## API keys and tokens

Copy `.env.example` to `.env` and fill in your own values. `.env` is ignored by Git, so keys are never committed.

## Folder layout

```text
src/            code (src/normalize.py and src/labels.py are shared)
data/           datasets
notebooks/      notebooks (e.g. Google Colab)
results/        small result files
models/         model weights (ignored by Git)
tests/          checks for the data
docs/           documentation
app/            web prototype
```

## Working together

See [CONTRIBUTING.md](CONTRIBUTING.md) for branches and pull requests.

## Licence

- Code: [MIT](LICENSE).
- Data: derived from the Bitext dataset, CDLA-Sharing-1.0. See [NOTICE.md](NOTICE.md) and [LICENSE-DATA.txt](LICENSE-DATA.txt).
