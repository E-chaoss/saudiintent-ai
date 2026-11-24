# Data attribution and licence

The training and validation data in `data/` are derived from the
**Bitext Customer Support LLM Chatbot Training Dataset** by Bitext Innovations
(https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset),
licensed under the Community Data License Agreement – Sharing, Version 1.0
(CDLA-Sharing-1.0, full text in `LICENSE-DATA.txt`, also at https://cdla.dev/sharing-1-0/).

Changes made by team G21 (Samsung Innovation Campus × Misk, 2026):
selection of 8 of the 27 intents; removal of exact and near-duplicate messages;
replacement of template placeholders with random values; translation of the
`instruction` text into Modern Standard Arabic with an LLM; manual corrections and
relabelling of 23 rows (`data/interim/manual_fixes.csv`); wording rules in
`src/fix_translations.py`; Arabic normalization; a train/validation split.

As required by CDLA-Sharing-1.0, the derived data files are shared under the same licence.
The code in `src/` is the team's own work.
