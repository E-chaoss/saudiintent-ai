# How the translation was produced

- Model: a general-purpose large language model, used through its chat interface, on 23–24 Nov 2026.
- Input: `data/interim/bitext_selected_en.csv` (1,570 rows), sent per intent in batches of up to 100 rows as JSON `[{"id": ..., "text_en": ...}]`.
- Instruction given with every batch: the 8 rules in `docs/task1/translation_guidelines.md`, plus the intent name of the batch and its definition from `docs/intents.md`, so the model knew which meaning had to be preserved.
- Output: JSON `[{"id": ..., "text_ar": ...}]`, saved to `data/interim/tr_out/<intent>.json`. Ids were checked to match the input one-to-one (1,570/1,570).
- The raw model output is kept unchanged in the column `text_ar_llm` of `data/interim/bitext_ar.csv`; all later corrections are in `text_ar` and are reproducible from `src/data_prep/fix_translations.py` and `data/interim/manual_fixes.csv`.
- To redo the translation with another LLM, send the same batches with the same instruction and save the output in the same JSON format; the rest of the pipeline does not change.
