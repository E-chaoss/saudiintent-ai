# Task 1: Data preparation and translation

Install the requirements with `pip install -r requirements.txt`, then run the steps from the **repo root**:

1. `python src/data_prep/prepare_bitext.py --src <Bitext CSV>`
   - Selects the 8 intents, fills `{{placeholders}}`, removes exact and near-duplicates (ratio ≥ 90 within an intent), and samples up to 200 per intent.
   - Output: `data/interim/bitext_selected_en.csv`.
   - Download Bitext from https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
2. Translation to MSA with an LLM. See `docs/task1/translation_prompt.md` and `docs/task1/translation_guidelines.md`.
   - Output: `data/interim/tr_out/*.json`.
3. `python src/data_prep/postprocess_translation.py`
   - Runs the automatic checks, then applies the quality-check corrections (`src/data_prep/fix_translations.py` and `data/interim/manual_fixes.csv`).
   - Normalizes the text (`src/normalize.py`), checks label conflicts and removes duplicates.
   - Output: `data/interim/bitext_ar.csv`.
4. `python src/data_prep/make_review_sheet.py`
   - Picks 50 rows per intent for human review and writes `data/interim/translation_review_sheet.csv`.
   - It refuses to overwrite a sheet that is already filled.
5. After the review: `python src/data_prep/apply_review_and_split.py`
   - Produces `data/processed/train.csv` and `data/processed/validation.csv`.
   - The split is 80/20, stratified by intent, and done by near-duplicate group.
6. `python src/data_prep/baseline_tfidf.py`
   - Runs the sanity baseline and a quick Saudi-dialect check.

Other files:
- **Saudi test set:** in `data/saudi_test/`, written by the team (see `docs/saudi_test_set_guide.md`) and used only for the final test.
- **Every model and the demo:** must call `normalize_ar()` from `src/normalize.py`.
- **Reports:** the data preparation report is `docs/task1/data_preparation_report.md`, and the text for the final report is `docs/task1/report_sections_task1.md`.
- **Data licence:** CDLA-Sharing-1.0 (Bitext). See `NOTICE.md`.
