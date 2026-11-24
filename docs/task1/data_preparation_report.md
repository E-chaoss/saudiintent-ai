# Task 1: Data Preparation and Translation Report (final)

## 1. Source data
- **Dataset:** Bitext Customer Support LLM Chatbot Training Dataset (`Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv`).
  - Size: 26,872 English messages across 27 intents.
  - Licence: CDLA-Sharing-1.0. See `NOTICE.md` and `LICENSE-DATA.txt`.
  - The messages are **synthetic**: Bitext generated them with natural-language-generation tools from seed texts.
- **Columns used:** only `instruction` (the customer message) and `intent`.
- **Intents (8):** track_order, cancel_order, change_order, get_refund, track_refund, payment_issue, complaint, contact_human_agent. Definitions and decision rules are in `docs/intents.md`.

## 2. Pipeline and counts

| Step | Script | Rows |
|---|---|---|
| Rows of the 8 intents in Bitext | `src/data_prep/prepare_bitext.py` | 7,983 |
| After removing exact duplicates within each intent (numbers, punctuation and case ignored) | same | 6,544 |
| After removing near-duplicates within each intent (rapidfuzz `fuzz.ratio` ≥ 90) | same | 4,167 |
| Sample for translation (up to 200 per intent, seed 42; cancel_order had only 170 distinct messages) | same | 1,570 |
| Translated to MSA by an LLM (method in `docs/task1/translation_prompt.md`) | `tr_out/*.json` | 1,570 |
| Failing the automatic checks (missing output, no Arabic, Latin words, changed numbers) | `src/data_prep/postprocess_translation.py` | 0 |
| Corrected after the quality check (15 rewritten, 8 relabelled) | `data/interim/manual_fixes.csv` | 23 |
| Changed by the wording rules (section 6) | `src/data_prep/fix_translations.py` | 429 |
| After normalization and removing duplicates created by translation | `src/data_prep/postprocess_translation.py` | 1,206 |
| Train / validation (80/20, stratified by intent, split by near-duplicate group) | `src/data_prep/apply_review_and_split.py` | 965 / 241 |

**Placeholders:** before translation, `{{Order Number}}` and `{{Refund Amount}}` were replaced with random realistic values: 6-digit order numbers and amounts of 50–2,000 SAR. "SAR" is written as ريال.

**Second deduplication:** sentences that differ only in their numbers, or only in the word رقم before an order number, count as duplicates.

**Raw output kept:** the raw LLM output is stored unchanged in the `text_ar_llm` column. 318 of the 1,206 final rows differ from it.

## 3. Final counts per intent

| Intent | Train | Validation | Total |
|---|---|---|---|
| contact_human_agent | 140 | 35 | 175 |
| track_refund | 136 | 34 | 170 |
| change_order | 131 | 33 | 164 |
| complaint | 122 | 30 | 152 |
| payment_issue | 119 | 30 | 149 |
| track_order | 119 | 29 | 148 |
| get_refund | 117 | 30 | 147 |
| cancel_order | 81 | 20 | 101 |
| **Total** | **965** | **241** | **1,206** |

**Imbalance:** cancel_order is the smallest class. The largest class has 1.7 times as many rows. Report macro-F1, and use `class_weight="balanced"` or an equivalent setting.

## 4. Data leakage control
- **Exact overlap:** no id and no normalized text appears in both train and validation. The script asserts this.
- **Near-duplicates:** translation produces many near-identical sentences, for example "كيف يمكنني تتبع الطلب رقم 213346؟" and "هل يمكنني تتبع الطلب رقم 582662؟". With a plain stratified random split (seed 42), 124 of 242 validation sentences (51%) had a training sentence of the same intent with ratio ≥ 90, numbers ignored.
- **Grouped split:** all same-intent pairs with ratio ≥ 90 are joined into groups (734 groups), and the split uses `StratifiedGroupKFold`. As a result, **no validation sentence has a same-intent training sentence with ratio ≥ 90**, which the grouping guarantees.
- **Leakage at lower thresholds:**
  - At ratio ≥ 85, 144 of 241 validation sentences (60%) still have a close match in training.
  - At ratio ≥ 80, 205 of 241 (85%) do.
  - Grouping at 85 or 80 is not possible: the groups chain together into a few very large ones (one group holds up to 96% of an intent), so an 80/20 stratified split can no longer be made.
  - Conclusion: grouping reduces leakage but does not remove it. Bitext is template-based, and every validation set taken from it will look like the training set.
- **Hard pairs across intents:** 4 validation sentences have a ratio ≥ 90 match in a *different* intent, for example "ألغي الطلب #" against "أعدل الطلب #". These are hard cases for the model, not leakage.
- **Saudi test set:** it is never used here.

## 5. Normalization (`src/normalize.py`)
- **Removed:** diacritics, tatweel and invisible direction marks.
- **Unified:** alef forms (إأآٱ → ا), ى → ي, Persian ک/ی → ك/ي, and Arabic-Indic digits → 0–9.
- **Repeated letters:** capped at two. Digits are not affected: the first version also shortened repeated digits and changed 17 order numbers, and this is fixed.
- **Kept:** ة and English words, so code-switched messages are not damaged.
- **Empty values:** None or NaN becomes an empty string.
- **Use in later tasks:** the same function must be imported in Task 2 (models), in the Saudi test evaluation and in the demo.

## 6. Quality review
The work was checked on four points: translation quality, the pipeline and data leakage, the task requirements, and the accuracy of the numbers in this report.

- **Automatic checks:** 0 issues in 1,570 rows. The Latin-word check flags runs of 3 or more letters, and the number check covers numbers with 3 or more digits.
- **Full check of all 1,260 rows of the earlier version (the final 1,206 rows are a subset of them):**
  - No errors of meaning were found, and every number was preserved.
  - There is no dialect, English, profanity or diacritics.
- **Problems found and fixed:**
  1. **Shortcut word رقم:** the word appeared before order numbers only in track_order (98% of its rows, 0% of the others), and track_order also used طلب الشراء. A model could learn "contains رقم" as a rule for track_order. Now طلب الشراء → الطلب, and رقم is used in about half of the rows of every intent that has order numbers.
  2. **Wrong labels from the source:** 8 get_refund messages asked *when* an existing refund arrives, so they were relabelled track_refund. 6 cancel_order messages only said "I can't pay for order N", so "أريد إلغاءه" was added.
  3. **Word choice:** تعويض (compensation for damage) was replaced with استرداد in refund intents (69 rows in the reviewed version, 58 of them in the final data).
  4. **Literal phrases:** موظف مباشر → موظف خدمة العملاء, and تحديد موقع الطلب → معرفة مكان الطلب. 3 unnatural sentences were rewritten.
- **Known remaining shortcut:** order numbers appear only in track_order, cancel_order and change_order, because this is how Bitext is written. A Saudi message such as "عندي شكوى على الطلب 482913" is therefore classified as change_order. Recommendation for Task 2: add an order number to some complaint, payment_issue and refund training messages as augmentation, or replace numbers with a token and test both options.
- **Sanity baseline (`src/data_prep/baseline_tfidf.py`):** TF-IDF on character 2–5-grams with Logistic Regression gives validation macro-F1 = 0.992.
  - This shows the classes are easy to separate in translated Bitext. It does **not** measure translation quality or real performance.
  - On 10 Saudi-dialect sentences written for a quick check (not an evaluation set) the same model gets 5 of 10 right.
  - This gap is the main reason the Saudi test set is needed.
- **Choosing hyper-parameters:** validation is almost saturated, so it cannot be used to choose between settings. Two options for the team to decide:
  - (a) use fixed, standard settings for every model;
  - (b) hold out 20% of the Saudi *scenarios* (by `scenario_id`) as a development set, and report the final results only on the other 80%.
- **Human review (done, 24 Nov 2026):** one team member, Bushra Abdulrahman, reviewed a random sample of 400 translations, 50 per intent, from the corrected data.
  - For each sentence she checked two things: whether the meaning matches the English, and whether the sentence still belongs to its intent.
  - She accepted all 400 sentences with no corrections and no removals, so the counts in sections 2–4 did not change.
  - The review was done by a single reviewer, after the automatic corrections above, so it confirms the corrected version only.
  - The results are in `data/interim/translation_review_sheet.csv`.

## 7. Outputs
- `data/processed/train.csv` and `data/processed/validation.csv`, with columns id, intent and text (normalized Arabic).
- `data/interim/bitext_ar.csv`: the English, the raw LLM Arabic, the corrected Arabic and the normalized text, side by side.
- `data/interim/manual_fixes.csv` and `data/interim/fixes_applied.csv`: every manual change and the reason for it.
- `data/interim/translation_review_sheet.csv`: the human review results (400 rows).
- `docs/intents.md`, `docs/task1/translation_guidelines.md` and `docs/task1/translation_prompt.md`.
