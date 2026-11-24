# Final Report text: sections written from Task 1 (v2)

## 2.1 Data Acquisition

The training data comes from the Bitext Customer Support LLM Chatbot Training Dataset. It is a public dataset of 26,872 English customer-support messages labelled with 27 intents, released under the CDLA-Sharing-1.0 licence. The messages are synthetic: Bitext generated them from seed texts with natural-language-generation tools, and linguists curated the result. We chose this dataset for three reasons: it has an open licence, its labels are consistent, and it is large enough to sample from. No comparable labelled Arabic customer-support dataset was available to us.

We kept eight intents. Together they cover the order, refund, payment and support requests that an online store's chat assistant has to route:

| Intent | What the customer wants | Example (MSA) |
|---|---|---|
| track_order | Know the status or arrival time of an order | أريد معرفة حالة طلبي |
| cancel_order | Cancel an order that was placed | أريد إلغاء طلبي |
| change_order | Modify the items of an existing order | أريد إضافة منتج إلى طلبي |
| get_refund | Get money back | أريد استرداد أموالي |
| track_refund | Check the status of a refund already requested | ما حالة طلب استرداد المبلغ الذي قدمته؟ |
| payment_issue | A problem while paying | رُفضت بطاقتي عند الدفع |
| complaint | File a complaint about the service | أريد تقديم شكوى |
| contact_human_agent | Speak to a human agent | أريد التحدث مع موظف خدمة العملاء |

Bitext contains 7,983 messages under these eight intents. Bitext is in English and our target users write in Arabic, so we translated a deduplicated sample (Section 3.1) into Modern Standard Arabic (MSA) with a large language model. The translation followed written guidelines:

- Preserve the intent exactly.
- Keep every order number and amount.
- Use simple MSA, with no dialect words, spelling mistakes or English words.
- Drop profanity.

We translated into MSA rather than Saudi dialect on purpose. Because the dialect, the spelling mistakes and the English words appear only in the test set, the test set measures how well the model generalises to them.

The test data is a separate Saudi dataset that the team is writing. It has 120 original requests, 15 per intent. Each request is written in four styles: MSA, Saudi dialect, Saudi dialect with spelling mistakes, and Saudi dialect with English words, which gives 480 messages. The four versions of a request share one scenario ID. The set is used only for evaluation, never for training.

## 3.1 Data Preprocessing

Preprocessing was done before and after translation:

1. **Deduplication (English).** Many Bitext messages differ only in an order number, punctuation or one word. Translating them would cost effort and add no information.
   - We first removed exact duplicates within each intent, ignoring numbers, punctuation and case (7,983 → 6,544).
   - We then removed near-duplicates within each intent, using a string similarity of at least 90 with rapidfuzz (6,544 → 4,167).
2. **Sampling.** To keep the classes balanced and the translation effort manageable, we sampled up to 200 messages per intent, giving 1,570 messages. cancel_order had only 170 distinct messages.
3. **Placeholder filling.** Bitext uses template tokens such as {{Order Number}} and {{Refund Amount}}. Before translation we replaced them with realistic random values, because real users write actual numbers.
4. **Translation and automatic checks.** All 1,570 messages were translated. An automatic check looked for missing output, text without Arabic, leftover English words and changed numbers, and it found no issues.
5. **Quality check and corrections.** Every translated sentence was checked against its English source.
   - No errors of meaning were found.
   - The check did find a label shortcut: the word رقم ("number") appeared before order numbers only in track_order messages. A model could have learned that word instead of the meaning. We made the wording consistent across all intents.
   - We also relabelled 8 messages whose Bitext label did not match our intent definitions. For example, "when will I get my money back" asks about an existing refund, so it belongs to track_refund, not get_refund.
   - We replaced تعويض (compensation) with استرداد (refund), and rewrote a few literal phrases.
   - The original model output is kept, so that every change can be traced.
   - Finally, a team member reviewed a random sample of 400 corrected translations (50 per intent) for meaning and intent, and accepted all of them without changes.
6. **Arabic normalization.** Arabic allows several spellings of the same word, and users often vary hamza and other marks. To reduce this variation we:
   - removed diacritics and tatweel;
   - unified the alef forms;
   - mapped ى to ي;
   - converted Arabic-Indic digits to Latin digits;
   - limited repeated letters to two.

   We kept ة and English words unchanged, so that code-switched messages are not damaged. The same function is used for training, validation, testing and the demo.
7. **Second deduplication.** Translation turned many different English sentences into the same Arabic sentence. We removed these duplicates after normalization (1,570 → 1,206).
8. **Train/validation split.** If a sentence is in training and a close paraphrase of it is in validation, the validation score is inflated. With a plain random 80/20 split, 51% of validation sentences had a training sentence of the same intent with a similarity of at least 90.
   - We therefore grouped such near-duplicates and split by group with stratified group k-fold (80/20, stratified by intent). The result is 965 training and 241 validation sentences, and no validation sentence has a training sentence of the same intent at this similarity level.
   - Paraphrases with lower similarity still cross the split, because Bitext is built from templates. The grouping therefore reduces the leakage but does not remove it.

A simple baseline (character n-gram TF-IDF with logistic regression) reached a macro-F1 of 0.99 on the validation set. This shows that the eight intents are easy to separate in the translated data, which is simple and repetitive. It does not show real-world performance. The same model correctly classified only half of a small set of Saudi-dialect sentences, which is why the Saudi test set is the main evaluation of this project.

**Limitations.**
- **Synthetic, MSA-only training data:** the training data is synthetic and in MSA only, while the test data is dialectal, so a large gap between validation and test results is expected.
- **Order numbers as a cue:** in Bitext, order numbers appear only in order-related intents, so a model may treat the presence of a number as a cue. This will be checked in the Saudi test set and addressed with data augmentation if needed.
- **Single human reviewer:** a team member reviewed a random sample of 400 corrected translations (50 per intent) and accepted all of them. The review covers about a third of the data and was done by one person.
