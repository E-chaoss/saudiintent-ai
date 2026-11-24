# The Saudi test set (480 messages)

File: `data/saudi_test/saudi_test_FINAL.csv`

> **Status: complete, review pending.** All 480 messages are written. The second-member
> review described below has **not been done yet**. Record it in
> `data/saudi_test/saudi_test_review.csv` (see "Review" below), never in the test file itself.
> The test set becomes **frozen** once the review is recorded (see
> [ADR 0002](decisions/0002-saudi-test-set.md)).

## Design

- 8 intents x 15 base requests = 120 base requests; each one is written in 4 styles = 480 messages.
- All four styles of a base request share the same `base_request_id`, so they always stay together.
- Change ONLY the target factor between styles. Keep the meaning, order number and amount identical.
- Write from a situation, not from the intent name (e.g. "your order is a week late and you are annoyed"), and vary the wording across requests.
- Do not copy Bitext sentences. These 480 messages are **never** used for training or for choosing hyper-parameters.
- Use the same decision rules for borderline messages as [intents.md](intents.md).

## Columns

| Column | Meaning | Example |
|---|---|---|
| `id` | Unique message id | `SA-001` |
| `base_request_id` | The base request this message belongs to (`<intent>-<nn>`) | `track_order-01` |
| `text` | The customer message | `وين وصل طلبي الحين؟` |
| `intent` | Gold label, one of the 8 intents in [intents.md](intents.md) | `track_order` |
| `style` | One of the 4 style values below | `Saudi Dialect` |

## Style values (exactly as written in the file)

| `style` value | What it is |
|---|---|
| `MSA` | Simple Modern Standard Arabic |
| `Saudi Dialect` | Saudi dialect, correct spelling |
| `Spelling Mistakes` | The same dialect sentence with 1–2 realistic typos (missing/extra/swapped letters, ة/ه, ا/أ) |
| `Arabic-English Code-switching` | Dialect with 1–2 English words such as order, cancel, refund, track, payment |

Evaluation code must read these exact strings. Do not rename them in the file; if short codes are
needed in code, map them in one place.

The count per intent and style is in [saudi_test_distribution.md](saudi_test_distribution.md).

## Review

A second member checks each message: is the label right, does it keep the same meaning as the
other styles of its base request, and does the style match its definition.

Record the result in `data/saudi_test/saudi_test_review.csv` (one row per message `id`):

| Column | Value |
|---|---|
| `id` | Message id from the test file |
| `reviewer` | Reviewer name |
| `review_ok` | `Y` or `N` |
| `corrected_intent` | Only if the label is wrong |
| `notes` | Optional |

A template with the header row is in `data/saudi_test/saudi_test_review_template.csv`.
If a message must change after review, the change is made once, by agreement, before the set is
frozen, and is listed in the ADR.
