# Translation guidelines (Bitext EN -> Modern Standard Arabic)

1. Translate each English customer message into simple, natural Modern Standard Arabic (فصحى بسيطة) as a customer would write it to a store's support chat.
2. Preserve the intent exactly. Do not add, drop, or soften the request.
3. Keep every number exactly as written (order numbers, amounts). Write "SAR" as "ريال".
4. Do NOT use Saudi dialect words. Do NOT add Arabic spelling mistakes. Do NOT keep English words (except a brand or product name, if any).
5. Fix English typos silently (e.g. "cancep" -> cancel); do not try to imitate them in Arabic.
6. Profanity ("damn", "goddamn", "f***ing"): drop it; keep the frustration only with neutral wording if needed.
7. One message in, one message out. No quotes, no explanations, no diacritics (تشكيل).
8. Output JSON: a list of objects {"id": ..., "text_ar": ...} in the same order and with the same ids.
