"""Step 1c: sample 50 translated messages per intent for human review.
Reviewers fill: meaning_ok (Y/N), intent_ok (Y/N), corrected_text, reviewer.
Output: data/interim/translation_review_sheet.csv (open in Google Sheets)."""
import os, sys, pandas as pd
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repo root
out = "data/interim/translation_review_sheet.csv"
if os.path.exists(out) and "--force" not in sys.argv:
    done = (pd.read_csv(out, dtype=str).fillna("")[["meaning_ok", "intent_ok", "corrected_text"]] != "").any(axis=1).sum()
    if done:
        sys.exit(f"{out} already has {done} reviewed rows; not overwriting (use --force).")
df = pd.read_csv("data/interim/bitext_ar.csv")
s = pd.concat([g.sample(min(50, len(g)), random_state=7) for _, g in df.groupby("intent")])
s = s[["id", "intent", "text_en", "text_ar"]].assign(meaning_ok="", intent_ok="", corrected_text="", reviewer="")
s.to_csv("data/interim/translation_review_sheet.csv", index=False, encoding="utf-8-sig")
print(len(s), "rows ->", "data/interim/translation_review_sheet.csv")
