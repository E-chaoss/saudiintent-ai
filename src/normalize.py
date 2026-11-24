"""One Arabic normalization function, used identically for training, validation,
the Saudi test set, and the demo."""
import re

_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

_ZERO_WIDTH = re.compile("[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")

def normalize_ar(text) -> str:
    if text is None or (isinstance(text, float) and text != text):  # None / NaN
        return ""
    t = str(text)
    t = _ZERO_WIDTH.sub("", t)                 # invisible direction marks
    t = t.replace("ک", "ك").replace("ی", "ي")  # Persian keyboard letters
    t = _DIACRITICS.sub("", t)                 # remove tashkeel
    t = t.replace(_TATWEEL, "")                # remove tatweel
    t = re.sub("[إأآٱ]", "ا", t)               # unify alef forms
    t = t.replace("ى", "ي")                    # alef maqsura -> ya
    t = t.translate(_AR_DIGITS)                 # Arabic-Indic digits -> 0-9
    t = re.sub(r"([^\W\d_])\1{2,}", r"\1\1", t)  # cap repeated LETTERS at 2 (digits untouched)
    t = re.sub(r"\s+", " ", t).strip()
    return t
# Note: ة is kept (not mapped to ه) and English words are kept as-is (lower-cased
# by the vectorizer/tokenizer), so code-switched messages are not destroyed.
