"""Corrections found in the translation quality check.
Applied to the raw Arabic before normalization, so the original LLM output in
tr_out/ stays untouched and every change is traceable."""
import re, zlib
import pandas as pd

# Consistent wording across intents (fixes literal phrases and style differences
# between translation batches).
PHRASES = [
    (r"لطلب الشراء", "للطلب"), (r"طلب الشراء", "الطلب"),
    (r"تحديد موقع", "معرفة مكان"),
    (r"موظف مباشر", "موظف خدمة العملاء"), (r"موظفين مباشرين", "موظفي خدمة العملاء"),
]
REFUND = [  # تعويض means compensation for damage; the intent is a refund
    (r"تعويض مالي", "استرداد المبلغ"), (r"التعويض", "الاسترداد"),
    (r"بتعويض", "باسترداد"), (r"تعويضي", "المبلغ المسترد"), (r"تعويضات", "استرداد"),
    (r"تعويض", "استرداد"),
]
ORDER_NUM = re.compile(r"(الطلب|للطلب|طلبي|طلب)( رقم)? (\d{5,})")

def _half(id_):  # deterministic 50/50 choice per row
    return zlib.crc32(id_.encode()) % 2 == 0

def balance_raqm(text, id_):
    """'رقم' before an order number appeared only in track_order (a label shortcut).
    Use 'الطلب رقم N' in half of the rows and 'الطلب N' in the other half, in every intent."""
    return ORDER_NUM.sub(lambda m: f"{m.group(1)}{' رقم' if _half(id_) else ''} {m.group(3)}", text)

def apply_fixes(df, fixes_path="data/interim/manual_fixes.csv"):
    df = df.copy()
    log = []
    fx = pd.read_csv(fixes_path, dtype=str).fillna("")
    for r in fx.itertuples():
        m = df["id"] == r.id
        if not m.any():
            continue
        if r.action == "drop":
            df = df[~m]; log.append((r.id, "drop", r.reason)); continue
        if r.new_intent:
            df.loc[m, "intent"] = r.new_intent
        if r.new_text_ar:
            df.loc[m, "text_ar"] = r.new_text_ar
        log.append((r.id, r.action, r.reason))
    def fix_text(row):
        t = row.text_ar
        for a, b in PHRASES:
            t = re.sub(a, b, t)
        if row.intent in ("get_refund", "track_refund"):
            for a, b in REFUND:
                t = re.sub(a, b, t)
        return balance_raqm(t, row.id)
    new = df.apply(fix_text, axis=1)
    n_changed = int((new != df["text_ar"]).sum())
    df["text_ar"] = new
    return df, log, n_changed
