"""Step 1d: apply reviewer corrections, drop rows marked wrong, then split 80/20
(train/validation) stratified by intent. The Saudi set is never touched here.
Outputs: data/processed/train.csv, data/processed/validation.csv"""
import os, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from rapidfuzz import fuzz
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/, for the shared normalize.py
from normalize import normalize_ar
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repo root
df = pd.read_csv("data/interim/bitext_ar.csv")
rev_path = "data/interim/translation_review_sheet.csv"
if os.path.exists(rev_path):
    rev = pd.read_csv(rev_path, dtype=str).fillna("")
    no = lambda c: rev[c].str.strip().str.upper().str.startswith(("N", "لا"))
    has_fix = rev["corrected_text"].str.strip() != ""
    fix = rev[has_fix].set_index("id")["corrected_text"].str.strip()
    df.loc[df["id"].isin(fix.index), "text_ar"] = df["id"].map(fix)
    # drop: wrong intent, or wrong meaning with no correction given
    bad = rev[no("intent_ok") | (no("meaning_ok") & ~has_fix)]["id"]
    df = df[~df["id"].isin(bad)]
    df["text"] = df["text_ar"].map(normalize_ar)
    k = df["text"].str.replace(r"\d+", "#", regex=True).str.replace(" رقم #", " #")       # corrections may create duplicates/conflicts
    clash = k.map(df.assign(k=k).groupby("k")["intent"].nunique()) > 1
    df = df[~clash & ~k.duplicated()]
    print(f"after review: conflicts dropped={int(clash.sum())}, rows={len(df)}")
    reviewed = (rev["meaning_ok"] != "").sum()
    print(f"review rows filled: {reviewed}/{len(rev)}; corrected={len(fix)}; dropped={len(bad)}")
# Group near-duplicates (ratio >= 90 after masking digits) so a sentence and its
# paraphrase never end up on both sides of the split.
df = df.reset_index(drop=True)
key = df["text"].str.replace(r"\d+", "#", regex=True).str.replace(" رقم #", " #").tolist()
parent = list(range(len(df)))
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]; i = parent[i]
    return i
for _, idx in df.groupby("intent").groups.items():
    idx = list(idx)
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            if fuzz.ratio(key[idx[a]], key[idx[b]]) >= 90:
                parent[find(idx[a])] = find(idx[b])
df["group"] = [find(i) for i in range(len(df))]
print("near-duplicate groups:", df["group"].nunique(), "for", len(df), "rows")
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
tr_i, va_i = next(sgkf.split(df, df["intent"], df["group"]))
tr, va = df.iloc[tr_i], df.iloc[va_i]
assert not set(tr["group"]) & set(va["group"]), "near-duplicate group on both sides"
assert not set(tr["text"]) & set(va["text"]), "same text in train and validation"
tr[["id", "intent", "text"]].to_csv("data/processed/train.csv", index=False)
va[["id", "intent", "text"]].to_csv("data/processed/validation.csv", index=False)
print("train", len(tr), "validation", len(va)); print(pd.crosstab(df["intent"], df["id"].isin(tr["id"]).map({True: "train", False: "val"})))
