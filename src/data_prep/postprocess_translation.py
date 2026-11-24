"""Step 1b: merge LLM translations, run automatic quality checks, normalize,
and remove duplicates created by translation.

Output: data/interim/bitext_ar.csv  and  data/interim/translation_issues.csv
"""
import json, glob, os, re
import pandas as pd
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/, for the shared normalize.py
from normalize import normalize_ar
from fix_translations import apply_fixes

def main():
    en = pd.read_csv("data/interim/bitext_selected_en.csv")
    rows = []
    for f in glob.glob("data/interim/tr_out/*.json"):
        rows += json.load(open(f, encoding="utf-8"))
    ar = pd.DataFrame(rows)
    df = en.merge(ar, on="id", how="left")
    issues = []
    for r in df.itertuples():
        t = r.text_ar if isinstance(r.text_ar, str) else ""
        why = []
        if not t.strip(): why.append("missing")
        if re.search(r"[A-Za-z]{3,}", t): why.append("latin_words")
        if not re.search(r"[؀-ۿ]", t): why.append("no_arabic")
        n_en = re.findall(r"\d{3,}", r.text_en); n_ar = re.findall(r"\d{3,}", t.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩","0123456789")))
        if sorted(n_en) != sorted(n_ar): why.append("number_mismatch")
        if why: issues.append({"id": r.id, "intent": r.intent, "text_en": r.text_en, "text_ar": t, "issue": ",".join(why)})
    pd.DataFrame(issues, columns=["id", "intent", "text_en", "text_ar", "issue"]).to_csv(
        "data/interim/translation_issues.csv", index=False)
    df = df[~df["id"].isin([i["id"] for i in issues if "missing" in i["issue"] or "no_arabic" in i["issue"]])]
    df["text_ar_llm"] = df["text_ar"]                    # keep the original LLM output
    df, log, n_changed = apply_fixes(df)                 # corrections from the quality check
    pd.DataFrame(log, columns=["id", "action", "reason"]).to_csv("data/interim/fixes_applied.csv", index=False)
    df["text"] = df["text_ar"].map(normalize_ar)
    key = df["text"].str.replace(r"\d+", "#", regex=True).str.replace(" رقم #", " #")
    conflict = df.assign(k=key).groupby("k")["intent"].nunique()   # checked BEFORE dedup
    if conflict.max() > 1:
        bad = conflict[conflict > 1].index
        print("label conflicts dropped:", df[key.isin(bad)][["id", "intent", "text"]].to_string())
        df = df[~key.isin(bad)]; key = key[df.index]
    before = len(df)
    df = df.loc[~key.duplicated()]                       # duplicates introduced by translation
    df[["id", "intent", "text_en", "text_ar_llm", "text_ar", "text"]].to_csv("data/interim/bitext_ar.csv", index=False)
    print(f"translated={len(ar)} issues={len(issues)} manual_fixes={len(log)} rows_changed_by_rules={n_changed} after_dedup={len(df)} (removed {before-len(df)})")
    print(df["intent"].value_counts().to_string())

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repo root
    main()
