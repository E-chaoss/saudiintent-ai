"""Step 1a: select the 8 intents from Bitext, fill placeholders, remove duplicates
and near-duplicates, and draw a balanced sample for translation.

Output: data/interim/bitext_selected_en.csv  (id, intent, text_en)
"""
import argparse, random, re
import pandas as pd
from rapidfuzz import fuzz

INTENTS = ["track_order", "cancel_order", "change_order", "get_refund",
           "track_refund", "payment_issue", "complaint", "contact_human_agent"]

def fill_placeholders(text, rng):
    text = re.sub(r"\{\{Order Number\}\}", lambda _: str(rng.randint(100000, 999999)), text)
    text = re.sub(r"\{\{Refund Amount\}\}", lambda _: str(rng.choice(range(50, 2001, 25))), text)
    text = re.sub(r"\{\{Currency Symbol\}\}", "SAR ", text)
    return re.sub(r"\s+", " ", text).strip()

def dedup_key(text):
    t = text.lower()
    t = re.sub(r"\{\{[^}]+\}\}", " X ", t)
    t = re.sub(r"\d+", " X ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def main(src, out, per_intent, fuzzy, seed):
    rng = random.Random(seed)
    df = pd.read_csv(src)
    df = df[df["intent"].isin(INTENTS)][["instruction", "intent"]].copy()
    df["key"] = df["instruction"].map(dedup_key)
    n0 = len(df)
    df = df.drop_duplicates(["intent", "key"])          # exact duplicates (ignoring numbers/punctuation)
    n1 = len(df)
    keep = []
    for _, g in df.groupby("intent"):                 # near-duplicates inside each intent
        kept = []
        for idx, k in zip(g.index, g["key"]):
            if all(fuzz.ratio(k, x) < fuzzy for x in kept):
                kept.append(k); keep.append(idx)
    df = df.loc[keep]
    n2 = len(df)
    assert df.groupby("key")["intent"].nunique().max() == 1, "same text under two intents"
    parts = [g.sample(min(per_intent, len(g)), random_state=seed) for _, g in df.groupby("intent")]
    s = pd.concat(parts)
    s["text_en"] = s["instruction"].map(lambda t: fill_placeholders(t, rng))
    s = s.sort_values(["intent"]).reset_index(drop=True)
    s.insert(0, "id", [f"bx{i:05d}" for i in range(len(s))])
    s[["id", "intent", "text_en"]].to_csv(out, index=False)
    print(f"rows: selected={n0} exact-dedup={n1} near-dedup={n2} sampled={len(s)}")
    print(s["intent"].value_counts().to_string())

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", default="data/interim/bitext_selected_en.csv")
    p.add_argument("--per-intent", type=int, default=200)
    p.add_argument("--fuzzy", type=int, default=90)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    main(a.src, a.out, a.per_intent, a.fuzzy, a.seed)
