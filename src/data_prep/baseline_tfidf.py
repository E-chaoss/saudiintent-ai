"""Sanity baseline: TF-IDF (character 2-5-grams) + Logistic Regression.
Trains on train.csv, reports validation macro-F1, and runs a few
Saudi-dialect check sentences to show the gap that the Saudi test set must measure.
(These sentences are NOT the test set and are not used for any decision.)"""
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/, for the shared normalize.py
from normalize import normalize_ar
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repo root
tr = pd.read_csv("data/processed/train.csv"); va = pd.read_csv("data/processed/validation.csv")
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
clf = LogisticRegression(C=10, max_iter=3000, class_weight="balanced").fit(vec.fit_transform(tr["text"]), tr["intent"])
pred = clf.predict(vec.transform(va["text"]))
print(f"validation macro-F1 = {f1_score(va['intent'], pred, average='macro'):.3f}")
print(classification_report(va["intent"], pred, digits=3))
check = [("track_order", "ابي اعرف وين طلبي 482913"), ("track_order", "متى يوصل طلبي"),
         ("cancel_order", "الغو الطلب الله يعافيكم"), ("change_order", "ابي اغير العنوان حق الطلب"),
         ("get_refund", "ابي فلوسي ترجع"), ("track_refund", "وش صار على الاسترجاع حقي"),
         ("payment_issue", "البطاقة ما قبلت"), ("payment_issue", "انسحب المبلغ مرتين"),
         ("complaint", "عندي شكوى على الطلب 482913"), ("contact_human_agent", "ابي اكلم موظف")]
p = clf.predict(vec.transform([normalize_ar(t) for _, t in check]))
print("quick check (Saudi dialect):", sum(a == b for (a, _), b in zip(check, p)), "/", len(check))
for (gold, t), b in zip(check, p):
    print(("OK " if gold == b else "XX ") + f"{t}  gold={gold} pred={b}")
