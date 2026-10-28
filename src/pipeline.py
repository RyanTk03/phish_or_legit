"""Lab 04 - Phish or Legit?: TF-IDF + Naive Bayes / Logistic Regression (no UI code here).

The split happens BEFORE any fitting.  Everything that learns from data (TF-IDF vocabulary and idf weights,
classifier weights) lives inside one scikit-learn Pipeline fitted on the TRAIN part only - this is what
prevents data leakage.
"""
import re

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

SEED = 42
URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b\w+\.(?:com|org|info|ly)\S*", re.I)
MONEY_RE = re.compile(r"\$\s?\d+(?:[.,]\d+)?")
NUM_RE = re.compile(r"\d+")


def preprocess(text: str) -> str:
    """Lower-case and replace URLs / money / numbers by placeholder tokens (normalisation, as in Lab 01)."""
    text = text.lower()
    text = URL_RE.sub(" urltok ", text)
    text = MONEY_RE.sub(" moneytok ", text)
    return NUM_RE.sub(" numtok ", text)


def load_data(path="data/messages.csv"):
    df = pd.read_csv(path)
    df["y"] = (df["label"] == "phish").astype(int)          # 1 = phish, 0 = legit
    return df


def split(df, test_size=0.25):
    """Stratified, seeded split -> keeps the 20 % phishing rate in both parts."""
    return train_test_split(df["text"], df["y"], test_size=test_size, stratify=df["y"], random_state=SEED)


def make_pipeline(kind="lr", analyzer="word"):
    """kind: 'nb' | 'lr' | 'nb_cal' (calibrated NB) | 'lr_cal'.  analyzer: 'word' | 'char_wb'."""
    if analyzer == "word":
        tfidf = TfidfVectorizer(preprocessor=preprocess, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    else:
        tfidf = TfidfVectorizer(preprocessor=preprocess, analyzer="char_wb", ngram_range=(2, 5),
                                min_df=3, sublinear_tf=True)
    if kind == "nb":
        clf = MultinomialNB(alpha=0.1)
    elif kind == "lr":
        clf = LogisticRegression(max_iter=1000, C=3.0, random_state=SEED)
    elif kind == "nb_cal":
        clf = CalibratedClassifierCV(MultinomialNB(alpha=0.1), method="sigmoid", cv=5)
    elif kind == "lr_cal":
        clf = CalibratedClassifierCV(LogisticRegression(max_iter=1000, C=3.0, random_state=SEED),
                                     method="sigmoid", cv=5)
    else:
        raise ValueError(kind)
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def predict(pipe, text: str):
    """Returns (label, score) where score = model output for the 'phish' class (NOT a calibrated probability
    unless the model was calibrated and checked)."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The message is empty - paste some text to classify.")
    score = float(pipe.predict_proba([text])[0][1])
    return ("phish" if score >= 0.5 else "legit"), score


def contributions(pipe, text: str):
    """Per-feature contribution to the log-odds of 'phish' for a LOGISTIC REGRESSION pipeline:
    contribution = tf-idf value x coefficient.  Positive -> pushes toward phish."""
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    X = vec.transform([text])
    names = vec.get_feature_names_out()
    row = X.tocoo()
    contrib = {names[j]: float(v * clf.coef_[0][j]) for j, v in zip(row.col, row.data)}
    return dict(sorted(contrib.items(), key=lambda kv: -abs(kv[1])))


def word_weights(pipe, text: str):
    """Contribution of each lower-cased word of the ORIGINAL text (unigram features) - used to highlight."""
    contrib = contributions(pipe, text)
    out = {}
    for w in re.findall(r"\w+", preprocess(text)):
        out[w] = contrib.get(w, 0.0)
    return out


def global_top_features(pipe, k=8):
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    names, coef = vec.get_feature_names_out(), clf.coef_[0]
    order = np.argsort(coef)
    return [(names[i], round(float(coef[i]), 2)) for i in order[::-1][:k]], \
           [(names[i], round(float(coef[i]), 2)) for i in order[:k]]


if __name__ == "__main__":
    df = load_data()
    Xtr, Xte, ytr, yte = split(df)
    pipe = make_pipeline("lr").fit(Xtr, ytr)
    print(predict(pipe, "Northbank: your account has been locked. Verify now http://bit.ly/abc123"))
