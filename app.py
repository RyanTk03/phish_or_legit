"""Lab 04 - Phish or Legit? (Streamlit UI). Run:  streamlit run app.py"""
import re

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import precision_score, recall_score

from src.pipeline import load_data, split, make_pipeline, predict, word_weights, contributions

st.set_page_config(page_title="Phish or Legit?")
st.title("Phish or Legit?")
st.warning("This tool supports a human decision and does not block messages by itself. "
           "The data is synthetic.")


@st.cache_resource
def load_everything():
    df = load_data()
    Xtr, Xte, ytr, yte = split(df)
    models = {
        "Logistic regression": make_pipeline("lr").fit(Xtr, ytr),
        "Naive Bayes": make_pipeline("nb").fit(Xtr, ytr),
    }
    texts, y = np.asarray(Xte), np.asarray(yte)
    probs = {name: m.predict_proba(texts)[:, 1] for name, m in models.items()}
    return models, texts, y, probs


def pick_examples(texts, y, p):
    """2 phishing + 2 legit messages from the TEST set, plus one the logistic regression gets wrong."""
    pred = (p >= 0.5).astype(int)

    def first(mask, n):
        return [texts[i] for i in np.where(mask)[0][:n]]

    ex = {}
    for i, t in enumerate(first((y == 1) & (pred == 1), 2), 1):
        ex[f"Phishing #{i} (correct)"] = (t, "phish")
    for i, t in enumerate(first((y == 0) & (pred == 0), 2), 1):
        ex[f"Legit #{i} (correct)"] = (t, "legit")
    fn, fp = first((y == 1) & (pred == 0), 1), first((y == 0) & (pred == 1), 1)
    if fn:
        ex["Misclassified: phishing seen as legit"] = (fn[0], "phish")
    elif fp:
        ex["Misclassified: legit seen as phishing"] = (fp[0], "legit")
    return ex


models, test_texts, test_y, test_probs = load_everything()
examples = pick_examples(test_texts, test_y, test_probs["Logistic regression"])

st.session_state.setdefault("msg", next(iter(examples.values()))[0])
st.session_state.setdefault("done", False)


def load_example():
    choice = st.session_state.example
    if choice in examples:
        st.session_state.msg = examples[choice][0]
        st.session_state.done = False


st.selectbox("Load an example", ["- write your own -"] + list(examples), key="example",
             on_change=load_example)
st.text_area("Message", key="msg", height=110)
if st.session_state.example in examples:
    st.caption(f"Ground-truth label of this example: **{examples[st.session_state.example][1]}**")

c1, c2 = st.columns(2)
model_name = c1.selectbox("Classifier", list(models))
threshold = c2.slider("Decision threshold (score ≥ threshold → phish)", 0.05, 0.95, 0.50, 0.05)

if st.button("Classify", type="primary"):
    st.session_state.done = True

if st.session_state.done:
    text = st.session_state.msg
    try:
        _, score = predict(models[model_name], text)
    except ValueError as err:
        st.error(str(err))
        st.stop()

    label = "phish" if score >= threshold else "legit"
    if label == "phish":
        st.error(f"⚠️ Predicted class: **PHISH**  ·  score {score:.2f} ≥ threshold {threshold:.2f}")
    else:
        st.success(f"✅ Predicted class: **LEGIT**  ·  score {score:.2f} < threshold {threshold:.2f}")
    st.caption("The score is the model's output for the phish class, not a calibrated probability.")

    st.subheader("Why this score?")
    lr = models["Logistic regression"]
    if model_name != "Logistic regression":
        st.caption("Explanations always come from the logistic regression model "
                   "(Naive Bayes has no per-message coefficients to show).")

    weights = word_weights(lr, text)
    parts = []
    for tok in re.findall(r"\w+|\W+", text):
        w = weights.get(tok.lower(), 0.0)
        if tok.strip() and re.match(r"\w+", tok) and abs(w) > 0.3:
            parts.append(f":red[**{tok}**]" if w > 0 else f":green[**{tok}**]")
        else:
            parts.append(tok)
    st.markdown("".join(parts))
    st.caption("red = pushes toward phish, green = pushes toward legit (strong weights only)")

    contrib = contributions(lr, text)
    up = [(t, v) for t, v in contrib.items() if v > 0][:5]
    down = [(t, v) for t, v in contrib.items() if v < 0][:5]
    left, right = st.columns(2)
    left.markdown("**Increase the score (→ phish)**")
    right.markdown("**Decrease the score (→ legit)**")
    for col, rows in ((left, up), (right, down)):
        if rows:
            col.markdown("\n".join(f"- `{t}`  {v:+.2f}" for t, v in rows))
        else:
            col.caption("none")
    st.caption("Values are contributions to the log-odds of 'phish'.")

with st.expander("Precision / recall vs threshold (test set)"):
    p = test_probs[model_name]
    ths = np.round(np.arange(0.05, 0.96, 0.05), 2)
    curve = pd.DataFrame({
        "precision": [precision_score(test_y, (p >= t).astype(int), zero_division=0) for t in ths],
        "recall": [recall_score(test_y, (p >= t).astype(int), zero_division=0) for t in ths],
    }, index=pd.Index(ths, name="threshold"))
    st.line_chart(curve)
    st.caption(f"Classifier: {model_name}. Raising the threshold usually trades recall for precision.")