"""Lab 04 - Phish or Legit? (Streamlit UI). Run:  streamlit run app.py"""
import re

import streamlit as st

from src.pipeline import load_data, split, make_pipeline, predict, word_weights, contributions

st.set_page_config(page_title="Phish or Legit?")
st.title("Phish or Legit?")
st.caption("TF-IDF + logistic regression on a SYNTHETIC dataset (fictional brands). "
           "Labels are 'phishing-style' vs 'legit', not real-world ground truth.")


@st.cache_resource
def get_model():
    df = load_data()
    Xtr, _, ytr, _ = split(df)
    return make_pipeline("lr").fit(Xtr, ytr), make_pipeline("lr_cal").fit(Xtr, ytr)


model, model_cal = get_model()
text = st.text_area("Paste a message:", "Northbank: your account has been locked. Verify now http://bit.ly/ab12cd",
                    height=100)

if st.button("Classify", type="primary"):
    try:
        label, score = predict(model, text)
        _, cal_score = predict(model_cal, text)
    except ValueError as err:
        st.error(str(err))
    else:
        if label == "phish":
            st.error(f"⚠️ Looks like PHISHING-style text  ·  model score {score:.2f}")
        else:
            st.success(f"✅ Looks legit  ·  model score for phish {score:.2f}")
        st.caption(f"Calibrated score (sigmoid calibration on training data): {cal_score:.2f}. "
                   "Scores are only meaningful as probabilities if calibration was checked - see the lab report.")

        weights = word_weights(model, text)
        parts = []
        for tok in re.findall(r"\w+|\W+", text):
            w = weights.get(tok.lower(), 0.0)
            if tok.strip() and re.match(r"\w+", tok) and abs(w) > 0.3:
                parts.append(f":red[**{tok}**]" if w > 0 else f":green[**{tok}**]")
            else:
                parts.append(tok)
        st.markdown("".join(parts))
        st.caption("red = pushes toward phishing, green = pushes toward legit (words with a strong weight only)")
        top = list(contributions(model, text).items())[:6]
        st.table({"feature": [t for t, _ in top], "contribution to log-odds": [round(v, 2) for _, v in top]})
