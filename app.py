"""Banking Intent Classifier: Streamlit demo.

Run:  streamlit run app.py
Needs the folder `saved_model/` next to this file (tfidf_logreg.joblib, lstm_model.keras,
vocab.json, classes.json), the same files the prediction notebook uses.
"""
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Text preparation. It must match the training code exactly.
# The TF-IDF model was saved with this tokenizer, so these functions must exist
# in __main__ before the model is loaded.
# ---------------------------------------------------------------------------
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")


def clean_text(text):
    return re.sub(r"\s+", " ", str(text).lower().strip())


def tokenize(text):
    return TOKEN_PATTERN.findall(clean_text(text))


_main = sys.modules["__main__"]
_main.clean_text = clean_text
_main.tokenize = tokenize

SAVE_DIR = Path(__file__).parent / "saved_model"
OOV_ID = 1

EXAMPLES = {
    "Lost card": "I lost my card yesterday, please block it",
    "Declined transfer": "Why was my transfer declined?",
    "Top-up time": "How long does a top up take?",
    "Card not arrived": "My card has not arrived yet",
}

st.set_page_config(page_title="Banking Intent Classifier", page_icon="🏦", layout="centered")


@st.cache_resource(show_spinner="Loading the main model...")
def load_main():
    """TF-IDF + Logistic Regression, the class names and the vocabulary. Loads in seconds."""
    tfidf = joblib.load(SAVE_DIR / "tfidf_logreg.joblib")
    with open(SAVE_DIR / "classes.json") as f:
        classes = json.load(f)
    with open(SAVE_DIR / "vocab.json") as f:
        vocab = json.load(f)
    stoi = {tok: i for i, tok in enumerate(vocab["itos"])}
    return tfidf, classes, stoi, int(vocab["max_len"])


@st.cache_resource(show_spinner="Loading the LSTM (this can take a minute)...")
def load_lstm():
    """The LSTM needs TensorFlow, which is slow to load, so it is optional."""
    try:
        from tensorflow import keras
        return keras.models.load_model(SAVE_DIR / "lstm_model.keras")
    except Exception:
        return None


def encode(text, stoi, max_len):
    ids = [stoi.get(tok, OOV_ID) for tok in tokenize(text)][:max_len]
    return np.array([ids + [0] * (max_len - len(ids))], dtype="int32")


def pretty(label):
    return str(label).replace("_", " ")


def bars(items):
    """items: list of (label, probability). Simple horizontal bars."""
    rows = []
    for label, p in items:
        width = max(p * 100, 0.5)
        rows.append(
            f"<div style='display:flex;align-items:center;gap:12px;margin:6px 0'>"
            f"<div style='width:46%;font-size:0.95rem'>{label}</div>"
            f"<div style='flex:1;background:#24424A;border-radius:6px'>"
            f"<div style='width:{width:.1f}%;background:#2DD4BF;height:22px;border-radius:6px'></div></div>"
            f"<div style='width:48px;text-align:right;font-size:0.95rem'>{p:.0%}</div></div>"
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("🏦 Banking Intent Classifier")
st.write("Type a customer message. The saved model tells us what the customer wants (one of 77 intents). "
         "No training happens here.")

try:
    tfidf_model, CLASSES, STOI, MAX_LEN = load_main()
except Exception as err:
    st.error(f"Could not load the models from `{SAVE_DIR}`. Check that the folder exists. ({err})")
    st.stop()

with st.sidebar:
    st.header("About the models")
    st.markdown(
        "**TF-IDF + Logistic Regression** (main model)  \n"
        "Test Macro F1: 0.886  \n\n"
        "**LSTM** (comparison)  \n"
        "Test Macro F1: 0.834"
    )
    st.caption("Trained on PolyAI Banking77: 77 intents, about 10,000 messages.")
    use_lstm = st.checkbox("Also show the LSTM (slower to load)", value=False)

if "text" not in st.session_state:
    st.session_state["text"] = ""

st.caption("Try an example:")
cols = st.columns(len(EXAMPLES))
for col, (name, sentence) in zip(cols, EXAMPLES.items()):
    if col.button(name, use_container_width=True):
        st.session_state["text"] = sentence

text = st.text_area("Customer message", key="text", height=110,
                    placeholder="For example: I lost my card yesterday, please block it")

if st.button("Predict", type="primary"):
    if not text.strip():
        st.warning("Please type a message first.")
    else:
        proba = tfidf_model.predict_proba([text])[0]
        order = proba.argsort()[::-1][:3]
        best = int(order[0])

        st.subheader("Prediction")
        st.success(f"We predict this customer intent is **{pretty(CLASSES[best])}** "
                   f"(confidence {proba[best]:.0%}).")
        if proba[best] < 0.5:
            st.info("The model is not very sure. A person should check this message.")

        st.markdown("**Top 3 (TF-IDF + Logistic Regression)**")
        bars([(pretty(CLASSES[i]), float(proba[i])) for i in order])

        lstm_model = load_lstm() if use_lstm else None
        if use_lstm and lstm_model is None:
            st.caption("The LSTM could not be loaded, so only TF-IDF is shown.")
        if lstm_model is not None:
            lstm_proba = lstm_model.predict(encode(text, STOI, MAX_LEN), verbose=0)[0]
            lstm_best = int(lstm_proba.argmax())
            st.markdown("**For comparison, the LSTM predicts**")
            bars([(pretty(CLASSES[i]), float(lstm_proba[i]))
                  for i in lstm_proba.argsort()[::-1][:3]])
            if lstm_best != best:
                st.caption("The two models disagree on this message.")
