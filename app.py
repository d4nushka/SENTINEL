import pickle
import shap
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.sparse import hstack, csr_matrix
import re, math
from collections import Counter

st.set_page_config(page_title="SENTINEL", page_icon="🛡️", layout="wide")

FEATURE_COLS = ['word_count','char_count','avg_word_len','exclamation_cnt',
                'question_cnt','caps_ratio','type_token_ratio','repetition_score',
                'sentiment_score','vagueness_score','burstiness','perplexity_proxy',
                'avg_sent_len','punct_ratio','rating']

def word_count(text): return len(text.split())
def char_count(text): return len(text)
def avg_word_length(text):
    words = text.split()
    return sum(len(w) for w in words) / len(words) if words else 0
def exclamation_count(text): return text.count('!')
def question_count(text): return text.count('?')
def caps_ratio(text):
    return sum(1 for c in text if c.isupper()) / len(text) if text else 0
def type_token_ratio(text):
    words = text.lower().split()
    return len(set(words)) / len(words) if words else 0
def repetition_score(text):
    words = text.lower().split()
    if len(words) < 2: return 0
    counts = Counter(words)
    repeated = sum(1 for w, c in counts.items() if c > 1 and len(w) > 3)
    return repeated / len(counts)
def sentiment_score(text):
    pos = ['amazing','excellent','perfect','love','great','fantastic','outstanding',
           'brilliant','wonderful','superb','incredible','best','awesome','exceptional']
    neg = ['terrible','awful','horrible','worst','bad','poor','disappointing',
           'useless','broken','cheap','fake','waste','defective','return']
    t = text.lower()
    p = sum(1 for w in pos if w in t)
    n = sum(1 for w in neg if w in t)
    return (p - n) / (p + n + 1)
def vagueness_score(text):
    vague = ['everything','everyone','always','never','perfect','best','greatest',
             'amazing','incredible','unbelievable','absolutely','completely',
             'totally','literally','honestly','genuinely','definitely','certainly']
    t = text.lower()
    count = sum(1 for w in vague if w in t)
    return count / (len(text.split()) + 1)
def burstiness(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 2: return 0
    lengths = [len(s.split()) for s in sentences]
    mean = np.mean(lengths)
    std  = np.std(lengths)
    return std / (mean + 1e-9)
def perplexity_proxy(text):
    words = text.lower().split()
    if len(words) < 2: return 0
    freq = Counter(words)
    total = len(words)
    probs = [freq[w] / total for w in words]
    return -sum(p * math.log(p + 1e-9) for p in probs)
def avg_sentence_length(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences: return 0
    return np.mean([len(s.split()) for s in sentences])
def punctuation_ratio(text):
    if not text: return 0
    return sum(1 for c in text if c in '.,!?;:') / len(text)

def extract_features(text, rating=4.0):
    return {
        'word_count':       word_count(text),
        'char_count':       char_count(text),
        'avg_word_len':     avg_word_length(text),
        'exclamation_cnt':  exclamation_count(text),
        'question_cnt':     question_count(text),
        'caps_ratio':       caps_ratio(text),
        'type_token_ratio': type_token_ratio(text),
        'repetition_score': repetition_score(text),
        'sentiment_score':  sentiment_score(text),
        'vagueness_score':  vagueness_score(text),
        'burstiness':       burstiness(text),
        'perplexity_proxy': perplexity_proxy(text),
        'avg_sent_len':     avg_sentence_length(text),
        'punct_ratio':      punctuation_ratio(text),
        'rating':           rating,
    }

@st.cache_resource
def load_models():
    baseline  = pickle.load(open("outputs/model_baseline.pkl",  "rb"))
    gbm       = pickle.load(open("outputs/model_gbm.pkl",       "rb"))
    sentinel  = pickle.load(open("outputs/model_sentinel.pkl",  "rb"))
    scaler    = pickle.load(open("outputs/scaler.pkl",          "rb"))
    tfidf_vec = pickle.load(open("outputs/tfidf_vec.pkl",       "rb"))
    return baseline, gbm, sentinel, scaler, tfidf_vec

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🛡️ SENTINEL")
st.markdown("**LLM-Generated Fake Review Detector** | Trained on 40,431 reviews · Cohen Kappa: 0.87 · AUC: 0.98")
st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    review_text = st.text_area("Paste a product review to analyze:", height=150,
        placeholder="Paste any product review here...")
    rating = st.slider("Product rating given by reviewer", 1.0, 5.0, 4.0, 0.5)
    analyze = st.button("🔍 Analyze Review", use_container_width=True)

with col2:
    st.markdown("**Try these examples:**")
    samples = {
        "🚨 Likely Fake": "This product is absolutely AMAZING!!! Best purchase EVER! Changed my life completely! Everyone needs this NOW! Five stars is not enough!!!",
        "✅ Likely Real": "Decent product for the price. Shipping was a bit slow but it does what it says. Nothing fancy but no complaints.",
        "🤔 Tricky case": "Good product. Works as expected. Delivery was fast. Would recommend to friends.",
    }
    for label, text in samples.items():
        if st.button(label, use_container_width=True):
            st.session_state['sample_text'] = text
            st.session_state['sample_rating'] = 5.0 if "Fake" in label else 4.0

if 'sample_text' in st.session_state:
    review_text = st.session_state['sample_text']

if analyze and review_text.strip():
    try:
        baseline, gbm, sentinel, scaler, tfidf_vec = load_models()

        features = extract_features(review_text, rating)
        feat_df  = pd.DataFrame([features])
        feat_scaled = scaler.transform(feat_df[FEATURE_COLS])

        bl_prob  = baseline.predict_proba([review_text])[0][1]
        gbm_prob = gbm.predict_proba(feat_scaled)[0][1]
        tfidf_v  = tfidf_vec.transform([review_text])
        hyb_feat = hstack([tfidf_v, csr_matrix(feat_scaled)])
        sen_prob = sentinel.predict_proba(hyb_feat)[0][1]

        ensemble_prob = (bl_prob * 0.35 + gbm_prob * 0.45 + sen_prob * 0.20)
        is_fake = ensemble_prob > 0.5

        st.divider()
        v1, v2, v3 = st.columns(3)
        with v1:
            if is_fake:
                st.error("⚠️ FAKE REVIEW DETECTED")
            else:
                st.success("✅ LIKELY GENUINE")
        with v2:
            st.metric("Fake Probability", f"{ensemble_prob:.1%}")
        with v3:
            confidence = abs(ensemble_prob - 0.5) * 2
            st.metric("Confidence", f"{confidence:.1%}")

        st.divider()
        st.subheader("Model Breakdown")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("TF-IDF + LR", f"{bl_prob:.1%}", help="Baseline model")
        mc2.metric("Linguistic GBM", f"{gbm_prob:.1%}", help="Feature-based model")
        mc3.metric("SENTINEL Hybrid", f"{sen_prob:.1%}", help="Combined model")

        st.divider()
        st.subheader("Linguistic Signal Analysis")
        fc1, fc2, fc3, fc4, fc5 = st.columns(5)
        fc1.metric("Words", features['word_count'])
        fc2.metric("Exclamations", features['exclamation_cnt'])
        fc3.metric("Vocab Diversity", f"{features['type_token_ratio']:.2f}")
        fc4.metric("Vagueness", f"{features['vagueness_score']:.3f}")
        fc5.metric("Caps Ratio", f"{features['caps_ratio']:.2%}")

        # SHAP explanation
        try:
            explainer   = shap.TreeExplainer(gbm)
            shap_values = explainer.shap_values(pd.DataFrame(feat_scaled, columns=FEATURE_COLS))
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
            elif len(np.array(shap_values).shape) == 3:
                sv = shap_values[0, :, 1]
            else:
                sv = shap_values[0]

            shap_df = pd.DataFrame({'Feature': FEATURE_COLS, 'SHAP': sv})
            shap_df = shap_df.reindex(shap_df['SHAP'].abs().sort_values(ascending=True).index)

            st.divider()
            st.subheader("🔍 Why SENTINEL made this decision")
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in shap_df['SHAP']]
            ax.barh(shap_df['Feature'], shap_df['SHAP'], color=colors)
            ax.axvline(0, color='black', linewidth=0.8)
            ax.set_xlabel("SHAP value  (red → pushed toward FAKE  |  green → pushed toward REAL)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.info(f"SHAP visualization skipped: {e}")

    except FileNotFoundError:
        st.error("Models not found! Run: python src/train.py first.")

st.divider()
st.caption("SENTINEL · 40,431 reviews · F1=0.93 · AUC=0.98 · Cohen Kappa=0.87 · Built by Anushka Das, VIT Bhopal")