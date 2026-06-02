import pickle
import shap
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.sparse import hstack, csr_matrix
import re, math, sys
from collections import Counter

st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

def get_css(dark):
    if dark:
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
        :root {
            --bg-primary: #0A0C10;
            --bg-secondary: #111318;
            --bg-card: #161A22;
            --bg-card-hover: #1C2130;
            --accent: #00E5FF;
            --accent-dim: #00E5FF22;
            --accent-red: #FF4757;
            --accent-green: #2ED573;
            --text-primary: #E8EAF0;
            --text-secondary: #8B92A5;
            --text-muted: #4A5066;
            --border: #1E2435;
            --border-accent: #00E5FF33;
        }
        .stApp { background: var(--bg-primary); font-family: 'DM Sans', sans-serif; }
        .stApp > header { background: transparent !important; }
        #MainMenu, footer, header { visibility: hidden; }
        .stDeployButton { display: none; }
        .main .block-container { padding: 2rem 3rem; max-width: 1200px; }

        .hero-wrap {
            padding-bottom: 2rem;
            margin-bottom: 2rem;
            border-bottom: 1px solid var(--border);
        }
        .hero-eyebrow {
            font-family: 'Space Mono', monospace;
            font-size: 10px;
            letter-spacing: 0.2em;
            color: var(--accent);
            text-transform: uppercase;
            margin-bottom: 10px;
            opacity: 0.8;
        }
        .hero-title {
            font-family: 'Space Mono', monospace;
            font-size: 3rem;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.02em;
            line-height: 1;
            margin: 0 0 10px;
        }
        .hero-title span { color: var(--accent); }
        .hero-sub {
            font-size: 15px;
            color: var(--text-secondary);
            font-weight: 300;
            max-width: 520px;
            line-height: 1.6;
        }
        .hero-pills {
            display: flex;
            gap: 8px;
            margin-top: 16px;
            flex-wrap: wrap;
        }
        .hero-pill {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 99px;
            padding: 4px 12px;
            font-size: 11px;
            color: var(--text-muted);
            font-family: 'Space Mono', monospace;
        }

        .section-header {
            font-family: 'Space Mono', monospace;
            font-size: 10px;
            letter-spacing: 0.2em;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .section-header::after {
            content: '';
            flex: 1;
            height: 1px;
            background: var(--border);
        }

        .verdict-fake {
            background: linear-gradient(135deg, #FF475718 0%, transparent 100%);
            border: 1px solid #FF475740;
            border-radius: 10px;
            padding: 1.25rem 1.5rem;
            display: flex;
            align-items: center;
            gap: 14px;
        }
        .verdict-real {
            background: linear-gradient(135deg, #2ED57318 0%, transparent 100%);
            border: 1px solid #2ED57340;
            border-radius: 10px;
            padding: 1.25rem 1.5rem;
            display: flex;
            align-items: center;
            gap: 14px;
        }
        .verdict-icon { font-size: 2rem; line-height: 1; }
        .verdict-eyebrow {
            font-family: 'Space Mono', monospace;
            font-size: 10px;
            letter-spacing: 0.15em;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 3px;
        }
        .verdict-main {
            font-size: 1.25rem;
            font-weight: 600;
            margin: 0;
        }
        .verdict-fake .verdict-main { color: #FF4757; }
        .verdict-real .verdict-main { color: #2ED573; }

        .chips {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 12px;
        }
        .chip {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px 18px;
            flex: 1;
            min-width: 90px;
            text-align: center;
        }
        .chip-val {
            font-family: 'Space Mono', monospace;
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text-primary);
            display: block;
            line-height: 1.2;
        }
        .chip-lbl {
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 4px;
            display: block;
        }

        .bar-row { margin-bottom: 12px; }
        .bar-meta {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        .bar-name { font-size: 12px; color: var(--text-secondary); }
        .bar-pct { font-family: 'Space Mono', monospace; font-size: 12px; color: var(--accent); }
        .bar-track { background: var(--bg-secondary); border-radius: 3px; height: 5px; }
        .bar-fill { height: 100%; border-radius: 3px; background: linear-gradient(90deg, var(--accent), #007A99); transition: width 0.5s; }
        .bar-fill-hot { background: linear-gradient(90deg, #FF4757, #CC0033); }

        .sample-pill {
            display: inline-block;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 8px 14px;
            font-size: 12px;
            color: var(--text-secondary);
            cursor: pointer;
            margin-bottom: 6px;
            width: 100%;
            text-align: left;
            transition: border-color 0.15s, color 0.15s;
        }
        .sample-pill:hover { border-color: var(--border-accent); color: var(--text-primary); }

        .stTextArea textarea {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            color: var(--text-primary) !important;
            font-size: 14px !important;
            line-height: 1.6 !important;
        }
        .stTextArea textarea:focus { border-color: var(--border-accent) !important; }
        .stButton > button {
            background: var(--accent) !important;
            color: #000 !important;
            border: none !important;
            border-radius: 6px !important;
            font-family: 'Space Mono', monospace !important;
            font-size: 11px !important;
            font-weight: 700 !important;
            letter-spacing: 0.1em !important;
            padding: 12px !important;
            text-transform: uppercase !important;
            width: 100% !important;
            transition: opacity 0.15s !important;
        }
        .stButton > button:hover { opacity: 0.85 !important; }
        hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }
        label { color: var(--text-secondary) !important; font-size: 12px !important; }
        .stMarkdown p { color: var(--text-secondary); }

        .footer {
            font-family: 'Space Mono', monospace;
            font-size: 10px;
            color: var(--text-muted);
            letter-spacing: 0.1em;
            text-align: center;
            padding: 1rem 0;
        }
        </style>
        """
    else:
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
        :root {
            --bg-primary: #F4F5F9;
            --bg-secondary: #EAECF2;
            --bg-card: #FFFFFF;
            --bg-card-hover: #F0F2F8;
            --accent: #0055CC;
            --accent-dim: #0055CC15;
            --accent-red: #CC2200;
            --accent-green: #1A7A38;
            --text-primary: #151822;
            --text-secondary: #4A5168;
            --text-muted: #9AA0B8;
            --border: #DDE0EC;
            --border-accent: #0055CC33;
        }
        .stApp { background: var(--bg-primary); font-family: 'DM Sans', sans-serif; }
        .stApp > header { background: transparent !important; }
        #MainMenu, footer, header { visibility: hidden; }
        .stDeployButton { display: none; }
        .main .block-container { padding: 2rem 3rem; max-width: 1200px; }

        .hero-wrap { padding-bottom: 2rem; margin-bottom: 2rem; border-bottom: 1px solid var(--border); }
        .hero-eyebrow { font-family: 'Space Mono', monospace; font-size: 10px; letter-spacing: 0.2em; color: var(--accent); text-transform: uppercase; margin-bottom: 10px; opacity: 0.8; }
        .hero-title { font-family: 'Space Mono', monospace; font-size: 3rem; font-weight: 700; color: var(--text-primary); letter-spacing: -0.02em; line-height: 1; margin: 0 0 10px; }
        .hero-title span { color: var(--accent); }
        .hero-sub { font-size: 15px; color: var(--text-secondary); font-weight: 300; max-width: 520px; line-height: 1.6; }
        .hero-pills { display: flex; gap: 8px; margin-top: 16px; flex-wrap: wrap; }
        .hero-pill { background: var(--bg-card); border: 1px solid var(--border); border-radius: 99px; padding: 4px 12px; font-size: 11px; color: var(--text-muted); font-family: 'Space Mono', monospace; }

        .section-header { font-family: 'Space Mono', monospace; font-size: 10px; letter-spacing: 0.2em; color: var(--text-muted); text-transform: uppercase; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
        .section-header::after { content: ''; flex: 1; height: 1px; background: var(--border); }

        .verdict-fake { background: #FFF0EE; border: 1px solid #FFBBAA; border-radius: 10px; padding: 1.25rem 1.5rem; display: flex; align-items: center; gap: 14px; }
        .verdict-real { background: #EDFBF1; border: 1px solid #A8E6BC; border-radius: 10px; padding: 1.25rem 1.5rem; display: flex; align-items: center; gap: 14px; }
        .verdict-icon { font-size: 2rem; line-height: 1; }
        .verdict-eyebrow { font-family: 'Space Mono', monospace; font-size: 10px; letter-spacing: 0.15em; color: var(--text-muted); text-transform: uppercase; margin-bottom: 3px; }
        .verdict-main { font-size: 1.25rem; font-weight: 600; margin: 0; }
        .verdict-fake .verdict-main { color: #CC2200; }
        .verdict-real .verdict-main { color: #1A7A38; }

        .chips { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
        .chip { background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 8px; padding: 10px 18px; flex: 1; min-width: 90px; text-align: center; }
        .chip-val { font-family: 'Space Mono', monospace; font-size: 1.05rem; font-weight: 700; color: var(--text-primary); display: block; line-height: 1.2; }
        .chip-lbl { font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; display: block; }

        .bar-row { margin-bottom: 12px; }
        .bar-meta { display: flex; justify-content: space-between; margin-bottom: 5px; }
        .bar-name { font-size: 12px; color: var(--text-secondary); }
        .bar-pct { font-family: 'Space Mono', monospace; font-size: 12px; color: var(--accent); }
        .bar-track { background: var(--bg-secondary); border-radius: 3px; height: 5px; }
        .bar-fill { height: 100%; border-radius: 3px; background: linear-gradient(90deg, var(--accent), #003399); }
        .bar-fill-hot { background: linear-gradient(90deg, #CC2200, #990000); }

        .stTextArea textarea { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; color: var(--text-primary) !important; font-size: 14px !important; box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important; }
        .stButton > button { background: var(--accent) !important; color: #FFF !important; border: none !important; border-radius: 6px !important; font-family: 'Space Mono', monospace !important; font-size: 11px !important; font-weight: 700 !important; letter-spacing: 0.1em !important; padding: 12px !important; text-transform: uppercase !important; width: 100% !important; }
        .stButton > button:hover { opacity: 0.88 !important; }
        hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }
        label { color: var(--text-secondary) !important; font-size: 12px !important; }
        .stMarkdown p { color: var(--text-secondary); }
        .footer { font-family: 'Space Mono', monospace; font-size: 10px; color: var(--text-muted); letter-spacing: 0.1em; text-align: center; padding: 1rem 0; }
        </style>
        """

def extract_features(text, rating=4.0):
    words = text.split()
    lower_words = text.lower().split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    freq = Counter(lower_words)
    total_w = max(len(lower_words), 1)
    probs = [freq[w] / total_w for w in lower_words]
    pos_words = ['amazing','excellent','perfect','love','great','fantastic','outstanding',
                 'brilliant','wonderful','superb','incredible','best','awesome','exceptional']
    neg_words = ['terrible','awful','horrible','worst','bad','poor','disappointing',
                 'useless','broken','cheap','fake','waste','defective','return']
    vague_words = ['everything','everyone','always','never','perfect','best','greatest',
                   'amazing','incredible','unbelievable','absolutely','completely',
                   'totally','literally','honestly','genuinely','definitely','certainly']
    t = text.lower()
    counts = Counter(lower_words)
    repeated = sum(1 for w, c in counts.items() if c > 1 and len(w) > 3)
    sent_lengths = [len(s.split()) for s in sentences] if sentences else [len(words)]
    mean_sl = np.mean(sent_lengths)
    p_pos = sum(1 for w in pos_words if w in t)
    p_neg = sum(1 for w in neg_words if w in t)
    return {
        'word_count':       len(words),
        'char_count':       len(text),
        'avg_word_len':     sum(len(w) for w in words) / len(words) if words else 0,
        'exclamation_cnt':  text.count('!'),
        'question_cnt':     text.count('?'),
        'caps_ratio':       sum(1 for c in text if c.isupper()) / len(text) if text else 0,
        'type_token_ratio': len(set(lower_words)) / len(lower_words) if lower_words else 0,
        'repetition_score': repeated / len(counts) if counts else 0,
        'sentiment_score':  (p_pos - p_neg) / (p_pos + p_neg + 1),
        'vagueness_score':  sum(1 for w in vague_words if w in t) / (len(words) + 1),
        'burstiness':       np.std(sent_lengths) / (mean_sl + 1e-9) if len(sent_lengths) > 1 else 0,
        'perplexity_proxy': -sum(p * math.log(p + 1e-9) for p in probs),
        'avg_sent_len':     mean_sl,
        'punct_ratio':      sum(1 for c in text if c in '.,!?;:') / len(text) if text else 0,
        'rating':           rating,
    }

@st.cache_resource
def load_models():
    baseline  = pickle.load(open("outputs/model_baseline.pkl", "rb"))
    gbm       = pickle.load(open("outputs/model_gbm.pkl",      "rb"))
    sentinel  = pickle.load(open("outputs/model_sentinel.pkl", "rb"))
    scaler    = pickle.load(open("outputs/scaler.pkl",         "rb"))
    tfidf_vec = pickle.load(open("outputs/tfidf_vec.pkl",      "rb"))
    return baseline, gbm, sentinel, scaler, tfidf_vec

@st.cache_resource
def load_distilbert():
    try:
        sys.path.insert(0, '.')
        from src.distilbert_predict import load_distilbert as _load
        return _load()
    except Exception:
        return None, None, None

# ── Inject CSS ────────────────────────────────────────────────────────────────
st.markdown(get_css(st.session_state.dark_mode), unsafe_allow_html=True)

# ── Top bar ───────────────────────────────────────────────────────────────────
_, toggle_col = st.columns([6, 1])
with toggle_col:
    label = "☀ Light" if st.session_state.dark_mode else "⬤ Dark"
    if st.button(label, key="theme_toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-eyebrow">◈ AI-Powered Detection</div>
  <div class="hero-title">Fake Review<span> Detector</span></div>
  <p class="hero-sub">
    Paste any product review and find out if it was written by a real person
    or generated by an AI. Trained on 40,000+ reviews across 10 product categories.
  </p>
  <div class="hero-pills">
    <span class="hero-pill">DistilBERT</span>
    <span class="hero-pill">Explainable AI</span>
    <span class="hero-pill">10 Domains</span>
    <span class="hero-pill">40K+ Reviews</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown('<div class="section-header">Paste a review</div>', unsafe_allow_html=True)
    review_text = st.text_area("", height=160,
        placeholder="e.g. This product is absolutely AMAZING!!! Best purchase EVER!!!\n\nor\n\nDecent product. Nothing fancy but gets the job done.",
        label_visibility="collapsed")
    rating = st.slider("Star rating given by the reviewer", 1.0, 5.0, 4.0, 0.5)
    st.button("⟶  CHECK THIS REVIEW", key="analyze_btn")

with right:
    st.markdown('<div class="section-header">Try an example</div>', unsafe_allow_html=True)
    samples = {
        "⚠  Suspicious — lots of caps & exclamations": ("This product is absolutely AMAZING!!! Best purchase EVER! Changed my life completely! Five stars is not enough!!!", 5.0),
        "⚠  Suspicious — sounds too polished":         ("This pillow saved my back. I love the look and feel of this pillow.", 5.0),
        "✓  Genuine — balanced, specific":             ("Decent product for the price. Shipping was slow but it does what it says. Nothing fancy but no complaints.", 4.0),
        "✓  Genuine — honest negative":                ("Terrible experience. Stopped working after 2 days. Very disappointed.", 1.0),
        "◈  Borderline — short and vague":             ("Good product. Works as expected. Would recommend.", 4.0),
    }
    for label, (text, r) in samples.items():
        if st.button(label, use_container_width=True, key=f"sample_{label}"):
            st.session_state['sample_text']   = text
            st.session_state['sample_rating'] = r
            st.rerun()

if 'sample_text' in st.session_state:
    review_text = st.session_state.pop('sample_text')
    rating = st.session_state.pop('sample_rating', 4.0)

analyze = st.session_state.get("analyze_btn", False)

if review_text and review_text.strip():
    with st.spinner("Analyzing..."):
        try:
            baseline, gbm, sentinel, scaler, tfidf_vec = load_models()
            db_model, db_tokenizer, db_device = load_distilbert()

            features    = extract_features(review_text, rating)
            FEATURE_COLS = list(features.keys())
            feat_df     = pd.DataFrame([features])
            feat_scaled = scaler.transform(feat_df[FEATURE_COLS])

            bl_prob  = baseline.predict_proba([review_text])[0][1]
            gbm_prob = gbm.predict_proba(feat_scaled)[0][1]
            tfidf_v  = tfidf_vec.transform([review_text])
            hyb_feat = hstack([tfidf_v, csr_matrix(feat_scaled)])
            sen_prob = sentinel.predict_proba(hyb_feat)[0][1]

            if db_model is not None:
                from src.distilbert_predict import predict as db_predict
                db_prob, _ = db_predict(review_text, db_model, db_tokenizer, db_device)
                spam = features['exclamation_cnt'] > 3 or features['caps_ratio'] > 0.08
                if spam:
                    ensemble_prob = db_prob*0.15 + bl_prob*0.15 + gbm_prob*0.60 + sen_prob*0.10
                else:
                    ensemble_prob = db_prob*0.50 + bl_prob*0.20 + gbm_prob*0.20 + sen_prob*0.10
                using_db = True
            else:
                db_prob = None
                ensemble_prob = bl_prob*0.35 + gbm_prob*0.45 + sen_prob*0.20
                using_db = False

            is_fake    = ensemble_prob > 0.5
            confidence = abs(ensemble_prob - 0.5) * 2

        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    st.markdown("---")
    st.markdown('<div class="section-header">Result</div>', unsafe_allow_html=True)

    if is_fake:
        st.markdown("""
        <div class="verdict-fake">
          <div class="verdict-icon">⚠</div>
          <div>
            <div class="verdict-eyebrow">Our assessment</div>
            <div class="verdict-main">This review looks fake</div>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="verdict-real">
          <div class="verdict-icon">✓</div>
          <div>
            <div class="verdict-eyebrow">Our assessment</div>
            <div class="verdict-main">This review looks genuine</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="chips">
      <div class="chip">
        <span class="chip-val">{ensemble_prob:.0%}</span>
        <span class="chip-lbl">Fake probability</span>
      </div>
      <div class="chip">
        <span class="chip-val">{confidence:.0%}</span>
        <span class="chip-lbl">Confidence</span>
      </div>
      <div class="chip">
        <span class="chip-val">{features['word_count']}</span>
        <span class="chip-lbl">Word count</span>
      </div>
      <div class="chip">
        <span class="chip-val">{features['exclamation_cnt']}</span>
        <span class="chip-lbl">Exclamations</span>
      </div>
      <div class="chip">
        <span class="chip-val">{features['type_token_ratio']:.2f}</span>
        <span class="chip-lbl">Vocab diversity</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">What each model thinks</div>', unsafe_allow_html=True)

    models = []
    if using_db and db_prob is not None:
        models.append(("DistilBERT (language model)", db_prob))
    models += [
        ("Vocabulary patterns",          bl_prob),
        ("Writing style signals",        gbm_prob),
        ("Combined analysis",            sen_prob),
    ]

    bars_html = ""
    for name, prob in models:
        hot = "bar-fill-hot" if prob > 0.5 else "bar-fill"
        bars_html += f"""
        <div class="bar-row">
          <div class="bar-meta">
            <span class="bar-name">{name}</span>
            <span class="bar-pct">{prob:.0%}</span>
          </div>
          <div class="bar-track">
            <div class="{hot}" style="width:{prob*100:.1f}%"></div>
          </div>
        </div>"""
    st.markdown(bars_html, unsafe_allow_html=True)

    # SHAP
    try:
        st.markdown("---")
        st.markdown('<div class="section-header">Why SENTINEL made this call</div>', unsafe_allow_html=True)
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

        bg = '#161A22' if st.session_state.dark_mode else '#FFFFFF'
        tc = '#E8EAF0' if st.session_state.dark_mode else '#151822'
        gc = '#1E2435' if st.session_state.dark_mode else '#DDE0EC'

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        colors = ['#FF4757' if v > 0 else '#2ED573' for v in shap_df['SHAP']]
        ax.barh(shap_df['Feature'], shap_df['SHAP'], color=colors, height=0.55)
        ax.axvline(0, color=gc, linewidth=1)
        ax.tick_params(colors=tc, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(gc)
            spine.set_linewidth(0.5)
        ax.set_xlabel("← more genuine    |    more suspicious →", color=tc, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
    except Exception:
        pass

st.markdown("---")
st.markdown('<div class="footer">Fake Review Detector · Built by Anushka Das</div>', unsafe_allow_html=True)