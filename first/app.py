import os
import re
import pickle
import numpy as np
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#  Page config 
st.set_page_config(page_title="Fake News Detector", page_icon="🔍", layout="centered")

#  Custom CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

.hero {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero p {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-top: 0;
}

.verdict-fake {
    background: linear-gradient(135deg, #ff416c, #ff4b2b);
    color: white;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
    font-size: 1.8rem;
    font-weight: 800;
    margin: 1rem 0;
    box-shadow: 0 8px 32px rgba(255, 65, 108, 0.4);
}
.verdict-real {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    color: white;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
    font-size: 1.8rem;
    font-weight: 800;
    margin: 1rem 0;
    box-shadow: 0 8px 32px rgba(56, 239, 125, 0.35);
}

.confidence-text {
    text-align: center;
    color: #cbd5e1;
    font-size: 1rem;
    margin-top: -0.5rem;
}

.note-box {
    background: rgba(234, 179, 8, 0.1);
    border-left: 4px solid #eab308;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    color: #fde68a;
    font-size: 0.9rem;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# NLTK assets 
@st.cache_resource
def load_nlp():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    return WordNetLemmatizer(), set(stopwords.words('english'))

# Load saved model & vectorizer 
@st.cache_resource
def load_models():
    here = os.path.dirname(__file__)
    model_path = os.path.join(here, 'baseline_model.pkl')
    vec_path   = os.path.join(here, 'tfidf_vectorizer.pkl')

    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        return None, None

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vec_path, 'rb') as f:
        vec = pickle.load(f)
    return model, vec

#  Text cleaning (must match train.py) 
def clean_text(text, lemmatizer, stop_words):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    # Adding len(w) > 2 just like we used in the training script!
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

#Header 
st.markdown("""
<div class="hero">
    <h1>🔍 Fake News Detector</h1>
    <p>Paste any news article below to check if it is real or fake</p>
</div>
""", unsafe_allow_html=True)

lemmatizer, stop_words = load_nlp()
model, vec = load_models()

if model is None:
    st.error(
        "⚠️ **Model files not found!** Please train the model first by running:\n\n"
        "```bash\n"
        "python first/train.py\n"
        "```\n\n"
        "This creates `baseline_model.pkl` and `tfidf_vectorizer.pkl` in the `first/` folder."
    )
    st.stop()

st.markdown("#### 📰 Paste a News Article")
news_text = st.text_area(
    label="article_input",
    placeholder="Paste the full news article text here...",
    height=220,
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_btn = st.button("🔍 Analyze Article", type="primary", use_container_width=True)

#  Result 
if analyze_btn:
    if not news_text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            # Clean → vectorize → predict
            cleaned = clean_text(news_text, lemmatizer, stop_words)
            vec_text = vec.transform([cleaned])
            
            # Using Sigmoid for accurate probability estimation (matching how we setup LIME in train.py)
            probs = model.predict_proba(vec_text)[0]
            prob_fake = probs[1]  # 0.0 to 1.0 probability of being FAKE
            
            prediction = 1 if prob_fake >= 0.5 else 0
            
            # Confidence is how far we are from the 0.5 decision boundary
            confidence = prob_fake if prediction == 1 else (1 - prob_fake)

        if prediction == 1:
            st.markdown('<div class="verdict-fake">🚨 FAKE NEWS</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="verdict-real">✅ REAL NEWS</div>', unsafe_allow_html=True)

        st.markdown(f'<p class="confidence-text">Model confidence: <strong>{confidence:.1%}</strong></p>', unsafe_allow_html=True)

        # Confidence bars
        st.markdown("#### Confidence Breakdown")
        col_real, col_fake = st.columns(2)
        with col_real:
            st.metric("🟢 Real Probability", f"{(1 - prob_fake)*100:.1f}%")
            st.progress(float(1 - prob_fake))
        with col_fake:
            st.metric("🔴 Fake Probability", f"{prob_fake*100:.1f}%")
            st.progress(float(prob_fake))

        st.markdown("""
        <div class="note-box">
            ⚠️ This is an ML model trained on a limited dataset. Always verify important news from multiple trusted sources.
        </div>
        """, unsafe_allow_html=True)

#  Footer 
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#475569; font-size:0.85rem;'>"
    "Built with Scikit-learn · NLTK · Streamlit &nbsp;|&nbsp; BuzzFeed Fake News Dataset"
    "</p>",
    unsafe_allow_html=True
)
