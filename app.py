"""
MailGuard AI — Email Spam Detector
"""

import streamlit as st
import torch
import json

from src.transformer_model import TransformerClassifier
from src.preprocessing import TextPreprocessor

CLASS_NAMES = ['ham', 'spam']

EXAMPLES = {
    "Legitimate — Work meeting": "Hi team, just a reminder about our meeting tomorrow at 10am in the main conference room. Please bring your quarterly reports. Best regards, Sarah.",
    "Legitimate — Shipping update": "Your order #4829 has shipped and is expected to arrive on Thursday. You can track your package using the link in your account dashboard.",
    "Spam — Prize scam": "CONGRATULATIONS! You've been selected as today's LUCKY WINNER of $1,000,000! Click here NOW to claim your FREE prize before it expires!",
    "Spam — Urgent action": "URGENT: Your bank account has been compromised. Verify your identity immediately by clicking this secure link or your account will be suspended within 24 hours.",
}


@st.cache_resource
def load_model():
    preprocessor = TextPreprocessor.load('models/preprocessor.pkl')
    with open('models/config.json') as f:
        config = json.load(f)
    model = TransformerClassifier(**config)
    model.load_state_dict(torch.load('models/transformer_best.pth', map_location='cpu'))
    model.eval()
    metrics = None
    try:
        with open('models/metrics.json') as f:
            metrics = json.load(f)
    except FileNotFoundError:
        pass
    return model, preprocessor, metrics


def predict(model, preprocessor, text):
    seq = torch.LongTensor([preprocessor.text_to_sequence(text)])
    with torch.no_grad():
        probs = torch.softmax(model(seq), dim=1)[0]
    idx = torch.argmax(probs).item()
    return {
        'class': CLASS_NAMES[idx],
        'confidence': probs[idx].item(),
        'probabilities': {n: probs[i].item() for i, n in enumerate(CLASS_NAMES)}
    }


st.set_page_config(page_title="MailGuard AI", page_icon="🛡️", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .block-container { max-width: 720px; padding-top: 2rem; }

    h1 { font-size: 2rem !important; font-weight: 700 !important; color: #0f172a !important; margin-bottom: 0 !important; }

    .subtitle { color: #64748b; font-size: 1.05rem; margin-bottom: 2rem; }

    .stTextArea textarea {
        border: 2px solid #e2e8f0 !important;
        border-radius: 12px !important;
        font-size: 0.95rem !important;
        padding: 1rem !important;
        transition: border-color 0.2s;
    }
    .stTextArea textarea:focus { border-color: #3b82f6 !important; box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important; }

    .stButton > button[kind="primary"] {
        background: #3b82f6 !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: background 0.2s;
    }
    .stButton > button[kind="primary"]:hover { background: #2563eb !important; }

    .result-card {
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin: 1.5rem 0;
        text-align: center;
    }
    .result-ham {
        background: #f0f9ff;
        border: 2px solid #3b82f6;
        color: #1e40af;
    }
    .result-spam {
        background: #fef2f2;
        border: 2px solid #ef4444;
        color: #991b1b;
    }
    .result-label { font-size: 1.4rem; font-weight: 700; margin-bottom: 0.25rem; }
    .result-conf { font-size: 1rem; font-weight: 500; opacity: 0.8; }

    .example-btn button {
        background: #f8fafc !important;
        border: 1.5px solid #e2e8f0 !important;
        border-radius: 8px !important;
        color: #334155 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-align: left !important;
        padding: 0.5rem 0.75rem !important;
        transition: all 0.15s;
    }
    .example-btn button:hover {
        border-color: #3b82f6 !important;
        background: #f0f7ff !important;
        color: #1e40af !important;
    }

    .stat-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stat-value { font-size: 1.3rem; font-weight: 700; color: #0f172a; }
    .stat-label { font-size: 0.8rem; color: #64748b; margin-top: 0.2rem; }

    div[data-testid="stSidebar"] { background: #f8fafc; }
    div[data-testid="stSidebar"] h2 { font-size: 1.1rem !important; color: #0f172a !important; }
</style>
""", unsafe_allow_html=True)

st.title("MailGuard AI")
st.markdown('<p class="subtitle">Paste an email below to check whether it\'s legitimate or spam.</p>', unsafe_allow_html=True)

model, preprocessor, metrics = load_model()
if model is None:
    st.error("Model not loaded. Train the model first.")
    st.stop()

# --- Sidebar ---
if metrics:
    st.sidebar.markdown("## About the model")
    st.sidebar.markdown(
        "A **Transformer neural network** built from scratch in PyTorch, "
        "trained on 7,584 real emails from SpamAssassin and Enron corpora."
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f'<div class="stat-box"><div class="stat-value">{metrics["test_accuracy"]:.1%}</div><div class="stat-label">Test accuracy</div></div>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown("")
    cols = st.sidebar.columns(2)
    cols[0].markdown(
        f'<div class="stat-box"><div class="stat-value">{metrics["total_params"]/1e6:.1f}M</div><div class="stat-label">Parameters</div></div>',
        unsafe_allow_html=True
    )
    cols[1].markdown(
        f'<div class="stat-box"><div class="stat-value">{metrics["training_time_min"]}min</div><div class="stat-label">Training time</div></div>',
        unsafe_allow_html=True
    )
    if 'report' in metrics:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Per-class performance**")
        for cls in CLASS_NAMES:
            if cls in metrics['report']:
                r = metrics['report'][cls]
                st.sidebar.markdown(
                    f"**{cls.capitalize()}** — F1: {r['f1-score']:.2f} · Precision: {r['precision']:.2f} · Recall: {r['recall']:.2f}"
                )

# --- Examples ---
st.markdown("**Try an example**")
example_cols = st.columns(2)
for i, (label, text) in enumerate(EXAMPLES.items()):
    with example_cols[i % 2]:
        st.markdown('<div class="example-btn">', unsafe_allow_html=True)
        if st.button(label, key=f"ex_{i}", use_container_width=True):
            st.session_state.email_text = text
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("")
email_text = st.text_area(
    "Email content",
    value=st.session_state.get('email_text', ''),
    height=180,
    placeholder="Paste the email text you want to analyze...",
    label_visibility="collapsed"
)

st.button("Analyze", type="primary", use_container_width=True, key="analyze")

if st.session_state.get("analyze"):
    if len(email_text.strip()) < 10:
        st.warning("Please enter at least 10 characters.")
    else:
        result = predict(model, preprocessor, email_text)
        is_spam = result['class'] == 'spam'
        card_class = "result-spam" if is_spam else "result-ham"
        label = "Spam detected" if is_spam else "Legitimate email"

        st.markdown(
            f'<div class="result-card {card_class}">'
            f'<div class="result-label">{label}</div>'
            f'<div class="result-conf">{result["confidence"]:.1%} confidence</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)
        for col, name in zip([c1, c2], CLASS_NAMES):
            p = result['probabilities'][name]
            col.markdown(
                f'<div class="stat-box"><div class="stat-value">{p:.1%}</div><div class="stat-label">{name.capitalize()}</div></div>',
                unsafe_allow_html=True
            )
