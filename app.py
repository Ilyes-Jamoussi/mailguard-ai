"""MailGuard AI - Streamlit interface for the email spam detector.

This module only renders the UI; every model and tensor operation lives in
``src.inference`` and is imported from there.
"""

from __future__ import annotations

import streamlit as st

from src.config import STYLES_PATH
from src.inference import Prediction, SpamClassifier, is_analyzable, load_metrics

EXAMPLES: dict[str, str] = {
    "Legitimate - Project update": (
        "Just following up on the project timeline. Could we schedule a short call "
        "this week to review the remaining tasks and assign owners before Friday?"
    ),
    "Legitimate - Quarterly report": (
        "The quarterly figures are ready for your review. I summarized the key changes "
        "in the attached document and flagged the items that need a decision."
    ),
    "Spam - Prize scam": (
        "CONGRATULATIONS! You've been selected as today's LUCKY WINNER of $1,000,000! "
        "Click here NOW to claim your FREE prize before it expires!"
    ),
    "Spam - Phishing": (
        "URGENT: Your bank account has been compromised. Verify your identity "
        "immediately by clicking this secure link or your account will be suspended "
        "within 24 hours."
    ),
}


@st.cache_resource
def get_classifier() -> SpamClassifier:
    """Load the trained classifier once and cache it across reruns."""
    return SpamClassifier.load()


def _inject_css() -> None:
    st.markdown(f"<style>{STYLES_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _render_sidebar(metrics: dict) -> None:
    st.sidebar.markdown("## About the model")
    st.sidebar.markdown(
        "A **Transformer neural network** built from scratch in PyTorch, trained on "
        "real emails from the SpamAssassin public corpus."
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f'<div class="stat-box"><div class="stat-value">{metrics["test_accuracy"]:.1%}</div>'
        '<div class="stat-label">Test accuracy</div></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("")
    columns = st.sidebar.columns(2)
    columns[0].markdown(
        f'<div class="stat-box"><div class="stat-value">{metrics["total_params"] / 1e6:.1f}M</div>'
        '<div class="stat-label">Parameters</div></div>',
        unsafe_allow_html=True,
    )
    columns[1].markdown(
        f'<div class="stat-box"><div class="stat-value">{metrics["training_time_min"]}min</div>'
        '<div class="stat-label">Training time</div></div>',
        unsafe_allow_html=True,
    )
    report = metrics.get("report", {})
    if report:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Per-class performance**")
        for class_name in ("ham", "spam"):
            scores = report.get(class_name)
            if scores:
                st.sidebar.markdown(
                    f"**{class_name.capitalize()}** - F1: {scores['f1-score']:.2f} - "
                    f"Precision: {scores['precision']:.2f} - Recall: {scores['recall']:.2f}"
                )


def _render_probability_bar(prediction: Prediction) -> None:
    spam_pct = prediction.probabilities["spam"] * 100
    fill_class = "prob-fill-spam" if prediction.label == "spam" else "prob-fill-ham"
    st.markdown(
        '<div class="prob-section">'
        '<div class="prob-row"><span>Spam probability</span>'
        f"<span>{spam_pct:.1f}%</span></div>"
        '<div class="prob-track">'
        f'<div class="prob-fill {fill_class}" style="width: {spam_pct:.1f}%"></div>'
        "</div></div>",
        unsafe_allow_html=True,
    )


def _render_result(prediction: Prediction) -> None:
    is_spam = prediction.label == "spam"
    card_class = "result-spam" if is_spam else "result-ham"
    label = "Spam detected" if is_spam else "Legitimate email"
    st.markdown(
        f'<div class="result-card {card_class}"><div class="result-label">{label}</div>'
        f'<div class="result-conf">{prediction.confidence:.1%} confidence</div></div>',
        unsafe_allow_html=True,
    )
    _render_probability_bar(prediction)


def _render_examples() -> None:
    st.markdown("**Try an example**")
    columns = st.columns(2)
    for index, (label, text) in enumerate(EXAMPLES.items()):
        with columns[index % 2]:
            st.markdown('<div class="example-btn">', unsafe_allow_html=True)
            if st.button(label, key=f"example_{index}", use_container_width=True):
                st.session_state.email_text = text
            st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="MailGuard AI", page_icon="🛡️", layout="centered")
    _inject_css()

    st.title("MailGuard AI")
    st.markdown(
        '<p class="subtitle">Paste an email below to check whether it\'s legitimate or spam.</p>',
        unsafe_allow_html=True,
    )

    classifier = get_classifier()
    metrics = load_metrics()
    if metrics:
        _render_sidebar(metrics)

    _render_examples()

    st.markdown("")
    email_text = st.text_area(
        "Email content",
        value=st.session_state.get("email_text", ""),
        height=180,
        placeholder="Paste the email text you want to analyze...",
        label_visibility="collapsed",
    )

    if st.button("Analyze", type="primary", use_container_width=True):
        if not is_analyzable(email_text):
            st.warning("Please enter at least 10 characters.")
        else:
            _render_result(classifier.predict(email_text))


main()
