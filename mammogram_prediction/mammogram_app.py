from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "artifacts_mammogram" / "mammogram_model.keras"
METRICS_PATH = PROJECT_DIR / "artifacts_mammogram" / "mammogram_metrics.json"

st.set_page_config(page_title="MammoLens | Mammogram Review", page_icon="+", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap');
    .stApp { background: #f3f5ef; color: #18231e; }
    .block-container { max-width: 1080px; padding-top: 2.5rem; padding-bottom: 3rem; }
    h1 { color: #18231e; font: 500 3rem 'Fraunces', Georgia, serif; }
    h2, h3, p, label { font-family: 'DM Sans', sans-serif; }
    .eyebrow { color: #24735f; font: 500 .76rem 'DM Mono', monospace; letter-spacing: .08em; }
    .lede, .small { color: #59685f; font-family: 'DM Sans', sans-serif; line-height: 1.5; }
    .panel, .note { background: #fff; border: 1px solid #dce2d8; border-radius: 8px; padding: 22px; }
    .result { background: #e1f0e8; border-left: 5px solid #24735f; border-radius: 6px; padding: 18px; }
    .result-title { color: #153f35; font: 600 1.5rem 'DM Sans', sans-serif; }
    [data-testid='stMetric'] { background: #fff; border: 1px solid #dce2d8; border-radius: 8px; padding: 14px; }
    [data-testid='stMetricValue'] { color: #18231e; font-family: 'DM Mono', monospace; }
    [data-testid='stFileUploader'] { background: #f8faf6; border: 1px dashed #8eaa9d; border-radius: 6px; padding: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def prepare_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((224, 224))
    return np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)


st.markdown("<div class='eyebrow'>MAMMOLENS / DDSM + INBREAST + MIAS</div>", unsafe_allow_html=True)
st.title("Mammogram malignancy signal")
st.markdown(
    "<div class='lede'>Upload a mammogram lesion image to inspect whether it resembles benign or malignant training examples.</div>",
    unsafe_allow_html=True,
)
st.write("")

if not MODEL_PATH.exists():
    st.warning("The mammogram model has not been trained yet.")
    st.code(
        "python mammogram_prediction/train_mammogram.py --dataset-root \"C:/Users/graff/Downloads/Dataset of Mammography with Benign Malignant Breast Masses\"",
        language="powershell",
    )
    st.info("Train the model first, then restart this app.")
    st.stop()

upload_column, guide_column = st.columns([1.05, 0.95], gap="large")
with upload_column:
    st.markdown("### Upload a mammogram image")
    uploaded_file = st.file_uploader("JPG, JPEG, or PNG", type=["jpg", "jpeg", "png"])
with guide_column:
    st.markdown("### Use a matching image")
    st.markdown(
        "<div class='small'>Use a lesion crop similar to the training images in the DDSM, INbreast, or MIAS folders. "
        "Do not upload a phone photo, ultrasound, CT, MRI, or an unrelated medical image.</div>",
        unsafe_allow_html=True,
    )

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    model = load_model()
    malignant_signal = float(model.predict(prepare_image(image), verbose=0)[0][0])
    predicted_class = "malignant" if malignant_signal >= 0.5 else "benign"
    confidence = malignant_signal if predicted_class == "malignant" else 1 - malignant_signal
    image_column, result_column = st.columns([1, 1], gap="large")
    with image_column:
        st.image(image, caption="Uploaded mammogram", width="stretch")
    with result_column:
        st.markdown(
            f"<div class='result'><div class='result-title'>{predicted_class.title()}</div>"
            f"<div class='small'>Highest model class: {confidence:.1%}</div></div>",
            unsafe_allow_html=True,
        )
        st.metric("Model-estimated malignant signal", f"{malignant_signal:.1%}")
        st.progress(malignant_signal)
        st.markdown(
            "<div class='small'><b>Interpretation:</b> this is a model score, not a calibrated patient-risk percentage. "
            "It does not diagnose cancer or rule it out.</div>",
            unsafe_allow_html=True,
        )
        if METRICS_PATH.exists():
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            st.caption(f"Grouped validation ROC-AUC: {metrics['validation_roc_auc']:.1%}.")

st.divider()
st.markdown(
    "<div class='note'><strong>Educational use only.</strong> This model is trained on labeled image crops. "
    "A high malignant signal requires review by a qualified radiologist and appropriate clinical follow-up. "
    "Do not use this website for medical decisions.</div>",
    unsafe_allow_html=True,
)