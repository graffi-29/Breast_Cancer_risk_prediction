from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "artifacts" / "ultrasound_model.keras"
METRICS_PATH = PROJECT_DIR / "artifacts" / "ultrasound_metrics.json"
CLASS_NAMES = ["benign", "malignant", "normal"]
IMAGE_SIZE = (224, 224)

st.set_page_config(page_title="SonoLens | Ultrasound Review", page_icon="+", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap');
    .stApp { background: #f3f5ef; color: #18231e; }
    .block-container { max-width: 1180px; padding-top: 2.5rem; padding-bottom: 3rem; }
    h1 { color: #18231e; font: 500 3.1rem 'Fraunces', Georgia, serif; letter-spacing: 0; }
    h2, h3, p, label { font-family: 'DM Sans', sans-serif; }
    .eyebrow { color: #24735f; font: 500 0.76rem 'DM Mono', monospace; letter-spacing: 0.08em; }
    .lede { color: #59685f; font: 400 1.05rem 'DM Sans', sans-serif; max-width: 680px; }
    .panel { background: #ffffff; border: 1px solid #dce2d8; border-radius: 8px; padding: 24px; min-height: 255px; }
    .panel-label { color: #24735f; font: 500 0.72rem 'DM Mono', monospace; letter-spacing: 0.08em; margin-bottom: 12px; }
    .result { background: #e1f0e8; border-left: 5px solid #24735f; border-radius: 6px; padding: 18px 20px; }
    .result-title { color: #153f35; font: 600 1.5rem 'DM Sans', sans-serif; }
    .small { color: #59685f; font: 400 0.88rem 'DM Sans', sans-serif; line-height: 1.5; }
    .note { background: #ffffff; border: 1px solid #dce2d8; border-radius: 8px; padding: 18px; }
    [data-testid='stMetric'] { background: #ffffff; border: 1px solid #dce2d8; border-radius: 8px; padding: 14px; }
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
    image = image.convert("RGB").resize(IMAGE_SIZE)
    return np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)


st.markdown("<div class='eyebrow'>SONOLENS / MAMMOBENCH V2 / BUSI ULTRASOUND</div>", unsafe_allow_html=True)
st.title("A second look at the image")
st.markdown(
    "<div class='lede'>A visual demonstration of a three-class ultrasound model trained on BUSI images. "
    "Upload one matching scan to inspect the model's signal.</div>",
    unsafe_allow_html=True,
)
st.write("")

if not MODEL_PATH.exists():
    st.warning("The ultrasound model has not been trained yet.")
    st.code("python train_ultrasound.py --data-dir data/ultrasound/images", language="powershell")
    st.info("Download MammoBench v2 from Kaggle and place the benign, malignant, and normal folders under data/ultrasound/images.")
    st.stop()

upload_column, guide_column = st.columns([1.1, 0.9], gap="large")
with upload_column:
    st.markdown("<div class='panel-label'>01 / UPLOAD A SCAN</div>", unsafe_allow_html=True)
    st.markdown("### Choose an ultrasound image")
    uploaded_file = st.file_uploader(
        "Supported formats: JPG, JPEG, PNG",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible",
    )
    st.markdown(
        "<div class='small'>Use a grayscale breast ultrasound scan similar to the BUSI dataset. "
        "The model was not trained on mammograms, phone photos, CT, MRI, or pathology slides.</div>",
        unsafe_allow_html=True,
    )
with guide_column:
    st.markdown("<div class='panel-label'>02 / WHAT WORKS BEST</div>", unsafe_allow_html=True)
    st.markdown("### Upload the right kind of image")
    st.markdown(
        "<div class='small'><b>Good:</b> an original breast ultrasound frame, with the lesion visible and the scan readable.</div>"
        "<div class='small' style='margin-top:10px'><b>Avoid:</b> screenshots with heavy text overlays, selfies, mammograms, or unrelated medical images.</div>",
        unsafe_allow_html=True,
    )

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_column, result_column = st.columns([1, 1], gap="large")
    with image_column:
        st.markdown("<div class='panel-label'>03 / IMAGE PREVIEW</div>", unsafe_allow_html=True)
        st.image(image, caption="Uploaded ultrasound", width="stretch")
    model = load_model()
    probabilities = model.predict(prepare_image(image), verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])
    malignancy_signal = float(probabilities[CLASS_NAMES.index("malignant")])

    with result_column:
        st.markdown("<div class='panel-label'>04 / MODEL SIGNAL</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='result'><div class='result-title'>{predicted_class.title()}</div>"
            f"<div class='small'>Highest model score: {confidence:.1%}</div></div>",
            unsafe_allow_html=True,
        )
        st.write("")
        st.metric("Model confidence", f"{confidence:.1%}")
        st.progress(confidence)
        st.metric("Model-estimated malignancy signal", f"{malignancy_signal:.1%}")
        st.markdown("**Class score breakdown**")
        for class_name, probability in zip(CLASS_NAMES, probabilities):
            st.write(f"{class_name.title()}  ·  **{float(probability):.1%}**")
        st.markdown(
            "<div class='small'><b>Important:</b> the malignancy signal is the model's score for the malignant class. "
            "It is not calibrated patient risk and is not a probability that a person has cancer.</div>",
            unsafe_allow_html=True,
        )

    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.caption(f"Held-out validation accuracy: {metrics['classification_report']['accuracy']:.1%}.")

st.divider()
st.markdown(
    "<div class='note'><strong>What this project predicts:</strong> it estimates whether an ultrasound image resembles "
    "the benign, malignant, or normal images in its training dataset. The malignant score is a model signal, not a "
    "personal cancer-risk percentage. A malignant prediction is not a diagnosis; a benign or normal prediction does "
    "not rule out disease. Clinical risk requires a qualified clinician, the original examination, patient history, "
    "and appropriate follow-up. This demonstration must not be used for medical decisions.</div>",
    unsafe_allow_html=True,
)