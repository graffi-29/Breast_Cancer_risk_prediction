from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent
MAMMOGRAM_MODEL_PATH = PROJECT_DIR / "artifacts_mammogram" / "mammogram_model.keras"
MAMMOGRAM_METRICS_PATH = PROJECT_DIR / "artifacts_mammogram" / "mammogram_metrics.json"
ULTRASOUND_MODEL_PATH = PROJECT_DIR / "artifacts" / "ultrasound_model.keras"
ULTRASOUND_METRICS_PATH = PROJECT_DIR / "artifacts" / "ultrasound_metrics.json"

st.set_page_config(page_title="BreastLens | Combined Review", page_icon="+", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap');
    .stApp { background: #0d1320; color: #f6f3f7; }
    [data-testid='stSidebar'] { display: none; }
    .css-1d391kg, .css-17lntkn { display: none !important; }
    .block-container { max-width: 1240px; padding-top: 1.8rem; padding-bottom: 3rem; }
    h1 { color: #f6f3f7; font: 500 3.2rem 'Fraunces', Georgia, serif; line-height: 1.05; }
    h2, h3, p, label { font-family: 'DM Sans', sans-serif; }
    .eyebrow { color: #f05a9b; font: 500 .76rem 'DM Mono', monospace; letter-spacing: .08em; }
    .lede, .small { color: #a7adbd; font-family: 'DM Sans', sans-serif; line-height: 1.5; }
    .masthead { background: linear-gradient(135deg, #171e2f, #101522); border: 1px solid #273149; border-radius: 13px; padding: 24px 28px 26px; margin-bottom: 18px; }
    .masthead h1 { color: #f5f7f1; margin: 10px 0 12px; }
    .masthead .lede { color: #cbd0dc; max-width: 700px; }
    .step { color: #f05a9b; font: 500 .72rem 'DM Mono', monospace; letter-spacing: .08em; }
    .panel, .note { background: #171e2d; border: 1px solid #293349; border-radius: 10px; padding: 20px; }
    .panel { min-height: 190px; box-shadow: 0 12px 30px rgba(0,0,0,.16); }
    .signal-panel { background: #171e2d; color: #f6f3f7; border: 1px solid #293349; border-radius: 10px; padding: 22px; margin-top: 18px; }
    .signal-panel h3 { color: #f6f3f7; }
    .result { background: linear-gradient(135deg, #d4f5e5, #a7e4c5); border-left: 5px solid #36b879; border-radius: 8px; padding: 20px; }
    .result-title { color: #153f35; font: 600 1.5rem 'DM Sans', sans-serif; }
    .risk-number { color: #f5f7f1; font: 600 3.2rem 'DM Mono', monospace; line-height: 1.1; }
    .risk-label { color: #f05a9b; font: 600 .78rem 'DM Mono', monospace; letter-spacing: .08em; }
    .risk-track { display: flex; height: 12px; border-radius: 20px; overflow: hidden; margin: 16px 0 7px; }
    .risk-low { background: #36b879; width: 34%; }
    .risk-mid { background: #e7be70; width: 33%; }
    .risk-high { background: #f05a9b; width: 33%; }
    .risk-scale { display: flex; justify-content: space-between; color: #a7adbd; font: 500 .68rem 'DM Mono', monospace; }
    [data-testid='stMetric'] { background: #1c2435; border: 1px solid #303b53; border-radius: 9px; padding: 14px; }
    [data-testid='stMetricLabel'] { color: #a7adbd; }
    [data-testid='stMetricValue'] { color: #f6f3f7; font-family: 'DM Mono', monospace; }
    [data-testid='stFileUploader'] { background: #111827; border: 1px dashed #5e6a84; border-radius: 6px; padding: 8px; }
    [data-testid='stProgressBar'] > div > div { background: #f05a9b; }
    .brand { color: #f6f3f7; font: 700 1.05rem 'DM Sans', sans-serif; line-height: 1.2; }
    .brand-mark { color: #f05a9b; font-size: 2.5rem; line-height: 1; }
    .nav-active { background: #9c3c78; border-radius: 7px; padding: 10px 12px; color: #fff; }
    .nav-item { padding: 10px 12px; color: #cbd0dc; }
    .section-card { background: #171e2d; border: 1px solid #293349; border-radius: 10px; padding: 18px; min-height: 210px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path)


def prepare_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((224, 224))
    return np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)


def ultrasound_signal(model, image: Image.Image) -> float:
    scores = model.predict(prepare_image(image), verbose=0)[0]
    return float(scores[1])


def mammogram_signal(model, image: Image.Image) -> float:
    return float(model.predict(prepare_image(image), verbose=0)[0][0])


def validation_text(path: Path, metric_key: str) -> str:
    if not path.exists():
        return ""
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if metric_key == "classification_report":
        accuracy = metrics.get(metric_key, {}).get("accuracy")
        return f"Ultrasound validation accuracy: {accuracy:.1%}" if accuracy is not None else ""
    value = metrics.get(metric_key)
    return f"Mammogram validation ROC-AUC: {value:.1%}" if value is not None else ""


mammogram_metrics = json.loads(MAMMOGRAM_METRICS_PATH.read_text(encoding="utf-8"))
ultrasound_metrics = json.loads(ULTRASOUND_METRICS_PATH.read_text(encoding="utf-8"))
mammogram_auc = mammogram_metrics["validation_roc_auc"]
ultrasound_accuracy = ultrasound_metrics["classification_report"]["accuracy"]

st.markdown(
    "<div class='masthead'><div class='eyebrow'>BREASTLENS / AI IMAGE REVIEW</div>"
    "<h1>Breast Cancer Prediction</h1>"
    "<div class='lede'>Upload a mammogram and ultrasound image to inspect the two trained model signals.</div></div>",
    unsafe_allow_html=True,
)
st.write("")
metric_columns = st.columns(4)
metric_columns[0].metric("Dataset used", "25K+", "Mammogram images")
metric_columns[1].metric("Modalities", "2", "Mammogram + ultrasound")
metric_columns[2].metric("Mammo ROC-AUC", f"{mammogram_auc:.1%}", "Grouped validation")
metric_columns[3].metric("Ultrasound", f"{ultrasound_accuracy:.1%}", "Validation accuracy")
st.write("")

if not MAMMOGRAM_MODEL_PATH.exists() or not ULTRASOUND_MODEL_PATH.exists():
    st.error("Both trained models are required. Train the mammogram and ultrasound projects first.")
    st.stop()

mammogram_column, ultrasound_column = st.columns(2, gap="large")
with mammogram_column:
    st.markdown("<div class='section-card'><div class='step'>01 / INPUT IMAGE</div>", unsafe_allow_html=True)
    st.markdown("### Mammogram image")
    mammogram_file = st.file_uploader(
        "Upload a mammogram lesion crop", type=["jpg", "jpeg", "png"], key="mammogram"
    )
    st.caption("DDSM, INbreast, or MIAS image crop")
    st.markdown("</div>", unsafe_allow_html=True)
with ultrasound_column:
    st.markdown("<div class='section-card'><div class='step'>02 / INPUT IMAGE</div>", unsafe_allow_html=True)
    st.markdown("### Ultrasound image")
    ultrasound_file = st.file_uploader(
        "Upload a BUSI-style ultrasound image", type=["jpg", "jpeg", "png"], key="ultrasound"
    )
    st.caption("BUSI-style ultrasound image")
    st.markdown("</div>", unsafe_allow_html=True)

mammogram_score = None
ultrasound_score = None
if mammogram_file is not None:
    mammogram_image = Image.open(mammogram_file)
    mammogram_score = mammogram_signal(load_model(str(MAMMOGRAM_MODEL_PATH)), mammogram_image)
if ultrasound_file is not None:
    ultrasound_image = Image.open(ultrasound_file)
    ultrasound_score = ultrasound_signal(load_model(str(ULTRASOUND_MODEL_PATH)), ultrasound_image)

if mammogram_score is not None or ultrasound_score is not None:
    st.divider()
    st.markdown("<div class='step'>03 / PREDICTION RESULT</div>", unsafe_allow_html=True)
    st.info("The two models are shown separately. No artificial combined-risk percentage is used.")

    preview_columns = st.columns(2, gap="large")
    if mammogram_file is not None:
        with preview_columns[0]:
            st.image(mammogram_image, caption="Mammogram input", width="stretch")
    if ultrasound_file is not None:
        with preview_columns[1]:
            st.image(ultrasound_image, caption="Ultrasound input", width="stretch")

    result_columns = st.columns(2, gap="large")
    if mammogram_score is not None:
        with result_columns[0]:
            st.markdown(
                f"<div class='result'><div class='risk-label'>MAMMOGRAM RESULT</div>"
                f"<div class='risk-number' style='color:#153f35'>{mammogram_score:.1%}</div>"
                "<div class='result-title'>Malignant signal</div></div>",
                unsafe_allow_html=True,
            )
    if ultrasound_score is not None:
        with result_columns[1]:
            st.markdown(
                f"<div class='result'><div class='risk-label'>ULTRASOUND RESULT</div>"
                f"<div class='risk-number' style='color:#153f35'>{ultrasound_score:.1%}</div>"
                "<div class='result-title'>Malignant signal</div></div>",
                unsafe_allow_html=True,
            )

st.write("")
st.divider()
st.markdown(
    "<div class='note'><strong>Review required:</strong> these separate model signals are not diagnoses or calibrated patient-risk percentages. "
    "A qualified medical professional must review any result.</div>",
    unsafe_allow_html=True,
)