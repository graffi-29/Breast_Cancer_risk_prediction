from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

from train import load_dataset


PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

st.set_page_config(
    page_title="Breast Cancer Model Evaluation",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap');
    .stApp { background: #f3f5ef; color: #18231e; }
    .block-container { max-width: 1240px; padding-top: 2.4rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: #18231e; }
    h1 { font-family: 'Fraunces', Georgia, serif; font-size: 2.8rem; font-weight: 500; }
    h2, h3, p, label, [data-testid='stMetricLabel'] { font-family: 'DM Sans', sans-serif; }
    [data-testid='stMetric'] {
        background: #ffffff; border: 1px solid #dce2d8; border-radius: 8px;
        padding: 16px 18px; min-height: 112px;
    }
    [data-testid='stMetricLabel'] { color: #59685f; font-size: 0.82rem; }
    [data-testid='stMetricValue'] { color: #18231e; font-family: 'DM Mono', monospace; }
    .eyebrow { color: #24735f; font: 500 0.76rem 'DM Mono', monospace; letter-spacing: 0.08em; }
    .subhead { color: #59685f; font: 400 1rem 'DM Sans', sans-serif; margin-top: -0.7rem; }
    .section-label { color: #24735f; font: 500 0.72rem 'DM Mono', monospace; letter-spacing: 0.08em; }
    .model-tag { color: #153f35; background: #dceee5; border-radius: 4px; padding: 7px 10px; font: 500 0.8rem 'DM Mono', monospace; }
    [data-testid='stAlert'] { border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def read_metrics() -> dict:
    with METRICS_PATH.open(encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


@st.cache_resource
def read_model(model_path: str):
    return joblib.load(model_path)


def chart_layout(figure: go.Figure, height: int = 340) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=24, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#34443b", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return figure


if not METRICS_PATH.exists():
    st.title("Model evaluation")
    st.error("No trained model found. Run `python train.py` from the project folder first.")
    st.stop()

metrics = read_metrics()
if "candidate_cv_roc_auc" not in metrics:
    st.title("Model evaluation")
    st.error("The saved results are out of date. Run `python train.py` to refresh them.")
    st.stop()

model_path = ARTIFACTS_DIR / metrics["model_file"]
if not model_path.exists():
    st.title("Model evaluation")
    st.error("The saved model file is missing. Run `python train.py` to recreate it.")
    st.stop()

model = read_model(str(model_path))
features, target, _ = load_dataset()
_, X_test, _, y_test = train_test_split(
    features,
    target,
    test_size=metrics["test_size"],
    random_state=metrics["random_state"],
    stratify=target,
)
test_scores = (
    model.decision_function(X_test)
    if hasattr(model, "decision_function")
    else model.predict_proba(X_test)[:, 1]
)
false_positive_rate, true_positive_rate, _ = roc_curve(y_test, test_scores)

st.markdown("<div class='eyebrow'>WISCONSIN DIAGNOSTIC DATASET · MODEL REVIEW</div>", unsafe_allow_html=True)
heading, selected = st.columns([0.78, 0.22], vertical_alignment="center")
with heading:
    st.title("What did the model learn?")
    st.markdown(
        "<div class='subhead'>A held-out evaluation of three classifiers on 30 tumor measurements.</div>",
        unsafe_allow_html=True,
    )
with selected:
    display_name = metrics["selected_model"].replace("_", " ").title()
    st.markdown(
        f"<span class='model-tag'>SELECTED · {display_name.upper()}</span>",
        unsafe_allow_html=True,
    )

st.write("")
metric_columns = st.columns(4)
metric_columns[0].metric("Test ROC-AUC", f"{metrics['test_roc_auc']:.3f}")
metric_columns[1].metric("Malignant recall", f"{metrics['test_malignant_recall']:.1%}")
metric_columns[2].metric("Benign recall / specificity", f"{metrics['test_benign_recall_specificity']:.1%}")
metric_columns[3].metric("Malignant precision", f"{metrics['test_malignant_precision']:.1%}")

st.write("")
comparison_column, roc_column = st.columns([0.9, 1.1], gap="large")

with comparison_column:
    st.markdown("<div class='section-label'>01 / MODEL SELECTION</div>", unsafe_allow_html=True)
    st.subheader("Cross-validation comparison")
    scores = metrics["candidate_cv_roc_auc"]
    model_names = [name.replace("_", " ").title() for name in scores]
    values = list(scores.values())
    colors = ["#24735f" if name == metrics["selected_model"] else "#bdc9be" for name in scores]
    comparison = go.Figure(
        go.Bar(
            x=values,
            y=model_names,
            orientation="h",
            marker_color=colors,
            text=[f"{value:.3f}" for value in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>Mean CV ROC-AUC: %{x:.3f}<extra></extra>",
        )
    )
    comparison.update_xaxes(range=[0.9, 1.015], title="Mean ROC-AUC across folds", gridcolor="#e2e7df")
    comparison.update_yaxes(autorange="reversed", showgrid=False)
    comparison.update_layout(showlegend=False)
    st.plotly_chart(chart_layout(comparison), use_container_width=True, config={"displayModeBar": False})
    st.caption("Hyperparameters were selected using training data only. Higher ROC-AUC is better.")

with roc_column:
    st.markdown("<div class='section-label'>02 / HELD-OUT TEST SET</div>", unsafe_allow_html=True)
    st.subheader("Receiver operating characteristic")
    roc_figure = go.Figure()
    roc_figure.add_trace(
        go.Scatter(
            x=false_positive_rate,
            y=true_positive_rate,
            mode="lines",
            name=f"Selected model · AUC {metrics['test_roc_auc']:.3f}",
            line=dict(color="#24735f", width=3),
            hovertemplate="False positive rate: %{x:.2f}<br>True positive rate: %{y:.2f}<extra></extra>",
        )
    )
    roc_figure.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Chance",
            line=dict(color="#d28a52", width=1.5, dash="dash"),
        )
    )
    roc_figure.update_xaxes(title="False positive rate", range=[0, 1], gridcolor="#e2e7df")
    roc_figure.update_yaxes(title="True positive rate", range=[0, 1.02], gridcolor="#e2e7df")
    st.plotly_chart(chart_layout(roc_figure), use_container_width=True, config={"displayModeBar": False})
    st.caption("The curve shows the sensitivity/specificity trade-off across thresholds.")

st.markdown("<div class='section-label'>03 / CLASSIFICATION ERRORS</div>", unsafe_allow_html=True)
st.subheader("What was correctly classified, and what was missed?")
matrix_column, explanation_column = st.columns([0.9, 1.1], gap="large", vertical_alignment="center")
with matrix_column:
    matrix = np.asarray(metrics["confusion_matrix"])
    labels = ["Benign", "Malignant"]
    heatmap = go.Figure(
        go.Heatmap(
            z=matrix,
            x=[f"Predicted {label.lower()}" for label in labels],
            y=[f"Actual {label.lower()}" for label in labels],
            text=matrix,
            texttemplate="%{text}",
            textfont=dict(size=22, color="#18231e"),
            colorscale=[[0, "#edf1e9"], [1, "#79b5a0"]],
            showscale=False,
            hovertemplate="%{y}<br>%{x}: %{z} cases<extra></extra>",
        )
    )
    heatmap.update_xaxes(side="bottom", showgrid=False)
    heatmap.update_yaxes(autorange="reversed", showgrid=False)
    st.plotly_chart(chart_layout(heatmap, height=310), use_container_width=True, config={"displayModeBar": False})

with explanation_column:
    false_negatives = int(matrix[1, 0])
    false_positives = int(matrix[0, 1])
    st.markdown(
        f"**{false_negatives} malignant case(s)** were classified as benign in this test split. "
        f"There were **{false_positives} benign case(s)** classified as malignant."
    )
    st.caption(
        f"Evaluation used {int(matrix.sum())} held-out cases from a stratified split. "
        "Rows are actual labels; columns are model predictions."
    )

st.divider()
st.markdown(
    "**Educational project only.** This small public dataset cannot establish safety or performance "
    "for real patients. The dashboard is not a diagnostic tool and must not be used for medical decisions."
)