# music.py — Streamlit app for Music Genre Classification
# -------------------------------------------------------
# Features:
# • Single-image predict (one model or compare all models)
# • Batch predict (one selected model) + Batch compare all models
# • Sidebar: validation metrics (Accuracy / F1 / Precision / Recall), table, and bar chart
# • Single-image "Compare All": winner badge, highlighted row, colored bar chart with labels, top-3 per model
# • Batch "Compare All": unified table + single bar chart of average confidence per model across files
# • CSV downloads, spinners, graceful errors, SVM probability fallback

# ----------------- Imports -----------------
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from PIL import Image
from tensorflow.keras.models import load_model
from skimage.color import rgb2gray
from skimage.feature import hog

# NEW: robust model fetch
from pathlib import Path
import requests

# ----------------- Page setup -----------------
st.set_page_config(
    page_title="Music Genre Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🎵 Music Genre Classification")
st.write(
    "Upload spectrograms to predict the genre using **CNN**, **Logistic Regression**, or **SVM** — "
    "and compare each model’s **validation performance** (Accuracy/F1/Precision/Recall)."
)

# ----------------- Labels -----------------
GENRE_LABELS = [
    "Classical", "Jazz", "Pop", "Rock", "Hip-Hop",
    "Country", "Electronic", "Reggae", "Blues", "Metal"
]

# ----------------- Validation metrics -----------------
@st.cache_data(show_spinner=False)
def load_validation_metrics():
    """
    Load per-model validation metrics from metrics.json, else use defaults.
    Expected keys per model: accuracy, precision, recall, f1, report (str).
    """
    default = {
        "CNN": {
            "accuracy": 0.62, "precision": 0.65, "recall": 0.62, "f1": 0.61,
            "report": """precision    recall  f1-score   support
1  0.50 0.25 0.33 20
2  0.90 0.90 0.90 20
3  0.73 0.55 0.63 20
4  0.50 0.35 0.41 20
5  0.77 0.50 0.61 20
6  0.67 0.90 0.77 20
7  0.88 0.75 0.81 20
8  0.58 0.70 0.64 20
9  0.36 0.80 0.50 20
10 0.62 0.50 0.56 20

accuracy 0.62 (n=200)"""
        },
        "Logistic Regression": {
            "accuracy": 0.63, "precision": 0.64, "recall": 0.63, "f1": 0.62,
            "report": """precision    recall  f1-score   support
1  0.77 0.50 0.61 20
2  0.79 0.95 0.86 20
3  0.57 0.65 0.60 20
4  0.44 0.20 0.28 20
5  0.72 0.65 0.68 20
6  0.82 0.70 0.76 20
7  0.79 0.75 0.77 20
8  0.60 0.75 0.67 20
9  0.48 0.60 0.53 20
10 0.41 0.55 0.47 20

accuracy 0.63 (n=200)"""
        },
        "SVM": {
            "accuracy": 0.65, "precision": 0.66, "recall": 0.65, "f1": 0.64,
            "report": """precision    recall  f1-score   support
1  0.75 0.75 0.75 20
2  0.86 0.95 0.90 20
3  0.45 0.65 0.53 20
4  0.56 0.25 0.34 20
5  0.68 0.75 0.71 20
6  0.78 0.70 0.74 20
7  0.82 0.70 0.76 20
8  0.63 0.60 0.62 20
9  0.52 0.60 0.56 20
10 0.52 0.55 0.54 20

accuracy 0.65 (n=200)"""
        }
    }
    try:
        with open("metrics.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            for k in list(data.keys()):
                for m in ("accuracy", "precision", "recall", "f1"):
                    data[k].setdefault(m, None)
                data[k].setdefault("report", "")
            return data
    except Exception:
        return default

MODEL_METRICS = load_validation_metrics()

@st.cache_data(show_spinner=False)
def metrics_table():
    rows = []
    for name, m in MODEL_METRICS.items():
        rows.append({
            "Model": name,
            "Accuracy": m.get("accuracy"),
            "F1": m.get("f1"),
            "Precision": m.get("precision"),
            "Recall": m.get("recall"),
        })
    df = pd.DataFrame(rows)
    return df[["Model", "Accuracy", "F1", "Precision", "Recall"]]

# ----------------- Model URLs & download helpers (NEW) -----------------
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

CNN_MODEL_URL = "https://github.com/Manjotkaur226/music-genre-classification-cnn/releases/download/v1-models/best_cnn_model.h5"
LR_MODEL_URL  = "https://github.com/Manjotkaur226/music-genre-classification-cnn/releases/download/v1-models/logistic_regression_model.pkl"
SVM_MODEL_URL = "https://github.com/Manjotkaur226/music-genre-classification-cnn/releases/download/v1-models/svm_model.pkl"

def _download_if_missing(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    with st.spinner(f"Downloading model: {dest.name}"):
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return dest

# ----------------- Load models -----------------
@st.cache_resource(show_spinner=False)
def load_models():
    # ensure files exist (download once, then cached)
    cnn_path = _download_if_missing(CNN_MODEL_URL, MODEL_DIR / "best_cnn_model.h5")
    lr_path  = _download_if_missing(LR_MODEL_URL,  MODEL_DIR / "logistic_regression_model.pkl")
    svm_path = _download_if_missing(SVM_MODEL_URL, MODEL_DIR / "svm_model.pkl")

    # Keras CNN
    cnn_model = load_model(str(cnn_path))

    # Traditional models
    with open(lr_path, "rb") as f:
        lr_model = pickle.load(f)
    with open(svm_path, "rb") as f:
        svm_model = pickle.load(f)

    return cnn_model, lr_model, svm_model

cnn_model, lr_model, svm_model = load_models()

# ----------------- Helpers -----------------
def preprocess_for_cnn(img: Image.Image):
    arr = np.array(img.resize((224, 224))) / 255.0
    return arr[np.newaxis, ...]  # (1, 224, 224, 3)

def extract_hog_features(image: Image.Image):
    gray = rgb2gray(np.array(image.resize((224, 224))))
    feats = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
    return feats.reshape(1, -1)

def safe_predict_proba(model, X):
    """Use predict_proba if available; else softmax(decision_function)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    scores = model.decision_function(X)
    if scores.ndim == 1:
        scores = np.vstack([scores, -scores]).T
    e = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def topk(probs, k=3):
    idx = np.argsort(probs[0])[::-1][:k]
    return [(int(i), float(probs[0][i])) for i in idx]

# ----------------- Sidebar: validation metrics -----------------
with st.sidebar:
    st.header("📊 Validation (held-out test set)")
    df_metrics = metrics_table()
    metric_choice = st.radio(
        "Compare by metric",
        ["Accuracy", "F1", "Precision", "Recall"],
        index=0
    )

    view = df_metrics[["Model", metric_choice]].sort_values(metric_choice, ascending=False)
    st.dataframe(
        view.rename(columns={metric_choice: f"Validation {metric_choice}"}).style
            .format({f"Validation {metric_choice}": "{:.2%}"}),
        hide_index=True, use_container_width=True
    )

    # Small bar chart of the chosen validation metric
    chart_df = view.sort_values(metric_choice, ascending=True)
    st.bar_chart(chart_df.set_index("Model"))

    # Full reports
    with st.expander("View classification reports"):
        for name, m in MODEL_METRICS.items():
            val = m.get(metric_choice.lower())
            st.markdown(f"**{name}** — {metric_choice}: `{val:.2%}`" if val is not None else f"**{name}**")
            if m.get("report"):
                st.code(m["report"])

# ----------------- Tabs -----------------
tab_pred, tab_reports, tab_about = st.tabs(["🔮 Predict", "📜 Reports", "ℹ️ About"])

with tab_reports:
    st.subheader("Full Classification Reports")
    for name, m in MODEL_METRICS.items():
        st.markdown(f"**{name}** — Accuracy: `{m.get('accuracy', 0):.2%}` • F1: `{m.get('f1', 0):.2%}`")
        st.code(m.get("report", ""))

with tab_about:
    st.markdown("""
**Data:** Spectrogram images of audio clips  
**Models:** CNN (Keras), Logistic Regression (HOG), SVM (HOG)  
**Preprocessing:** 224×224 RGB for CNN; HOG (16×16 cells, 2×2 blocks) for classical models  
**Output:** Predicted genre + confidence; comparison against validation metrics  
""")

# ----------------- Predict Tab -----------------
with tab_pred:
    # -------- Single Predict --------
    st.subheader("Single Image")
    uploaded_file = st.file_uploader("Upload a spectrogram image", type=["jpg", "jpeg", "png"], key="single")
    model_choice = st.selectbox("Select Model", ("CNN", "Logistic Regression", "SVM", "Compare All Models"))

    if uploaded_file and uploaded_file.size > 5_000_000:
        st.warning("Large image (>5MB). Resizing may be slow.")

    if uploaded_file and st.button("Predict"):
        try:
            with st.spinner("Running inference…"):
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Image", use_container_width=True)

                if model_choice == "CNN":
                    x = preprocess_for_cnn(image)
                    p = cnn_model.predict(x)
                    cls = int(np.argmax(p)); conf = float(p[0][cls])
                    st.success(f"🎯 Predicted: **{GENRE_LABELS[cls]}**")
                    st.write(f"💡 Confidence: **{conf:.2%}**")
                    st.caption("Top-3 predictions")
                    for i, c in topk(p, k=3):
                        st.write(f"- {GENRE_LABELS[i]} — {c:.2%}")

                elif model_choice == "Logistic Regression":
                    x = extract_hog_features(image)
                    p = safe_predict_proba(lr_model, x)
                    cls = int(np.argmax(p)); conf = float(p[0][cls])
                    st.success(f"🎯 Predicted: **{GENRE_LABELS[cls]}**")
                    st.write(f"💡 Confidence: **{conf:.2%}**")
                    st.caption("Top-3 predictions")
                    for i, c in topk(p, k=3):
                        st.write(f"- {GENRE_LABELS[i]} — {c:.2%}")

                elif model_choice == "SVM":
                    x = extract_hog_features(image)
                    p = safe_predict_proba(svm_model, x)
                    cls = int(np.argmax(p)); conf = float(p[0][cls])
                    st.success(f"🎯 Predicted: **{GENRE_LABELS[cls]}**")
                    st.write(f"💡 Confidence: **{conf:.2%}**")
                    st.caption("Top-3 predictions")
                    for i, c in topk(p, k=3):
                        st.write(f"- {GENRE_LABELS[i]} — {c:.2%}")

                else:  # Compare All Models (single)
                    rows = []

                    # CNN
                    x_cnn = preprocess_for_cnn(image)
                    p_cnn = cnn_model.predict(x_cnn)
                    cls_cnn = int(np.argmax(p_cnn)); conf_cnn = float(p_cnn[0][cls_cnn])
                    rows.append(["CNN", GENRE_LABELS[cls_cnn], conf_cnn, MODEL_METRICS["CNN"][metric_choice.lower()]])

                    # HOG once
                    x_hog = extract_hog_features(image)

                    # Logistic Regression
                    p_lr = safe_predict_proba(lr_model, x_hog)
                    cls_lr = int(np.argmax(p_lr)); conf_lr = float(p_lr[0][cls_lr])
                    rows.append(["Logistic Regression", GENRE_LABELS[cls_lr], conf_lr,
                                 MODEL_METRICS["Logistic Regression"][metric_choice.lower()]])

                    # SVM
                    p_svm = safe_predict_proba(svm_model, x_hog)
                    cls_svm = int(np.argmax(p_svm)); conf_svm = float(p_svm[0][cls_svm])
                    rows.append(["SVM", GENRE_LABELS[cls_svm], conf_svm, MODEL_METRICS["SVM"][metric_choice.lower()]])

                    compare_df = pd.DataFrame(
                        rows, columns=["Model", "Predicted Genre", "Confidence", f"Validation {metric_choice}"]
                    )

                    # Winner badge
                    best_idx = compare_df["Confidence"].idxmax()
                    best_model = compare_df.loc[best_idx, "Model"]
                    best_genre = compare_df.loc[best_idx, "Predicted Genre"]
                    best_conf  = compare_df.loc[best_idx, "Confidence"]
                    st.success(f"🏆 Highest confidence: **{best_model}** → **{best_genre}** ({best_conf:.2%})")

                    st.subheader("🔍 Predictions from All Models")
                    st.dataframe(
                        compare_df.style
                            .format({"Confidence": "{:.2%}", f"Validation {metric_choice}": "{:.2%}"})
                            .highlight_max(subset=["Confidence"], color="#E6FFEE"),
                        hide_index=True, use_container_width=True
                    )

                    # Colored bar chart (Altair)
                    st.subheader("📊 Confidence by Model")
                    chart_df = compare_df[["Model", "Confidence"]]
                    chart = (
                        alt.Chart(chart_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("Model:N", sort="-y"),
                            y=alt.Y("Confidence:Q", axis=alt.Axis(format="%")),
                            color=alt.Color("Model:N", legend=None),
                            tooltip=[alt.Tooltip("Model:N"), alt.Tooltip("Confidence:Q", format=".2%")]
                        )
                    )
                    labels = (
                        alt.Chart(chart_df)
                        .mark_text(dy=-6)
                        .encode(
                            x=alt.X("Model:N", sort="-y"),
                            y=alt.Y("Confidence:Q"),
                            text=alt.Text("Confidence:Q", format=".1%")
                        )
                    )
                    st.altair_chart(chart + labels, use_container_width=True)

                    # Top-3 per model
                    with st.expander("Show Top-3 predictions per model"):
                        def _topk_table(name, probs):
                            idx = np.argsort(probs[0])[::-1][:3]
                            tk = [(GENRE_LABELS[i], float(probs[0][i])) for i in idx]
                            st.markdown(f"**{name}**")
                            st.table(pd.DataFrame(tk, columns=["Class", "Probability"]).assign(
                                Probability=lambda d: (d["Probability"]*100).round(2).astype(str) + "%"
                            ))

                        _topk_table("CNN", p_cnn)
                        _topk_table("Logistic Regression", p_lr)
                        _topk_table("SVM", p_svm)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # -------- Batch Predict --------
    st.subheader("Batch Predict")
    files = st.file_uploader(
        "Upload multiple spectrograms",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch"
    )

    # Option A: Run batch for ONE selected model
    if files and st.button("Run Batch – One Model"):
        rows = []
        try:
            with st.spinner(f"Running batch inference for {model_choice}…"):
                for f in files:
                    img = Image.open(f).convert("RGB")
                    if model_choice == "CNN":
                        p = cnn_model.predict(preprocess_for_cnn(img))
                    elif model_choice == "Logistic Regression":
                        p = safe_predict_proba(lr_model, extract_hog_features(img))
                    elif model_choice == "SVM":
                        p = safe_predict_proba(svm_model, extract_hog_features(img))
                    else:
                        # If user picked "Compare All Models" but clicked One Model, default to CNN
                        p = cnn_model.predict(preprocess_for_cnn(img))
                    cls = int(np.argmax(p)); conf = float(p[0][cls])
                    rows.append({"File": f.name, "Predicted Genre": GENRE_LABELS[cls], "Confidence": conf})

            batch_df = pd.DataFrame(rows)
            st.dataframe(batch_df.style.format({"Confidence": "{:.2%}"}), use_container_width=True, hide_index=True)

            csv = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download results (CSV)", data=csv, file_name="batch_results.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

    # Option B: Run batch for ALL models (Compare All) + average-confidence chart
    if files and st.button("Run Batch – Compare All Models"):
        all_rows = []
        try:
            with st.spinner("Running batch inference for all models…"):
                for f in files:
                    img = Image.open(f).convert("RGB")

                    # CNN
                    p_cnn = cnn_model.predict(preprocess_for_cnn(img))
                    cls_cnn = int(np.argmax(p_cnn)); conf_cnn = float(p_cnn[0][cls_cnn])
                    all_rows.append([
                        f.name, "CNN", GENRE_LABELS[cls_cnn], conf_cnn,
                        MODEL_METRICS["CNN"][metric_choice.lower()]
                    ])

                    # HOG once
                    hog_feat = extract_hog_features(img)

                    # Logistic Regression
                    p_lr = safe_predict_proba(lr_model, hog_feat)
                    cls_lr = int(np.argmax(p_lr)); conf_lr = float(p_lr[0][cls_lr])
                    all_rows.append([
                        f.name, "Logistic Regression", GENRE_LABELS[cls_lr], conf_lr,
                        MODEL_METRICS["Logistic Regression"][metric_choice.lower()]
                    ])

                    # SVM
                    p_svm = safe_predict_proba(svm_model, hog_feat)
                    cls_svm = int(np.argmax(p_svm)); conf_svm = float(p_svm[0][cls_svm])
                    all_rows.append([
                        f.name, "SVM", GENRE_LABELS[cls_svm], conf_svm,
                        MODEL_METRICS["SVM"][metric_choice.lower()]
                    ])

            compare_df = pd.DataFrame(
                all_rows,
                columns=["File", "Model", "Predicted Genre", "Confidence", f"Validation {metric_choice}"]
            )

            # Table of all file-model results
            st.dataframe(
                compare_df.style.format({"Confidence": "{:.2%}", f"Validation {metric_choice}": "{:.2%}"}),
                hide_index=True, use_container_width=True
            )

            # Single bar chart of average confidence per model (across files)
            st.subheader("📊 Average Confidence per Model (across uploaded files)")
            avg_conf = (compare_df.groupby("Model", as_index=False)["Confidence"]
                        .mean()
                        .sort_values("Confidence", ascending=True))
            avg_chart = (
                alt.Chart(avg_conf)
                .mark_bar()
                .encode(
                    x=alt.X("Model:N"),
                    y=alt.Y("Confidence:Q", axis=alt.Axis(format="%")),
                    color=alt.Color("Model:N", legend=None),
                    tooltip=[alt.Tooltip("Model:N"), alt.Tooltip("Confidence:Q", format=".2%")]
                )
            )
            avg_labels = (
                alt.Chart(avg_conf)
                .mark_text(dy=-6)
                .encode(
                    x=alt.X("Model:N"),
                    y=alt.Y("Confidence:Q"),
                    text=alt.Text("Confidence:Q", format=".1%")
                )
            )
            st.altair_chart(avg_chart + avg_labels, use_container_width=True)

            # Download results
            csv = compare_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download results (CSV)", data=csv, file_name="batch_compare_results.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Batch comparison failed: {e}")
