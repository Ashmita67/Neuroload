# app.py

import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import io
import shap

# ---------------- Load Data & Model ----------------
X = np.load("data/processed/X_features.npy")
y = np.load("data/processed/y.npy")
feature_names = np.load("data/processed/feature_names.npy", allow_pickle=True)
model = joblib.load("models/random_forest_eeg.pkl")

# SHAP Explainer
explainer = shap.Explainer(model.named_steps['randomforestclassifier'])
X_scaled = model.named_steps['standardscaler'].transform(X)
shap_values = explainer(X_scaled)

# ---------------- Sidebar Navigation ----------------
st.set_page_config(page_title="NeuroLoad Dashboard", layout="wide")
st.sidebar.title("🧠 NeuroLoad Dashboard")
section = st.sidebar.radio("Navigation", ["Overview", "Visualizations", "Test Sample", "Upload EEG & Predict", "Report Generator"])

# ---------------- Section: Overview ----------------
if section == "Overview":
    st.title("🧠 NeuroLoad: EEG-Based Cognitive Load Detection")
    st.markdown("A machine learning system to detect mental fatigue using EEG signals.")

    st.subheader("📊 Dataset Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Samples", len(X))
        st.metric("Feature Dimensions", X.shape[1])
    with col2:
        counts = np.bincount(y)
        st.metric("Relaxed (0)", counts[0])
        st.metric("Load (1)", counts[1])

# ---------------- Section: Visualizations ----------------
elif section == "Visualizations":
    st.title("📈 Feature Space Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        st.image("reports/figures/pca_projection.png", caption="PCA Projection", use_container_width=True)
    with col2:
        st.image("reports/figures/tsne_projection.png", caption="t-SNE Projection", use_container_width=True)

    st.title("📌 SHAP Explainability")
    st.image("reports/figures/shap_feature_importance_class1.png", caption="Feature Importance", use_container_width=True)
    st.image("reports/figures/shap_summary_class1.png", caption="SHAP Summary", use_container_width=True)

elif section == "Test Sample":
    st.title("🧪 Test the Model on a Dataset Sample")
    sample_idx = st.slider("Select a sample index", 0, len(X) - 1, 0)

    X_sample = X[sample_idx].reshape(1, -1)
    pred = model.predict(X_sample)[0]
    proba = model.predict_proba(X_sample)[0][pred]

    st.write(f"Prediction: **{'Cognitive Load' if pred else 'Relaxed'}**")
    st.progress(int(proba * 100))

# ---- SHAP explanation for this sample ----
    st.subheader("🔍 SHAP Explanation for this Sample")

    sample_scaled = model.named_steps["standardscaler"].transform(X_sample)
    sample_expl = explainer(sample_scaled)

    # If explainer returns both classes, slice the predicted class
    if sample_expl.values.ndim == 3:
        sample_expl.values      = sample_expl.values[0, pred]
        sample_expl.base_values = sample_expl.base_values[0, pred]
        sample_expl.data        = sample_expl.data[0]

    shap.plots.waterfall(sample_expl,max_display=10, show=False)
    st.pyplot(plt.gcf())


# ---------------- Section: Upload EEG & Predict ----------------
elif section == "Upload EEG & Predict":
    st.title("📤 Upload EEG Window (.npy) and Predict")

    uploaded = st.file_uploader("Upload a NumPy file shaped (14, N_samples)", type="npy")

    if uploaded:
        try:
            eeg_window = np.load(uploaded)
            if eeg_window.shape[0] != 14:
                st.error(f"Expected 14 channels, got {eeg_window.shape[0]}.")
            else:
                # 1️⃣  Feature extraction (reuse the same function you used offline)
                from src.features import extract_features        # returns list
                feat_vector = np.array(extract_features(eeg_window)).reshape(1, -1)

                # 2️⃣  Prediction
                pred_class = model.predict(feat_vector)[0]
                pred_prob  = model.predict_proba(feat_vector)[0][pred_class]

                st.success(f"Prediction: {'Cognitive Load' if pred_class else 'Relaxed'} "
                           f"({pred_prob:.2%} confidence)")

                # 3️⃣  SHAP explanation for this sample
                #     - scale first, then create a single-sample explanation
                feat_scaled = model.named_steps['standardscaler'].transform(feat_vector)
                single_expl = explainer(feat_scaled)

                # If explainer returns both classes, select the predicted class
                if isinstance(single_expl.values, np.ndarray) and single_expl.values.ndim == 3:
                    single_expl.values      = single_expl.values[0, pred_class]
                    single_expl.base_values = single_expl.base_values[0, pred_class]
                    single_expl.data        = single_expl.data[0]

                st.subheader("🔍 SHAP Explanation for Uploaded Sample")
                shap.plots.waterfall(single_expl, feature_names=feature_names,
                                     max_display=10, show=False)
                st.pyplot(bbox_inches="tight")

        except Exception as e:
            st.error(f"Error while processing file: {e}")

# ---------------- Section: Report Generator ----------------
elif section == "Report Generator":
    st.title("📝 Auto-Generated Report")
    st.markdown("Edit the summary below and download as needed.")
    report_text = st.text_area("Generated Report", value=f"""
Project: NeuroLoad
Model: Random Forest Classifier
Feature Count: {X.shape[1]}
Accuracy: ~70%
Top Features (via SHAP): AF3_theta_rel, FC6_entropy, O2_beta_alpha_ratio

This project uses EEG signals to detect cognitive load using machine learning.
Trained on UCI EEG Eye State dataset, the model helps identify mental fatigue and focus levels in real time.
""", height=300)

    st.download_button("📥 Download Report as .txt", data=report_text, file_name="neuroload_report.txt")
