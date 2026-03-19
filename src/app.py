"""
app.py

This file creates a Streamlit dashboard to demonstrate the
Network Intrusion Detection system trained on the UNSW-NB15 dataset.

Users can upload network traffic CSV files and the trained model
will classify each record as Normal or Attack.
"""

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from PIL import Image
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Network Intrusion Detection Dashboard",
    layout="wide"
)

st.title("🔐 Network Intrusion Detection System")
st.caption("Dataset: UNSW-NB15")

st.markdown("""
This dashboard demonstrates a **Machine Learning based Intrusion Detection System**.

The system analyzes **network traffic features** and classifies them into:

🟢 **Normal Traffic**  
🔴 **Attack Traffic**

### How to use
1. Upload a network traffic **CSV file**
2. The model processes the data
3. Predictions and summary statistics are displayed
""")

# -----------------------------
# Load Model + Pipeline Objects
# -----------------------------
@st.cache_resource
def load_models():

    model = joblib.load("models/best_model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
    scaler = joblib.load("models/scaler.pkl")
    pca = joblib.load("models/pca.pkl")

    return model, preprocessor, scaler, pca


model, preprocessor, scaler, pca = load_models()

# -----------------------------
# Sidebar Upload
# -----------------------------
st.sidebar.header("Upload Network Traffic Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

# -----------------------------
# Prediction Section
# -----------------------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # Remove label if present
    if "label" in df.columns:
        X = df.drop("label", axis=1)
    else:
        X = df.copy()

    

    try:

        # -----------------------------
        # Apply Saved Preprocessing
        # -----------------------------
        X_processed = preprocessor.transform(X)

        # Scaling
        X_scaled = scaler.transform(X_processed)

        # PCA
        X_pca = pca.transform(X_scaled)

        # -----------------------------
        # Prediction
        # -----------------------------
        predictions = model.predict(X_pca)

        results = df.copy()
        results["Prediction"] = predictions

        results["Prediction"] = results["Prediction"].map({
            0: "Normal",
            1: "Attack"
        })

        st.subheader("Prediction Results")
        st.dataframe(results)

        # -----------------------------
        # Summary Metrics
        # -----------------------------
        attack_count = (results["Prediction"] == "Attack").sum()
        normal_count = (results["Prediction"] == "Normal").sum()

        col1, col2 = st.columns(2)

        col1.metric("Normal Traffic", normal_count)
        col2.metric("Attack Traffic", attack_count)

        # -----------------------------
        # Prediction Pie Chart
        # -----------------------------
        st.subheader("Traffic Distribution")

        fig, ax = plt.subplots()

        ax.pie(
            [normal_count, attack_count],
            labels=["Normal", "Attack"],
            autopct="%1.1f%%",
            colors=["green", "red"]
        )

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction Error: {e}")


# -----------------------------
# Dataset Analysis Section
# -----------------------------
st.markdown("---")
st.header("Dataset Analysis")

col1, col2 = st.columns(2)

if os.path.exists("outputs/before_smote.png"):
    with col1:
        st.subheader("Before SMOTE")
        st.image("outputs/before_smote.png")

if os.path.exists("outputs/after_smote.png"):
    with col2:
        st.subheader("After SMOTE")
        st.image("outputs/after_smote.png")

if os.path.exists("outputs/correlation_heatmap.png"):
    st.subheader("Feature Correlation Heatmap")
    st.image("outputs/correlation_heatmap.png")

if os.path.exists("outputs/feature_importance.png"):
    st.subheader("Feature Importance")
    st.image("outputs/feature_importance.png")

# -----------------------------
# Model Comparison
# -----------------------------
st.markdown("---")
st.header("Model Performance Comparison")

st.markdown("""
Different machine learning models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

The best performing model was selected for deployment.
""")

if os.path.exists("outputs/model_comparison.png"):
    st.image(
        "outputs/model_comparison.png",
        caption="Model Performance Comparison"
    )

# -----------------------------
# Best Model Information
# -----------------------------
st.markdown("---")
st.header("Best Model")

st.markdown("""
The best performing model selected for intrusion detection is:

### Support Vector Machine (RBF Kernel)

Reasons for selecting SVM:

• Works well with **high dimensional datasets**  
• Handles **non-linear decision boundaries**  
• Provides strong performance in **network intrusion detection**

The trained SVM model is used to classify incoming network traffic.
""")

st.success("Active Detection Model: SVM (RBF Kernel)")