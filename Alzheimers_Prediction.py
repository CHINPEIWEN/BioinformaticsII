import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore")

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(
    page_title="Alzheimer's Disease Prediction",
    layout="wide"
)

# =========================
# CONSTANTS
# =========================
TARGET_COLUMN = "Diagnosis"
FEATURES_TO_REMOVE = [
    'PatientID', 'DoctorInCharge',
    'Gender', 'Ethnicity',
    'AlcoholConsumption', 'SleepQuality'
]

# =========================
# TITLE
# =========================
st.title("🧠 Alzheimer's Disease Prediction Application")
st.markdown("Machine Learning–based Clinical Decision Support Demo")

# =========================
# TABS
# =========================
tab1, tab2 = st.tabs([
    "📂 Dataset Upload & Model Training",
    "🧍 Manual Alzheimer Prediction"
])

# =====================================================
# TAB 1 — DATASET UPLOAD & TRAINING
# =====================================================
with tab1:

    uploaded_file = st.file_uploader(
        "📂 Upload Alzheimer’s Dataset (CSV)",
        type=["csv"]
    )

    if uploaded_file is None:
        st.info("Please upload a dataset to begin.")
        st.stop()

    # Load data
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully!")

    # =========================
    # DATA OVERVIEW
    # =========================
    st.header("📄 Dataset Overview")
    st.write(f"Shape: {df.shape}")
    st.dataframe(df.head())

    # =========================
    # FEATURE REMOVAL
    # =========================
    removed_cols = [c for c in FEATURES_TO_REMOVE if c in df.columns]
    df = df.drop(columns=removed_cols)

    # =========================
    # EDA
    # =========================
    st.header("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        df[TARGET_COLUMN].value_counts().plot(kind="bar", ax=ax)
        ax.set_title("Diagnosis Distribution")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        df[TARGET_COLUMN].value_counts().plot(
            kind="pie", autopct="%1.1f%%", ax=ax
        )
        ax.set_ylabel("")
        st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # =========================
    # PREPROCESSING
    # =========================
    st.header("⚙️ Data Preprocessing")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Encode target if needed
    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Handle missing values
    X = X.fillna(X.median())

    # Keep numeric only
    X = X.select_dtypes(include=np.number)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    st.success("Preprocessing completed.")

    # =========================
    # MODEL TRAINING
    # =========================
    st.header("🤖 Model Training & Evaluation")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Support Vector Machine": SVC(probability=True),
        "K-Nearest Neighbors": KNeighborsClassifier()
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_prob),
            "ModelObject": model,
            "Prob": y_prob
        })

    results_df = pd.DataFrame(results).drop(columns=["ModelObject", "Prob"])
    st.dataframe(results_df)

    # =========================
    # ROC CURVES
    # =========================
    st.subheader("📈 ROC Curve Comparison")

    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["Prob"])
        ax.plot(fpr, tpr, label=f"{r['Model']} (AUC={r['AUC']:.2f})")

    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

    # =========================
    # BEST MODEL SELECTION
    # =========================
    best = max(results, key=lambda x: x["Accuracy"])
    best_model = best["ModelObject"]

    st.success(
        f"🏆 Best Model: **{best['Model']}** "
        f"(Accuracy = {best['Accuracy']:.4f})"
    )

    # =========================
    # FEATURE IMPORTANCE (TOP 5)
    # =========================
    feature_names = X.columns

    if hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_[0])

    elif hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_

    else:
        importances = np.abs(
            pd.DataFrame(X, columns=feature_names)
            .corrwith(pd.Series(y))
            .values
        )

    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    top_5_features = feature_importance_df.head(5)["Feature"].tolist()

    st.subheader("⭐ Top 5 Most Important Features")
    st.table(feature_importance_df.head(5))

    # Save model and scaler
    joblib.dump(best_model, "best_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

# =====================================================
# TAB 2 — MANUAL PREDICTION
# =====================================================
with tab2:

    st.header("🔍 Manual Alzheimer’s Disease Prediction")
    st.markdown(
        "This prediction uses **only the Top 5 most important clinical features** "
        "identified by the best-performing model."
    )

    st.write("### 🧠 Selected Features")
    st.write(top_5_features)

    with st.form("manual_prediction_form"):
        manual_inputs = {}

        for feature in top_5_features:
            manual_inputs[feature] = st.number_input(
                feature,
                value=float(X[feature].median())
            )

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Build full feature vector
        full_input = pd.DataFrame(
            [{col: X[col].median() for col in X.columns}]
        )

        for feature in top_5_features:
            full_input[feature] = manual_inputs[feature]

        input_scaled = scaler.transform(full_input)

        prediction = best_model.predict(input_scaled)[0]
        probability = best_model.predict_proba(input_scaled)[0][1]

        st.subheader("🧾 Prediction Result")

        if prediction == 1:
            st.error("🧠 Alzheimer’s Disease Detected")
        else:
            st.success("✅ No Alzheimer’s Disease Detected")

        st.info(f"Probability of Alzheimer’s Disease: **{probability:.2f}**")
