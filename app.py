"""
app.py
Diabetes Risk Prediction — Streamlit Web Application
Educational Machine Learning Mini-Project
"""

from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Prediction | ML Mini Project",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# CONSTANTS & PATHS
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "diabetes.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
METRICS_PATH = PROJECT_ROOT / "models" / "model_metrics.json"
METADATA_PATH = PROJECT_ROOT / "models" / "feature_metadata.json"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

DISCLAIMER_TEXT = (
    "Disclaimer: This application is an educational machine learning project. "
    "Predictions are based on patterns in the training dataset and should not be "
    "considered medical advice or a medical diagnosis."
)

# ---------------------------------------------------------
# CACHED LOADERS
# ---------------------------------------------------------
@st.cache_resource
def load_trained_model():
    """Loads the serialized scikit-learn pipeline."""
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading trained pipeline: {e}")
        return None


@st.cache_data
def load_project_data():
    """Loads the raw dataset."""
    if not DATA_PATH.exists():
        return None
    try:
        return pd.read_csv(DATA_PATH)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


@st.cache_data
def load_metrics_json():
    """Loads calculated evaluation metrics and model comparison data."""
    if not METRICS_PATH.exists():
        return None
    try:
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading metrics metadata: {e}")
        return None


@st.cache_data
def load_feature_metadata():
    """Loads feature schema and observed statistical limits."""
    if not METADATA_PATH.exists():
        return None
    try:
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading feature metadata: {e}")
        return None


# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
st.sidebar.title("🩺 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Data Analysis", "Risk Prediction", "Model Performance", "About Project"],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Project:** Diabetes Risk Prediction\n\n"
    "**Type:** Academic ML Mini-Project\n\n"
    "**Target:** Binary Classification (Outcome)"
)
st.sidebar.warning(f"⚠️ **Note:** {DISCLAIMER_TEXT}")

# ---------------------------------------------------------
# PAGE 1: HOME
# ---------------------------------------------------------
if page == "Home":
    st.title("🩺 Diabetes Risk Prediction Using Machine Learning")
    st.caption("An End-to-End Supervised Binary Classification Academic Mini-Project")

    st.warning(f"**Medical Framing Disclaimer:** {DISCLAIMER_TEXT}")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("📌 Project Overview")
        st.write(
            "This mini-project demonstrates the rigorous application of machine learning techniques "
            "for binary classification. Using historical diagnostic and physiological metrics, candidate "
            "algorithms are trained, tuned via stratified cross-validation, and evaluated to assess their "
            "ability to distinguish positive from negative risk classes."
        )

        st.subheader("🎯 Primary Objectives")
        st.markdown(
            """
            - **Leak-Free Preprocessing:** Handle hidden missing values (implausible zeros in physiological measures) strictly within scikit-learn pipelines.
            - **Multi-Model Comparison:** Train and compare Logistic Regression, K-Nearest Neighbors, Decision Trees, and Random Forests.
            - **Cross-Validation & Tuning:** Optimize hyperparameter configurations using 5-Fold Stratified Cross-Validation.
            - **Interactive Demonstration:** Provide an intuitive Streamlit interface for academic presentation and real-time inference.
            """
        )

    with col2:
        metrics_data = load_metrics_json()
        if metrics_data:
            st.subheader("🏆 Selected Model Summary")
            st.metric(
                label="Best Model",
                value=metrics_data.get("selected_model_name", "Decision Tree"),
            )
            tm = metrics_data.get("test_metrics", {})
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                st.metric("Test F1-Score", f"{tm.get('f1', 0.0):.4f}")
                st.metric("Test Accuracy", f"{tm.get('accuracy', 0.0):.4f}")
            with mcol2:
                st.metric("Test Recall", f"{tm.get('recall', 0.0):.4f}")
                st.metric("Test ROC-AUC", f"{tm.get('roc_auc', 0.0):.4f}")

    st.markdown("---")
    st.subheader("🔄 Machine Learning Project Workflow")
    workflow_steps = [
        "**1. Dataset Acquisition**\n\nLoad public benchmark records",
        "**2. Data Validation**\n\nInspect data types & biologically implausible zeros",
        "**3. EDA**\n\nAnalyze distributions & correlations",
        "**4. Train/Test Split**\n\n80/20 Stratified split prior to any fitting",
        "**5. Pipeline Fitting**\n\nMedian imputation & standard scaling",
        "**6. Model Tuning**\n\n5-Fold CV parameter optimization",
        "**7. Evaluation**\n\nF1, Recall, ROC-AUC on held-out test data",
        "**8. Web Interface**\n\nReal-time inference & interpretation",
    ]
    w_cols = st.columns(4)
    for i, step in enumerate(workflow_steps[:4]):
        with w_cols[i]:
            st.info(step)
    w_cols2 = st.columns(4)
    for i, step in enumerate(workflow_steps[4:]):
        with w_cols2[i]:
            st.success(step)

# ---------------------------------------------------------
# PAGE 2: DATA ANALYSIS
# ---------------------------------------------------------
elif page == "Data Analysis":
    st.title("📊 Exploratory Data Analysis & Quality Assessment")
    st.write(
        "Explore distributions, identify hidden missing entries, and examine correlation patterns within the training data."
    )

    df = load_project_data()
    if df is None:
        st.error("Dataset not found at `data/diabetes.csv`. Please verify file placement.")
    else:
        tab1, tab2, tab3 = st.tabs(["📋 Data Overview", "📈 Distributions", "🔗 Correlations & Relationships"])

        with tab1:
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)

            scol1, scol2, scol3, scol4 = st.columns(4)
            with scol1:
                st.metric("Total Records", len(df))
            with scol2:
                st.metric("Total Features", df.shape[1] - 1)
            with scol3:
                st.metric("Class 0 (Negative)", int((df["Outcome"] == 0).sum()))
            with scol4:
                st.metric("Class 1 (Positive)", int((df["Outcome"] == 1).sum()))

            st.markdown("### 🔍 Zero-Value Analysis (Biologically Implausible Entries)")
            st.write(
                "In physiological features, 0 values indicate missing readings rather than actual measurements. "
                "The table below shows the exact zero counts across all features:"
            )
            zero_summary = pd.DataFrame({
                "Feature": df.columns,
                "Zero Count": [(df[c] == 0).sum() for c in df.columns],
                "Percentage (%)": [round((df[c] == 0).mean() * 100, 2) for c in df.columns],
                "Methodological Interpretation": [
                    "Legitimate 0 (Nulliparous / never pregnant)" if c == "Pregnancies"
                    else "Target label (Negative class)" if c == "Outcome"
                    else "Strictly positive continuous feature" if c in ["DiabetesPedigreeFunction", "Age"]
                    else "Biologically implausible (Treated as missing; imputed via training median)"
                    for c in df.columns
                ]
            })
            st.dataframe(zero_summary, use_container_width=True)

        with tab2:
            st.subheader("Feature Distributions by Outcome")
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                p_class = FIGURES_DIR / "class_distribution.png"
                if p_class.exists():
                    st.image(str(p_class), caption="Target Class Distribution (Outcome)")
                p_gluc = FIGURES_DIR / "glucose_distribution.png"
                if p_gluc.exists():
                    st.image(str(p_gluc), caption="Plasma Glucose Distribution by Class")
            with dcol2:
                p_bmi = FIGURES_DIR / "bmi_distribution.png"
                if p_bmi.exists():
                    st.image(str(p_bmi), caption="Body Mass Index (BMI) Distribution by Class")
                p_age = FIGURES_DIR / "age_distribution.png"
                if p_age.exists():
                    st.image(str(p_age), caption="Age Distribution by Class")

        with tab3:
            st.subheader("Correlation Heatmap")
            st.write(
                "Pearson correlation coefficients between all diagnostic features and the target variable (`Outcome`)."
            )
            p_corr = FIGURES_DIR / "correlation_heatmap.png"
            if p_corr.exists():
                st.image(str(p_corr), caption="Feature Correlation Heatmap", use_container_width=True)

# ---------------------------------------------------------
# PAGE 3: RISK PREDICTION
# ---------------------------------------------------------
elif page == "Risk Prediction":
    st.title("🎯 Diabetes Risk Classification Prediction")
    st.caption("Interactive Supervised Prediction via Saved Scikit-Learn Pipeline")

    st.warning(f"**Academic Notice:** {DISCLAIMER_TEXT}")

    pipeline = load_trained_model()
    metadata = load_feature_metadata()

    if pipeline is None or metadata is None:
        st.error("Model artifacts or feature metadata missing. Please run `python -m src.train_model` first.")
    else:
        st.markdown("### ⚙️ Input Feature Parameters")
        st.write("Enter values manually or select a preset sample profile:")

        profile_choice = st.selectbox(
            "Select Sample Preset Profile",
            ["Custom Input", "Sample Profile A (Lower Historical Risk)", "Sample Profile B (Higher Historical Risk)"],
        )

        presets = {
            "Sample Profile A (Lower Historical Risk)": {
                "Pregnancies": 1, "Glucose": 85, "BloodPressure": 66, "SkinThickness": 29,
                "Insulin": 94, "BMI": 26.6, "DiabetesPedigreeFunction": 0.351, "Age": 31
            },
            "Sample Profile B (Higher Historical Risk)": {
                "Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35,
                "Insulin": 160, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50
            }
        }

        features_info = metadata["features"]
        user_inputs = {}

        icol1, icol2 = st.columns(2)
        half = len(metadata["feature_names"]) // 2
        cols_col1 = metadata["feature_names"][:half]
        cols_col2 = metadata["feature_names"][half:]

        def get_val(f_name):
            if profile_choice in presets:
                return presets[profile_choice][f_name]
            return features_info[f_name]["default_ui_value"]

        with icol1:
            for f_name in cols_col1:
                info = features_info[f_name]
                step_val = 0.01 if "float" in info["dtype"] else 1
                user_inputs[f_name] = st.number_input(
                    label=f"{f_name} ({info['description'][:40]}...)",
                    min_value=float(0.0),
                    max_value=float(info["max_observed"] * 1.5),
                    value=float(get_val(f_name)),
                    step=float(step_val),
                    help=info["description"],
                    key=f"input_{f_name}",
                )

        with icol2:
            for f_name in cols_col2:
                info = features_info[f_name]
                step_val = 0.01 if "float" in info["dtype"] else 1
                user_inputs[f_name] = st.number_input(
                    label=f"{f_name} ({info['description'][:40]}...)",
                    min_value=float(0.0),
                    max_value=float(info["max_observed"] * 1.5),
                    value=float(get_val(f_name)),
                    step=float(step_val),
                    help=info["description"],
                    key=f"input_{f_name}",
                )

        predict_button = st.button("🔍 Predict Risk Class", type="primary", use_container_width=True)

        if predict_button:
            # Construct DataFrame in exact feature order
            input_df = pd.DataFrame([user_inputs])[metadata["feature_names"]]

            try:
                # 1. Pipeline Prediction
                raw_pred = pipeline.predict(input_df)[0]

                # 2. Probability extraction (locate positive class '1' safely)
                if hasattr(pipeline, "predict_proba"):
                    classes = list(pipeline.classes_)
                    pos_idx = classes.index(1) if 1 in classes else 1
                    prob_array = pipeline.predict_proba(input_df)[0]
                    pos_prob = float(prob_array[pos_idx])
                else:
                    pos_prob = None

                st.markdown("---")
                st.subheader("📋 Model Prediction Output")

                rcol1, rcol2 = st.columns([1, 1])

                with rcol1:
                    if raw_pred == 1:
                        st.error("### Predicted Class: Positive (1)")
                    else:
                        st.success("### Predicted Class: Negative (0)")

                    if pos_prob is not None:
                        st.info(
                            f"**Statistical Class Probability:** The model assigned a **{pos_prob * 100:.2f}%** "
                            f"probability to the positive class based on patterns learned from the training dataset."
                        )
                        st.progress(pos_prob)
                    else:
                        st.info("Probability estimation is not supported by this classifier.")

                with rcol2:
                    st.markdown("#### Input Feature Summary")
                    summary_df = pd.DataFrame([user_inputs]).T.reset_index()
                    summary_df.columns = ["Feature", "Submitted Value"]
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)

                st.caption(
                    "Note: This output is a supervised machine learning classification score derived strictly "
                    "from historical training data patterns and does not constitute a clinical assessment."
                )

            except Exception as ex:
                st.error(f"Inference error: {ex}")

# ---------------------------------------------------------
# PAGE 4: MODEL PERFORMANCE
# ---------------------------------------------------------
elif page == "Model Performance":
    st.title("📈 Model Evaluation & Performance Metrics")
    st.write(
        "Review empirical performance, confusion matrix breakdowns, ROC-AUC analysis, and cross-validation comparisons."
    )

    metrics_data = load_metrics_json()
    if not metrics_data:
        st.error("Model metrics file missing. Please run `python -m src.train_model` first.")
    else:
        st.subheader(f"Selected Best Model: {metrics_data.get('selected_model_name')}")
        st.markdown(f"> **Selection Rationale:** {metrics_data.get('selection_rationale')}")

        st.markdown("### 🏆 Held-Out Test Set Performance")
        tm = metrics_data.get("test_metrics", {})
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{tm.get('accuracy', 0.0):.4f}")
        m2.metric("Precision", f"{tm.get('precision', 0.0):.4f}")
        m3.metric("Recall", f"{tm.get('recall', 0.0):.4f}")
        m4.metric("F1-Score", f"{tm.get('f1', 0.0):.4f}")
        m5.metric("ROC-AUC", f"{tm.get('roc_auc', 0.0):.4f}")

        st.markdown("### 📊 Model Comparison Table")
        comp_df = pd.DataFrame(metrics_data.get("model_comparison", []))
        if not comp_df.empty:
            display_cols = ["model_name", "cv_f1", "accuracy", "precision", "recall", "f1", "roc_auc"]
            renamed_cols = {
                "model_name": "Model",
                "cv_f1": "5-Fold CV F1",
                "accuracy": "Test Accuracy",
                "precision": "Test Precision",
                "recall": "Test Recall",
                "f1": "Test F1-Score",
                "roc_auc": "Test ROC-AUC",
            }
            st.dataframe(
                comp_df[display_cols].rename(columns=renamed_cols).sort_values(by="Test F1-Score", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### 🖼️ Diagnostic Evaluation Figures")
        col_fig1, col_fig2 = st.columns(2)
        with col_fig1:
            p_cm = FIGURES_DIR / "confusion_matrix.png"
            if p_cm.exists():
                st.image(str(p_cm), caption="Confusion Matrix with TP, TN, FP, FN counts", use_container_width=True)
            p_comp = FIGURES_DIR / "model_comparison.png"
            if p_comp.exists():
                st.image(str(p_comp), caption="Candidate Model Comparison on Held-Out Test Set", use_container_width=True)

        with col_fig2:
            p_roc = FIGURES_DIR / "roc_curve.png"
            if p_roc.exists():
                st.image(str(p_roc), caption="Receiver Operating Characteristic (ROC) Curve", use_container_width=True)
            p_imp = FIGURES_DIR / "feature_importance.png"
            if p_imp.exists():
                st.image(str(p_imp), caption="Feature Importance (Pattern Influence, Not Causation)", use_container_width=True)

# ---------------------------------------------------------
# PAGE 5: ABOUT PROJECT
# ---------------------------------------------------------
elif page == "About Project":
    st.title("ℹ️ About This Project")
    st.caption("Academic Machine Learning Mini-Project Documentation")

    st.subheader("1. Problem Statement")
    st.write(
        "Early identification of diabetes risk indicators allows for timely lifestyle modifications and clinical monitoring. "
        "The goal of this educational project is to explore whether standard supervised classification algorithms "
        "can predict high-risk diabetes classifications from diagnostic features such as plasma glucose, body mass index, "
        "and age, without introducing methodological data leakage."
    )

    st.subheader("2. Dataset Overview")
    st.write(
        "This project utilizes the public Pima Indians Diabetes Database (National Institute of Diabetes and Digestive and "
        "Kidney Diseases). It contains 768 patient records with 8 continuous predictors and 1 binary outcome target."
    )
    st.markdown(
        """
        - **Total Records:** 768
        - **Target:** `Outcome` (0 = Negative / Non-diabetic, 1 = Positive / Diabetic)
        - **Key Features:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
        """
    )

    st.subheader("3. Methodological Rigor & Data Leakage Prevention")
    st.markdown(
        """
        1. **Strict Partitioning:** The dataset is partitioned into an 80% training set and 20% test set using `stratify=y` before any imputer or scaler statistics are computed.
        2. **Biologically Implausible Zeros:** Five features (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) contained 0 values representing missing entries. These are imputed via median learned exclusively on training folds.
        3. **Standardization:** Features are standardized using `StandardScaler` fitted strictly on training data inside a `Pipeline`.
        4. **Hyperparameter Selection:** Optimization is conducted via 5-fold cross-validation (`GridSearchCV`) solely on training data.
        """
    )

    st.subheader("4. Technologies Used")
    st.markdown(
        """
        - **Language:** Python 3.13
        - **Data Analysis & Processing:** Pandas, NumPy, Scikit-Learn
        - **Visualizations:** Matplotlib, Seaborn
        - **Web Framework:** Streamlit
        - **Serialization:** Joblib, JSON
        """
    )

    st.subheader("5. Limitations & Future Scope")
    st.markdown(
        """
        - **Dataset Size:** 768 records is relatively small; larger, multi-institutional datasets would improve generalization.
        - **Demographic Homogeneity:** The sample is restricted to female subjects of Pima Indian heritage aged 21 and older. Models trained on this cohort cannot be generalized to diverse global populations.
        - **Future Scope:** Incorporation of longitudinal clinical records, electronic health record (EHR) features, and modern ensemble architectures (e.g. LightGBM, CatBoost) with SHAP value interpretability.
        """
    )

    st.warning(f"**Medical Framing Disclaimer:** {DISCLAIMER_TEXT}")
