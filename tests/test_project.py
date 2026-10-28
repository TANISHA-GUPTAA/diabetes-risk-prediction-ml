"""
test_project.py
Comprehensive automated sanity tests for the Diabetes Risk Prediction project.
Tests data loading, preprocessing, model loading, predictions, metadata alignment, and Streamlit compatibility.
"""

from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "diabetes.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
METRICS_PATH = PROJECT_ROOT / "models" / "model_metrics.json"
METADATA_PATH = PROJECT_ROOT / "models" / "feature_metadata.json"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def test_artifacts_exist():
    """Verify that all core project artifacts exist."""
    assert DATA_PATH.exists(), f"Dataset missing at {DATA_PATH}"
    assert MODEL_PATH.exists(), f"Model missing at {MODEL_PATH}"
    assert METRICS_PATH.exists(), f"Metrics JSON missing at {METRICS_PATH}"
    assert METADATA_PATH.exists(), f"Metadata JSON missing at {METADATA_PATH}"
    assert (PROJECT_ROOT / "app.py").exists(), "app.py missing"
    assert (PROJECT_ROOT / "requirements.txt").exists(), "requirements.txt missing"


def test_dataset_integrity():
    """Verify dataset columns, shape, and types."""
    df = pd.read_csv(DATA_PATH)
    assert df.shape[0] == 768, f"Expected 768 rows, got {df.shape[0]}"
    assert df.shape[1] == 9, f"Expected 9 columns, got {df.shape[1]}"
    expected_cols = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    assert list(df.columns) == expected_cols


def test_feature_metadata_consistency():
    """Verify feature metadata strictly matches model expectations."""
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    assert "feature_names" in metadata
    assert len(metadata["feature_names"]) == 8
    assert "Outcome" not in metadata["feature_names"]
    assert metrics["selected_model_name"] in [
        "Decision Tree", "Random Forest", "Logistic Regression", "K-Nearest Neighbors"
    ]


def test_pipeline_predictions():
    """Verify model inference on various representative cases."""
    pipeline = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    feature_names = metadata["feature_names"]

    test_cases = [
        # 1. Typical low-risk case
        {
            "Pregnancies": 1, "Glucose": 85, "BloodPressure": 66, "SkinThickness": 29,
            "Insulin": 94, "BMI": 26.6, "DiabetesPedigreeFunction": 0.351, "Age": 31
        },
        # 2. Typical high-risk case
        {
            "Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35,
            "Insulin": 160, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50
        },
        # 3. Imputation case (zeros in glucose, insulin, skin thickness)
        {
            "Pregnancies": 0, "Glucose": 0, "BloodPressure": 70, "SkinThickness": 0,
            "Insulin": 0, "BMI": 28.0, "DiabetesPedigreeFunction": 0.45, "Age": 25
        },
        # 4. Extreme values
        {
            "Pregnancies": 8, "Glucose": 195, "BloodPressure": 90, "SkinThickness": 40,
            "Insulin": 300, "BMI": 45.0, "DiabetesPedigreeFunction": 1.2, "Age": 55
        },
    ]

    for i, tc in enumerate(test_cases):
        df_in = pd.DataFrame([tc])[feature_names]
        pred = pipeline.predict(df_in)
        assert pred[0] in [0, 1], f"Invalid prediction class {pred[0]}"
        
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(df_in)[0]
            assert len(proba) == 2, "Binary probability distribution must have 2 entries"
            assert np.isclose(np.sum(proba), 1.0), "Probabilities must sum to 1.0"


def test_figures_exist():
    """Verify that all core analysis and evaluation figures are present."""
    expected_figures = [
        "class_distribution.png",
        "correlation_heatmap.png",
        "glucose_distribution.png",
        "bmi_distribution.png",
        "age_distribution.png",
        "confusion_matrix.png",
        "roc_curve.png",
        "feature_importance.png",
        "model_comparison.png",
    ]
    for fig_name in expected_figures:
        fig_file = FIGURES_DIR / fig_name
        assert fig_file.exists(), f"Figure missing: {fig_name}"
        assert fig_file.stat().st_size > 1000, f"Figure file {fig_name} appears empty"


if __name__ == "__main__":
    test_artifacts_exist()
    test_dataset_integrity()
    test_feature_metadata_consistency()
    test_pipeline_predictions()
    test_figures_exist()
    print("ALL TESTS PASSED SUCCESSFULLY!")
