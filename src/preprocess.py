"""
preprocess.py
Robust, leak-free preprocessing pipeline definitions for diabetes risk classification.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Clinically and empirically justified column groupings:
# In the Pima dataset, 0 is biologically impossible for these 5 vital measures and represents missing entries:
ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# For pregnancies, 0 is a completely legitimate value (never pregnant).
# DiabetesPedigreeFunction and Age are strictly positive.
LEGITIMATE_COLS = ["Pregnancies", "DiabetesPedigreeFunction", "Age"]

ALL_FEATURE_COLS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def create_preprocessor() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that:
    1. Imputes zero values as missing for Glucose, BloodPressure, SkinThickness, Insulin, and BMI using median.
    2. Imputes any general missing values for other columns using median.
    3. Scales all numeric features using StandardScaler.
    All statistics are learned strictly from the training partition during pipeline.fit().
    """
    zero_impute_pipeline = Pipeline([
        ("imputer", SimpleImputer(missing_values=0, strategy="median")),
        ("scaler", StandardScaler()),
    ])

    normal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("zero_impute", zero_impute_pipeline, ZERO_AS_MISSING_COLS),
            ("normal", normal_pipeline, LEGITIMATE_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def create_pipeline(classifier) -> Pipeline:
    """
    Combines the leak-free preprocessing transformer and a supervised classifier
    into an end-to-end scikit-learn Pipeline.
    """
    preprocessor = create_preprocessor()
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])
    return pipeline
