"""
data_utils.py
Utilities for loading, inspecting, and visualizing the diabetes dataset.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_default_data_path() -> Path:
    """Returns the default relative path to the diabetes dataset."""
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "data" / "diabetes.csv"


def load_data(filepath: str | Path | None = None) -> pd.DataFrame:
    """Loads the dataset from CSV and returns a pandas DataFrame."""
    if filepath is None:
        filepath = get_default_data_path()
    path_obj = Path(filepath)
    if not path_obj.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path_obj}")
    return pd.read_csv(path_obj)


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Computes basic structural and statistical summary of the dataset."""
    zero_counts = {
        col: int((df[col] == 0).sum())
        for col in df.columns
    }
    class_counts = df["Outcome"].value_counts().to_dict()
    class_proportions = df["Outcome"].value_counts(normalize=True).to_dict()

    return {
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_explicit_nulls": {col: int(val) for col, val in df.isnull().sum().items()},
        "zero_counts": zero_counts,
        "class_distribution": {
            "negative_0": int(class_counts.get(0, 0)),
            "positive_1": int(class_counts.get(1, 0)),
            "negative_0_ratio": float(class_proportions.get(0, 0.0)),
            "positive_1_ratio": float(class_proportions.get(1, 0.0)),
        },
    }


def generate_eda_plots(df: pd.DataFrame, output_dir: str | Path | None = None) -> list[str]:
    """
    Generates meaningful EDA figures and saves them to outputs/figures/.
    Returns a list of saved file paths.
    """
    if output_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        output_dir = project_root / "outputs" / "figures"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved_files = []

    # Set aesthetic style
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    # 1. Target Class Distribution
    fig, ax = plt.subplots(figsize=(6, 4.5))
    counts = df["Outcome"].value_counts()
    bars = ax.bar(["Negative (0)", "Positive (1)"], [counts.get(0, 0), counts.get(1, 0)], color=["#2b5c8f", "#d95f02"], width=0.5)
    ax.set_title("Target Class Distribution (Outcome)", fontsize=13, weight="bold")
    ax.set_xlabel("Class Label", fontsize=11)
    ax.set_ylabel("Number of Samples", fontsize=11)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height} ({height/len(df)*100:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, weight="bold")
    p1 = out_path / "class_distribution.png"
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    saved_files.append(str(p1))

    # 2. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", mask=mask, cbar_kws={"shrink": 0.8}, ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap", fontsize=13, weight="bold")
    p2 = out_path / "correlation_heatmap.png"
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    saved_files.append(str(p2))

    # 3. Glucose Distribution by Outcome
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.kdeplot(data=df[df["Glucose"] > 0], x="Glucose", hue="Outcome", common_norm=False, fill=True, alpha=0.3, ax=ax, palette=["#2b5c8f", "#d95f02"])
    ax.set_title("Plasma Glucose Concentration Distribution by Outcome", fontsize=13, weight="bold")
    ax.set_xlabel("Glucose (mg/dL)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    p3 = out_path / "glucose_distribution.png"
    fig.savefig(p3, dpi=300)
    plt.close(fig)
    saved_files.append(str(p3))

    # 4. BMI Distribution by Outcome
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.kdeplot(data=df[df["BMI"] > 0], x="BMI", hue="Outcome", common_norm=False, fill=True, alpha=0.3, ax=ax, palette=["#2b5c8f", "#d95f02"])
    ax.set_title("Body Mass Index (BMI) Distribution by Outcome", fontsize=13, weight="bold")
    ax.set_xlabel("BMI (kg/m²)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    p4 = out_path / "bmi_distribution.png"
    fig.savefig(p4, dpi=300)
    plt.close(fig)
    saved_files.append(str(p4))

    # 5. Age Distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(data=df, x="Age", hue="Outcome", multiple="stack", bins=20, ax=ax, palette=["#2b5c8f", "#d95f02"])
    ax.set_title("Age Distribution by Outcome", fontsize=13, weight="bold")
    ax.set_xlabel("Age (Years)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    p5 = out_path / "age_distribution.png"
    fig.savefig(p5, dpi=300)
    plt.close(fig)
    saved_files.append(str(p5))

    return saved_files
