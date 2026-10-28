"""
evaluate.py
Evaluation metrics calculation, confusion matrix plotting, ROC curves, and feature importance.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)


def calculate_metrics(y_true, y_pred, y_proba=None) -> dict:
    """
    Computes standard binary classification evaluation metrics.
    Handles zero-division safely.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def get_confusion_matrix_breakdown(y_true, y_pred) -> dict:
    """
    Computes TN, FP, FN, TP counts from binary ground truth and predictions.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "matrix": cm.tolist(),
    }


def plot_confusion_matrix(y_true, y_pred, save_path: str | Path | None = None, model_name: str = "Final Model"):
    """
    Plots a professionally labeled confusion matrix with clear clinical interpretation notes.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(6.5, 5))
    labels = [
        [f"TN\n{tn}\n(Correct Neg)", f"FP\n{fp}\n(False Alarm)"],
        [f"FN\n{fn}\n(Missed Positive)", f"TP\n{tp}\n(Correct Pos)"],
    ]

    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted Negative (0)", "Predicted Positive (1)"],
        yticklabels=["Actual Negative (0)", "Actual Positive (1)"],
        ax=ax,
        linewidths=1,
        annot_kws={"fontsize": 11, "weight": "bold"},
    )

    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, weight="bold")
    ax.set_ylabel("Actual Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_roc_curve(y_true, y_proba, save_path: str | Path | None = None, model_name: str = "Final Model"):
    """
    Plots the Receiver Operating Characteristic (ROC) curve with AUC score.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(fpr, tpr, color="#2b5c8f", lw=2.5, label=f"{model_name} (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], color="#999999", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=13, weight="bold")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_feature_importance(pipeline, feature_names: list[str], save_path: str | Path | None = None, model_name: str = "Model"):
    """
    Plots feature importance (for tree models) or absolute standardized coefficients (for linear models).
    Uses non-causal phrasing.
    """
    classifier = pipeline.named_steps["classifier"]
    
    if hasattr(classifier, "feature_importances_"):
        raw_vals = classifier.feature_importances_
        metric_label = "Relative Importance Score (Gini Importance)"
    elif hasattr(classifier, "coef_"):
        raw_vals = np.abs(classifier.coef_[0])
        metric_label = "Absolute Standardized Coefficient Magnitude (|β|)"
    else:
        return None

    # Sort in descending order
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": raw_vals,
    }).sort_values(by="Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.barh(importance_df["Feature"], importance_df["Importance"], color="#2b5c8f", edgecolor="#1a3857")
    ax.set_xlabel(metric_label, fontsize=11)
    ax.set_ylabel("Feature Name", fontsize=11)
    ax.set_title(f"Features Most Influential in Trained {model_name}\n(Pattern Correlation, Not Causation)", fontsize=12, weight="bold")
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.3f}",
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=9)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig
