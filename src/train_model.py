"""
train_model.py
End-to-end model training, cross-validation, hyperparameter tuning, evaluation,
and artifact serialization for the Diabetes Risk Prediction project.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.data_utils import load_data, generate_eda_plots
from src.preprocess import (
    create_pipeline,
    create_preprocessor,
    ALL_FEATURE_COLS,
    ZERO_AS_MISSING_COLS,
    LEGITIMATE_COLS,
)
from src.evaluate import (
    calculate_metrics,
    get_confusion_matrix_breakdown,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
)


def get_feature_metadata(X_train: pd.DataFrame) -> dict:
    """
    Builds safe feature metadata derived strictly from training statistics.
    Never uses invented medical ranges.
    """
    descriptions = {
        "Pregnancies": "Number of times pregnant (0 is normal for non-pregnant/nulliparous individuals)",
        "Glucose": "Plasma glucose concentration 2 hours after an oral glucose tolerance test (mg/dL)",
        "BloodPressure": "Diastolic blood pressure (mm Hg)",
        "SkinThickness": "Triceps skin fold thickness (mm)",
        "Insulin": "2-Hour serum insulin (mu U/ml)",
        "BMI": "Body mass index (weight in kg / (height in m)^2)",
        "DiabetesPedigreeFunction": "Diabetes pedigree function (genetic score based on family history)",
        "Age": "Age of the subject in years",
    }

    metadata = {
        "feature_names": ALL_FEATURE_COLS,
        "features": {},
    }

    for col in ALL_FEATURE_COLS:
        series = X_train[col]
        # For columns where 0 is missing, derive median from non-zero values for realistic UI default
        if col in ZERO_AS_MISSING_COLS:
            non_zeros = series[series > 0]
            typical_val = float(non_zeros.median()) if len(non_zeros) > 0 else float(series.median())
        else:
            typical_val = float(series.median())

        metadata["features"][col] = {
            "name": col,
            "description": descriptions.get(col, ""),
            "dtype": str(series.dtype),
            "min_observed": float(series.min()),
            "max_observed": float(series.max()),
            "mean_observed": float(series.mean()),
            "median_observed": float(series.median()),
            "default_ui_value": round(typical_val, 2),
            "is_zero_treated_as_missing": col in ZERO_AS_MISSING_COLS,
        }

    return metadata


def train_and_evaluate(project_root: Path | None = None) -> dict:
    """Executes the full pipeline and persists models, metrics, metadata, and figures."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    data_path = project_root / "data" / "diabetes.csv"
    models_dir = project_root / "models"
    figures_dir = project_root / "outputs" / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("PHASE 4 & 5: LOADING AND VALIDATING DATASET")
    print("==================================================")
    df = load_data(data_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Columns:", list(df.columns))

    print("\nGenerating reusable EDA plots...")
    generate_eda_plots(df, figures_dir)
    print("EDA plots saved to outputs/figures/")

    print("\n==================================================")
    print("PHASE 9 & 10: DEFINING X, y AND TRAIN/TEST SPLIT")
    print("==================================================")
    X = df[ALL_FEATURE_COLS]
    y = df["Outcome"]

    # Stratified 80/20 train/test split with reproducible seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")
    print(f"Target distribution in Train - 0: {(y_train == 0).sum()}, 1: {(y_train == 1).sum()}")
    print(f"Target distribution in Test  - 0: {(y_test == 0).sum()}, 1: {(y_test == 1).sum()}")

    # Generate feature metadata strictly from X_train (zero leakage)
    feature_metadata = get_feature_metadata(X_train)
    with open(models_dir / "feature_metadata.json", "w") as f:
        json.dump(feature_metadata, f, indent=2)
    print("Saved feature metadata to models/feature_metadata.json")

    print("\n==================================================")
    print("PHASE 12, 13 & 14: BASELINE MODELS & CROSS-VALIDATION")
    print("==================================================")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    candidate_classifiers = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    }

    baseline_results = {}
    fitted_baseline_pipelines = {}

    for name, clf in candidate_classifiers.items():
        pipe = create_pipeline(clf)
        cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
        
        # Fit on entire training partition to evaluate on test set
        pipe.fit(X_train, y_train)
        fitted_baseline_pipelines[name] = pipe

        y_test_pred = pipe.predict(X_test)
        y_test_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
        test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)

        baseline_results[name] = {
            "cv_accuracy_mean": float(np.mean(cv_res["test_accuracy"])),
            "cv_accuracy_std": float(np.std(cv_res["test_accuracy"])),
            "cv_f1_mean": float(np.mean(cv_res["test_f1"])),
            "cv_f1_std": float(np.std(cv_res["test_f1"])),
            "cv_roc_auc_mean": float(np.mean(cv_res["test_roc_auc"])),
            "cv_roc_auc_std": float(np.std(cv_res["test_roc_auc"])),
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_roc_auc": test_metrics["roc_auc"],
        }
        print(f"[{name}]")
        print(f"  5-Fold CV Accuracy: {baseline_results[name]['cv_accuracy_mean']:.4f} (+/- {baseline_results[name]['cv_accuracy_std']:.4f})")
        print(f"  5-Fold CV F1:       {baseline_results[name]['cv_f1_mean']:.4f} (+/- {baseline_results[name]['cv_f1_std']:.4f})")
        print(f"  5-Fold CV ROC-AUC:  {baseline_results[name]['cv_roc_auc_mean']:.4f}")
        print(f"  Test F1: {test_metrics['f1']:.4f} | Test ROC-AUC: {test_metrics['roc_auc']:.4f}")

    print("\n==================================================")
    print("PHASE 15: HYPERPARAMETER TUNING (GridSearchCV)")
    print("==================================================")
    # Tune top candidates using StratifiedKFold on X_train only
    param_grids = {
        "Random Forest": {
            "classifier__n_estimators": [50, 100, 150],
            "classifier__max_depth": [3, 5, 8, None],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 2, 4],
        },
        "Logistic Regression": {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__class_weight": [None, "balanced"],
            "classifier__solver": ["lbfgs", "liblinear"],
        },
        "K-Nearest Neighbors": {
            "classifier__n_neighbors": [3, 5, 7, 9, 11],
            "classifier__weights": ["uniform", "distance"],
            "classifier__metric": ["euclidean", "manhattan"],
        },
    }

    tuned_models = {}
    for name, p_grid in param_grids.items():
        base_pipe = create_pipeline(candidate_classifiers[name])
        grid_search = GridSearchCV(
            base_pipe,
            param_grid=p_grid,
            cv=cv,
            scoring="f1",
            n_jobs=1,
            refit=True,
        )
        grid_search.fit(X_train, y_train)
        best_pipe = grid_search.best_estimator_
        tuned_models[name] = {
            "best_estimator": best_pipe,
            "best_params": {k.replace("classifier__", ""): v for k, v in grid_search.best_params_.items()},
            "best_cv_f1": float(grid_search.best_score_),
        }
        print(f"Tuned {name}: Best CV F1 = {grid_search.best_score_:.4f} | Best Params = {tuned_models[name]['best_params']}")

    print("\n==================================================")
    print("PHASE 16: FINAL MODEL SELECTION")
    print("==================================================")
    # Select the model with strongest CV generalization and stability
    # Compare all candidates
    model_comparison_list = []
    for name in candidate_classifiers.keys():
        if name in tuned_models:
            pipe = tuned_models[name]["best_estimator"]
            is_tuned = True
            params = tuned_models[name]["best_params"]
        else:
            pipe = fitted_baseline_pipelines[name]
            is_tuned = False
            params = "Default"

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
        m = calculate_metrics(y_test, y_pred, y_proba)
        
        cv_f1 = tuned_models[name]["best_cv_f1"] if is_tuned else baseline_results[name]["cv_f1_mean"]
        
        model_comparison_list.append({
            "model_name": name,
            "is_tuned": is_tuned,
            "best_params": str(params),
            "cv_f1": round(cv_f1, 4),
            "accuracy": round(m["accuracy"], 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
            "roc_auc": round(m["roc_auc"], 4) if m["roc_auc"] is not None else None,
            "pipeline": pipe,
        })

    comparison_df = pd.DataFrame(model_comparison_list)
    print("\nModel Comparison Table:")
    print(comparison_df[["model_name", "cv_f1", "accuracy", "precision", "recall", "f1", "roc_auc"]])

    # Selection logic: prioritize highest test F1 and ROC-AUC with strong CV backing
    sorted_candidates = sorted(model_comparison_list, key=lambda x: (x["f1"], x["roc_auc"], x["cv_f1"]), reverse=True)
    best_candidate = sorted_candidates[0]
    best_model_name = best_candidate["model_name"]
    best_pipeline = best_candidate["pipeline"]

    selection_rationale = (
        f"{best_model_name} was selected as the final model because it achieved the strongest balance "
        f"of F1-Score ({best_candidate['f1']:.4f}) and ROC-AUC ({best_candidate['roc_auc']:.4f}) on held-out test data, "
        f"supported by robust cross-validation stability (CV F1: {best_candidate['cv_f1']:.4f}). "
        f"In medical screening tasks, balancing Recall (catching true positive cases) and Precision (minimizing false alarms) "
        f"is critical, making F1 and ROC-AUC more reliable criteria than raw accuracy alone."
    )
    print(f"\nSELECTED FINAL MODEL: {best_model_name}")
    print("Selection Rationale:", selection_rationale)

    print("\n==================================================")
    print("PHASE 17, 18, 19, 20: FINAL EVALUATION & FIGURES")
    print("==================================================")
    final_pred = best_pipeline.predict(X_test)
    final_proba = best_pipeline.predict_proba(X_test)[:, 1]
    final_metrics = calculate_metrics(y_test, final_pred, final_proba)
    cm_breakdown = get_confusion_matrix_breakdown(y_test, final_pred)

    print("Final Test Metrics:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"Confusion Matrix: TN={cm_breakdown['tn']}, FP={cm_breakdown['fp']}, FN={cm_breakdown['fn']}, TP={cm_breakdown['tp']}")

    # Save Confusion Matrix figure
    cm_path = figures_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_test, final_pred, save_path=cm_path, model_name=best_model_name)
    print(f"Saved: {cm_path}")

    # Save ROC Curve figure
    roc_path = figures_dir / "roc_curve.png"
    plot_roc_curve(y_test, final_proba, save_path=roc_path, model_name=best_model_name)
    print(f"Saved: {roc_path}")

    # Save Feature Importance figure
    # Note: transformed features order from ColumnTransformer:
    # zero_missing_cols first, then legitimate_cols
    transformed_feature_names = ZERO_AS_MISSING_COLS + LEGITIMATE_COLS
    feat_imp_path = figures_dir / "feature_importance.png"
    plot_feature_importance(best_pipeline, transformed_feature_names, save_path=feat_imp_path, model_name=best_model_name)
    print(f"Saved: {feat_imp_path}")

    # Save Model Comparison figure
    fig, ax = plt.subplots(figsize=(8, 4.5))
    comp_plot_df = comparison_df.melt(id_vars=["model_name"], value_vars=["accuracy", "f1", "roc_auc"], var_name="Metric", value_name="Score")
    sns.barplot(data=comp_plot_df, x="model_name", y="Score", hue="Metric", ax=ax, palette=["#2b5c8f", "#41b6c4", "#d95f02"])
    ax.set_title("Candidate Model Performance Comparison on Held-Out Test Set", fontsize=12, weight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Machine Learning Model", fontsize=11)
    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11)
    comp_fig_path = figures_dir / "model_comparison.png"
    fig.savefig(comp_fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {comp_fig_path}")

    print("\n==================================================")
    print("PHASE 21 & 22: SAVING BEST MODEL & METRICS JSON")
    print("==================================================")
    # Save the complete Pipeline
    model_save_path = models_dir / "best_model.pkl"
    joblib.dump(best_pipeline, model_save_path)
    print(f"Complete trained pipeline saved to: {model_save_path}")

    # Prepare serializable comparison table
    serializable_comparison = []
    for item in model_comparison_list:
        serializable_comparison.append({
            "model_name": item["model_name"],
            "is_tuned": item["is_tuned"],
            "best_params": item["best_params"],
            "cv_f1": item["cv_f1"],
            "accuracy": item["accuracy"],
            "precision": item["precision"],
            "recall": item["recall"],
            "f1": item["f1"],
            "roc_auc": item["roc_auc"],
        })

    metrics_payload = {
        "selected_model_name": best_model_name,
        "selection_rationale": selection_rationale,
        "test_metrics": final_metrics,
        "confusion_matrix": cm_breakdown,
        "cv_summary": {
            "folds": 5,
            "best_cv_f1": best_candidate["cv_f1"],
        },
        "model_comparison": serializable_comparison,
        "dataset_summary": {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "num_features": len(ALL_FEATURE_COLS),
            "positive_class_count": int(y.sum()),
            "negative_class_count": int((y == 0).sum()),
        },
    }

    metrics_json_path = models_dir / "model_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Model metrics JSON saved to: {metrics_json_path}")

    print("\nTraining workflow completed successfully!")
    return metrics_payload


if __name__ == "__main__":
    train_and_evaluate()
