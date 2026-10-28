# Academic Viva Voce Preparation Guide
## 35 Likely Examination Questions & Concise Student Answers

---

### 1. General Machine Learning Concepts

#### Q1: What is Machine Learning, and how does supervised learning apply to this project?
**Answer:** Machine Learning is the science of training algorithms to discover patterns in historical data rather than hand-coding deterministic rules. In our project, we use **supervised binary classification**: the dataset contains input features (like glucose, BMI, and age) paired with known ground-truth labels (`Outcome`: 0 or 1), allowing our models to learn a predictive mapping function.

#### Q2: What is the difference between classification and regression?
**Answer:** Classification predicts discrete category labels (such as Diabetic vs. Non-diabetic, or 1 vs. 0), whereas regression predicts continuous numeric values (such as predicting a person's exact blood glucose number). Because our target is a binary outcome, this is a classification problem.

#### Q3: What is the difference between overfitting and underfitting?
**Answer:** 
- **Underfitting** occurs when a model is too simple to capture patterns in the data, performing poorly on both training and test data (high bias).
- **Overfitting** occurs when a model memorizes noise and idiosyncrasies in the training set, achieving near-perfect training scores but failing on unseen test data (high variance). We prevented overfitting using depth constraints and cross-validation.

---

### 2. Dataset & Quality Assessment

#### Q4: Which dataset did you use, and what are its key characteristics?
**Answer:** We used the public **Pima Indians Diabetes Database** from the NIDDK/UCI repository. It contains **768 rows** and **9 columns** (8 diagnostic predictor features and 1 binary target outcome). The subjects are all females of Pima Indian heritage aged 21 and older.

#### Q5: What was the critical issue regarding "zeros" in this dataset?
**Answer:** In raw records, features like `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` had values of `0`. In living humans, a glucose level or blood pressure of zero is biologically impossible. These zeros actually represented missing data. However, for `Pregnancies`, a value of `0` is completely legitimate, meaning the woman has never been pregnant. We therefore treated zeros as missing strictly in the 5 physiological columns.

#### Q6: How did you handle these missing values without causing data leakage?
**Answer:** We used scikit-learn's `ColumnTransformer` with `SimpleImputer(missing_values=0, strategy='median')` applied strictly inside our training pipeline. The median values were calculated exclusively from the training partition (`X_train`) and then applied to transform test samples, guaranteeing zero leakage.

---

### 3. Preprocessing & Data Leakage

#### Q7: What is data leakage, and how did you prevent it?
**Answer:** Data leakage happens when information from outside the training partition (like test set statistics) accidentally influences model training, inflating performance artificially. We prevented this by splitting our data into train and test sets **before** fitting any imputers or scalers, and bundling all transformations into an end-to-end `Pipeline`.

#### Q8: Why did you scale the features, and which scaler did you choose?
**Answer:** Features in our dataset exist on very different scales (e.g., `Age` ranges from 21–81, while `Insulin` reaches 846, and `DiabetesPedigreeFunction` is between 0.08 and 2.42). Distance-based algorithms like KNN and gradient-based models like Logistic Regression are heavily distorted by large numbers. We used `StandardScaler` to transform features to have zero mean and unit variance.

#### Q9: Why is train/test stratification important?
**Answer:** Our dataset has a class imbalance (65.1% negative vs. 34.9% positive). A standard random split might accidentally assign too many positive cases to one set. Stratification (`stratify=y`) ensures both the 80% training set and the 20% test set have the exact same class ratio.

---

### 4. Algorithms & Model Mechanics

#### Q10: How does Logistic Regression work for classification?
**Answer:** Despite its name, Logistic Regression is a classification algorithm. It calculates a linear combination of weighted features, passes the score through a sigmoid activation function, and maps the output into a probability between 0 and 1. If the probability exceeds 0.5, it predicts the positive class.

#### Q11: How does K-Nearest Neighbors (KNN) make predictions?
**Answer:** KNN is an instance-based non-parametric algorithm. When predicting a new sample, it computes the distance (e.g., Euclidean) between that sample and all points in the training set, finds the $k$ nearest data points, and assigns the majority class label among them.

#### Q12: How does a Decision Tree split data?
**Answer:** A Decision Tree recursively evaluates candidate feature thresholds and chooses the split that minimizes impurity (such as Gini impurity or entropy). In our project, we constrained the tree to `max_depth=5` to keep it interpretable and avoid overfitting small leaf subsets.

#### Q13: What is Random Forest, and how does it differ from a single Decision Tree?
**Answer:** A Random Forest is an ensemble of multiple decision trees. It uses **bagging** (bootstrap aggregating) to train each tree on a random sample of rows with replacement, and **feature subsampling** at each split. The final prediction is the majority vote of all trees, reducing variance and improving stability.

---

### 5. Evaluation Metrics & Confusion Matrix

#### Q14: Why is Accuracy not enough for evaluating this project?
**Answer:** Because our dataset is imbalanced (65.1% negative), a naive model that always predicts "negative" would achieve 65.1% accuracy while failing to detect a single diabetic patient! In medical applications, Recall and F1-Score are far more meaningful than accuracy alone.

#### Q15: Explain Precision, Recall, and the F1-Score.
**Answer:**
- **Recall (Sensitivity):** Out of all actual positive cases, how many did the model catch? ($\frac{TP}{TP + FN}$). In medical screening, high recall is vital so we don't miss sick patients.
- **Precision:** Out of all cases predicted positive, how many were truly positive? ($\frac{TP}{TP + FP}$).
- **F1-Score:** The harmonic mean of Precision and Recall ($\frac{2 \cdot P \cdot R}{P + R}$), providing a single balanced metric.

#### Q16: Walk through the 4 cells of your Confusion Matrix.
**Answer:** On our 154 held-out test samples:
- **True Negatives (TN = 78):** Actual negative patients correctly predicted as negative.
- **False Positives (FP = 22):** Actual negative patients incorrectly predicted as positive (false alarm).
- **False Negatives (FN = 15):** Actual positive patients missed by the model.
- **True Positives (TP = 39):** Actual positive patients correctly identified.

#### Q17: What does the ROC Curve and ROC-AUC score represent?
**Answer:** The ROC (Receiver Operating Characteristic) curve plots True Positive Rate (Recall) against False Positive Rate across all possible classification probability thresholds. The Area Under the Curve (ROC-AUC) measures overall separability. A score of 0.5 represents random guessing, while our final models achieved ~0.76–0.81, showing strong class separation capability.

---

### 6. Validation, Tuning & Pipelines

#### Q18: What is K-Fold Cross-Validation, and why use Stratified K-Fold?
**Answer:** In 5-Fold Cross-Validation, the training set is split into 5 equal parts. The model trains on 4 folds and evaluates on the 5th, repeating 5 times so every fold serves as validation. Stratified K-Fold preserves the exact class ratio in every fold, preventing biased splits in imbalanced data.

#### Q19: What is `GridSearchCV`?
**Answer:** `GridSearchCV` performs an exhaustive search over a specified dictionary of hyperparameter values, evaluating every combination using cross-validation. It automatically identifies the optimal parameter set that maximizes our chosen metric (like F1-score).

#### Q20: What are hyperparameters vs. parameters?
**Answer:**
- **Parameters** are learned automatically from data during training (e.g., the weights $\beta$ in Logistic Regression or split thresholds in a tree).
- **Hyperparameters** are architectural settings defined before training (e.g., `n_neighbors` in KNN, `max_depth` in Decision Trees, or `C` in Logistic Regression).

#### Q21: Why use scikit-learn `Pipeline`?
**Answer:** A `Pipeline` chains together preprocessing steps (imputer, scaler) and the estimator into a single object. When you call `fit()` or `predict()`, data flows sequentially through the pipeline. This prevents data leakage and ensures incoming production data in Streamlit is transformed identically to training data.

---

### 7. Model Selection & Interpretation

#### Q22: Which model did you choose as your final model, and why?
**Answer:** We selected the **Decision Tree Classifier (`max_depth=5`)**. Although Random Forest and Logistic Regression achieved high ROC-AUC, the Decision Tree delivered the highest **Recall (72.22%)** and highest **F1-Score (0.6783)** on unseen test data, catching 39 of 54 true positive cases. In addition, its decision rules are transparent and clinically auditable.

#### Q23: What is Feature Importance, and what does it tell us in this project?
**Answer:** Feature importance in our Decision Tree measures how frequently and effectively a feature was used to reduce Gini impurity across splits. `Glucose` and `BMI` were the most influential features. Crucially, feature importance reflects **statistical correlation within this trained model, not biological causation**.

---

### 8. Deployment & Software Engineering

#### Q24: How did you serialize and load the model?
**Answer:** We saved the complete pipeline using `joblib.dump(best_pipeline, 'models/best_model.pkl')`. In `app.py`, we load it with `@st.cache_resource` so the model stays in memory across interactions without reloading from disk on every click.

#### Q25: Why did you save `feature_metadata.json`?
**Answer:** It acts as a single source of truth between training and deployment. It stores the exact feature names, their required order, data types, and empirical min/max/median values derived from `X_train`. Streamlit uses this to build safe UI inputs matching the exact schema expected by the pipeline.

#### Q26: How does the Streamlit app extract class probability?
**Answer:** We call `pipe.predict_proba(input_df)[0]`. Instead of assuming index 1 is positive, our code queries `list(pipeline.classes_).index(1)` to dynamically locate the positive class index, avoiding bugs if class order changes.

---

### 9. Version Control & Reproducibility

#### Q27: What is the purpose of `.gitignore`?
**Answer:** `.gitignore` prevents unnecessary or sensitive files from entering Git. In our project, it explicitly ignores `.venv/` (which is large and machine-specific), `__pycache__/`, `.DS_Store`, and environment secret files.

#### Q28: How do you ensure your results are reproducible?
**Answer:** We fixed random seeds across all probabilistic steps (`random_state=42` in `train_test_split`, `StratifiedKFold`, and model initializers), pinned our dependencies in `requirements.txt`, and automated all steps via `python -m src.train_model`.

---

### 10. Medical Ethics & Project Scope

#### Q29: Is this application a clinical diagnostic tool?
**Answer:** **No, absolutely not.** This is strictly an educational machine learning mini-project. The predictions reflect statistical correlations in a historical academic dataset and must never be considered medical advice, a diagnosis, or a substitute for a licensed healthcare professional.

#### Q30: Why should the app never say "You have diabetes"?
**Answer:** In educational and non-clinical ML, diagnosing an individual is medically unsafe, unethical, and legally prohibited. Instead, the application displays objective terminology such as: *"Predicted class: Positive (1)"* and *"The model assigned a 72.4% probability to the positive class based on training dataset patterns."*

#### Q31: What are the main limitations of the Pima Indians dataset?
**Answer:** 
1. Small sample size (768 records).
2. Population-specific: all patients are adult females of Pima Indian heritage. Results cannot be generalized to broader global populations or male demographics.
3. Hidden missing values required imputation, which introduces statistical assumptions.

#### Q32: What would you do in the future to improve this project?
**Answer:** 
1. Test on modern, diverse, multi-center health cohorts (e.g. NHANES).
2. Benchmark gradient-boosted trees (LightGBM, XGBoost, CatBoost).
3. Implement SHAP (SHapley Additive exPlanations) for local instance-level explainability.

#### Q33: What is the difference between `joblib` and `pickle`?
**Answer:** `joblib` is optimized for Python objects containing large NumPy arrays and machine learning models. It serializes and deserializes numerical arrays much faster and with lower memory overhead than standard `pickle`.

#### Q34: What is Gini Impurity?
**Answer:** Gini impurity measures the probability that a randomly chosen element from a set would be incorrectly labeled if it were randomly labeled according to the class distribution. A pure node with only one class has a Gini impurity of 0.0.

#### Q35: If a user enters 0 for Glucose in the Streamlit app, will the model crash?
**Answer:** No! Because the pipeline contains `SimpleImputer(missing_values=0, strategy='median')`, it automatically replaces the zero with the median learned during training and continues inference smoothly.
