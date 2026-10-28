# Academic Mini-Project Report
# Diabetes Risk Prediction Using Machine Learning

---

## 1. Project Title
**DIABETES RISK PREDICTION USING MACHINE LEARNING: A LEAK-FREE SUPERVISED CLASSIFICATION PIPELINE**

---

## 2. Abstract
Type 2 diabetes represents one of the fastest-growing metabolic disorders globally. This academic mini-project investigates the application of supervised binary machine learning algorithms to predict high-risk diabetes classification from non-invasive physiological and diagnostic parameters. Using the benchmark Pima Indians Diabetes Database (768 patient records), we developed a leak-free machine learning workflow. A critical challenge identified in raw medical data was the presence of biologically implausible zero values in continuous features (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI`), which functioned as undocumented missing data. To prevent methodological data leakage, all imputation (median-based) and standardization statistics were computed strictly on training folds within a unified `scikit-learn` Pipeline. Four candidate algorithms—Logistic Regression, K-Nearest Neighbors (KNN), Decision Trees, and Random Forests—were trained and optimized using 5-Fold Stratified Cross-Validation. On the unseen held-out test partition (154 samples), the constrained Decision Tree classifier achieved the highest balance of sensitivity and precision, yielding a Test Accuracy of 75.97%, F1-Score of 0.6783, Recall of 72.22%, and ROC-AUC of 0.7622. An interactive, multi-page Streamlit web application was constructed for real-time inference, model auditing, and academic viva demonstration.

---

## 3. Introduction
Machine learning in healthcare has demonstrated notable potential for early risk stratification. Rather than replacing physician judgment, algorithmic screening tools serve as assistive mechanisms to flag patients who may benefit from follow-up diagnostic testing. This project explores the end-to-end development cycle of an educational machine learning application, emphasizing reproducibility, rigorous validation, and ethical medical framing.

---

## 4. Problem Statement
Many medical records contain subtle collection anomalies, hidden missing representations, and class imbalances. In naive machine learning implementations, preprocessing transformations such as global mean imputation and whole-dataset scaling introduce subtle information leakage from test partitions into training models, producing overly optimistic and unreplicable evaluations. The objective is to design, execute, and validate an end-to-end classification system that isolates test data completely, treats missing data appropriately, and surfaces realistic model performance metrics.

---

## 5. Objectives
1. Perform automated data validation and identify biologically impossible measurements.
2. Build reusable, leak-free scikit-learn preprocessing pipelines.
3. Conduct meaningful exploratory data analysis (EDA) with publication-quality visualizations.
4. Train and cross-validate four candidate classifiers (Logistic Regression, KNN, Decision Tree, Random Forest).
5. Tune model hyperparameters using `GridSearchCV` on training folds.
6. Select the best final model based on balanced clinical metrics (F1-Score and Recall).
7. Deploy the model into an interactive Streamlit application with explicit medical disclaimers.

---

## 6. Dataset Description
The project employs the **Pima Indians Diabetes Database**, originally curated by the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) and hosted by the UCI Machine Learning Repository.
- **Sample Population:** 768 female patients of Pima Indian heritage aged 21 years and older.
- **Attributes:** 8 numeric physiological predictor features and 1 binary class target (`Outcome`).
- **Target Distribution:**
  - Class 0 (Negative / Non-diabetic): 500 instances (65.1%)
  - Class 1 (Positive / Diabetic): 268 instances (34.9%)

### Features:
1. `Pregnancies`: Number of times pregnant (Range: 0–17)
2. `Glucose`: 2-hour plasma glucose concentration in oral glucose tolerance test (Range: 0–199 mg/dL)
3. `BloodPressure`: Diastolic blood pressure (Range: 0–122 mm Hg)
4. `SkinThickness`: Triceps skin fold thickness (Range: 0–99 mm)
5. `Insulin`: 2-Hour serum insulin (Range: 0–846 μU/ml)
6. `BMI`: Body mass index (Range: 0–67.1 kg/m²)
7. `DiabetesPedigreeFunction`: Genetic pedigree score based on family history (Range: 0.078–2.42)
8. `Age`: Subject age in years (Range: 21–81)

---

## 7. Tools and Technologies
- **Programming Language:** Python 3.13.5
- **Libraries:**
  - `pandas` (v3.1.0): Data wrangling and DataFrame manipulation
  - `numpy` (v2.5.4): Numerical array computations
  - `scikit-learn` (v1.9.1): Preprocessing, pipelines, modeling, and evaluation
  - `matplotlib` (v3.11.2) & `seaborn` (v0.13.2): High-resolution statistical plots
  - `joblib` (v1.6.0): Pipeline serialization
  - `streamlit` (v1.66.1): Web application framework
- **Environment & Version Control:** PowerShell, Python Virtual Environment (`.venv`), Git 2.42, GitHub

---

## 8. Methodology
The project enforces a strict, leak-free pipeline methodology:
1. **Raw Data Ingestion:** Loading `data/diabetes.csv` without modifying source files.
2. **Train/Test Splitting:** An 80/20 stratified split (`random_state=42`) partitions the 768 rows into 614 training instances and 154 held-out test instances.
3. **Partition-Isolated Preprocessing:** Imputation statistics (median) and scaling statistics (mean and standard deviation) are learned strictly on `X_train`.
4. **Internal Cross-Validation:** 5-Fold Stratified K-Fold CV evaluates algorithm generalization and guides hyperparameter tuning.
5. **Final Testing:** Held-out test set is evaluated exactly once after all tuning and selection decisions are finalized.

---

## 9. Data Preprocessing
- **Handling Biologically Implausible Zeros:**
  In physiological features, 0 values are biologically impossible for living individuals (`Glucose`: 5 zeros, `BloodPressure`: 35 zeros, `SkinThickness`: 227 zeros, `Insulin`: 374 zeros, `BMI`: 11 zeros). These were mapped to missing representations. For `Pregnancies`, 0 is legitimate (nulliparous women) and was untouched.
- **Scikit-Learn ColumnTransformer:**
  - Branch 1: `SimpleImputer(missing_values=0, strategy='median')` + `StandardScaler()` applied to the 5 zero-missing columns.
  - Branch 2: `SimpleImputer(strategy='median')` + `StandardScaler()` applied to `Pregnancies`, `DiabetesPedigreeFunction`, and `Age`.
- **Packaging:** Assembled into an atomic `Pipeline` combining preprocessor and estimator.

---

## 10. Exploratory Data Analysis (EDA)
- **Outcome Balance:** Demonstrates a 1.86:1 negative-to-positive ratio, requiring stratified evaluation metrics (F1-score rather than accuracy alone).
- **Glucose & BMI Distributions:** KDE plots revealed distinct rightward shifts for diabetic individuals, indicating high discriminatory potential.
- **Correlation Analysis:** Glucose had the highest correlation with Outcome ($r = 0.47$), followed by BMI ($r = 0.29$) and Age ($r = 0.24$).

---

## 11. Machine Learning Algorithms
1. **Logistic Regression:** Linear decision boundary via log-odds modeling with L2 regularization.
2. **K-Nearest Neighbors (KNN):** Distance-weighted neighborhood classification with Euclidean distance.
3. **Decision Tree:** Recursive binary splitting using Gini impurity with maximum depth constraint (`max_depth=5`) to prevent memorization.
4. **Random Forest Classifier:** Bagged ensemble averaging predictions over 100 decorrelated decision trees.

---

## 12. Model Training & Cross-Validation
Each candidate model pipeline was trained across 5 stratified folds on the 614 training rows.
- Logistic Regression 5-Fold CV Accuracy: 78.82% (CV F1: 0.6545, CV ROC-AUC: 0.8434)
- Random Forest 5-Fold CV Accuracy: 76.39% (CV F1: 0.6324, CV ROC-AUC: 0.8165)
- K-Nearest Neighbors 5-Fold CV Accuracy: 72.96% (CV F1: 0.5939, CV ROC-AUC: 0.7843)
- Decision Tree 5-Fold CV Accuracy: 73.13% (CV F1: 0.5866, CV ROC-AUC: 0.7719)

---

## 13. Hyperparameter Tuning
Using `GridSearchCV` on the training partition:
- **Logistic Regression:** Tuned over $C \in [0.01, 0.1, 1.0, 10.0]$, solver, and class weighting. Best params: `C=0.1, class_weight='balanced'`, improving CV F1 to **0.6823**.
- **Random Forest:** Tuned over tree counts, depths, and min leaf samples. Best params: `n_estimators=150, max_depth=None, min_samples_leaf=2, min_samples_split=5`, reaching CV F1 of **0.6536**.
- **K-Nearest Neighbors:** Tuned over neighbor counts and distance weighting. Best params: `n_neighbors=9, metric='euclidean', weights='distance'`, reaching CV F1 of **0.6366**.

---

## 14. Model Comparison

| Algorithm | 5-Fold CV F1 | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
|---|---|---|---|---|---|---|
| **Decision Tree** | 0.5866 | **75.97%** | **63.93%** | **72.22%** | **0.6783** | **0.7622** |
| **Random Forest** | 0.6536 | 74.68% | 65.31% | 59.26% | 0.6214 | 0.8146 |
| **K-Nearest Neighbors** | 0.6366 | 74.03% | 63.46% | 61.11% | 0.6226 | 0.7969 |
| **Logistic Regression** | 0.6823 | 70.78% | 57.14% | 66.67% | 0.6154 | 0.8104 |

---

## 15. Final Model Selection
The constrained **Decision Tree Classifier (`max_depth=5`)** was selected as the final operational model.
- **Clinical Rationale:** In medical risk screening, False Negatives (missing patients with the condition) carry far higher clinical consequence than False Positives. The Decision Tree achieved a superior **Recall of 72.22%** (identifying 39 out of 54 actual cases) and the highest **F1-Score of 0.6783**.
- **Interpretability:** Unlike black-box ensembles, a constrained decision tree generates transparent decision paths that can be visualized and explained during faculty viva examination.

---

## 16. Streamlit Application
A multi-page web application (`app.py`) provides:
1. **Home:** Overview, architecture flow, and prominent medical disclaimer.
2. **Data Analysis:** Interactive dataset viewer, zero-value breakdown, and high-resolution plots.
3. **Risk Prediction:** Form with preset profiles, real-time inference, and class probability output.
4. **Model Performance:** Comparison table, confusion matrix, ROC curve, and feature importances.
5. **About Project:** Background, methodology, and limitations.

---

## 17. Results Summary
- **Held-Out Test Set Size:** 154 samples (100 Negative, 54 Positive)
- **Accuracy:** 75.97% (117 correct classifications out of 154)
- **Sensitivity / Recall:** 72.22% (39 true positives identified)
- **Specificity:** 78.00% (78 true negatives identified)
- **Precision:** 63.93%
- **F1-Score:** 0.6783
- **ROC-AUC:** 0.7622
- **Confusion Matrix Breakdown:** TN = 78, FP = 22, FN = 15, TP = 39

---

## 18. Limitations
1. **Dataset Size:** 768 samples is relatively small; larger multi-center cohorts are needed for production clinical validation.
2. **Demographic Homogeneity:** The Pima Indian female cohort exhibits specific genetic risk profiles that may not transfer directly to other ethnic groups or male demographics.
3. **Cross-Sectional Nature:** The dataset records instantaneous measurements without longitudinal tracking over time.

---

## 19. Future Scope
- Testing on modern heterogeneous medical datasets (e.g. NHANES, CDC BRFSS).
- Implementing advanced ensemble algorithms (XGBoost, LightGBM, CatBoost) with SHAP tree explainer values.
- Incorporating lifestyle and dietary history into predictive feature schemas.

---

## 20. Conclusion
This academic mini-project successfully engineered an end-to-end Machine Learning pipeline for diabetes risk prediction. By eliminating data leakage through strict pipeline architecture, correctly imputing biologically implausible zeros, comparing diverse algorithms, and deploying an interactive web application, the project demonstrates sound machine learning engineering principles.

---

## 21. References
1. Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. *Proceedings of the Symposium on Computer Applications in Medical Care*, IEEE Computer Society Press, pp. 261-265.
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, pp. 2825-2830.
3. National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK). Pima Indians Diabetes Database. UCI Machine Learning Repository.
