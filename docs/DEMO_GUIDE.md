# Faculty Demo Walkthrough Guide (5–7 Minute Sequence)
## Step-by-Step Presentation Script for Viva Examination

---

### Timing Breakdown
- **Minute 1: Problem & Motivation**
- **Minute 2: Dataset, Zeros & Leakage-Free Preprocessing**
- **Minute 3: Models, Cross-Validation & Metric Comparison**
- **Minute 4: Final Model Rationale & Visual Evidence**
- **Minute 5: Live Streamlit Application Demo**
- **Minute 6: Feature Influence & Educational Medical Framing**
- **Minute 7: Limitations, Codebase & Conclusion**

---

### Step 1: Problem Introduction (1 Minute)
> "Good morning / afternoon professors. Today I am presenting my Machine Learning mini-project: **Diabetes Risk Prediction Using Machine Learning**."
> 
> "Our goal is to explore whether standard supervised classification algorithms can detect early diabetes risk patterns from physiological parameters such as blood glucose, BMI, blood pressure, and age. Crucially, this is an **educational project** focused on ML methodology, data leakage prevention, and model interpretability."

---

### Step 2: Dataset & Preprocessing Pipeline (1 Minute)
> "We used the benchmark **Pima Indians Diabetes dataset** (768 records, 8 predictive features, 1 binary outcome).
> 
> While exploring the raw data, we discovered a vital data quality issue: continuous biological features like `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` contained `0` values. For living humans, a glucose level or blood pressure of zero is biologically impossible—they represented undocumented missing values. For `Pregnancies`, however, zero is completely valid.
> 
> To prevent **data leakage**, we split the dataset 80/20 with stratification *before* fitting any transformer. Then, using scikit-learn's `ColumnTransformer`, we applied median imputation exclusively to the 5 physiological features and standardized all features using `StandardScaler` inside a single unified `Pipeline`."

---

### Step 3: Models & Cross-Validation (1 Minute)
> "We trained and evaluated four diverse candidate algorithms:
> 1. **Logistic Regression**
> 2. **K-Nearest Neighbors**
> 3. **Decision Tree Classifier**
> 4. **Random Forest Classifier**
> 
> Each model was validated using **5-Fold Stratified Cross-Validation** on the training partition. We also tuned hyperparameters using `GridSearchCV` on the training folds, optimizing for F1-score rather than raw accuracy due to class imbalance (65% negative, 35% positive)."

---

### Step 4: Model Comparison & Final Selection (1 Minute)
> *(Open the **Model Performance** page in Streamlit or show the comparison table)*
> 
> "On our held-out test set of 154 unseen samples:
> - **Decision Tree** achieved: **75.97% Accuracy**, **72.22% Recall**, and an **F1-Score of 0.6783**.
> - **Random Forest** achieved: 74.68% Accuracy and 0.6214 F1.
> - **Logistic Regression** achieved: 70.78% Accuracy and 0.6154 F1.
> 
> We selected the **Decision Tree Classifier (`max_depth=5`)** as our final model because in medical screening, **Recall** is critical: we want to catch as many true positive cases as possible with minimal false negatives. The Decision Tree caught 39 out of 54 actual positive cases, while offering full rule transparency."

---

### Step 5: Live App Demonstration (1.5 Minutes)
> *(Navigate to the **Risk Prediction** page)*
> 
> 1. **Show Preset A (Lower Historical Risk):**
>    - Select `Sample Profile A (Lower Historical Risk)` from the dropdown.
>    - Click **Predict Risk Class**.
>    - Explain output: *"The pipeline outputs Class 0 (Negative). The model assigned a low probability to the positive class."*
> 
> 2. **Show Preset B (Higher Historical Risk):**
>    - Select `Sample Profile B (Higher Historical Risk)`.
>    - Click **Predict Risk Class**.
>    - Explain output: *"The pipeline outputs Class 1 (Positive) with elevated probability. Notice our phrasing: 'The model assigned a XX% probability to the positive class based on training data patterns.' We never diagnose the user."*
> 
> 3. **Show Robustness (Zero Imputation):**
>    - Enter `0` for `Glucose` and `Insulin`.
>    - Click **Predict Risk Class**.
>    - Point out: *"The app does not crash or throw errors because the saved pipeline automatically imputes non-zero median values learned during training."*

---

### Step 6: Feature Influence & Medical Disclaimer (1 Minute)
> *(Show the **Feature Importance** plot)*
> 
> "Examining the trained model's feature importance, `Glucose` and `BMI` were the most influential split variables. We emphasize that this represents **statistical correlation within our trained model, not biological causation**.
> 
> Prominently displayed across every page is our medical disclaimer: this application is strictly an academic demonstration and not a clinical diagnosis."

---

### Step 7: Architecture & GitHub Readiness (30 Seconds)
> "The entire codebase is structured professionally with:
> - Clean modular `src/` packages (`preprocess.py`, `train_model.py`, `evaluate.py`, `data_utils.py`)
> - Automated assertion test suite (`tests/test_project.py`) passing 100%
> - Reproducible `requirements.txt`
> - Complete Jupyter Notebook (`notebooks/diabetes_analysis.ipynb`)
> - Git repository with clean commits and `.gitignore` ready to publish to GitHub.
> 
> Thank you, professors! I am now ready for your questions."
