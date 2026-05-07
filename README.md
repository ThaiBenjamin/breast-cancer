# 🔬 Breast Cancer Predictor

A workshop exercise in supervised machine learning — training a logistic regression classifier on the Wisconsin Breast Cancer Dataset to predict whether a tumor is malignant or benign.

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)

---

## 📚 What I Was Learning

The end-to-end supervised ML workflow — loading and cleaning a real dataset, splitting into train/test sets, training a model, evaluating with classification metrics, and exposing a prediction function.

---

## 🔑 Key Things Practiced

- **Data Preparation** — Loading CSV data with pandas, dropping irrelevant columns, encoding categorical labels
- **Train/Test Split** — Using `train_test_split` for honest evaluation on held-out data
- **Feature Scaling** — Wrapping logistic regression in a `StandardScaler` pipeline
- **Model Training** — Fitting `LogisticRegression` from scikit-learn
- **Evaluation** — Accuracy score, precision, recall, F1 via `classification_report`
- **Prediction Function** — Wrapping the model in a clean API for single-sample predictions

---

## 💡 What It Taught Me

Using a real medical dataset made this more meaningful than a toy example. The Wisconsin Breast Cancer Dataset has 30 numeric features (cell radius, texture, perimeter, etc.) and a binary target — a clean setup for understanding classification. The most important lesson was why feature scaling matters: logistic regression with gradient descent converges much faster and more reliably when all features are on the same scale, which is why `StandardScaler` gets bundled into the pipeline. Using `classification_report` also taught me that accuracy alone is misleading for medical data — you care a lot more about false negatives than false positives.
