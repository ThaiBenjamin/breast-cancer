# breast_cancer_predictor.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# ================================
# 1. Load Data
# ================================
data = pd.read_csv("data.csv")

# Drop unused columns
df = data.drop(columns=["id", "Unnamed: 32"], errors="ignore")

# Encode target: M=1 (malignant), B=0 (benign)
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Features (X) and target (y)
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

# ================================
# 2. Train/Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# 3. Model (Logistic Regression + Scaling)
# ================================
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, solver="lbfgs")
)

# Train
model.fit(X_train, y_train)

# ================================
# 4. Evaluation
# ================================
y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# ================================
# 5. Prediction Function
# ================================
def predict_tumor(features):
    """
    features: list or numpy array of 30 numeric values (same order as dataset)
    Returns: 'Malignant' or 'Benign'
    """
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)[0]
    return "Malignant" if prediction == 1 else "Benign"


# ================================
# 6. Example Usage
# ================================
if __name__ == "__main__":
    sample = X.iloc[0].values
    print("\nPrediction for first row in dataset:", predict_tumor(sample))

    new_data = [17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
                1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003,
                0.006193, 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
    
    print("Prediction for manual input:", predict_tumor(new_data))

