import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

df = pd.read_csv("synthetic_groundwater.csv")

thresholds = {
    "pH": (6.5, 8.5),
    "TDS": (None, 500),
    "turbidity": (None, 5),
    "nitrate": (None, 50),
    "arsenic": (None, 0.01),
    "lead": (None, 0.01),
    "fluoride": (None, 1.5),
    "hardness": (None, 300),
    "temperature": (None, 35),
    "EC": (None, 1500),
    "coliform": (0, 0),
}

def within_threshold(value, low, high):
    if pd.isna(value):
        return False
    if low is not None and value < low:
        return False
    if high is not None and value > high:
        return False
    return True
if "safe_water" not in df.columns:
    df["safe_water"] = df.apply(
        lambda row: int(all(within_threshold(row[col], *thresholds[col])  for col in thresholds if col in df.columns)), axis=1 )

df.to_csv("processed_with_target.csv", index=False)
print("[INFO] Target column added → processed_with_target.csv")

X = df.drop(columns=["safe_water"])
y = df["safe_water"]
X = X.fillna(X.median(numeric_only=True))

if y.nunique() > 1:
    stratify_opt = y
else:
    stratify_opt = None

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, stratify=stratify_opt, random_state=42)
log_reg = Pipeline([ ("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])

rf = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=300, random_state=42))])

models = {"Logistic Regression": log_reg, "Random Forest": rf}
results = {}

for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"\n{name} Results:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=[0, 1]))
        results[name] = model
    except Exception as e:
        print(f"\n[ERROR] Could not train {name}: {e}")

if results:
    if "Random Forest" in results:
        best_model_name = "Random Forest"
    else:
        best_model_name = list(results.keys())[0]
    best_model = results[best_model_name]
    joblib.dump(best_model, "groundwater_best_model.pkl")
    print(f"\n[INFO] Best model ({best_model_name}) saved → groundwater_best_model.pkl")
else:
    print("\n[WARNING] No model could be trained. Check dataset (safe_water column).")

