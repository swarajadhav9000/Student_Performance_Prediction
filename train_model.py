import json
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("data/student_performance_clean.csv")

FEATURES = [
    "attendance",
    "study_hours",
    "previous_score",
    "participation",
    "internal_marks",
]

X = df[FEATURES]
y_class = df["performance_level"]
y_reg = df["final_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
}

results = {}
best_model = None
best_name = None
best_acc = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"\n{name} accuracy: {acc:.3f}")
    print(classification_report(y_test, preds))
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)
reg = LinearRegression()
reg.fit(Xr_train, yr_train)
r2 = r2_score(yr_test, reg.predict(Xr_test))
print(f"\nLinear Regression R2 (score prediction): {r2:.3f}")

with open("model_classifier.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("model_regressor.pkl", "wb") as f:
    pickle.dump(reg, f)

metadata = {
    "features": FEATURES,
    "best_classifier": best_name,
    "classifier_accuracy": round(best_acc, 4),
    "regressor_r2": round(r2, 4),
    "all_classifier_results": {k: round(v, 4) for k, v in results.items()},
}
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nBest classifier: {best_name} ({best_acc:.3f}) saved to model_classifier.pkl")
print("Regressor saved to model_regressor.pkl")
