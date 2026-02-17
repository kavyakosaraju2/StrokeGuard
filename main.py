# StrokeGuard
# Phase 2 - Hyperparameter Tuning + SHAP

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from scipy.stats import randint, uniform

# -----------------------
# Load Dataset
# -----------------------
df = pd.read_csv("data/healthcare-dataset-stroke-data.csv")

# -----------------------
# Preprocessing
# -----------------------
df = df.drop("id", axis=1)
df["bmi"] = df["bmi"].fillna(df["bmi"].median())
df = pd.get_dummies(df, drop_first=True)

X = df.drop("stroke", axis=1)
y = df["stroke"]

print("Preprocessing Completed!\n")

# -----------------------
# Train-Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train-Test Split Completed!")
print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# -----------------------
# Handle Class Imbalance
# -----------------------
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
print("Scale Pos Weight:", scale_pos_weight)

# -----------------------
# Hyperparameter Tuning
# -----------------------
print("\nStarting Hyperparameter Tuning...\n")

xgb = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss",
    use_label_encoder=False
)

param_dist = {
    "n_estimators": randint(200, 600),
    "max_depth": randint(3, 8),
    "learning_rate": uniform(0.01, 0.2),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "gamma": uniform(0, 2)
}

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=20,
    scoring="recall",
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

print("\nBest Parameters Found:")
print(random_search.best_params_)

xgb_model = random_search.best_estimator_

print("\nModel Training Completed!\n")

# -----------------------
# Evaluation
# -----------------------
y_pred = xgb_model.predict(X_test)
y_probs = xgb_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probs)

print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nROC-AUC Score:", roc_auc)
print("\n--- Threshold Analysis ---")

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

for t in thresholds:
    y_pred_threshold = (y_probs >= t).astype(int)
    recall = classification_report(y_test, y_pred_threshold, output_dict=True)['1']['recall']
    precision = classification_report(y_test, y_pred_threshold, output_dict=True)['1']['precision']
    print(f"Threshold: {t} | Recall: {recall:.2f} | Precision: {precision:.2f}")
import json

performance_metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "roc_auc": float(roc_auc),
    "recall_stroke": float(classification_report(y_test, y_pred, output_dict=True)["1"]["recall"])
}

with open("models/model_metrics.json", "w") as f:
    json.dump(performance_metrics, f)

print("Model performance metrics saved!")


# -----------------------
# Save Model & Features
# -----------------------
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")
joblib.dump(xgb_model, "models/strokeguard_xgb_model.pkl")

print("\nModel Saved Successfully!")

# -----------------------
# SHAP Explainability
# -----------------------
print("\nGenerating SHAP explanations...")

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("models/shap_feature_importance.png")
plt.close()

print("SHAP Feature Importance Plot Saved!")




