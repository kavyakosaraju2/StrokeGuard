
# StrokeGuard
# Train-Test Split + XGBoost Training


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# -----------------------
# Load Dataset
# -----------------------
df = pd.read_csv("data/healthcare-dataset-stroke-data.csv")

# Preprocessing
df = df.drop("id", axis=1)
df["bmi"] = df["bmi"].fillna(df["bmi"].median())
df = pd.get_dummies(df, drop_first=True)

X = df.drop("stroke", axis=1)
y = df["stroke"]

print("Preprocessing Completed!\n")


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train-Test Split Completed!\n")
print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


# Handle Imbalance
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])


# Train XGBoost
xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    subsample=1,
    colsample_bytree=1,
    eval_metric='logloss'
)


xgb_model.fit(X_train, y_train)

print("\nModel Training Completed!\n")


# Evaluation
y_pred = xgb_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
from sklearn.metrics import roc_auc_score
import joblib

# ROC-AUC Score
y_probs = xgb_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_probs)

print("\nROC-AUC Score:", roc_auc)
import joblib

# Save feature columns
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

# Save Model
joblib.dump(xgb_model, "models/strokeguard_xgb_model.pkl")

print("\nModel Saved Successfully!")

import shap
import matplotlib.pyplot as plt

print("\nGenerating SHAP explanations...")

# Create SHAP explainer
explainer = shap.TreeExplainer(xgb_model)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test)

# Create summary bar plot
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)

plt.tight_layout()
plt.savefig("models/shap_feature_importance.png")
plt.close()

print("SHAP Feature Importance Plot Saved!")



