# StrokeGuard - Web Application

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import json

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="StrokeGuard",
    layout="centered"
)

# -------------------------
# Load Model & Features
# -------------------------
model = joblib.load("models/strokeguard_xgb_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")




with open("models/model_metrics.json", "r") as f:
    model_metrics = json.load(f)
    


# Create SHAP explainer
explainer = shap.TreeExplainer(model)
# Clean feature name mapping for UI display
feature_name_map = {
    "age": "Age",
    "hypertension": "Hypertension",
    "heart_disease": "Heart Disease",
    "avg_glucose_level": "Average Glucose Level",
    "bmi": "BMI",
    "ever_married_Yes": "Married (Yes)",
    "gender_Male": "Gender (Male)",
    "work_type_Private": "Work Type (Private)",
    "work_type_Self-employed": "Work Type (Self-employed)",
    "work_type_children": "Work Type (Children)",
    "work_type_Never_worked": "Work Type (Never Worked)",
    "Residence_type_Urban": "Residence Type (Urban)",
    "smoking_status_formerly smoked": "Smoking (Former)",
    "smoking_status_never smoked": "Smoking (Never)",
    "smoking_status_smokes": "Smoking (Current)"
}


# -------------------------
# Title
# -------------------------
st.title("StrokeGuard")
st.subheader("Boosted & Explainable Stroke Risk Prediction System")

st.write("Enter patient details below to predict stroke risk.")

st.markdown("###  Patient Information")

# -------------------------
# Inputs
# -------------------------
age = st.slider("Age", 0, 100, 50)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
avg_glucose_level = st.number_input("Average Glucose Level", value=100.0)
bmi = st.number_input("BMI", value=25.0)
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
gender = st.selectbox("Gender", ["Male", "Female"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes"])

# -------------------------
# Prepare Input
# -------------------------
input_dict = {
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
}

input_dict["ever_married_Yes"] = 1 if ever_married == "Yes" else 0
input_dict["gender_Male"] = 1 if gender == "Male" else 0
input_dict["work_type_Private"] = 1 if work_type == "Private" else 0
input_dict["work_type_Self-employed"] = 1 if work_type == "Self-employed" else 0
input_dict["work_type_children"] = 1 if work_type == "children" else 0
input_dict["work_type_Never_worked"] = 1 if work_type == "Never_worked" else 0
input_dict["Residence_type_Urban"] = 1 if residence_type == "Urban" else 0
input_dict["smoking_status_formerly smoked"] = 1 if smoking_status == "formerly smoked" else 0
input_dict["smoking_status_never smoked"] = 1 if smoking_status == "never smoked" else 0
input_dict["smoking_status_smokes"] = 1 if smoking_status == "smokes" else 0

input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=feature_columns, fill_value=0)
# -------------------------
# Prediction
# -------------------------
if st.button(" Predict Stroke Risk"):

    with st.spinner("Analyzing patient data..."):

        probability = model.predict_proba(input_df)[0][1]
        risk_percent = probability * 100

        # Custom tuned threshold
        threshold = 0.6
        prediction = 1 if probability >= threshold else 0

        st.markdown("##  Prediction Result")
        st.markdown(f"### Stroke Probability: {risk_percent:.2f}%")
        st.progress(min(int(risk_percent), 100))

        # Risk Category
        if probability < 0.30:
            st.success("🟢 Low Stroke Risk")
            st.info("Maintain healthy lifestyle and regular checkups.")

        elif 0.30 <= probability < threshold:
            st.warning("🟡 Moderate Stroke Risk")
            st.info("Lifestyle improvement and monitoring recommended.")

        else:
            st.error("🔴 High Stroke Risk")
            st.info("Immediate medical consultation recommended.")

        st.caption("Decision threshold set at 0.60 for optimized medical recall-performance balance.")

        # -------------------------
        # SHAP Explanation
        # -------------------------
        st.markdown("##  Why This Prediction Was Made")

        shap_values = explainer.shap_values(input_df)

        shap_df = pd.DataFrame({
            "Feature": input_df.columns,
            "Impact": shap_values[0]
        })

        shap_df["AbsImpact"] = shap_df["Impact"].abs()
        shap_df = shap_df.sort_values(by="AbsImpact", ascending=False).head(5)

        shap_df["Feature"] = shap_df["Feature"].map(feature_name_map)

        st.markdown("### 🔎 Top Factors Influencing This Prediction")

        colors = ["red" if val > 0 else "blue" for val in shap_df["Impact"]]

        fig, ax = plt.subplots()
        ax.barh(shap_df["Feature"], shap_df["Impact"], color=colors)
        ax.set_xlabel("Impact on Stroke Risk")
        ax.set_title("Feature Contribution")
        ax.axvline(0, color="black", linewidth=1)
        ax.invert_yaxis()

        st.pyplot(fig)
        st.caption("🔴 Red bars increase stroke risk | 🔵 Blue bars decrease stroke risk")

