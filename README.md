#  StrokeGuard
### Explainable Stroke Risk Prediction System

StrokeGuard is a Machine Learning web application that predicts the probability of stroke risk using patient health parameters.

It uses XGBoost with hyperparameter tuning and SHAP explainability, and is deployed using Streamlit.

## Architecutre Diagram
![Architecutre Diagram](docs/System architecture diagram.png)


##  Features
- XGBoost classifier with hyperparameter tuning
- Handles class imbalance using scale_pos_weight
- ROC-AUC optimized model
- Threshold-based risk prediction
- SHAP explainability (feature impact visualization)
- Interactive Streamlit web app



##  Model Performance
- ROC-AUC: ~0.85
- Optimized Threshold: 0.6
- High recall for stroke class (medical priority)



##  Tech Stack
- Python
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Matplotlib



##  Project Structure
StrokeGuard/
│── app.py
│── main.py
│── requirements.txt
│── data/
│── models/




##  Live Demo
🔗 https://kavyakosaraju2-strokeguard-app-rcyxcs.streamlit.app/




 Disclaimer

This project is for educational purposes only and should not replace medical advice.
Author
Kavya Kosaraju






