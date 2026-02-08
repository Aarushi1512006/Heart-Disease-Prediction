# ❤️ Heart Disease Prediction Web App (CardioCheck AI)

An end-to-end **Machine Learning web application** that predicts the likelihood of heart disease based on patient clinical parameters.  
The project combines **ML model training, data preprocessing, and a user-friendly frontend** built using **Streamlit**.

> ⚠️ **Disclaimer:** This application is for **educational purposes only** and must not be considered a medical diagnosis tool.

---

## 🚀 Live Demo
👉 https://heart-disease-prediction-g9rz8emrgkuzwj2bc2qcdf.streamlit.app/

---

## 📌 Features
- Interactive and clean **Streamlit UI**
- Predicts **heart disease risk** with probability score
- Uses trained **Logistic Regression** model
- Handles categorical data using **One-Hot Encoding**
- Feature scaling with **StandardScaler**
- End-to-end ML pipeline (training → saving → deployment)

---

## 🧠 Machine Learning Details
- **Algorithm:** Logistic Regression  
- **Problem Type:** Binary Classification  
- **Preprocessing:**
  - One-Hot Encoding for categorical features
  - Feature scaling using StandardScaler
- **Output:**
  - Prediction (High Risk / Normal)
  - Probability of heart disease

---

## 🩺 Input Parameters
The model uses the following clinical inputs:
- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Max Heart Rate
- Exercise Induced Angina
- Oldpeak (ST Depression)
- ST Slope

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Joblib
- Git & GitHub

---

## 📁 Project Structure
heart-disease-prediction-app/
│
├── app.py

├── heart_disease_model.ipynb

├── heart_disease_model.pkl

├── scaler.pkl

├── model_columns.pkl

├── requirements.txt

├── README.md

└── .gitignore

---

## ▶️ Run Locally
1. Clone the repository:
```bash
git clone https://github.com/your-username/heart-disease-prediction-app.git
cd heart-disease-prediction-app
pip install -r requirements.txt
streamlit run app.py
