import streamlit as st
import pandas as pd
import joblib

# Load the model components
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('columns.pkl')

# --- Page Configuration ---
st.set_page_config(page_title="CardioCheck AI", page_icon="❤️", layout="centered")

# --- Custom CSS for Aesthetics ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .stNumberInput, .stSelectbox {
        border-radius: 10px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header Section ---
st.title("❤️ CardioCheck AI")
st.subheader("Predictive Analytics for Heart Health")
st.write("Please fill in the clinical parameters below to assess heart disease risk.")
st.divider()

# --- Input Form ---
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 👤 Patient Info")
        age = st.number_input("Age", min_value=1, max_value=110, value=50)
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
        max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
        exercise_angina = st.selectbox("Exercise Induced Angina", ["N", "Y"])

    with col2:
        st.markdown("### 🩺 Clinical Tests")
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", value=120)
        cholesterol = st.number_input("Cholesterol (mg/dl)", value=200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
        resting_ecg = st.selectbox("Resting ECG Result", ["Normal", "ST", "LVH"])
        oldpeak = st.number_input("Oldpeak (ST Depression)", value=0.0, step=0.1)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.divider()

# --- Prediction Logic ---
if st.button("Generate Health Report"):
    input_data = {
        'Age': age, 'RestingBP': resting_bp, 'Cholesterol': cholesterol,
        'FastingBS': fasting_bs, 'MaxHR': max_hr, 'Oldpeak': oldpeak,
        'Sex': sex, 'ChestPainType': chest_pain, 'RestingECG': resting_ecg,
        'ExerciseAngina': exercise_angina, 'ST_Slope': st_slope
    }

    df = pd.DataFrame([input_data])

    # Preprocessing (matching your notebook logic)
    df_encoded = pd.get_dummies(df)
    df_final = df_encoded.reindex(columns=model_columns, fill_value=0)
    scaled_data = scaler.transform(df_final)

    prediction = model.predict(scaled_data)[0]
    prob = model.predict_proba(scaled_data)[0][1]

    # --- Result Display ---
    if prediction == 1:
        st.markdown(f"""
            <div class="prediction-box" style="background-color: #ffdce0; color: #af233a; border: 1px solid #af233a;">
                High Risk Detected<br>
                <span style="font-size: 16px;">Probability: {prob:.2%}</span>
            </div>
            """, unsafe_allow_html=True)
        st.warning("Recommendation: Please consult a cardiologist for further evaluation.")
    else:
        st.markdown(f"""
            <div class="prediction-box" style="background-color: #dcfce7; color: #15803d; border: 1px solid #15803d;">
                Normal Result<br>
                <span style="font-size: 16px;">Probability: {prob:.2%}</span>
            </div>
            """, unsafe_allow_html=True)
        st.success("The model suggests a low risk of heart disease.")

# Footer
st.caption("Disclaimer: This tool is for educational purposes and is not a substitute for professional medical advice.")
