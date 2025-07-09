import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("best_svm_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Digit Classifier", layout="centered")

# App title
st.title("🔢 Digit Classifier (0-9)")
st.write("Enter 64 pixel values (0–16) for an 8x8 grayscale image to predict the digit.")

# Input form
with st.form("digit_form"):
    inputs = []
    for i in range(8):
        row = st.columns(8)
        for j in range(8):
            value = row[j].number_input(f"{8*i + j}", min_value=0.0, max_value=16.0, step=1.0, key=f"pixel_{8*i+j}")
            inputs.append(value)

    submitted = st.form_submit_button("Predict Digit")

# Prediction
if submitted:
    # Reshape and scale input
    X_input = np.array(inputs).reshape(1, -1)
    X_scaled = scaler.transform(X_input)

    # Predict
    prediction = model.predict(X_scaled)[0]
    st.success(f"✅ Predicted Digit: **{prediction}**")

