# app.py

import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# 1. Load Model and Feature Columns
# -------------------------------
st.title("🌍 Air Quality Index (AQI) Prediction App")

# Replace with your latest saved files
model_filename = "aqi_model_2025m13_134449.pkl"
features_filename = "aqi_features_2025m13_134449.pkl"

model = joblib.load(model_filename)
feature_columns = joblib.load(features_filename)

st.success("Model and feature structure loaded successfully!")

# -------------------------------
# 2. User Input Form
# -------------------------------
st.header("Enter Pollution Levels")

# Pollutant inputs
pollutants = {
    "PM2.5": st.number_input("PM2.5", min_value=0.0, value=50.0),
    "PM10": st.number_input("PM10", min_value=0.0, value=80.0),
    "NO": st.number_input("NO", min_value=0.0, value=10.0),
    "NO2": st.number_input("NO2", min_value=0.0, value=20.0),
    "NOx": st.number_input("NOx", min_value=0.0, value=30.0),
    "NH3": st.number_input("NH3", min_value=0.0, value=15.0),
    "CO": st.number_input("CO", min_value=0.0, value=5.0),
    "SO2": st.number_input("SO2", min_value=0.0, value=40.0),
    "O3": st.number_input("O3", min_value=0.0, value=60.0),
    "Benzene": st.number_input("Benzene", min_value=0.0, value=0.5),
    "Toluene": st.number_input("Toluene", min_value=0.0, value=0.5),
}

# City selection
st.header("Select City")
selected_city = st.text_input("City Name (exactly as in dataset)", "Ahmedabad")

# -------------------------------
# 3. Prepare Input Data
# -------------------------------
if st.button("Predict AQI"):
    # Start with pollutant values
    input_data = pd.DataFrame([pollutants])

    # Add city column and apply one-hot encoding
    input_data["City"] = selected_city
    input_data = pd.get_dummies(input_data, columns=["City"], drop_first=True)

    # Ensure all training columns exist (add missing ones as 0)
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder columns to match training
    input_data = input_data[feature_columns]

    # -------------------------------
    # 4. Make Prediction
    # -------------------------------
    prediction = model.predict(input_data)[0]

    # Show result
    st.subheader(f"Predicted AQI: {prediction:.2f}")

    # -------------------------------
    # 5. AQI Category
    # -------------------------------
    if prediction <= 50:
        bucket = "Good 😊"
    elif prediction <= 100:
        bucket = "Satisfactory 🙂"
    elif prediction <= 200:
        bucket = "Moderate 😐"
    elif prediction <= 300:
        bucket = "Poor 😷"
    elif prediction <= 400:
        bucket = "Very Poor 🤢"
    else:
        bucket = "Severe ☠️"

    st.write(f"**AQI Category:** {bucket}")
