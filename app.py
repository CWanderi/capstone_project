import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model
model = joblib.load('xgboost_model.pkl')

# Title
st.title("🌾 Agricultural Suitability Predictor")

st.markdown("Fill in the following parameters to predict land suitability:")

# --- Inputs ---
ndvi = st.slider("NDVI (Normalized Difference Vegetation Index)", 0.0, 1.0, 0.5, 0.01)

degradation = st.selectbox("Degradation Level", ["None", "Low", "Moderate", "Severe"])

soil_texture = st.selectbox("Soil Texture", ["Sandy", "Loamy", "Clay", "Silty", "Other"])

crop_name = st.selectbox("Crop Name", ["Maize", "Wheat", "Rice", "Sorghum", "Beans", "Other"])

rainfall = st.number_input("Rainfall (mm/year)", min_value=0.0, max_value=3000.0, value=1000.0)

pH = st.number_input("Soil pH", min_value=3.0, max_value=10.0, value=6.5)

# NOTE: You can ignore soil_texture_name if it's redundant with soil_texture

# Submit button
if st.button("Predict Suitability"):
    # Encode categorical values manually if necessary
    degradation_dict = {"None": 0, "Low": 1, "Moderate": 2, "Severe": 3}
    soil_texture_dict = {"Sandy": 0, "Loamy": 1, "Clay": 2, "Silty": 3, "Other": 4}
    crop_name_dict = {"Maize": 0, "Wheat": 1, "Rice": 2, "Sorghum": 3, "Beans": 4, "Other": 5}

    # Create feature array
    input_data = np.array([[ndvi,
                            degradation_dict[degradation],
                            soil_texture_dict[soil_texture],
                            crop_name_dict[crop_name],
                            soil_texture_dict[soil_texture],  # reusing soil_texture as soil_texture_name
                            rainfall,
                            pH]])

    # Prediction
    prediction = model.predict(input_data)[0]

    # Display
    if prediction == 1:
        st.success("✅ The land is **suitable** for agriculture.")
    else:
        st.warning("❌ The land is **not suitable** for agriculture.")
