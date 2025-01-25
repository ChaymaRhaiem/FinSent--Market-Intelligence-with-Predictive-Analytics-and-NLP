
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import sklearn
import streamlit_shadcn_ui as shadcn


# Define categories and corresponding models
categories_models = {
    "Insurance Leasing": "Data/adaboost_classifier_model.joblib",
    "Bank": "Data/Bank_adaboost_classifier_model.joblib",
    "Bank Leasing Sicav": "Data/BankLeasing_adaboost_classifier_model.joblib",
    "Other": "Data/Autre_adaboost_classifier_model.joblib"
}

# Define risk descriptions and advice
risk_descriptions = {
    "Low": "This stock is considered low risk. It may provide steady returns with minimal fluctuations. It could be suitable for conservative investors or those seeking stability in their investment portfolio.",
    "High": "This stock is considered high risk. It may experience significant price fluctuations and could result in substantial gains or losses. It could be suitable for experienced investors willing to take on higher levels of risk."
}

# Streamlit App
st.title('Stock Risk Prediction')

# Sidebar
st.sidebar.title('Select Category')
selected_category = st.sidebar.selectbox(
    'Select Category', list(categories_models.keys()))

# Load the saved model based on selected category
loaded_model = load(categories_models[selected_category])

# Function to perform risk assessment


def perform_risk_assessment(loaded_model, selected_category):
    st.header(selected_category)
    st.sidebar.title('Enter Stock Features')
    rsi = st.sidebar.number_input(
        'RSI', min_value=0.0, max_value=100.0, value=50.0)
    volatility = st.sidebar.number_input(
        'Volatility', min_value=0.0, value=0.05)
    volume_change = st.sidebar.number_input('Volume Change', value=0.0)

    # Make prediction
    if st.sidebar.button('Predict Risk Level', key="predict_btn"):
        # Prepare data for prediction
        single_stock_data = np.array([[rsi, volatility, volume_change]])

        # Predict risk level for the single stock using the loaded model
        predicted_risk_level = loaded_model.predict(single_stock_data)

        # Display predicted risk level
        st.write("Predicted Risk Level:", predicted_risk_level[0])

        # Display risk description and advice
        if predicted_risk_level[0] in risk_descriptions:
            st.write("Risk Description:")
            st.write(risk_descriptions[predicted_risk_level[0]])


# Perform risk assessment based on selected category
perform_risk_assessment(loaded_model, selected_category)
