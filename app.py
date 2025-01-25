import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import sklearn

# Load the trained models for each category
categories_models = {
    "Insurance Leasing": "Data/best_clf3.pkl",
    "Bank": "Data/BankModel2.pkl",
    "Bank Leasing Sicav": "Data/BankLeasingModel.pkl",
    "Other": "Data/AutreModel.pkl"
}

# Define a function to load the model based on the selected category


def load_model(selected_category):
    model_path = categories_models[selected_category]
    return joblib.load(model_path)

# Define a function to extract features from a date and make predictions


def predict_price_up_down_for_date(date, model):
    # Extract features from the date
    day = date.day
    month = date.month
    year = date.year
    weekday = date.weekday()
    is_weekend = 1 if weekday >= 5 else 0

    # Dummy values for other features
    daily_price_change = 0.0
    seven_day_ma = 0.0
    thirty_day_ma = 0.0
    volume_change = 0.0
    rsi = 0.0
    volatility = 0.0

    # Create a DataFrame with the input data
    data = {
        'Day': [day],
        'Month': [month],
        'Year': [year],
        'Weekday': [weekday],
        'IsWeekend': [is_weekend],
        'DailyPriceChange': [daily_price_change],
        '7DayMA': [seven_day_ma],
        '30DayMA': [thirty_day_ma],
        'VolumeChange': [volume_change],
        'RSI': [rsi],
        'Volatility': [volatility]
    }
    df = pd.DataFrame(data)

    # Make prediction
    prediction = model.predict(df)

    # Return the prediction
    return prediction[0]


# Define categories and corresponding pages
categories = {
    "Insurance Leasing": "Insurance Leasing Price Prediction",
    "Bank": "Bank Price Prediction",
    "Bank Leasing Sicav": "Bank Leasing Sicav Price Prediction",
    "Other": "Other Price Prediction"
}


# Sidebar - Select Category
selected_category = st.sidebar.selectbox(
    'Select Category', list(categories_models.keys()))

# Display header for selected category
st.title(categories[selected_category])

# Load model for selected category
model = load_model(selected_category)

# Date input
date_input = st.date_input("Enter a date")

# Predict on date input
if st.button("Predict"):
    prediction = predict_price_up_down_for_date(date_input, model)
    if prediction == 1:
        st.write("The model predicts that the price will go up on", date_input)
        st.write("This means that the market expects the value of the asset to increase on the selected date. Investors may interpret this as a favorable signal to buy or hold onto the asset, anticipating potential future gains.")
    else:
        st.write("The model predicts that the price will go down on", date_input)
        st.write("This means that the market expects the value of the asset to decrease on the selected date. Investors may interpret this as a signal to sell the asset, or to refrain from buying until the price reaches a more favorable level.")
