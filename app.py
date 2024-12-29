import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the saved model and scaler
model = joblib.load('car_purchase_model.pkl')
scaler = joblib.load('scaler.pkl')  # Save and load the scaler used during training

# Define the app interface
st.title("Car Purchase Amount Prediction")
st.write("Enter customer details below to predict their car purchase amount.")

# Input fields for user data
age = st.number_input("Age", min_value=18, max_value=100, step=1, value=30)
annual_salary = st.number_input("Annual Salary ($)", min_value=0.0, step=1000.0, value=50000.0)
credit_card_debt = st.number_input("Credit Card Debt ($)", min_value=0.0, step=100.0, value=5000.0)
net_worth = st.number_input("Net Worth ($)", min_value=0.0, step=1000.0, value=100000.0)
gender = st.selectbox("Gender", options=["Male", "Female"])

# Encode gender as numerical value
gender_encoded = 0 if gender == "Male" else 1

# Predict button
if st.button("Predict"):
    # Prepare input data as a NumPy array
    input_data = np.array([[gender_encoded, age, annual_salary, credit_card_debt, net_worth]])

    # Apply the same scaling as during training
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Display the result
    st.success(f"Predicted Car Purchase Amount: ${prediction[0]:,.2f}")
