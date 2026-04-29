
# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd
import sklearn  # This is needed for the pickle file to load!

# Load the trained model
# --- Put the Model in Drive First---
with open("/content/Loan_model.pkl", "rb") as file:
    model = pickle.load(file)
# Title for the app
st.markdown(
    "<h1 style='text-align: center; background-color: #e0f7fa; padding: 10px; color: #006064;'><b>Loan Approval Predictor</b></h1>",
    unsafe_allow_html=True
)

st.header("Enter Applicant Details")

# --- Numeric Inputs based on your dataset  ---
granted_amount = st.number_input("Granted Loan Amount", min_value=5000, max_value=2000000, value=50000)
fico_score = st.slider("FICO Score", min_value=300, max_value=850, value=650)
monthly_income = st.number_input("Monthly Gross Income", min_value=0, value=5000)
housing_payment = st.number_input("Monthly Housing Payment", min_value=0, value=1200)
ever_bankrupt = st.selectbox("Ever Bankrupt or Foreclosed?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# --- Categorical Inputs based on your dataset  ---
# These must match the categories used during your encoding phase
reason_options = [
    "credit_card_refinancing", "debt_conslidation", "home_improvement", 
    "major_purchase", "other"
]
reason = st.selectbox("Reason for Loan", options=reason_options)

employment_status = st.selectbox("Employment Status", options=["full_time", "part_time", "unemployed"])

sector_options = [
    "communication_services", "consumer_discretionary", "consumer_staples", 
    "energy", "financials", "health_care", "industrials", 
    "information_technology", "materials", "real_estate", "utilities"
]
sector = st.selectbox("Employment Sector", options=sector_options)

lender = st.selectbox("Lending Partner", options=["A", "B", "C"])

# Create the input data as a DataFrame
input_data = pd.DataFrame({
    "Granted_Loan_Amount": [granted_amount],
    "FICO_score": [fico_score],
    "Monthly_Gross_Income": [monthly_income],
    "Monthly_Housing_Payment": [housing_payment],
    "Ever_Bankrupt_or_Foreclose": [ever_bankrupt],
    "Reason": [reason],
    "Employment_Status": [employment_status],
    "Employment_Sector": [sector],
    "Lender": [lender]
})

# --- Prepare Data for Prediction ---
# 1. One-hot encode the user's input
input_data_encoded = pd.get_dummies(input_data, columns=['Reason', 'Employment_Status', 'Employment_Sector', 'Lender'])

# 2. Add missing columns the model expects 
# Your model expected columns like 'Reason_credit_card_refinancing', 'Lender_B', etc.
model_columns = model.feature_names_in_
for col in model_columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

# 3. Ensure columns match the training order exactly
input_data_encoded = input_data_encoded[model_columns]

if st.button("Evaluate Loan Approval"):
    prediction = model.predict(input_data_encoded)[0]
    
    # In your project, the target 'Approved' is 1 for approved and 0 for denied 
    if prediction == 1:
        st.success("The prediction is: **Approved** ✅")
        # Calculate potential payout based on Instructions 
        payouts = {"A": 250, "B": 350, "C": 150}
        st.info(f"Potential Revenue from Lender {lender}: ${payouts.get(lender)}")
    else:
        st.error("The prediction is: **Denied** ❌")
