
# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd
import sklearn  # This is needed for the pickle file to load!

# Load the trained model
# --- Put the Model in Drive First---
with open("/content/Loan_model.pkl", "rb") as file:
    model = pickle.load(file)
    
CUSTOM_THRESHOLD = 0.7

# 2. App Title & Styling
st.markdown(
    """
    <div style='text-align: center; background-color: #ffcccc; padding: 15px; border-radius: 10px;'>
        <h1 style='color: #cc0000; margin-bottom: 0;'><b>Loan Approval Predictor</b></h1>
        <p style='color: #8b0000; font-size: 1.1em;'>Maximizing Revenue through Strategic Lender Matching</p>
    </div>
    """, 
    unsafe_allow_html=True
)

st.write("") 

# 3. Input Section
st.header("Enter Loan Applicant's Details")
col1, col2 = st.columns(2)

with col1:
    granted_amt = st.number_input("Granted Loan Amount ($)", min_value=0, value=25000)
    fico = st.slider("FICO Score", 300, 850, 700)
    income = st.number_input("Monthly Gross Income ($)", min_value=0, value=5000)

with col2:
    housing = st.number_input("Monthly Housing Payment ($)", min_value=0, value=1200)
    bankrupt = st.selectbox("Ever Bankrupt or Foreclosed?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Categorical inputs 
reason = st.selectbox("Reason for Loan", ["debt_conslidation", "credit_card_refinancing", "home_improvement", "major_purchase", "other"])
emp_status = st.selectbox("Employment Status", ["employed", "part_time", "unemployed"])
sector = st.selectbox("Employment Sector", [
    "information_technology", "financials", "health_care", "energy", 
    "communication_services", "consumer_discretionary", "consumer_staples", 
    "industrials", "materials", "real_estate", "utilities"
])
lender = st.selectbox("Select Lender", ["A", "B", "C"])

# Create the input data as a DataFrame
input_data = pd.DataFrame({
    "Granted_Loan_Amount": [granted_amt],
    "FICO_score": [fico],
    "Monthly_Gross_Income": [income],
    "Monthly_Housing_Payment": [housing],
    "Ever_Bankrupt_or_Foreclose": [bankrupt],
    "Reason": [reason],
    "Employment_Status": [emp_status],
    "Employment_Sector": [sector],
    "Lender": [lender]
})

# --- Prepare Data for Prediction ---

# 1. One-hot encode the user's input.
input_data_encoded = pd.get_dummies(input_data)

# 2. Add any "missing" columns the model expects (fill with 0).
model_columns = model.feature_names_in_
for col in model_columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

# 3. Reorder/filter columns to exactly match the model's training data.
input_data_encoded = input_data_encoded[model_columns]

"""
What happens if the user enters a value not in the training data?
Example: User enters Sector = 'Agriculture', but the model only knows 'Energy' and 'Financials'.

1. pd.get_dummies creates a new column: Employment_Sector_Agriculture = 1.
2. The code then adds the known columns: Employment_Sector_Energy = 0 and Employment_Sector_Financials = 0.
3. The final filtering step drops the unknown 'Agriculture' column because it's not in the 
   model's expected feature list.

Result: The model receives 0s for all known categories, which correctly 
treats the unknown input as "Other".
"""

# 4. Predict button
if st.button("Evaluate Loan"):
    # Get probability for the custom 0.7 threshold
    proba = model.predict_proba(input_data_encoded)[0][1]
    prediction = 1 if proba >= CUSTOM_THRESHOLD else 0

    # Lender Payouts (A=$250, B=$350, C=$150)
    payouts = {"A": 250, "B": 350, "C": 150}
    potential_payout = payouts[lender]

    # Display results
    st.write("---")
    st.write(f"**Approval Probability:** {proba:.2%}")

    if prediction == 1:
        st.success(f"### Prediction: **Approved** ✅")
        st.write(f"The platform payout for **Lender {lender}** is: **${potential_payout}**")
    else:
        st.error("### Prediction: **Denied** ❌")
        st.write(f"Probability is below the required **{CUSTOM_THRESHOLD}** threshold.")

