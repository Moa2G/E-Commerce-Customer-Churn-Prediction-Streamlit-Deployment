import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page title
st.title("🔮 Customer Churn Prediction App")

st.write("Fill in customer details below to predict churn (0 = Active, 1 = Churned).")

# Numerical inputs
freq = st.number_input("Frequency (# of orders)", min_value=0.0, step=1.0)
monetary = st.number_input("Monetary (total spent)", min_value=0.0, step=10.0)
recency = st.number_input("Recency (days since last purchase)", min_value=0.0, step=1.0)
tenure = st.number_input("Tenure (days since first purchase)", min_value=0.0, step=1.0)
aov = st.number_input("AOV (Average Order Value)", min_value=0.0, step=1.0)
avg_units = st.number_input("Avg Units per Order", min_value=0.0, step=1.0)
return_rate = st.number_input("Return Rate", min_value=0.0, max_value=1.0, step=0.01)
avg_gap = st.number_input("Avg Purchase Gap (days)", min_value=0.0, step=1.0)
discount_rate = st.number_input("Discount Amount Rate", min_value=0.0, max_value=1.0, step=0.01)
engagement = st.number_input("Engagement Rate (%)", min_value=0.0, max_value=100.0, value=30.0)

# Categorical inputs (one-hot encoded)
region = st.selectbox("Region", ["Central", "East", "South", "West"])
segment = st.selectbox("Segment", ["Consumer", "Corporate", "Home Office"])

# Prepare input row
input_dict = {
    'Frequency': freq,
    'Monetary': monetary,
    'Recency': recency,
    'Tenure': tenure,
    'AOV': aov,
    'AvgUnitsPerOrder': avg_units,
    'ReturnRate': return_rate,
    'AvgGap': avg_gap,
    'DiscountAmountRate': discount_rate,
    'EngagementRate': engagement,
    'Region_Central': 1 if region == "Central" else 0,
    'Region_East': 1 if region == "East" else 0,
    'Region_South': 1 if region == "South" else 0,
    'Region_West': 1 if region == "West" else 0,
    'Segment_Consumer': 1 if segment == "Consumer" else 0,
    'Segment_Corporate': 1 if segment == "Corporate" else 0,
    'Segment_Home Office': 1 if segment == "Home Office" else 0
}

input_df = pd.DataFrame([input_dict])

# Scale numerical features (must match training)
num_cols = ['Frequency','Monetary','Recency','Tenure','AOV','AvgUnitsPerOrder',
            'ReturnRate','AvgGap','DiscountAmountRate','EngagementRate']
input_df[num_cols] = scaler.transform(input_df[num_cols])

# Predict
if st.button("Predict Churn"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f"⚠️ This customer is likely to **CHURN** (probability: {prob:.2f})")
    else:
        st.success(f"✅ This customer is likely to stay **ACTIVE** (probability: {1-prob:.2f})")
