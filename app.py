import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="E-Commerce Churn Prediction",
    page_icon="🛒",
    layout="wide"
)

st.markdown(
    """
    <h1 style="text-align:center; color:#2E86C1;">
        🛍️ E-Commerce Customer Churn Prediction
    </h1>
    <p style="text-align:center; color:gray;">
        Enter customer details and predict whether they will churn.
    </p>
    """,
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs(["📊 Prediction", "📈 Model Info", "🗂️ Batch Prediction"])

with open("custom_style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.header("🔧 Customer Features")
    freq = st.number_input("Frequency (# of orders)", min_value=0.0, step=1.0)
    monetary = st.number_input("Monetary (total spent)", min_value=0.0, step=10.0)
    recency = st.number_input("Recency (days since last purchase)", min_value=0.0, step=1.0)
    tenure = st.number_input("Tenure (days since first purchase)", min_value=0.0, step=1.0)
    aov = st.number_input("AOV (Average Order Value)", min_value=0.0, step=1.0)
    avg_units = st.number_input("Avg Units per Order", min_value=0.0, step=1.0)
    return_rate = st.number_input("Return Rate", min_value=0.0, max_value=1.0, step=0.01)
    avg_gap = st.number_input("Avg Purchase Gap (days)", min_value=0.0, step=1.0)
    discount_rate = st.number_input("Discount Amount Rate (%)", min_value=0.0, max_value=100.0, value=0.0)
    engagement = st.number_input("Engagement Rate (%)", min_value=0.0, max_value=100.0, value=0.0)

    region = st.selectbox("Region", ["Central", "East", "South", "West"])
    segment = st.selectbox("Segment", ["Consumer", "Corporate", "Home Office"])

input_dict = {
    'Frequency': freq,
    'Monetary': monetary,
    'Recency': recency,
    'Tenure': tenure,
    'AOV': aov,
    'AvgUnitsPerOrder': avg_units,
    'ReturnRate': return_rate,
    'AvgGap': avg_gap,
    'DiscountAmountRate': discount_rate/100,
    'EngagementRate': engagement/100,
    'Region_Central': 1 if region == "Central" else 0,
    'Region_East': 1 if region == "East" else 0,
    'Region_South': 1 if region == "South" else 0,
    'Region_West': 1 if region == "West" else 0,
    'Segment_Consumer': 1 if segment == "Consumer" else 0,
    'Segment_Corporate': 1 if segment == "Corporate" else 0,
    'Segment_Home Office': 1 if segment == "Home Office" else 0
}

input_df = pd.DataFrame([input_dict])

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

with tab3:
    st.subheader("🗂️ Batch Churn Prediction (CSV Upload)")
    st.write("Upload a CSV file with customer features to predict churn for multiple customers at once.")
    example_cols = ['Frequency','Monetary','Recency','Tenure','AOV','AvgUnitsPerOrder','ReturnRate','AvgGap','DiscountAmountRate','EngagementRate',
                   'Region_Central','Region_East','Region_South','Region_West','Segment_Consumer','Segment_Corporate','Segment_Home Office']
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        try:
            batch_df = pd.read_csv(uploaded_file)
            missing_cols = [col for col in example_cols if col not in batch_df.columns]
            if missing_cols:
                st.error(f"Missing columns in uploaded CSV: {missing_cols}")
            else:
                batch_preds = model.predict(batch_df)
                batch_probs = model.predict_proba(batch_df)[:,1]
                batch_df['Churn_Prediction'] = np.where(batch_preds == 1, 'CHURN', 'ACTIVE')
                batch_df['Churn_Probability'] = batch_probs
                st.success(f"Batch prediction complete! Showing results:")
                st.dataframe(batch_df[[*example_cols, 'Churn_Prediction', 'Churn_Probability']])
                csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name='batch_churn_predictions.csv',
                    mime='text/csv',
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")

