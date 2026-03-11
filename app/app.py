import streamlit as st
import pandas as pd
import numpy as np
import joblib

# page configuration

st.set_page_config(
    page_title="E-Commerce Fraud Detector",
    page_icon="🛡️",
    layout="wide"
)

st.title("E-Commerce Fraud Detection System")
st.write(
    "This application predicts whether an e-commerce transaction is **fraudulent or legitimate** "
    "using a trained **XGBoost machine learning model**."
)

# load model

@st.cache_resource
def load_model():
    model = joblib.load("models/xgb_fraud_model.pkl")
    return model

model = load_model()

# sidebar input

st.sidebar.header("Transaction Input")

transaction_amount = st.sidebar.number_input("Transaction Amount",min_value=0.0,value=100.0)

quantity = st.sidebar.number_input("Quantity",min_value=1,value=1)

customer_age = st.sidebar.number_input("Customer Age",min_value=18,max_value=100,value=30)

account_age_days = st.sidebar.number_input("Account Age (days)",min_value=0,value=365)

transaction_hour = st.sidebar.slider("Transaction Hour",0,23,12)

month = st.sidebar.slider("Month",1,12,6)

weekday = st.sidebar.slider("Weekday",0,6,3)

is_weekend = st.sidebar.selectbox("Weekend Transaction",[0,1])

address_mismatch = st.sidebar.selectbox("Billing / Shipping Address Mismatch",[0,1])

ip_usage_count = st.sidebar.number_input("IP Usage Count",min_value=0,value=5)

customer_ip_count = st.sidebar.number_input("Customer IP Count",min_value=0,value=2)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["bank transfer","credit card","debit card"]
)

product_category = st.sidebar.selectbox(
    "Product Category",
    ["electronics","health & beauty","home & garden","toys & games"]
)

device_used = st.sidebar.selectbox(
    "Device Used",
    ["desktop","mobile","tablet"]
)

# create feature dictionary

data = {
    "Transaction Amount":transaction_amount,
    "Quantity":quantity,
    "Customer Age":customer_age,
    "Account Age Days":account_age_days,
    "Transaction Hour":transaction_hour,
    "Customer_IP_Count":customer_ip_count,
    "IP_Usage_Count":ip_usage_count,
    "Month":month,
    "Weekday":weekday,
    "IsWeekend":is_weekend,
    "Address_Mismatch":address_mismatch,

    "Payment Method_bank transfer":0,
    "Payment Method_credit card":0,
    "Payment Method_debit card":0,

    "Product Category_electronics":0,
    "Product Category_health & beauty":0,
    "Product Category_home & garden":0,
    "Product Category_toys & games":0,

    "Device Used_mobile":0,
    "Device Used_tablet":0
}

# activate selected categorical feature

data[f"Payment Method_{payment_method}"] = 1
data[f"Product Category_{product_category}"] = 1

if device_used != "desktop":
    data[f"Device Used_{device_used}"] = 1

# convert to dataframe

input_data = pd.DataFrame([data])

# prediction

st.subheader("Prediction")

if st.button("Predict Fraud"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error("Fraudulent Transaction Detected")
    else:
        st.success("Transaction Appears Legitimate")

    st.write(f"Fraud Probability: **{probability:.4f}**")

# display input data

st.subheader("Input Transaction Data")

st.dataframe(input_data)

# footer

st.markdown("---")
st.caption("Fraud Detection System using Machine Learning (XGBoost)")