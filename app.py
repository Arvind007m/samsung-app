import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Framework Performance AI", layout="wide")

# Load models
model = joblib.load(os.path.join(SCRIPT_DIR, "regression_model.pkl"))
xgb_clf = joblib.load(os.path.join(SCRIPT_DIR, "classifier_model.pkl"))
scaler = joblib.load(os.path.join(SCRIPT_DIR, "scaler.pkl"))
le_service = joblib.load(os.path.join(SCRIPT_DIR, "le_service.pkl"))
le_framework = joblib.load(os.path.join(SCRIPT_DIR, "le_framework.pkl"))
clf_le = joblib.load(os.path.join(SCRIPT_DIR, "clf_le.pkl"))

st.markdown("""
<style>
.big-title {
    font-size:40px !important;
    font-weight:700;
    color:#4B8BBE;
}
.metric-box {
    background-color:#111;
    padding:15px;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">Microservice Performance Predictor</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    service = st.selectbox("Service", le_service.classes_)
    tps = st.number_input("TPS", 1000, 100000, 10000)
    threadpool = st.number_input("Threadpool", 10, 500, 100)

with col2:
    cpu = st.slider("CPU Usage", 0.0, 1.0, 0.2)
    memory = st.number_input("Memory (MB)", 100, 2000, 500)
    latency = st.number_input("Latency (ms)", 1, 10000, 100)
    throughput = st.number_input("Throughput (rps)", 1, 20000, 5000)
    error_rate = st.slider("Error Rate", 0.0, 10.0, 0.0)

if st.button("Predict Performance"):

    sample_reg = pd.DataFrame([{
        "Service_enc": le_service.transform([service])[0],
        "FrameWork_enc": le_framework.transform(["boot"])[0],
        "TPS": tps,
        "Threadpool": threadpool
    }])

    sample_scaled = scaler.transform(sample_reg)
    reg_pred = model.predict(sample_scaled)

    reg_df = pd.DataFrame(reg_pred, columns=[
        'Avg Response Time(ms)','P95','P99',
        'Throughput(rps)','Error Rate','CPU Usage',
        'Memory','Latency(ms)','Request Timeouts'
    ])

    st.subheader("Predicted Metrics")
    st.dataframe(reg_df)

    sample_clf = pd.DataFrame([{
        'TPS': tps,
        'Threadpool': threadpool,
        'CPU Usage': cpu,
        'Memory': memory,
        'Latency(ms)': latency,
        'Throughput(rps)': throughput,
        'Error Rate': error_rate
    }])

    framework_pred = xgb_clf.predict(sample_clf)[0]
    framework_name = clf_le.inverse_transform([framework_pred])[0]

    st.subheader("Recommended Framework")
    st.success(framework_name.upper())
