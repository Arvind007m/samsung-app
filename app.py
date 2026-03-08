import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Microservice Performance AI",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_models():
    return {
        'model': joblib.load(os.path.join(SCRIPT_DIR, "regression_model.pkl")),
        'classifier': joblib.load(os.path.join(SCRIPT_DIR, "classifier_model.pkl")),
        'scaler': joblib.load(os.path.join(SCRIPT_DIR, "scaler.pkl")),
        'le_service': joblib.load(os.path.join(SCRIPT_DIR, "le_service.pkl")),
        'le_framework': joblib.load(os.path.join(SCRIPT_DIR, "le_framework.pkl")),
        'clf_le': joblib.load(os.path.join(SCRIPT_DIR, "clf_le.pkl"))
    }

models = load_models()

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #667eea;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #333;
        text-align: center;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #333;
        margin: 0.5rem 0;
        text-align: center;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #aaa;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #fff;
    }
    .framework-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .framework-label {
        font-size: 1rem;
        color: rgba(255,255,255,0.8);
        margin-bottom: 0.5rem;
    }
    .framework-name {
        font-size: 2.5rem;
        font-weight: 700;
        color: #fff;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    .predicted-metric {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.5rem 1rem;
        border: 1px solid #444;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.15);
        transition: all 0.3s ease;
    }
    .predicted-metric:hover {
        border-color: #667eea;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        transform: translateY(-3px);
    }
    .predicted-metric-label {
        font-size: 0.9rem;
        color: #aaa;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .predicted-metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .input-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #333;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Microservice Performance Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered performance prediction and framework recommendation</p>', unsafe_allow_html=True)

st.markdown('<p class="section-title">Select Service</p>', unsafe_allow_html=True)
service = st.selectbox(
    "Service",
    models['le_service'].classes_,
    label_visibility="collapsed"
)

st.markdown('<p class="section-title">Load Parameters</p>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    tps = st.slider(
        "Transactions Per Second (TPS)",
        min_value=1000,
        max_value=100000,
        value=10000,
        step=1000
    )
with col2:
    threadpool = st.slider(
        "Thread Pool Size",
        min_value=10,
        max_value=500,
        value=100,
        step=10
    )

st.markdown('<p class="section-title">Resource Metrics</p>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    cpu = st.slider(
        "CPU Usage (%)",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=5.0
    )
    memory = st.slider(
        "Memory (MB)",
        min_value=100,
        max_value=2000,
        value=500,
        step=50
    )
with col2:
    latency = st.slider(
        "Latency (ms)",
        min_value=1,
        max_value=1000,
        value=100,
        step=10
    )
    throughput = st.slider(
        "Throughput (rps)",
        min_value=100,
        max_value=20000,
        value=5000,
        step=100
    )

error_rate = st.slider(
    "Error Rate (%)",
    min_value=0.0,
    max_value=10.0,
    value=0.0,
    step=0.1
)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

predict_btn = st.button("Predict Performance", use_container_width=True)

if predict_btn:
    with st.spinner("Analyzing performance metrics..."):
        sample_reg = pd.DataFrame([{
            "Service_enc": models['le_service'].transform([service])[0],
            "FrameWork_enc": models['le_framework'].transform(["boot"])[0],
            "TPS": tps,
            "Threadpool": threadpool
        }])

        sample_scaled = models['scaler'].transform(sample_reg)
        reg_pred = models['model'].predict(sample_scaled)[0]
        
        reg_pred = np.maximum(reg_pred, 0)

        sample_clf = pd.DataFrame([{
            'TPS': tps,
            'Threadpool': threadpool,
            'CPU Usage': cpu / 100,
            'Memory': memory,
            'Latency(ms)': latency,
            'Throughput(rps)': throughput,
            'Error Rate': error_rate
        }])

        framework_pred = models['classifier'].predict(sample_clf)[0]
        framework_name = models['clf_le'].inverse_transform([framework_pred])[0]

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("<h3 style='text-align: center; margin-bottom: 1.5rem;'>Predicted Performance Metrics</h3>", unsafe_allow_html=True)
    
    metric_names = [
        'Avg Response Time(ms)', 'P95', 'P99', 'Throughput(rps)',
        'Error Rate', 'CPU Usage', 'Memory', 'Latency(ms)', 'Request Timeouts'
    ]
    
    metrics_display = [
        ("Response Time", f"{reg_pred[0]:.1f} ms"),
        ("P95 Latency", f"{reg_pred[1]:.1f} ms"),
        ("P99 Latency", f"{reg_pred[2]:.1f} ms"),
        ("Throughput", f"{reg_pred[3]:.0f} rps"),
        ("Error Rate", f"{reg_pred[4]:.2f}%"),
        ("CPU Usage", f"{reg_pred[5]:.1f}%"),
        ("Memory", f"{reg_pred[6]:.0f} MB"),
        ("Latency", f"{reg_pred[7]:.1f} ms"),
        ("Timeouts", f"{reg_pred[8]:.0f}"),
    ]
    
    row1 = st.columns(3)
    for i, col in enumerate(row1):
        with col:
            st.markdown(f"""
            <div class="predicted-metric">
                <p class="predicted-metric-label">{metrics_display[i][0]}</p>
                <p class="predicted-metric-value">{metrics_display[i][1]}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    row2 = st.columns(3)
    for i, col in enumerate(row2):
        with col:
            st.markdown(f"""
            <div class="predicted-metric">
                <p class="predicted-metric-label">{metrics_display[i+3][0]}</p>
                <p class="predicted-metric-value">{metrics_display[i+3][1]}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    row3 = st.columns(3)
    for i, col in enumerate(row3):
        with col:
            st.markdown(f"""
            <div class="predicted-metric">
                <p class="predicted-metric-label">{metrics_display[i+6][0]}</p>
                <p class="predicted-metric-value">{metrics_display[i+6][1]}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="framework-box">
        <p class="framework-label">Recommended Framework</p>
        <p class="framework-name">{framework_name}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.expander("View Raw Predictions"):
        pred_df = pd.DataFrame([reg_pred], columns=metric_names)
        st.dataframe(pred_df, use_container_width=True)

else:
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-label">Step 1</p>
            <p class="metric-value">Configure</p>
            <p style="color: #888; font-size: 0.9rem;">Set parameters above</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-label">Step 2</p>
            <p class="metric-value">Predict</p>
            <p style="color: #888; font-size: 0.9rem;">Click the button</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-label">Step 3</p>
            <p class="metric-value">Optimize</p>
            <p style="color: #888; font-size: 0.9rem;">Get recommendations</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666; font-size: 0.85rem;'>"
    "Powered by XGBoost ML Models"
    "</p>",
    unsafe_allow_html=True
)
