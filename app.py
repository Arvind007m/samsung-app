import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models_v2")

st.set_page_config(
    page_title="Microservice Performance AI",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_models():
    return {
        'model': joblib.load(os.path.join(MODELS_DIR, "model.pkl")),
        'scaler': joblib.load(os.path.join(MODELS_DIR, "scaler.pkl")),
        'le_framework': joblib.load(os.path.join(MODELS_DIR, "le_framework.pkl"))
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
    .framework-desc {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.7);
        margin-top: 0.5rem;
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
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Microservice Performance Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered framework recommendation</p>', unsafe_allow_html=True)

st.markdown('<p class="section-title">Load Configuration</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    tps = st.slider(
        "Transactions Per Second (TPS)",
        min_value=5000,
        max_value=50000,
        value=10000,
        step=1000
    )
with col2:
    threadpool = st.slider(
        "Thread Pool Size",
        min_value=50,
        max_value=500,
        value=100,
        step=10
    )

st.markdown('<p class="section-title">Desired Performance Targets</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    target_latency = st.slider(
        "Target Latency (ms)",
        min_value=1,
        max_value=5000,
        value=100,
        step=10,
        help="Desired response latency"
    )
    target_throughput = st.slider(
        "Target Throughput (rps)",
        min_value=100,
        max_value=30000,
        value=10000,
        step=100,
        help="Desired requests per second"
    )
    target_error_rate = st.slider(
        "Target Error Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=0.5,
        step=0.1,
        help="Maximum acceptable error rate"
    )
with col2:
    target_cpu = st.slider(
        "Target CPU Usage",
        min_value=0.05,
        max_value=1.0,
        value=0.3,
        step=0.05,
        format="%.2f",
        help="Desired CPU utilization"
    )
    target_memory = st.slider(
        "Target Memory (MB)",
        min_value=200,
        max_value=1000,
        value=500,
        step=50,
        help="Desired memory usage"
    )
    target_response_time = st.slider(
        "Target Avg Response Time (ms)",
        min_value=1,
        max_value=5000,
        value=100,
        step=10,
        help="Desired average response time"
    )

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

recommend_btn = st.button("Recommend Best Framework", use_container_width=True)

def predict_performance(framework_name):
    framework_enc = models['le_framework'].transform([framework_name])[0]
    
    tps_thread_ratio = tps / (threadpool + 1)
    tps_log = np.log1p(tps)
    thread_log = np.log1p(threadpool)
    tps_thread_product = tps * threadpool
    tps_sq = tps ** 2
    thread_sq = threadpool ** 2
    
    sample = pd.DataFrame([{
        'FrameWork_enc': framework_enc,
        'TPS': tps,
        'Threadpool': threadpool,
        'TPS_Thread_ratio': tps_thread_ratio,
        'TPS_log': tps_log,
        'Thread_log': thread_log,
        'TPS_Thread_product': tps_thread_product,
        'TPS_sq': tps_sq,
        'Thread_sq': thread_sq
    }])
    
    sample_scaled = models['scaler'].transform(sample)
    
    predictions = models['model'].predict(sample_scaled)[0]
    
    min_values = [1, 1, 1, 10, 0, 0.05, 100, 1, 0]
    predictions = np.maximum(predictions, min_values)
    
    return predictions

if recommend_btn:
    with st.spinner("Analyzing which framework best matches your targets..."):
        boot_pred = predict_performance("boot")
        webflux_pred = predict_performance("webflux")
        
        def calculate_distance(pred, targets):
            distances = []
            distances.append(abs(pred[0] - targets['response_time']) / max(targets['response_time'], 1))
            distances.append(abs(pred[3] - targets['throughput']) / max(targets['throughput'], 1))
            distances.append(abs(pred[4] - targets['error_rate']) / max(targets['error_rate'], 0.1))
            distances.append(abs(pred[5] - targets['cpu']) / max(targets['cpu'], 0.01))
            distances.append(abs(pred[6] - targets['memory']) / max(targets['memory'], 1))
            distances.append(abs(pred[7] - targets['latency']) / max(targets['latency'], 1))
            return sum(distances) / len(distances)
        
        targets = {
            'response_time': target_response_time,
            'throughput': target_throughput,
            'error_rate': target_error_rate,
            'cpu': target_cpu,
            'memory': target_memory,
            'latency': target_latency
        }
        
        boot_distance = calculate_distance(boot_pred, targets)
        webflux_distance = calculate_distance(webflux_pred, targets)
        
        if boot_distance < webflux_distance:
            framework_name = "boot"
            match_score = (1 - boot_distance / (boot_distance + webflux_distance)) * 100
        else:
            framework_name = "webflux"
            match_score = (1 - webflux_distance / (boot_distance + webflux_distance)) * 100
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    if framework_name == "webflux":
        desc = "Reactive, non-blocking - Best for high concurrency"
    else:
        desc = "Traditional, blocking - Best for simple workloads"
    
    st.markdown(f"""
    <div class="framework-box">
        <p class="framework-label">Recommended Framework</p>
        <p class="framework-name">{framework_name}</p>
        <p class="framework-desc">{desc}</p>
        <p style="color: rgba(255,255,255,0.9); font-size: 0.9rem; margin-top: 0.5rem;">Match Score: {match_score:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666; font-size: 0.85rem;'>"
    "Powered by XGBoost ML Model"
    "</p>",
    unsafe_allow_html=True
)
