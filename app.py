import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from groq import Groq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models_v2")

def get_api_key():
    """Get API key from secrets (cloud) or session state (local)"""
    # Try Streamlit secrets first (for cloud deployment)
    try:
        return st.secrets["GROQ_API_KEY"]
    except:
        pass
    # Fall back to session state (for local testing)
    return st.session_state.get('groq_api_key', '')

def get_ai_description(framework_name, match_score, boot_pred, webflux_pred, targets, tps, threadpool):
    api_key = get_api_key()
    if not api_key:
        return None
    
    try:
        client = Groq(api_key=api_key)
        
        # Determine which metrics are closer to user targets
        metrics_closer = []
        
        boot_lat_diff = abs(boot_pred[7] - targets['latency'])
        webflux_lat_diff = abs(webflux_pred[7] - targets['latency'])
        metrics_closer.append(("Spring Boot" if boot_lat_diff < webflux_lat_diff else "WebFlux", "latency"))
        
        boot_tp_diff = abs(boot_pred[3] - targets['throughput'])
        webflux_tp_diff = abs(webflux_pred[3] - targets['throughput'])
        metrics_closer.append(("Spring Boot" if boot_tp_diff < webflux_tp_diff else "WebFlux", "throughput"))
        
        boot_rt_diff = abs(boot_pred[0] - targets['response_time'])
        webflux_rt_diff = abs(webflux_pred[0] - targets['response_time'])
        metrics_closer.append(("Spring Boot" if boot_rt_diff < webflux_rt_diff else "WebFlux", "response time"))
        
        boot_cpu_diff = abs(boot_pred[5] - targets['cpu'])
        webflux_cpu_diff = abs(webflux_pred[5] - targets['cpu'])
        metrics_closer.append(("Spring Boot" if boot_cpu_diff < webflux_cpu_diff else "WebFlux", "CPU usage"))
        
        boot_mem_diff = abs(boot_pred[6] - targets['memory'])
        webflux_mem_diff = abs(webflux_pred[6] - targets['memory'])
        metrics_closer.append(("Spring Boot" if boot_mem_diff < webflux_mem_diff else "WebFlux", "memory usage"))
        
        rec_name = "Spring Boot" if framework_name == "boot" else "WebFlux"
        other_name = "WebFlux" if framework_name == "boot" else "Spring Boot"
        closer_metrics = [m[1] for m in metrics_closer if m[0] == rec_name]
        other_closer = [m[1] for m in metrics_closer if m[0] != rec_name]
        
        prompt = f"""Analyze this framework recommendation and explain why {rec_name} is the better choice.

Configuration: {tps} TPS, {threadpool} thread pool

User's Target Requirements:
- Latency: {targets['latency']} ms
- Throughput: {targets['throughput']} rps
- Response Time: {targets['response_time']} ms
- Error Rate: {targets['error_rate']}%
- CPU Usage: {targets['cpu']}
- Memory: {targets['memory']} MB

Spring Boot Predicted:
- Latency: {boot_pred[7]:.1f} ms
- Throughput: {boot_pred[3]:.0f} rps
- Response Time: {boot_pred[0]:.1f} ms
- Error Rate: {boot_pred[4]:.2f}%
- CPU Usage: {boot_pred[5]:.2f}
- Memory: {boot_pred[6]:.0f} MB

WebFlux Predicted:
- Latency: {webflux_pred[7]:.1f} ms
- Throughput: {webflux_pred[3]:.0f} rps
- Response Time: {webflux_pred[0]:.1f} ms
- Error Rate: {webflux_pred[4]:.2f}%
- CPU Usage: {webflux_pred[5]:.2f}
- Memory: {webflux_pred[6]:.0f} MB

Analysis:
- {rec_name} is closer to target for: {', '.join(closer_metrics) if closer_metrics else 'none'}
- {other_name} is closer to target for: {', '.join(other_closer) if other_closer else 'none'}

Recommendation: {rec_name} (Match Score: {match_score:.1f}%)

Write 3-4 bullet points explaining why {rec_name} is recommended:
- Each bullet should explain one reason
- Mention which metrics are closer to the user's targets
- Keep each bullet brief and clear

IMPORTANT: Do NOT include any numbers. Use comparative language like "closer to target", "better aligned" instead."""

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a microservice expert. Respond ONLY in bullet points (use • or -). Do NOT include any numbers. Keep each bullet brief. Be confident."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=200
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"AI Analysis unavailable: {str(e)}"

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

with st.sidebar:
    st.markdown("### AI Settings")
    # Check if API key is configured in secrets
    has_secret_key = False
    try:
        if st.secrets.get("GROQ_API_KEY"):
            has_secret_key = True
            st.success("API Key configured!")
    except:
        pass
    
    if not has_secret_key:
        groq_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Get free API key from console.groq.com"
        )
        if groq_key:
            st.session_state['groq_api_key'] = groq_key
            st.success("API Key saved!")
        
        st.markdown("---")
        st.markdown("**How to get API key:**")
        st.markdown("1. Go to [console.groq.com](https://console.groq.com)")
        st.markdown("2. Sign up / Log in")
        st.markdown("3. Create API key")
        st.markdown("4. Paste above")

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
        
        def calculate_weighted_score(pred, targets):
            weights = {
                'latency': 0.25,
                'response_time': 0.20,
                'throughput': 0.20,
                'error_rate': 0.15,
                'cpu': 0.10,
                'memory': 0.10
            }
            
            weighted_distance = 0
            weighted_distance += weights['response_time'] * abs(pred[0] - targets['response_time']) / max(targets['response_time'], 1)
            weighted_distance += weights['throughput'] * abs(pred[3] - targets['throughput']) / max(targets['throughput'], 1)
            weighted_distance += weights['error_rate'] * abs(pred[4] - targets['error_rate']) / max(targets['error_rate'], 0.1)
            weighted_distance += weights['cpu'] * abs(pred[5] - targets['cpu']) / max(targets['cpu'], 0.01)
            weighted_distance += weights['memory'] * abs(pred[6] - targets['memory']) / max(targets['memory'], 1)
            weighted_distance += weights['latency'] * abs(pred[7] - targets['latency']) / max(targets['latency'], 1)
            
            return weighted_distance
        
        targets = {
            'response_time': target_response_time,
            'throughput': target_throughput,
            'error_rate': target_error_rate,
            'cpu': target_cpu,
            'memory': target_memory,
            'latency': target_latency
        }
        
        boot_score = calculate_weighted_score(boot_pred, targets)
        webflux_score = calculate_weighted_score(webflux_pred, targets)
        
        if boot_score < webflux_score:
            framework_name = "boot"
            match_score = (1 - boot_score / (boot_score + webflux_score)) * 100
        else:
            framework_name = "webflux"
            match_score = (1 - webflux_score / (boot_score + webflux_score)) * 100
    
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
    
    if get_api_key():
        st.markdown("<br>", unsafe_allow_html=True)
        with st.spinner("Generating AI analysis..."):
            ai_description = get_ai_description(
                framework_name, match_score, boot_pred, webflux_pred, 
                targets, tps, threadpool
            )
        
        if ai_description:
            # Convert bullet points to HTML list
            lines = ai_description.strip().split('\n')
            bullets_html = ""
            for line in lines:
                line = line.strip()
                if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                    bullet_text = line.lstrip('•-* ').strip()
                    bullets_html += f"<li style='margin-bottom: 0.5rem;'>{bullet_text}</li>"
                elif line:
                    bullets_html += f"<li style='margin-bottom: 0.5rem;'>{line}</li>"
            
            st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.1); border: 1px solid rgba(102, 126, 234, 0.3); 
                        border-radius: 10px; padding: 1.2rem; margin-top: 1rem;">
                <p style="color: #667eea; font-weight: 600; margin-bottom: 0.8rem;">🤖 AI Analysis</p>
                <ul style="color: #ccc; font-size: 0.95rem; line-height: 1.6; margin: 0; padding-left: 1.2rem;">
                    {bullets_html}
                </ul>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(255, 193, 7, 0.1); border: 1px solid rgba(255, 193, 7, 0.3); 
                    border-radius: 10px; padding: 1rem; margin-top: 1rem; text-align: center;">
            <p style="color: #ffc107; font-size: 0.9rem; margin: 0;">
                💡 Add Groq API key in sidebar for AI-powered analysis
            </p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666; font-size: 0.85rem;'>"
    "Powered by XGBoost ML Model"
    "</p>",
    unsafe_allow_html=True
)
