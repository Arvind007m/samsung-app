import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print("="*60)
print("MICROSERVICE PERFORMANCE MODEL TRAINING v2")
print("="*60)

augmented_path = os.path.join(SCRIPT_DIR, "augmented_dataset_large.csv")
if os.path.exists(augmented_path):
    df = pd.read_csv(augmented_path)
    print(f"\nLoaded augmented dataset: {len(df)} rows")
else:
    data_path = os.path.join(SCRIPT_DIR, "final_expanded_dataset (1).csv")
    df = pd.read_csv(data_path)
    df = df.drop(columns=['Service'])
    df = df.dropna()
    print(f"\nLoaded original dataset: {len(df)} rows")

le_framework = LabelEncoder()
df['FrameWork_enc'] = le_framework.fit_transform(df['FrameWork'])
print(f"Frameworks: {list(le_framework.classes_)}")

df['TPS_Thread_ratio'] = df['TPS'] / (df['Threadpool'] + 1)
df['TPS_log'] = np.log1p(df['TPS'])
df['Thread_log'] = np.log1p(df['Threadpool'])
df['TPS_Thread_product'] = df['TPS'] * df['Threadpool']
df['TPS_sq'] = df['TPS'] ** 2
df['Thread_sq'] = df['Threadpool'] ** 2
print("Feature engineering completed")

print("\n" + "="*60)
print("Training XGBoost Regression Model...")
print("="*60)

feature_columns = ['FrameWork_enc', 'TPS', 'Threadpool', 'TPS_Thread_ratio', 
                   'TPS_log', 'Thread_log', 'TPS_Thread_product', 'TPS_sq', 'Thread_sq']

target_columns = [
    'Avg Response Time(ms)', 'P95', 'P99', 'Throughput(rps)',
    'Error Rate', 'CPU Usage', 'Memory', 'Latency(ms)', 'Request Timeouts'
]

X_reg = df[feature_columns]
Y_reg = df[target_columns]

X_train, X_test, Y_train, Y_test = train_test_split(
    X_reg, Y_reg, test_size=0.1, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("  Training XGBoost model...")
xgb_base = XGBRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)
model = MultiOutputRegressor(xgb_base)
model.fit(X_train_scaled, Y_train)

Y_pred = model.predict(X_test_scaled)
r2 = r2_score(Y_test, Y_pred)

print(f"\nOverall R2 Score: {r2*100:.2f}%")
print("\nPer-metric R2 scores:")
excellent_count = 0
good_count = 0
for i, col in enumerate(target_columns):
    r2_m = r2_score(Y_test.iloc[:, i], Y_pred[:, i])
    if r2_m >= 0.8:
        status = "EXCELLENT"
        excellent_count += 1
    elif r2_m >= 0.7:
        status = "GOOD"
        good_count += 1
    elif r2_m >= 0.5:
        status = "OK"
    else:
        status = "WEAK"
    print(f"  {col:25s}: {r2_m*100:.2f}% - {status}")

print(f"\nExcellent metrics (80%+): {excellent_count}/9")
print(f"Good or better (70%+): {excellent_count + good_count}/9")

print("\n" + "="*60)
print("Saving Models...")
print("="*60)

models_dir = os.path.join(SCRIPT_DIR, "models_v2")
os.makedirs(models_dir, exist_ok=True)

joblib.dump(model, os.path.join(models_dir, "model.pkl"))
joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
joblib.dump(le_framework, os.path.join(models_dir, "le_framework.pkl"))

model_size = os.path.getsize(os.path.join(models_dir, "model.pkl")) / (1024*1024)

print(f"Models saved to: {models_dir}")
print(f"  - model.pkl ({model_size:.1f} MB)")
print("  - scaler.pkl")
print("  - le_framework.pkl")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"Regression R2: {r2*100:.2f}%")
print(f"Excellent metrics: {excellent_count}/9")
print(f"Good or better: {excellent_count + good_count}/9")
print(f"Model size: {model_size:.1f} MB")
