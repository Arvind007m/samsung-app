import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor, XGBClassifier
import joblib
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load data
data_path = os.path.join(SCRIPT_DIR, "benchdata.csv")
df = pd.read_csv(data_path)
print("Data loaded successfully!")

# Drop NA values
df = df.dropna().reset_index(drop=True)


def clean_numeric(col):
    return (
        col.astype(str)
           .str.replace(r"[^\d\.]", "", regex=True)
           .replace("", np.nan)
           .astype(float)
    )


numeric_cols = [
    'TPS', 'Threadpool', 'Avg Response Time(ms)', 'P95', 'P99',
    'Throughput(rps)', 'Error Rate', 'CPU Usage', 'Memory',
    'Latency(ms)', 'Request Timeouts'
]

for c in numeric_cols:
    df[c] = clean_numeric(df[c])

# Label encoding
le_service = LabelEncoder()
le_framework = LabelEncoder()
df['Service_enc'] = le_service.fit_transform(df['Service'])
df['FrameWork_enc'] = le_framework.fit_transform(df['FrameWork'])

# Drop rows with NA in target columns
df = df.dropna(subset=[
    'Avg Response Time(ms)', 'P95', 'P99', 'Throughput(rps)',
    'Error Rate', 'CPU Usage', 'Memory', 'Latency(ms)', 'Request Timeouts'
]).reset_index(drop=True)

# Features and targets for regression model
X = df[['Service_enc', 'FrameWork_enc', 'TPS', 'Threadpool']]
Y = df[[
    'Avg Response Time(ms)', 'P95', 'P99', 'Throughput(rps)',
    'Error Rate', 'CPU Usage', 'Memory', 'Latency(ms)', 'Request Timeouts'
]]

# Train-test split for regression
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train regression model
print("Training regression model...")
xgb_base = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)
model = MultiOutputRegressor(xgb_base)
model.fit(X_train_scaled, Y_train)
print("Regression model trained!")

# Classification model for framework recommendation
clf_le = LabelEncoder()
df['framework_label'] = clf_le.fit_transform(df['FrameWork'])

X_clf = df[[
    'TPS', 'Threadpool', 'CPU Usage', 'Memory',
    'Latency(ms)', 'Throughput(rps)', 'Error Rate'
]]
y_clf = df['framework_label']

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# Train classification model
print("Training classification model...")
xgb_clf = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)
xgb_clf.fit(X_train_clf, y_train_clf)
print("Classification model trained!")

# Save all models
print("Saving models...")
joblib.dump(model, os.path.join(SCRIPT_DIR, "regression_model.pkl"))
joblib.dump(xgb_clf, os.path.join(SCRIPT_DIR, "classifier_model.pkl"))
joblib.dump(scaler, os.path.join(SCRIPT_DIR, "scaler.pkl"))
joblib.dump(le_service, os.path.join(SCRIPT_DIR, "le_service.pkl"))
joblib.dump(le_framework, os.path.join(SCRIPT_DIR, "le_framework.pkl"))
joblib.dump(clf_le, os.path.join(SCRIPT_DIR, "clf_le.pkl"))

print("All models saved successfully!")
