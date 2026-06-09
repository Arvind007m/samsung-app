
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# STEP 1: LOAD ORIGINAL DATASET
# ==============================================================================
print("=" * 70)
print("STEP 1: LOADING ORIGINAL DATASET")
print("=" * 70)

data_path = os.path.join(SCRIPT_DIR, "benchdata_cleaned.csv")
df_original = pd.read_csv(data_path)

# Drop any completely empty rows (trailing blank lines)
df_original = df_original.dropna(how='all')

# Rename Memory(Mi) -> Memory for consistency
if 'Memory(Mi)' in df_original.columns:
    df_original = df_original.rename(columns={'Memory(Mi)': 'Memory'})

print(f"  Original dataset: {len(df_original)} rows")
print(f"  Columns: {list(df_original.columns)}")
print(f"  Frameworks: {df_original['FrameWork'].value_counts().to_dict()}")

# ==============================================================================
# STEP 2: TRAIN/TEST SPLIT (BEFORE ANY INTERPOLATION)
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 2: TRAIN/TEST SPLIT (90/10, stratified by FrameWork)")
print("=" * 70)

df_train_real, df_test = train_test_split(
    df_original,
    test_size=0.1,
    random_state=42,
    stratify=df_original['FrameWork']
)

# Reset indices
df_train_real = df_train_real.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

print(f"  Training set (REAL): {len(df_train_real)} rows")
print(f"  Test set (FROZEN):   {len(df_test)} rows")

# Save frozen test set for reproducibility
df_test.to_csv(os.path.join(SCRIPT_DIR, "frozen_test_set.csv"), index=False)
print("  Saved: frozen_test_set.csv")

# ==============================================================================
# STEP 3: INTERPOLATION (TRAINING SET ONLY)
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 3: INTERPOLATING TRAINING SET (factor=10, per-framework)")
print("=" * 70)

FACTOR = 10

# All numerical columns to interpolate
numerical_cols = [
    'TPS', 'Threadpool', 'Avg Response Time(ms)', 'P95', 'P99',
    'Throughput(rps)', 'Error Rate', 'CPU Usage', 'Memory',
    'Latency(ms)', 'Request Timeouts'
]


def interpolate_framework(df_fw, factor=10):
    """
    For every two consecutive rows in df_fw, generate intermediate points.
    Formula: X_new = X_1 * (1 - alpha) + X_2 * alpha
    where alpha = j / factor, for j in 1..factor-1
    """
    df_fw = df_fw.sort_values(['TPS', 'Threadpool']).reset_index(drop=True)
    interpolated_rows = []
    framework_name = df_fw['FrameWork'].iloc[0]

    for i in range(len(df_fw) - 1):
        row_1 = df_fw.iloc[i]
        row_2 = df_fw.iloc[i + 1]

        for j in range(1, factor):
            alpha = j / factor
            new_row = {'FrameWork': framework_name}

            for col in numerical_cols:
                val_1 = row_1[col]
                val_2 = row_2[col]
                new_row[col] = val_1 * (1 - alpha) + val_2 * alpha

            interpolated_rows.append(new_row)

    return pd.DataFrame(interpolated_rows)


df_train_boot = df_train_real[df_train_real['FrameWork'] == 'boot']
df_train_webflux = df_train_real[df_train_real['FrameWork'] == 'webflux']

interp_boot = interpolate_framework(df_train_boot, FACTOR)
interp_webflux = interpolate_framework(df_train_webflux, FACTOR)

df_train_interpolated = pd.concat(
    [df_train_real, interp_boot, interp_webflux],
    ignore_index=True
)

print(f"  Total interpolated training set: {len(df_train_interpolated)} rows")
df_train_interpolated.to_csv(os.path.join(SCRIPT_DIR, "interpolated_train_set.csv"), index=False)
print("  Saved: interpolated_train_set.csv")

# ==============================================================================
# STEP 4 & 5: FEATURE ENGINEERING + MODEL TRAINING
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 4 & 5: FEATURE ENGINEERING + OPTIMIZED XGBOOST TRAINING")
print("=" * 70)

feature_columns = [
    'FrameWork_enc', 'TPS', 'Threadpool', 'TPS_Thread_ratio',
    'TPS_log', 'Thread_log', 'TPS_Thread_product', 'TPS_sq', 'Thread_sq'
]

target_columns = [
    'Avg Response Time(ms)', 'P95', 'P99', 'Throughput(rps)',
    'Error Rate', 'CPU Usage', 'Memory', 'Latency(ms)', 'Request Timeouts'
]


def add_features(df, le_framework):
    df = df.copy()
    df['FrameWork_enc'] = le_framework.transform(df['FrameWork'])
    df['TPS_Thread_ratio'] = df['TPS'] / (df['Threadpool'] + 1)
    df['TPS_log'] = np.log1p(df['TPS'])
    df['Thread_log'] = np.log1p(df['Threadpool'])
    df['TPS_Thread_product'] = df['TPS'] * df['Threadpool']
    df['TPS_sq'] = df['TPS'] ** 2
    df['Thread_sq'] = df['Threadpool'] ** 2
    return df


le_framework = LabelEncoder()
le_framework.fit(df_original['FrameWork'])

df_train_real_fe = add_features(df_train_real, le_framework)
df_train_interp_fe = add_features(df_train_interpolated, le_framework)
df_test_fe = add_features(df_test, le_framework)

X_train_real = df_train_real_fe[feature_columns]
Y_train_real = df_train_real_fe[target_columns]

X_train_interp = df_train_interp_fe[feature_columns]
Y_train_interp = df_train_interp_fe[target_columns]

X_test = df_test_fe[feature_columns]
Y_test = df_test_fe[target_columns]

scaler_a = StandardScaler()
X_train_real_scaled = scaler_a.fit_transform(X_train_real)
X_test_scaled_a = scaler_a.transform(X_test)

scaler_b = StandardScaler()
X_train_interp_scaled = scaler_b.fit_transform(X_train_interp)
X_test_scaled_b = scaler_b.transform(X_test)

# --- Optimized XGBoost parameters from grid search ---
xgb_params = dict(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)

# Train Model A
print("\n  Training Model A (Optimized XGBoost on Real Only)...")
model_a = MultiOutputRegressor(XGBRegressor(**xgb_params))
model_a.fit(X_train_real_scaled, Y_train_real)

# Train Model B
print("  Training Model B (Optimized XGBoost on 3000-row Interpolated)...")
model_b = MultiOutputRegressor(XGBRegressor(**xgb_params))
model_b.fit(X_train_interp_scaled, Y_train_interp)

# ==============================================================================
# STEP 6: EVALUATE BOTH ON FROZEN TEST SET
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 6: EVALUATION ON FROZEN REAL TEST SET ({} rows)".format(len(df_test)))
print("=" * 70)

Y_pred_a = model_a.predict(X_test_scaled_a)
Y_pred_b = model_b.predict(X_test_scaled_b)

# ==========================================================
# SAVE PREDICTIONS FOR PLOTS
# ==========================================================

pred_df = pd.DataFrame()

for i, col in enumerate(target_columns):
    pred_df[f'Actual_{col}'] = Y_test.iloc[:, i].values
    pred_df[f'PredA_{col}'] = Y_pred_a[:, i]
    pred_df[f'PredB_{col}'] = Y_pred_b[:, i]

pred_path = os.path.join(SCRIPT_DIR, "prediction_results.csv")
pred_df.to_csv(pred_path, index=False)

print(f"  Saved: prediction_results.csv")

r2_a_overall = r2_score(Y_test, Y_pred_a)
r2_b_overall = r2_score(Y_test, Y_pred_b)

print(f"\n  {'Metric':<28s} {'Model A (Real)':<18s} {'Model B (Interp)':<18s} {'Winner'}")
print(f"  {'-'*28} {'-'*18} {'-'*18} {'-'*10}")

results = []
for i, col in enumerate(target_columns):
    r2_a = r2_score(Y_test.iloc[:, i], Y_pred_a[:, i])
    r2_b = r2_score(Y_test.iloc[:, i], Y_pred_b[:, i])
    mae_a = mean_absolute_error(Y_test.iloc[:, i], Y_pred_a[:, i])
    mae_b = mean_absolute_error(Y_test.iloc[:, i], Y_pred_b[:, i])
    rmse_a = np.sqrt(mean_squared_error(Y_test.iloc[:, i], Y_pred_a[:, i]))
    rmse_b = np.sqrt(mean_squared_error(Y_test.iloc[:, i], Y_pred_b[:, i]))

    winner = "A (Real)" if r2_a >= r2_b else "B (Interp)"

    print(f"  {col:<28s} {r2_a*100:>7.2f}%          {r2_b*100:>7.2f}%          {winner}")

    results.append({
        'Metric': col,
        'R2_ModelA_Real': round(r2_a * 100, 2),
        'R2_ModelB_Interp': round(r2_b * 100, 2),
        'MAE_ModelA_Real': round(mae_a, 4),
        'MAE_ModelB_Interp': round(mae_b, 4),
        'RMSE_ModelA_Real': round(rmse_a, 4),
        'RMSE_ModelB_Interp': round(rmse_b, 4),
        'Winner_R2': winner
    })

print(f"\n  {'OVERALL R2':<28s} {r2_a_overall*100:>7.2f}%          {r2_b_overall*100:>7.2f}%          {'A (Real)' if r2_a_overall >= r2_b_overall else 'B (Interp)'}")

# Count wins
a_wins = sum(1 for r in results if r['Winner_R2'] == 'A (Real)')
b_wins = sum(1 for r in results if r['Winner_R2'] == 'B (Interp)')

results.append({
    'Metric': 'OVERALL',
    'R2_ModelA_Real': round(r2_a_overall * 100, 2),
    'R2_ModelB_Interp': round(r2_b_overall * 100, 2),
    'MAE_ModelA_Real': '',
    'MAE_ModelB_Interp': '',
    'RMSE_ModelA_Real': '',
    'RMSE_ModelB_Interp': '',
    'Winner_R2': 'A (Real)' if r2_a_overall >= r2_b_overall else 'B (Interp)'
})

df_results = pd.DataFrame(results)
results_path = os.path.join(SCRIPT_DIR, "comparison_results.csv")
df_results.to_csv(results_path, index=False)
print(f"\n  Saved: comparison_results.csv")

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model_b.estimators_[0].feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

importance_df.to_csv(
    os.path.join(SCRIPT_DIR, "feature_importance.csv"),
    index=False
)

print("  Saved: feature_importance.csv")

# ==============================================================================
# STEP 7: SAVE MODELS
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 7: SAVING MODELS")
print("=" * 70)

# Save Model B (interpolated) as production model
models_dir = os.path.join(SCRIPT_DIR, "models_v2")
os.makedirs(models_dir, exist_ok=True)

joblib.dump(model_b, os.path.join(models_dir, "model.pkl"))
joblib.dump(scaler_b, os.path.join(models_dir, "scaler.pkl"))
joblib.dump(le_framework, os.path.join(models_dir, "le_framework.pkl"))

model_b_size = os.path.getsize(os.path.join(models_dir, "model.pkl")) / (1024 * 1024)
print(f"  Model B (production) saved to: {models_dir}")
print(f"    - model.pkl ({model_b_size:.2f} MB)")

# Save Model A (real-only) for reference
models_a_dir = os.path.join(models_dir, "model_a_real")
os.makedirs(models_a_dir, exist_ok=True)

joblib.dump(model_a, os.path.join(models_a_dir, "model.pkl"))
joblib.dump(scaler_a, os.path.join(models_a_dir, "scaler.pkl"))
joblib.dump(le_framework, os.path.join(models_a_dir, "le_framework.pkl"))

model_a_size = os.path.getsize(os.path.join(models_a_dir, "model.pkl")) / (1024 * 1024)
print(f"  Model A (reference) saved to: {models_a_dir}")
print(f"    - model.pkl ({model_a_size:.2f} MB)")

# ==============================================================================
# STEP 8: SUMMARY
# ==============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Original dataset:          {len(df_original)} real rows")
print(f"  Training set (real):       {len(df_train_real)} rows")
print(f"  Training set (interpolated): {len(df_train_interpolated)} rows")
print(f"  Test set (frozen, real):   {len(df_test)} rows")
print(f"  Interpolation factor:      {FACTOR}")
print(f"  Model A (real) overall R2: {r2_a_overall*100:.2f}%")
print(f"  Model B (interp) overall R2: {r2_b_overall*100:.2f}%")
print(f"  Model A metric wins:       {a_wins}/9")
print(f"  Model B metric wins:       {b_wins}/9")
print("=" * 70)
print("DONE")
print("=" * 70)
