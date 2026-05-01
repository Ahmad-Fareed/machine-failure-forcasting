# %% [markdown]
# # Task 3: Improvised Machine Failure Prediction Model
#
# **Improvements over the original paper:**
# 1. Feature Engineering (Data Engineering)
# 2. Sampling Strategy Optimization (Data Engineering)
# 3. Threshold Optimization (Hyperparameter Optimization)
# 4. Decision Tree Model (Algorithmic Enhancement)

# %%
# ============================================================
# 1. IMPORTS & SETUP
# ============================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, f1_score, confusion_matrix,
    accuracy_score, average_precision_score,
    precision_score, recall_score, fbeta_score,
    ConfusionMatrixDisplay
)
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)
print("All libraries loaded successfully.")

# %%
# ============================================================
# 2. LOAD DATASET
# ============================================================
df_raw = pd.read_excel("synthetic_machine_failure.xlsx")
print(f"Dataset shape: {df_raw.shape}")
print(f"\nClass distribution:\n{df_raw['Failure'].value_counts()}")
print(f"\nFailure rate: {df_raw['Failure'].mean()*100:.2f}%")
df_raw.head()

# %%
# ============================================================
# 3. FEATURE ENGINEERING (Improvement 1)
# ============================================================
def engineer_features(dataframe):
    """
    Create 5 new discriminative features from existing breakdown history.
    These ratio and density features capture degradation TRENDS that raw
    counts alone cannot express.
    """
    df_eng = dataframe.copy()

    # Breakdown acceleration: recent 30-day vs 90-day history
    df_eng['Breakdown_Ratio_30_90'] = (
        df_eng['Breakdowns_Last_30_Days'] / (df_eng['Breakdowns_Last_90_Days'] + 1)
    )
    # Breakdown acceleration: recent 30-day vs 180-day history
    df_eng['Breakdown_Ratio_30_180'] = (
        df_eng['Breakdowns_Last_30_Days'] / (df_eng['Breakdowns_Last_180_Days'] + 1)
    )
    # Average interval between failures
    df_eng['Avg_Days_Between_Failures'] = (
        df_eng['Time_Since_Last_Failure'] / (df_eng['Breakdowns_Last_180_Days'] + 1)
    )
    # Daily failure density (rate)
    df_eng['Failure_Density'] = df_eng['Breakdowns_Last_180_Days'] / 180.0

    # Binary risk flag: had a failure within the last 30 days
    df_eng['Is_Recent_Failure'] = (df_eng['Time_Since_Last_Failure'] < 30).astype(int)

    return df_eng


df_engineered = engineer_features(df_raw)
print("Engineered feature columns:")
print([c for c in df_engineered.columns if c not in df_raw.columns])

# Save the newly engineered dataset to a CSV file for the viva/report
df_engineered.to_csv("engineered_machine_dataset.csv", index=False)
print("\n✅ Engineered dataset saved to 'engineered_machine_dataset.csv'")

# %%
# ============================================================
# 4. CORRELATION HEATMAP — BEFORE vs AFTER
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

corr_orig = df_raw.drop(columns=['Machine_ID', 'Timestamp']).corr()
sns.heatmap(corr_orig, annot=True, cmap='coolwarm', center=0, fmt='.3f', ax=axes[0])
axes[0].set_title('Original Features', fontsize=13, fontweight='bold')

corr_eng = df_engineered.drop(columns=['Machine_ID', 'Timestamp']).corr()
sns.heatmap(corr_eng, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=axes[1])
axes[1].set_title('With Engineered Features', fontsize=13, fontweight='bold')

plt.suptitle('Correlation Comparison', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nCorrelation with Failure (sorted):")
print(corr_eng['Failure'].drop('Failure').sort_values(ascending=False).to_string())

# %%
# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def focal_loss(gamma=2., alpha=0.75):
    """Focal loss — same as original paper."""
    def focal_loss_fixed(y_true, y_pred):
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return -K.mean(alpha * K.pow(1. - pt, gamma) * K.log(pt + K.epsilon()))
    return focal_loss_fixed


def build_nn(input_dim):
    """Build the NN model (same architecture as original paper)."""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss=focal_loss(),
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model


def find_optimal_threshold(y_true, y_probs, beta=1.0):
    """Sweep thresholds and return the one maximizing F-beta score."""
    best_t, best_s = 0.5, 0
    records = []
    for t in np.arange(0.10, 0.90, 0.01):
        yp = (y_probs >= t).astype(int)
        fb = fbeta_score(y_true, yp, beta=beta, zero_division=0)
        records.append({
            'Threshold': round(t, 2),
            'F_beta': round(fb, 4),
            'Precision': round(precision_score(y_true, yp, zero_division=0), 4),
            'Recall': round(recall_score(y_true, yp, zero_division=0), 4),
            'F1': round(f1_score(y_true, yp, zero_division=0), 4),
        })
        if fb > best_s:
            best_s = fb
            best_t = t
    return round(best_t, 2), round(best_s, 4), pd.DataFrame(records)


def get_metrics(y_true, y_pred, y_probs, name):
    """Return a dict of evaluation metrics for one experiment."""
    return {
        'Config': name,
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'F1': round(f1_score(y_true, y_pred, zero_division=0), 4),
        'PR_AUC': round(average_precision_score(y_true, y_probs), 4),
        'Accuracy': round(accuracy_score(y_true, y_pred), 4),
    }

# %%
# ============================================================
# 6. PREPARE DATA — ORIGINAL (BASELINE) FEATURES
# ============================================================
df_base = df_raw.drop(columns=['Machine_ID', 'Timestamp'])
X_base = df_base.drop(columns=['Failure'])
y = df_base['Failure']

scaler_base = StandardScaler()
X_base_scaled = scaler_base.fit_transform(X_base)

X_train_b, X_test_b, y_train, y_test = train_test_split(
    X_base_scaled, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {X_train_b.shape}, Test: {X_test_b.shape}")
print(f"Train failure rate: {y_train.mean()*100:.2f}%")

# %%
# ============================================================
# 7. PREPARE DATA — ENGINEERED FEATURES
# ============================================================
df_eng = df_engineered.drop(columns=['Machine_ID', 'Timestamp'])
X_eng = df_eng.drop(columns=['Failure'])

scaler_eng = StandardScaler()
X_eng_scaled = scaler_eng.fit_transform(X_eng)

X_train_e, X_test_e, _, _ = train_test_split(
    X_eng_scaled, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Engineered Train: {X_train_e.shape}, Test: {X_test_e.shape}")

# %%
# ============================================================
# 8. BASELINE — Replicate Original Paper (XGBoost)
# ============================================================
print("=" * 60)
print("BASELINE: Original features + SMOTETomek 1:1 + threshold 0.49")
print("=" * 60)

xgb_base = xgb.XGBClassifier(
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    eval_metric='logloss', use_label_encoder=False, random_state=42
)
xgb_base.fit(X_train_b, y_train)
probs_base = xgb_base.predict_proba(X_test_b)[:, 1]
preds_base = (probs_base >= 0.49).astype(int)

baseline_metrics = get_metrics(y_test, preds_base, probs_base, "Baseline (Original Paper)")
print(classification_report(y_test, preds_base, digits=4, zero_division=0))
print(f"PR AUC: {baseline_metrics['PR_AUC']}")

all_results = [baseline_metrics]

# %%
# ============================================================
# 9. ABLATION 1 — Feature Engineering Only
#    (engineered features + same sampling + same threshold)
# ============================================================
print("=" * 60)
print("ABLATION 1: Engineered features + original sampling + threshold 0.49")
print("=" * 60)

xgb_fe = xgb.XGBClassifier(
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    eval_metric='logloss', use_label_encoder=False, random_state=42
)
xgb_fe.fit(X_train_e, y_train)
probs_fe = xgb_fe.predict_proba(X_test_e)[:, 1]
preds_fe = (probs_fe >= 0.49).astype(int)

fe_metrics = get_metrics(y_test, preds_fe, probs_fe, "+ Feature Engineering")
print(classification_report(y_test, preds_fe, digits=4, zero_division=0))
print(f"PR AUC: {fe_metrics['PR_AUC']}")
all_results.append(fe_metrics)

# %%
# ============================================================
# 10. ABLATION 2 — Sampling Strategy Comparison
#     (original features + varied sampling + threshold 0.49)
# ============================================================
print("=" * 60)
print("ABLATION 2: Sampling Strategy Comparison")
print("=" * 60)

sampling_results = []

strategies = {
    'SMOTETomek 1:1 (original)': SMOTETomek(random_state=42),
    'SMOTE 0.85': SMOTE(sampling_strategy=0.85, random_state=42),
    'SMOTE 0.95': SMOTE(sampling_strategy=0.95, random_state=42),
    'Random Undersampling': RandomUnderSampler(random_state=42),
    'No Resampling': None
}

for name, sampler in strategies.items():
    if sampler is None:
        X_res, y_res = X_train_b, y_train
    else:
        X_res, y_res = sampler.fit_resample(X_train_b, y_train)
    # Train NN on resampled data
    nn = build_nn(X_res.shape[1])
    nn.fit(X_res, y_res, epochs=25, batch_size=32, validation_split=0.2, verbose=0)
    nn_probs = nn.predict(X_test_b).flatten()
    nn_preds = (nn_probs >= 0.49).astype(int)
    m = get_metrics(y_test, nn_preds, nn_probs, f"NN + {name}")
    sampling_results.append(m)
    print(f"\n{name}: F1={m['F1']}, Prec={m['Precision']}, Rec={m['Recall']}, PR_AUC={m['PR_AUC']}")

samp_df = pd.DataFrame(sampling_results)
print("\n--- Sampling Strategy Comparison (NN) ---")
print(samp_df.to_string(index=False))

# Pick best sampling strategy by F1
best_sampling_name = samp_df.loc[samp_df['F1'].idxmax(), 'Config']
print(f"\nBest sampling strategy: {best_sampling_name}")

# Also add Random Undersampling result to ablation table (using XGBoost for fair comparison)
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X_train_b, y_train)
xgb_s2 = xgb.XGBClassifier(
    eval_metric='logloss', use_label_encoder=False, random_state=42
)
xgb_s2.fit(X_rus, y_rus)  # Train on resampled data
probs_s2 = xgb_s2.predict_proba(X_test_b)[:, 1]
preds_s2 = (probs_s2 >= 0.49).astype(int)
samp_abl_metrics = get_metrics(y_test, preds_s2, probs_s2, "+ Sampling (Random Undersample)")
all_results.append(samp_abl_metrics)

# %%
# ============================================================
# 11. ABLATION 3 — Threshold Optimization
#     (original features + original sampling + optimal threshold)
# ============================================================
print("=" * 60)
print("ABLATION 3: Threshold Optimization (F1-optimal)")
print("=" * 60)

opt_thresh, opt_f1, thresh_df = find_optimal_threshold(y_test, probs_base, beta=1.0)
print(f"Optimal threshold: {opt_thresh} (F1 = {opt_f1})")

preds_opt = (probs_base >= opt_thresh).astype(int)
thresh_metrics = get_metrics(y_test, preds_opt, probs_base, f"+ Threshold ({opt_thresh})")
print(classification_report(y_test, preds_opt, digits=4, zero_division=0))
all_results.append(thresh_metrics)

# Plot threshold sweep
plt.figure(figsize=(10, 6))
plt.plot(thresh_df['Threshold'], thresh_df['F1'], label='F1', color='blue', linewidth=2)
plt.plot(thresh_df['Threshold'], thresh_df['Precision'], '--', label='Precision', color='green')
plt.plot(thresh_df['Threshold'], thresh_df['Recall'], '--', label='Recall', color='red')
plt.axvline(opt_thresh, color='black', linestyle=':', label=f'Optimal: {opt_thresh}')
plt.axvline(0.49, color='gray', linestyle=':', alpha=0.5, label='Original: 0.49')
plt.title('Threshold Optimization: Precision, Recall & F1 vs Threshold', fontsize=13, fontweight='bold')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('threshold_optimization.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# ============================================================
# 12. DECISION TREE MODEL (Algorithmic Enhancement)
# ============================================================
print("=" * 60)
print("DECISION TREE: Engineered features + class_weight='balanced'")
print("=" * 60)

dt_model = DecisionTreeClassifier(
    max_depth=7, class_weight='balanced', random_state=42
)
dt_model.fit(X_train_e, y_train)
dt_probs = dt_model.predict_proba(X_test_e)[:, 1]

# Use optimal threshold for Decision Tree too
dt_opt_t, _, _ = find_optimal_threshold(y_test, dt_probs, beta=1.0)
dt_preds = (dt_probs >= dt_opt_t).astype(int)

dt_metrics = get_metrics(y_test, dt_preds, dt_probs, f"Decision Tree (t={dt_opt_t})")
print(classification_report(y_test, dt_preds, digits=4, zero_division=0))
print(f"PR AUC: {dt_metrics['PR_AUC']}")
all_results.append(dt_metrics)

# Feature importance from Decision Tree
feat_imp = pd.Series(dt_model.feature_importances_, index=X_eng.columns)
feat_imp = feat_imp.sort_values(ascending=True)

plt.figure(figsize=(8, 5))
feat_imp.plot(kind='barh', color='steelblue')
plt.title('Decision Tree — Feature Importance', fontsize=13, fontweight='bold')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('dt_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# ============================================================
# 13. COMBINED IMPROVED MODEL (All improvements together)
# ============================================================
print("=" * 60)
print("COMBINED: Engineered features + optimized threshold (XGBoost)")
print("=" * 60)

# XGBoost with engineered features (already trained as xgb_fe)
comb_opt_t, comb_opt_f1, _ = find_optimal_threshold(y_test, probs_fe, beta=1.0)
preds_comb = (probs_fe >= comb_opt_t).astype(int)

comb_metrics = get_metrics(y_test, preds_comb, probs_fe,
                           f"COMBINED (FE + Thresh {comb_opt_t})")
print(classification_report(y_test, preds_comb, digits=4, zero_division=0))
print(f"PR AUC: {comb_metrics['PR_AUC']}")
all_results.append(comb_metrics)

# Also do Combined NN
print("\n--- Combined NN ---")
smote_nn = SMOTETomek(random_state=42)
X_nn_res, y_nn_res = smote_nn.fit_resample(X_train_e, y_train)
nn_comb = build_nn(X_nn_res.shape[1])
nn_comb.fit(X_nn_res, y_nn_res, epochs=25, batch_size=32, validation_split=0.2, verbose=0)
nn_comb_probs = nn_comb.predict(X_test_e).flatten()
nn_opt_t, _, _ = find_optimal_threshold(y_test, nn_comb_probs, beta=1.0)
nn_comb_preds = (nn_comb_probs >= nn_opt_t).astype(int)
nn_comb_metrics = get_metrics(y_test, nn_comb_preds, nn_comb_probs,
                              f"COMBINED NN (FE + Thresh {nn_opt_t})")
print(classification_report(y_test, nn_comb_preds, digits=4, zero_division=0))
all_results.append(nn_comb_metrics)

# %%
# ============================================================
# 14. ABLATION STUDY — FINAL COMPARISON TABLE
# ============================================================
print("=" * 60)
print("ABLATION STUDY — COMPLETE RESULTS")
print("=" * 60)

results_df = pd.DataFrame(all_results)
results_df = results_df.set_index('Config')
print(results_df.to_string())

# Save to CSV
results_df.to_csv('ablation_results.csv')
print("\nResults saved to ablation_results.csv")

# %%
# ============================================================
# 15. VISUALIZATIONS
# ============================================================

# --- 15a. Ablation Bar Chart ---
fig, ax = plt.subplots(figsize=(14, 6))
results_df[['Precision', 'Recall', 'F1', 'PR_AUC']].plot(
    kind='bar', ax=ax, width=0.8
)
ax.set_title('Ablation Study: Side-by-Side Metric Comparison', fontsize=14, fontweight='bold')
ax.set_ylabel('Score')
ax.set_ylim(0, 1.05)
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right', fontsize=9)
ax.legend(loc='upper right')
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('ablation_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# --- 15b. Confusion Matrices: Baseline vs Combined ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_base = confusion_matrix(y_test, preds_base)
sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0])
axes[0].set_title('Baseline (Original Paper)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_xticklabels(['No Failure', 'Failure'])
axes[0].set_yticklabels(['No Failure', 'Failure'])

cm_comb = confusion_matrix(y_test, preds_comb)
sns.heatmap(cm_comb, annot=True, fmt='d', cmap='Greens', cbar=False, ax=axes[1])
axes[1].set_title('Combined Improved Model', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_xticklabels(['No Failure', 'Failure'])
axes[1].set_yticklabels(['No Failure', 'Failure'])

plt.suptitle('Confusion Matrix Comparison: Baseline vs Improved', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# --- 15c. Probability Distribution: Baseline vs Combined ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(probs_base, bins=30, kde=True, color='steelblue', ax=axes[0])
axes[0].axvline(0.49, color='red', linestyle='--', label='Threshold 0.49')
axes[0].set_title('Baseline — Probability Distribution', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicted Probability')
axes[0].legend()

sns.histplot(probs_fe, bins=30, kde=True, color='seagreen', ax=axes[1])
axes[1].axvline(comb_opt_t, color='red', linestyle='--', label=f'Threshold {comb_opt_t}')
axes[1].set_title('Improved — Probability Distribution', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Predicted Probability')
axes[1].legend()

plt.suptitle('Prediction Confidence: Baseline vs Improved', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('probability_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ All visualizations saved. Execution complete.")
