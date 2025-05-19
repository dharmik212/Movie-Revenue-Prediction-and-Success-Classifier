import warnings, joblib, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             confusion_matrix, accuracy_score, f1_score,
                             precision_score, recall_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from feature_engineering import BoosterWrapper, KerasProbWrapper
from sklearn.calibration import calibration_curve, CalibrationDisplay
import os

warnings.filterwarnings("ignore")

# ─── Load Data ─────────────────────────────────────────────────────
df = pd.read_csv("movie_features.csv")
core_cols = joblib.load("feature_columns_core.pkl")
reg_cols  = joblib.load("feature_columns_reg.pkl")

X_cls = df[core_cols]
y_cls = df["label"]
y_reg = df["log_revenue"]

_, test_idx = train_test_split(df.index, test_size=0.3,
                               stratify=y_cls, random_state=42)
X_test_cls = X_cls.loc[test_idx]
y_cls_test = y_cls.loc[test_idx]
y_reg_test = y_reg.loc[test_idx]

cls_model = load_model("cls_model.keras")
reg_model = joblib.load("reg_model.pkl")   # BoosterWrapper

# ─── Classifier Evaluation ────────────────────────────────────────
cls_prob = cls_model.predict(X_test_cls).ravel()

# Find Best Threshold
best_thresh = 0.5
best_f1 = 0
thresholds = np.linspace(0.2, 0.8, 61)
for thresh in thresholds:
    preds = (cls_prob >= thresh).astype(int)
    f1 = f1_score(y_cls_test, preds)
    if f1 > best_f1:
        best_thresh = thresh
        best_f1 = f1

print(f"\n✅ Best threshold for F1 = {best_thresh:.3f} (F1 = {best_f1:.3f})")

cls_pred = (cls_prob >= best_thresh).astype(int)

# Calculate Metrics
fpr, tpr, _ = roc_curve(y_cls_test, cls_prob)
accuracy = accuracy_score(y_cls_test, cls_pred)
precision = precision_score(y_cls_test, cls_pred)
recall = recall_score(y_cls_test, cls_pred)
auc_score = auc(fpr, tpr)

print(f"\n📊 Classifier Performance:")
print(f"Accuracy = {accuracy:.3f}")
print(f"Precision = {precision:.3f}")
print(f"Recall = {recall:.3f}")
print(f"F1 Score = {best_f1:.3f}")
print(f"ROC-AUC = {auc_score:.3f}")

# ─── Regressor Evaluation ─────────────────────────────────────────
X_test_reg = X_test_cls.copy()
X_test_reg["success_prob"] = cls_prob
log_rev_pred = reg_model.predict(X_test_reg[reg_cols])

rev_pred = np.expm1(log_rev_pred)
rev_true = np.expm1(y_reg_test)

mae_dollar = mean_absolute_error(rev_true, rev_pred)
mae_log = mean_absolute_error(y_reg_test, log_rev_pred)

print(f"\n📈 Regressor Performance:")
print(f"XGB MAE (in dollars) = ${mae_dollar:,.0f}")
print(f"XGB MAE (in log scale) = {mae_log:.3f}")
print(f"XGB R²   = {r2_score(rev_true,rev_pred):.3f}")

# ─── Create Folder for Saving Plots ───────────────────────────────
os.makedirs("evaluation_plots", exist_ok=True)

# ─── ROC Curve ────────────────────────────────────────────────────
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
plt.plot([0,1], [0,1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - ANN Classifier')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig("evaluation_plots/roc_curve.png")
plt.show()

# ─── Precision-Recall Curve ───────────────────────────────────────
precision_curve, recall_curve, _ = precision_recall_curve(y_cls_test, cls_prob)

plt.figure(figsize=(6,5))
plt.plot(recall_curve, precision_curve, color='green', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - ANN Classifier')
plt.grid(True)
plt.tight_layout()
plt.savefig("evaluation_plots/pr_curve.png")
plt.show()

# ─── Confusion Matrix ─────────────────────────────────────────────
cm = confusion_matrix(y_cls_test, cls_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(5,4))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues',
            xticklabels=['Flop', 'Success'],
            yticklabels=['Flop', 'Success'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix - ANN Classifier')
plt.tight_layout()
plt.savefig("evaluation_plots/confusion_matrix.png")
plt.show()

# ─── Quick Statistics on All Training Data ────────────────────────
print("\nANN prediction on all training data:")
preds = cls_model.predict(X_cls)

raw_te_p   = cls_model.predict(X_test_cls, verbose=0).ravel()
calib_te_p = joblib.load("prob_calibrator.pkl").predict(raw_te_p)

frac_raw,  mean_raw  = calibration_curve(y_cls_test, raw_te_p,   n_bins=10, strategy="quantile")
frac_cal,  mean_cal  = calibration_curve(y_cls_test, calib_te_p, n_bins=10, strategy="quantile")

plt.figure(figsize=(6,5))
plt.plot(mean_raw,  frac_raw,  "--o", label="raw")
plt.plot(mean_cal,  frac_cal,  "-o",  label="calib.")
plt.plot([0,1], [0,1], ":", color="gray")
plt.xlabel("Predicted probability"); plt.ylabel("Empirical success rate")
plt.title("Reliability curve")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig("evaluation_plots/calibration_curve.png")
plt.show()

print(f"Predicted probability mean: {np.mean(preds):.4f}")
print(f"Predicted probability std deviation: {np.std(preds):.4f}")
