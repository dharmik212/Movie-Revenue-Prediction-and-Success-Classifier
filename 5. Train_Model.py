import warnings, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import class_weight
from tensorflow.keras import layers, callbacks, models, regularizers
from tensorflow.keras.models import clone_model
import xgboost as xgb
from feature_engineering import KerasProbWrapper
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

# ─── 1. load engineered features ─────────────────────────────────
df = pd.read_csv("movie_features.csv")
core_cols = joblib.load("feature_columns_core.pkl")

X      = df.drop(columns=["id", "label", "log_revenue"])
X_cls  = X[core_cols]
y_cls  = df["label"]
y_reg  = df["log_revenue"]

# ─── 1.1 Downscale TF-IDF features ───────────────────────────────
tfidf_cols = [c for c in core_cols if c.startswith("tfidf_")]
X_cls[tfidf_cols] *= 0.003 

# ─── 2. train/val/test split for ANN baseline fit ───────────────
X_tr, X_tmp, y_tr, y_tmp = train_test_split(
    X_cls, y_cls, test_size=0.30, stratify=y_cls, random_state=42
)
X_val, X_te, y_val, y_te = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42
)
# Split X_val further: 70 % val, 30 % calib
X_val, X_cal, y_val, y_cal = train_test_split(
    X_val, y_val,
    test_size=0.30, stratify=y_val, random_state=42
)

# ─── 3. ANN classifier ──────────────────────────────────────────
def make_classifier(input_dim: int):
    m = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(), layers.Dropout(0.2),
        layers.Dense(32, activation="relu"), layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=["accuracy"])
    return m

cls_model = make_classifier(len(core_cols))
cw = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_tr), y=y_tr)
cls_model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=64,
    class_weight=dict(enumerate(cw)),
    callbacks=[callbacks.EarlyStopping(monitor="val_loss",
              patience=8, restore_best_weights=True)],
    verbose=2
)
cls_model.save("cls_model.keras")

# ─── Probability calibration ─────────────────────────────────────
cal_base  = KerasProbWrapper(cls_model)
raw_cal_p = cal_base.predict_proba(X_cal)[:, 1]

calib_model = IsotonicRegression(out_of_bounds='clip').fit(raw_cal_p, y_cal)

joblib.dump(calib_model, "prob_calibrator.pkl")
print("✓ prob_calibrator.pkl saved")

# ─── 4. OOF success-prob feature ────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_prob = np.zeros(len(X_cls))

for tr_idx, v_idx in skf.split(X_cls, y_cls):
    Xm_tr, Xm_val = X_cls.iloc[tr_idx], X_cls.iloc[v_idx]
    yc_tr         = y_cls.iloc[tr_idx]

    m = make_classifier(len(core_cols))
    m.fit(
        Xm_tr, yc_tr,
        epochs=40, batch_size=64, verbose=0,
        class_weight=dict(enumerate(cw)),
        validation_split=0.1,
        callbacks=[callbacks.EarlyStopping(monitor="val_loss",
                                           patience=5,
                                           restore_best_weights=True)]
    )
    oof_prob[v_idx] = m.predict(Xm_val).ravel()

X_reg_base = X.copy()
X_reg_base["success_prob"] = oof_prob
joblib.dump(X_reg_base.columns.tolist(), "feature_columns_reg.pkl")

# ─── 5. XGB: CV to find best_iteration ──────────────────────────
dtrain = xgb.DMatrix(X_reg_base, label=y_reg)

params = {
    "objective": "reg:squarederror",
    "eta": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
}

cv = xgb.cv(
    params,
    dtrain,
    num_boost_round=2000,
    nfold=5,
    metrics="rmse",
    early_stopping_rounds=50,
    verbose_eval=False
)
best_round = len(cv)
print(f"✔ best_iteration from cv = {best_round}")

booster = xgb.train(params, dtrain, num_boost_round=best_round)

# thin wrapper so `.predict(df)` works like sklearn
class BoosterWrapper:
    def __init__(self, booster, cols):
        self.booster = booster
        self.cols = cols
    def predict(self, X):
        return self.booster.predict(xgb.DMatrix(X[self.cols]))

reg_model = BoosterWrapper(booster, X_reg_base.columns.tolist())
joblib.dump(reg_model, "reg_model.pkl")

print("\n✅ Training complete:")
print("   • cls_model.keras")
print("   • reg_model.pkl")
print("   • feature_columns_core.pkl / feature_columns_reg.pkl")
