# ── MENTAL HEALTH RISK ANALYSIS — Model Training Pipeline ──────────────────

import os, warnings
warnings.filterwarnings("ignore")

import pandas  as pd
import numpy   as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection    import train_test_split, cross_val_score
from sklearn.preprocessing      import LabelEncoder
from sklearn.base               import clone
from sklearn.metrics            import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost                    import XGBClassifier
from encoders                   import OrdinalMapEncoder

# ── 1. LOAD DATA ─────────────────────────────────────────────────────────────
CSV_FILE = "Mental Health & Social Media Usage Survey (Responses) - Form Responses 1 (1)_augmented_new.csv"
df = pd.read_csv(CSV_FILE)
print(f"[INFO] Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")

# ── 2. RENAME COLUMNS ────────────────────────────────────────────────────────
RENAME_MAP = {
    df.columns[0]:  "timestamp",
    df.columns[1]:  "age",
    df.columns[2]:  "occupation",
    df.columns[3]:  "gender",
    df.columns[4]:  "daily_sm_hours",
    df.columns[5]:  "platforms",
    df.columns[6]:  "peak_usage_time",
    df.columns[7]:  "before_sleep_usage",
    df.columns[8]:  "purpose",
    df.columns[9]:  "sleep_hours",
    df.columns[10]: "stress_freq_raw",
    df.columns[11]: "mood_raw",
    df.columns[12]: "stress_experience",
    df.columns[13]: "avg_mood",
    df.columns[14]: "productivity_impact",
    df.columns[15]: "comparison_frequency",
    df.columns[16]: "mental_health_score",
    df.columns[17]: "mental_health_label",
}
df.rename(columns=RENAME_MAP, inplace=True)

# ── 3. FEATURE SELECTION ─────────────────────────────────────────────────────
BASE_FEATURES = [
    "age", "occupation", "gender", "daily_sm_hours",
    "peak_usage_time", "before_sleep_usage", "purpose",
    "sleep_hours", "stress_experience", "avg_mood",
    "productivity_impact", "comparison_frequency",
]
TARGET = "mental_health"

# ── 4. BUILD TARGET (0–4) ────────────────────────────────────────────────────
label_to_class = {
    "Very Poor (Highly Negative)":           0,
    "Poor (Mostly Negative)":                1,
    "Moderate (Neutral/Mixed)":              2,
    "Good (Mostly Positive)":               3,
    "Excellent/Very Good (Highly Positive)": 4,
}
df[TARGET] = df["mental_health_label"].map(label_to_class)

# ── 5. HANDLE MISSING VALUES ─────────────────────────────────────────────────
df.dropna(subset=[TARGET], inplace=True)

for col in BASE_FEATURES:
    if df[col].isna().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

print(f"[INFO] Dataset size after cleaning: {df.shape[0]} rows")

# ── 6. WORKING DATAFRAME ─────────────────────────────────────────────────────
data = df[BASE_FEATURES + [TARGET]].copy()
data[TARGET] = data[TARGET].astype(int)

# ── 7. ENCODING ──────────────────────────────────────────────────────────────
label_encoders = {}

ORDINAL_MAPS = {
    "stress_experience":    {"Never": 0, "Sometimes": 1, "Often": 2, "Always": 3},
    "comparison_frequency": {"Never": 0, "Sometimes": 1, "Often": 2, "Always": 3},
    "avg_mood":             {"Sad": 0, "Neutral": 1, "Happy": 2},
    "before_sleep_usage":   {"No": 0, "Yes": 1},
    "productivity_impact":  {
        "No, it has no major impact": 0,
        "Maybe, sometimes it helps, sometimes it hinders": 1,
        "Yes, significantly reduces it": 2,
    },
}

categorical_cols = [
    "age", "occupation", "gender", "daily_sm_hours",
    "peak_usage_time", "before_sleep_usage", "purpose",
    "sleep_hours", "stress_experience", "avg_mood",
    "productivity_impact", "comparison_frequency",
]

for col in categorical_cols:
    if col in ORDINAL_MAPS:
        data[col] = data[col].astype(str).map(ORDINAL_MAPS[col]).fillna(0).astype(int)
        label_encoders[col] = OrdinalMapEncoder(ORDINAL_MAPS[col])
    else:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

# ── 8. INTERACTION FEATURES ──────────────────────────────────────────────────
data["stress_x_mood"] = data["stress_experience"] * data["avg_mood"]

df_sleep_str = df["sleep_hours"].astype(str)
data["sleep_ord"] = df_sleep_str.map(
    lambda s: 0 if "Less" in s else (1 if "5" in s[:3] else (2 if "7" in s[:3] else 3))
).values
data["sleep_x_stress"] = data["sleep_ord"] * (3 - data["stress_experience"])
data.drop(columns=["sleep_ord"], inplace=True)

FEATURES = BASE_FEATURES + ["stress_x_mood", "sleep_x_stress"]
print(f"[INFO] Feature matrix: {data[FEATURES].shape}")

# ── 9. TRAIN / TEST SPLIT ────────────────────────────────────────────────────
X = data[FEATURES]
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"[INFO] Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

# ── 10. CLASS WEIGHTS ────────────────────────────────────────────────────────
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

# ── 11. MODEL TRAINING ───────────────────────────────────────────────────────
model = XGBClassifier(
    n_estimators          = 600,
    max_depth             = 7,
    learning_rate         = 0.05,
    subsample             = 0.90,
    colsample_bytree      = 0.90,
    min_child_weight      = 1,
    gamma                 = 0,
    reg_alpha             = 0.05,
    reg_lambda            = 0.5,
    use_label_encoder     = False,
    eval_metric           = "mlogloss",
    early_stopping_rounds = 30,
    random_state          = 42,
    n_jobs                = -1,
)

model.fit(
    X_train, y_train,
    sample_weight = sample_weights,
    eval_set      = [(X_test, y_test)],
    verbose       = False,
)
print(f"[INFO] Training complete. Best iteration: {model.best_iteration}")

model_cv = clone(model)
model_cv.set_params(early_stopping_rounds=None, n_estimators=model.best_iteration or 300)
cv_scores = cross_val_score(model_cv, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
print(f"[INFO] 5-fold CV: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

# ── 12. EVALUATION ───────────────────────────────────────────────────────────
y_pred     = model.predict(X_test)
acc        = accuracy_score(y_test, y_pred)
CLASS_NAMES = ["Very Bad", "Bad", "Normal", "Good", "Very Healthy"]

print("\n" + "="*60)
print(f"  ACCURACY ON TEST SET: {acc*100:.2f}%")
print("="*60)
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# ── 13. CONFUSION MATRIX ─────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# ── 14. FEATURE IMPORTANCE ───────────────────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=FEATURES)
importances_sorted = importances.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importances_sorted)))
bars = ax.barh(importances_sorted.index, importances_sorted.values, color=colors)
ax.set_xlabel("Importance Score")
ax.set_title("XGBoost Feature Importance")
ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()

for feat, score in importances.sort_values(ascending=False).items():
    print(f"  {feat:<25s} => {score:.4f}")

# ── 15. SAVE MODEL ───────────────────────────────────────────────────────────
ARTEFACTS = {
    "model":          model,
    "label_encoders": label_encoders,
    "features":       FEATURES,
    "class_names":    CLASS_NAMES,
}
joblib.dump(ARTEFACTS, "mental_health_model.pkl")
print("\n[INFO] Saved -> mental_health_model.pkl")
print("[DONE] Run: python -m streamlit run app.py")
