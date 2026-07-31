import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# --- Load engineered features ---
df = pd.read_csv("data/features_engineered.csv", parse_dates=["date"])

exclude_cols = ["date", "player", "fatigue", "fatigue_class"]
feature_cols = [c for c in df.columns if c not in exclude_cols]

X = df[feature_cols]
y = df["fatigue_class"]
groups = df["player"]

label_map = {"Low": 0, "Medium": 1, "High": 2}
inv_label_map = {v: k for k, v in label_map.items()}
y_encoded = y.map(label_map)

# --- Use one GroupKFold split to get out-of-fold predictions for ALL rows ---
gkf = GroupKFold(n_splits=5)
all_preds = np.zeros(len(y_encoded), dtype=int)

best_params = {"max_depth": 20, "min_samples_leaf": 3, "n_estimators": 200}

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y_encoded, groups=groups)):
    model = RandomForestClassifier(class_weight="balanced", random_state=42, **best_params)
    model.fit(X.iloc[train_idx], y_encoded.iloc[train_idx])
    all_preds[test_idx] = model.predict(X.iloc[test_idx])

# --- Classification report across ALL out-of-fold predictions ---
print("=== Classification Report (out-of-fold, all 5 folds combined) ===")
print(classification_report(y_encoded, all_preds, target_names=["Low", "Medium", "High"]))

print("=== Confusion Matrix ===")
cm = confusion_matrix(y_encoded, all_preds)
print(cm)

# --- Train FINAL model on ALL data (for deployment in the dashboard) ---
final_model = RandomForestClassifier(class_weight="balanced", random_state=42, **best_params)
final_model.fit(X, y_encoded)

# Save model + label map for dashboard use later
joblib.dump(final_model, "models/random_forest_final.joblib")
joblib.dump(label_map, "models/label_map.joblib")
joblib.dump(feature_cols, "models/feature_cols.joblib")
print("\nSaved final model to models/random_forest_final.joblib")