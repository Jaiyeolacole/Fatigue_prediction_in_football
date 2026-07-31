import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
import joblib

df = pd.read_csv("data/features_engineered.csv", parse_dates=["date"])

# Same-day features ONLY (no lag/rolling) — what a coach can type in on the spot
same_day_features = [
    "mood", "stress", "readiness", "sleep_duration", "sleep_quality", "soreness",
    "daily_load", "weekly_load", "acwr", "atl", "ctl28", "ctl42", "monotony", "strain"
]

X = df[same_day_features]
y = df["fatigue_class"]
groups = df["player"]

label_map = {"Low": 0, "Medium": 1, "High": 2}
inv_label_map = {v: k for k, v in label_map.items()}
y_encoded = y.map(label_map)

gkf = GroupKFold(n_splits=5)
all_preds = np.zeros(len(y_encoded), dtype=int)
fold_scores = []

for train_idx, test_idx in gkf.split(X, y_encoded, groups=groups):
    model = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=3,
        class_weight="balanced", random_state=42
    )
    model.fit(X.iloc[train_idx], y_encoded.iloc[train_idx])
    preds = model.predict(X.iloc[test_idx])
    all_preds[test_idx] = preds
    fold_scores.append(f1_score(y_encoded.iloc[test_idx], preds, average="macro"))

print("Same-day model — Macro F1 per fold:", [round(s, 3) for s in fold_scores])
print("Average Macro F1:", round(np.mean(fold_scores), 3))
print()
print(classification_report(y_encoded, all_preds, target_names=["Low", "Medium", "High"]))

# Train final version on ALL data for deployment
final_model = RandomForestClassifier(
    n_estimators=200, max_depth=20, min_samples_leaf=3,
    class_weight="balanced", random_state=42
)
final_model.fit(X, y_encoded)

joblib.dump(final_model, "models/deployment_model.joblib")
joblib.dump(same_day_features, "models/deployment_features.joblib")
joblib.dump(label_map, "models/label_map.joblib")

# Save feature ranges for the dashboard (so sliders have sensible min/max/default)
feature_stats = X.describe().T[["min", "max", "mean"]]
feature_stats.to_csv("models/feature_stats.csv")

print("\nSaved deployment model, feature list, and feature stats.")