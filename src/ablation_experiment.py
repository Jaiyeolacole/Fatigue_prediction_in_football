import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

df = pd.read_csv("data/features_engineered.csv", parse_dates=["date"])

wellness_features = ["mood", "stress", "readiness", "sleep_duration", "sleep_quality", "soreness"]
load_features = ["daily_load", "weekly_load", "acwr", "atl", "ctl28", "ctl42", "monotony", "strain"]
combined_features = wellness_features + load_features

label_map = {"Low": 0, "Medium": 1, "High": 2}
y = df["fatigue_class"].map(label_map)
groups = df["player"]

feature_sets = {
    "Wellness Only": wellness_features,
    "Training Load Only": load_features,
    "Combined (Wellness + Training Load)": combined_features
}

gkf = GroupKFold(n_splits=5)

print("=== Ablation Experiment: Feature Set Comparison ===\n")

summary = {}

for set_name, cols in feature_sets.items():
    X = df[cols]
    fold_scores = []
    all_preds = np.zeros(len(y), dtype=int)

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        model = RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=3,
            class_weight="balanced", random_state=42
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        all_preds[test_idx] = preds
        fold_scores.append(f1_score(y.iloc[test_idx], preds, average="macro"))

    avg_f1 = np.mean(fold_scores)
    summary[set_name] = avg_f1

    print(f"--- {set_name} ({len(cols)} features) ---")
    print(f"Fold scores: {[round(s, 3) for s in fold_scores]}")
    print(f"Average Macro F1: {avg_f1:.3f}")
    print()
    print(classification_report(y, all_preds, target_names=["Low", "Medium", "High"]))
    print("=" * 60)
    print()

print("=== Summary ===")
for name, score in summary.items():
    print(f"{name:40s} Macro F1: {score:.3f}")