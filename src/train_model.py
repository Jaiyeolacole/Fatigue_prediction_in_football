import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.dummy import DummyClassifier
import xgboost as xgb

# --- Load engineered features ---
df = pd.read_csv("data/features_engineered.csv", parse_dates=["date"])

exclude_cols = ["date", "player", "fatigue", "fatigue_class"]
feature_cols = [c for c in df.columns if c not in exclude_cols]

X = df[feature_cols]
y = df["fatigue_class"]
groups = df["player"]

label_map = {"Low": 0, "Medium": 1, "High": 2}
y_encoded = y.map(label_map)

print("X shape:", X.shape)
print("Class distribution:\n", y.value_counts())
print()

# --- 5-fold cross-validation, grouped by player ---
gkf = GroupKFold(n_splits=5)

# --- Majority-class baseline ---
baseline_scores = []
for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y_encoded, groups=groups)):
    y_train, y_test = y_encoded.iloc[train_idx], y_encoded.iloc[test_idx]

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X.iloc[train_idx], y_train)
    preds = dummy.predict(X.iloc[test_idx])

    f1 = f1_score(y_test, preds, average="macro")
    baseline_scores.append(f1)
    print(f"Fold {fold+1} | Baseline (majority class) | Macro F1: {f1:.3f}")

print(f"\nBaseline average Macro F1: {np.mean(baseline_scores):.3f} (+/- {np.std(baseline_scores):.3f})")
print()

# --- Real models ---
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
    "XGBoost": xgb.XGBClassifier(n_estimators=200, eval_metric="mlogloss", random_state=42)
}

results = {name: [] for name in models}

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y_encoded, groups=groups)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_encoded.iloc[train_idx], y_encoded.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name, model in models.items():
        if name == "LogisticRegression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        f1 = f1_score(y_test, preds, average="macro")
        results[name].append(f1)
        print(f"Fold {fold+1} | {name} | Macro F1: {f1:.3f}")

print("\n=== Average Macro F1 across 5 folds ===")
for name, scores in results.items():
    print(f"{name}: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")