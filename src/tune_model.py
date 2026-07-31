import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
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

# Pre-compute the fold splits ONCE as a list (generators can't be pickled/reused)
cv_splits = list(GroupKFold(n_splits=5).split(X, y_encoded, groups=groups))

# --- Random Forest tuning ---
print("=== Hyperparameter Tuning: Random Forest ===")

rf_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 3, 5]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=42),
    rf_param_grid,
    scoring="f1_macro",
    cv=cv_splits,
    n_jobs=-1,
    verbose=1
)
rf_grid.fit(X, y_encoded)

print("Best RF params:", rf_grid.best_params_)
print("Best RF macro F1:", rf_grid.best_score_)
print()

# --- XGBoost tuning ---
print("=== Hyperparameter Tuning: XGBoost ===")

xgb_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2]
}

xgb_grid = GridSearchCV(
    xgb.XGBClassifier(eval_metric="mlogloss", random_state=42),
    xgb_param_grid,
    scoring="f1_macro",
    cv=cv_splits,
    n_jobs=-1,
    verbose=1
)
xgb_grid.fit(X, y_encoded)

print("Best XGBoost params:", xgb_grid.best_params_)
print("Best XGBoost macro F1:", xgb_grid.best_score_)