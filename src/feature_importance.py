import pandas as pd
import joblib

model = joblib.load("models/deployment_model.joblib")
features = joblib.load("models/deployment_features.joblib")

importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

print("=== Feature Importance (Random Forest, same-day model) ===")
for feat, score in importances.items():
    bar = "█" * int(score * 100)
    print(f"{feat:20s} {score:.4f}  {bar}")