import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load data ---
df = pd.read_csv("data/features_engineered.csv", parse_dates=["date"])
df["team"] = df["player"].str.split("-").str[0]

# === 1. Fatigue score distribution (raw 1-5) ===
plt.figure(figsize=(7, 5))
sns.countplot(x="fatigue", data=df, color="steelblue")
plt.title("Distribution of Raw Fatigue Scores (1-5)")
plt.xlabel("Fatigue Score")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUT_DIR / "01_fatigue_score_distribution.png", dpi=150)
plt.close()

# === 2. Fatigue CLASS distribution (Low/Medium/High) ===
plt.figure(figsize=(7, 5))
order = ["Low", "Medium", "High"]
sns.countplot(x="fatigue_class", data=df, order=order, palette=["#4CAF50", "#FFC107", "#F44336"])
plt.title("Distribution of Fatigue Classes")
plt.xlabel("Fatigue Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUT_DIR / "02_fatigue_class_distribution.png", dpi=150)
plt.close()

# === 3. Fatigue trend over time (weekly average, per team) ===
weekly = df.set_index("date").groupby("team")["fatigue"].resample("W").mean().reset_index()
plt.figure(figsize=(12, 5))
sns.lineplot(data=weekly, x="date", y="fatigue", hue="team")
plt.title("Weekly Average Fatigue Score Over Time by Team")
plt.xlabel("Date")
plt.ylabel("Average Fatigue Score")
plt.tight_layout()
plt.savefig(OUT_DIR / "03_fatigue_trend_over_time.png", dpi=150)
plt.close()

# === 4. Correlation heatmap ===
corr_cols = ["fatigue", "mood", "stress", "readiness", "sleep_duration", "sleep_quality",
             "soreness", "daily_load", "weekly_load", "acwr", "atl", "ctl28", "ctl42",
             "monotony", "strain"]
plt.figure(figsize=(12, 10))
corr = df[corr_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation Heatmap: Wellness and Training Load Metrics")
plt.tight_layout()
plt.savefig(OUT_DIR / "04_correlation_heatmap.png", dpi=150)
plt.close()

# === 5. Boxplots: key variables by fatigue class ===
box_vars = ["mood", "stress", "sleep_quality", "soreness", "daily_load", "acwr"]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, var in zip(axes.flatten(), box_vars):
    sns.boxplot(x="fatigue_class", y=var, data=df, order=order, ax=ax,
                palette=["#4CAF50", "#FFC107", "#F44336"])
    ax.set_title(f"{var} by Fatigue Class")
plt.tight_layout()
plt.savefig(OUT_DIR / "05_boxplots_by_class.png", dpi=150)
plt.close()

# === 6. Training load (daily_load) distribution over time ===
weekly_load = df.set_index("date").groupby("team")["daily_load"].resample("W").mean().reset_index()
plt.figure(figsize=(12, 5))
sns.lineplot(data=weekly_load, x="date", y="daily_load", hue="team")
plt.title("Weekly Average Daily Training Load Over Time by Team")
plt.xlabel("Date")
plt.ylabel("Average Daily Load")
plt.tight_layout()
plt.savefig(OUT_DIR / "06_training_load_trend.png", dpi=150)
plt.close()

# === 7. Model comparison bar chart (from our CV results) ===
model_results = {
    "Baseline\n(Majority Class)": 0.263,
    "Logistic\nRegression": 0.626,
    "XGBoost\n(tuned)": 0.635,
    "Random Forest\n(tuned)": 0.653,
}
plt.figure(figsize=(8, 5))
bars = plt.bar(model_results.keys(), model_results.values(),
                color=["#9E9E9E", "#2196F3", "#FF9800", "#4CAF50"])
plt.ylabel("Macro F1 Score")
plt.title("Model Comparison: Macro F1 Score (5-fold GroupKFold CV)")
plt.ylim(0, 0.8)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.3f}", ha="center")
plt.tight_layout()
plt.savefig(OUT_DIR / "07_model_comparison.png", dpi=150)
plt.close()

print("Plots 1-7 saved to outputs/figures/")

# === 8. Confusion matrix heatmap ===
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
import joblib
import shap

label_map = {"Low": 0, "Medium": 1, "High": 2}
y_encoded = df["fatigue_class"].map(label_map)
feature_cols = [c for c in df.columns if c not in ["date", "player", "fatigue", "fatigue_class", "team"]]
X = df[feature_cols]
groups = df["player"]

best_params = {"max_depth": 20, "min_samples_leaf": 3, "n_estimators": 200}
gkf = GroupKFold(n_splits=5)
all_preds = np.zeros(len(y_encoded), dtype=int)

for train_idx, test_idx in gkf.split(X, y_encoded, groups=groups):
    model = RandomForestClassifier(class_weight="balanced", random_state=42, **best_params)
    model.fit(X.iloc[train_idx], y_encoded.iloc[train_idx])
    all_preds[test_idx] = model.predict(X.iloc[test_idx])

cm = confusion_matrix(y_encoded, all_preds)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — Random Forest (Out-of-Fold Predictions)")
plt.tight_layout()
plt.savefig(OUT_DIR / "08_confusion_matrix.png", dpi=150)
plt.close()

print("Plot 8 (confusion matrix) saved.")

# === 9. SHAP summary plots ===
final_model = joblib.load("models/random_forest_final.joblib")

# Use a sample for speed (SHAP on 16,881 rows can be slow)
sample = X.sample(n=1500, random_state=42)

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(sample)

# shap_values is a list (one array per class) for multi-class RF
plt.figure()
shap.summary_plot(shap_values[:, :, 1], sample, show=False, max_display=15)  # class 1 = Medium
plt.title("SHAP Feature Importance (class: Medium)")
plt.tight_layout()
plt.savefig(OUT_DIR / "09_shap_summary_medium.png", dpi=150)
plt.close()

# High fatigue class (most practically important)
plt.figure()
shap.summary_plot(shap_values[:, :, 2], sample, show=False, max_display=15)
plt.title("SHAP Feature Importance (class: High Fatigue)")
plt.tight_layout()
plt.savefig(OUT_DIR / "10_shap_summary_high.png", dpi=150)
plt.close()

print("Plots 9-10 (SHAP) saved.")

print("\nAll plots saved to outputs/figures/")
for f in sorted(OUT_DIR.glob("*.png")):
    print(" -", f.name)