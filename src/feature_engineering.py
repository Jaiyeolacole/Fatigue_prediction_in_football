import pandas as pd
from pathlib import Path

# Load the cleaned merged dataset
df = pd.read_csv("data/merged_wellness_load.csv", parse_dates=["date"])

# Sort by player and date — essential for any lag/rolling features
df = df.sort_values(["player", "date"]).reset_index(drop=True)

# --- 1. Fatigue class label (Low / Medium / High) ---
def fatigue_class(score):
    if score <= 2:
        return "Low"
    elif score == 3:
        return "Medium"
    else:
        return "High"

df["fatigue_class"] = df["fatigue"].apply(fatigue_class)

print("Class distribution:")
print(df["fatigue_class"].value_counts())
print()
print((df["fatigue_class"].value_counts(normalize=True) * 100).round(1))

# --- 2. Lag features (yesterday's values) ---
lag_cols = ["daily_load", "sleep_duration", "sleep_quality", "soreness", "stress", "mood", "readiness"]

for col in lag_cols:
    df[f"{col}_lag1"] = df.groupby("player")[col].shift(1)

# --- 3. Rolling averages (3-day and 7-day, using only PAST data) ---
roll_cols = ["daily_load", "sleep_duration", "soreness", "stress"]

for col in roll_cols:
    df[f"{col}_roll3"] = (
        df.groupby("player")[col]
        .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    )
    df[f"{col}_roll7"] = (
        df.groupby("player")[col]
        .transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean())
    )

# Check how many rows now have missing lag/rolling values (expected: first day per player)
print("\nMissing values after adding lag/rolling features:")
print(df.isna().sum()[df.isna().sum() > 0])
print()
print("Shape before dropping first-day rows:", df.shape)

# Drop rows where lag features are missing (first recorded day per player — no history yet)
df_final = df.dropna(subset=[f"{c}_lag1" for c in lag_cols])
print("Shape after dropping:", df_final.shape)

# Save
df_final.to_csv("data/features_engineered.csv", index=False)
print("Saved to data/features_engineered.csv")