import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

def load_wellness_metric(filename):
    col_name = filename.replace(".csv", "")
    df = pd.read_csv(DATA_DIR / "wellness" / filename)
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")
    long = df.melt(id_vars="date", var_name="player", value_name=col_name)
    return long.dropna(subset=[col_name])

def load_training_load_metric(filename):
    col_name = filename.replace(".csv", "")
    df = pd.read_csv(DATA_DIR / "training-load" / filename)
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")
    long = df.melt(id_vars="date", var_name="player", value_name=col_name)
    return long.dropna(subset=[col_name])

# --- Wellness ---
wellness_metrics = ["fatigue", "mood", "stress", "readiness", "sleep_duration", "sleep_quality", "soreness"]
wellness_dfs = {m: load_wellness_metric(f"{m}.csv") for m in wellness_metrics}

wellness = wellness_dfs["fatigue"]
for m in wellness_metrics[1:]:
    wellness = wellness.merge(wellness_dfs[m], on=["date", "player"], how="outer")

# Drop rows with any missing wellness value (very small % of data)
wellness_clean = wellness.dropna(subset=wellness_metrics)
print("Wellness after dropping missing:", wellness_clean.shape)

# --- Training load ---
load_metrics = ["daily_load", "weekly_load", "acwr", "atl", "ctl28", "ctl42", "monotony", "strain"]
load_dfs = {m: load_training_load_metric(f"{m}.csv") for m in load_metrics}

training_load = load_dfs["daily_load"]
for m in load_metrics[1:]:
    training_load = training_load.merge(load_dfs[m], on=["date", "player"], how="outer")

print("Training load shape:", training_load.shape)
print(training_load.head())
print()
print("Missing values in training load:")
print(training_load.isna().sum())

# --- Merge wellness + training load ---
merged = wellness_clean.merge(training_load, on=["date", "player"], how="inner")

print("Final merged shape:", merged.shape)
print(merged.head())
print()
print("Missing values after merge:")
print(merged.isna().sum())
print()

# Save the cleaned, merged dataset for the next stage
merged.to_csv("data/merged_wellness_load.csv", index=False)
print("Saved to data/merged_wellness_load.csv")