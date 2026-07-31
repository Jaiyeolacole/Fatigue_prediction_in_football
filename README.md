# Mental Fatigue Prediction Framework for Football Players

A machine learning-based framework for predicting mental fatigue in football players using wellness and training-load performance metrics, developed as part of a final-year thesis project.

## Overview

This project builds an end-to-end pipeline that:
1. Cleans and merges athlete wellness and training-load data
2. Engineers predictive features (lag and rolling-window variables)
3. Trains and compares multiple machine learning classification models
4. Evaluates model performance and interprets feature importance
5. Deploys the final model via an interactive Streamlit dashboard for real-time fatigue classification

## Dataset

This project uses the **SoccerMon** dataset — a publicly available, peer-reviewed athlete-monitoring dataset containing wellness self-reports (fatigue, mood, stress, readiness, sleep duration/quality, soreness) and training-load metrics (session-RPE, daily/weekly load, ACWR, ATL, CTL28, CTL42, monotony, strain) collected from two elite women's football teams over two competitive seasons.

**Note on scope**: The dataset is drawn from elite female athletes. This is explicitly acknowledged as a scope limitation; the study's contribution is framed as a transferable machine learning framework rather than a claim of male-population-specific findings.

Source: Zenodo (DOI: 10.5281/zenodo.10033832)

## Project Structure

```
fatigue_prediction/
│
├── data/
│   ├── raw/                        # Original SoccerMon subjective data (wellness, training-load, etc.)
│   ├── merged_wellness_load.csv    # Cleaned, merged dataset
│   └── features_engineered.csv     # Final modeling dataset with engineered features
│
├── src/
│   ├── data_cleaning.py            # Loads and merges raw wellness + training-load CSVs
│   ├── feature_engineering.py      # Creates fatigue class labels, lag and rolling features
│   ├── train_model.py              # Baseline comparison + 5-fold GroupKFold model comparison
│   ├── tune_models.py              # Hyperparameter tuning (Random Forest, XGBoost)
│   ├── final_model.py              # Trains and saves the final research model
│   ├── train_deployment_model.py   # Trains the simplified same-day model for the dashboard
│   ├── feature_importance.py       # Extracts feature importance from the trained model
│   ├── generate_plots.py           # Generates all figures (distributions, correlations, SHAP, etc.)
│   └── dashboard.py                # Streamlit dashboard for interactive fatigue prediction
│
├── models/                         # Saved trained models and supporting artifacts (.joblib, .csv)
│
├── outputs/
│   └── figures/                    # All generated plots and charts
│
├── requirements.txt                # Python dependencies
└── README.md
```

## Setup

### 1. Clone or download the project folder
Ensure the folder is **not** located inside a cloud-synced directory (e.g., OneDrive), as this can cause virtual environment path conflicts.

### 2. Create and activate a virtual environment

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

Or, if setting up fresh:

```powershell
pip install pandas numpy scikit-learn xgboost matplotlib seaborn shap streamlit joblib plotly ipykernel
```

### 4. Add the dataset

Download the `subjective.zip` file from the SoccerMon Zenodo repository and unzip its contents into `data/raw/`, so the structure matches:

```
data/raw/wellness/fatigue.csv
data/raw/training-load/daily_load.csv
...
```

## Usage

Run the pipeline scripts in order from the project root:

```powershell
# 1. Clean and merge raw data
python src/data_cleaning.py

# 2. Engineer features and define fatigue classes
python src/feature_engineering.py

# 3. Compare baseline and candidate models
python src/train_model.py

# 4. Tune hyperparameters
python src/tune_models.py

# 5. Train and save the final research model
python src/final_model.py

# 6. Train the simplified deployment model (for the dashboard)
python src/train_deployment_model.py

# 7. Check feature importance
python src/feature_importance.py

# 8. Generate all figures
python src/generate_plots.py
```

### Launch the dashboard

```powershell
streamlit run src/dashboard.py
```

This opens an interactive web interface where wellness and training-load values can be entered to receive a real-time fatigue classification (Low / Medium / High), along with prediction confidence and a fatigue risk gauge.

## Models Used

- Logistic Regression (baseline linear model)
- Random Forest (final selected model)
- XGBoost (gradient-boosted comparison model)
- Majority-class Dummy Classifier (baseline reference)

All models are evaluated using **GroupKFold cross-validation, grouped by player**, to prevent data leakage between training and test sets.

## Limitations

- Dataset reflects elite **women's** football; generalizability to male populations has not been directly validated.
- The dashboard's deployment model uses same-day features only (no lag/rolling history), trading a small amount of predictive power for ease of use.
- Fatigue labels are based on subjective self-report, which carries inherent variability across individuals.

## Citation

If referencing the dataset used in this project, cite the original SoccerMon publication and Zenodo repository (DOI: 10.5281/zenodo.10033832).

## License / Usage Note

This project is developed for academic (final-year thesis) purposes. The SoccerMon dataset is used under its original public license terms; raw data files are not redistributed as part of this repository.