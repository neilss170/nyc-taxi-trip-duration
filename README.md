# NYC Taxi Trip Duration Analysis

This project studies New York City taxi trips and builds regression models for
trip duration. It includes separate notebooks for exploratory analysis, feature
engineering, and modelling, plus an interactive Streamlit dashboard.

## Project workflow

1. Inspect trip records, missing values, duplicates, and unusual coordinates.
2. Create time, distance, airport, direction, and interaction features.
3. Remove target-derived fields before modelling.
4. Compare regression models and evaluate the selected feature set.
5. Explore trip patterns through a filterable dashboard.

## Model results

`notebooks/model.ipynb` trains five regressors on the split and the 29 features selected in
`feature_engineering.ipynb` (random 80/20 split, 583,431 training and 145,858 test trips). The target is
`log1p(trip_duration)`, so the score is RMSLE, the metric of the original Kaggle competition. Every model is
scored once on the held-out 20%; the numbers are written to `reports/model_metrics.json`.

| Model | Test RMSLE | R² (log scale) | MAE (minutes) | Median abs. error (minutes) |
|---|---|---|---|---|
| **HistGradientBoosting** | **0.390** | **0.757** | **4.83** | **1.80** |
| Random Forest | 0.402 | 0.742 | 4.99 | 1.91 |
| Ridge / Linear Regression | 0.528 | 0.555 | 6.49 | 2.84 |
| Baseline (training mean) | 0.791 | 0.000 | 9.32 | 5.07 |

- Gradient boosting cuts RMSLE by 50.7% against the mean baseline.
- Straight-line trip distance dominates: shuffling it raises RMSLE by 0.55, against 0.08 for the next feature
  (trip direction) and 0.05 for pickup hour.
- Errors stay small for ordinary trips (1.3-2.7 minutes MAE up to 20 minutes) but not for the 1,210 test trips
  over an hour, which include records lasting days; those pull the overall MAE up to 4.8 minutes.

## Dashboard

The dashboard includes:

- Trip count, average duration, and distance summaries
- Hour, weekday, passenger, and distance filters
- Time-of-day and weekday comparisons
- Distance and duration distributions
- Airport-trip analysis
- Feature previews and data-quality checks

Run it from the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run dashboard\app.py
```

The repository includes a 25,000-row sample so the dashboard can run after
cloning. The full Kaggle NYC Taxi Trip Duration dataset and generated training
matrices are excluded because they are too large for a normal Git repository.

To rerun the notebooks, save the Kaggle `train.csv` as `data/nyc_taxi_trip_duration.csv`, then run
`eda.ipynb`, `feature_engineering.ipynb` (writes the split to `outputs/`) and `model.ipynb`, in that order.

## Project structure

```text
dashboard/    Streamlit application
data/         Small dashboard sample
notebooks/    EDA, feature engineering, and modelling
reports/      Written project reports and model_metrics.json
```

## Limitations

- The sample is intended for dashboard demonstration, not model retraining.
- Straight-line distance is only an approximation of the actual driven route.
- Traffic, weather, and road-closure information are not available.
- The test split is random, not by time, so scores reflect trips from the same months as training.
