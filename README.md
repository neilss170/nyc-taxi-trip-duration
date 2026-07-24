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

## Project structure

```text
dashboard/    Streamlit application
data/         Small dashboard sample
notebooks/    EDA, feature engineering, and modelling
reports/      Written project reports
```

## Limitations

- The sample is intended for dashboard demonstration, not model retraining.
- Straight-line distance is only an approximation of the actual driven route.
- Traffic, weather, and road-closure information are not available.
