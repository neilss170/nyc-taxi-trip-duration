"""Paths and constants shared by cleaning, features, training and the site export."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "nyc_taxi_trip_duration.csv"
SAMPLE_CSV = ROOT / "data" / "nyc_taxi_sample.csv"
REPORTS = ROOT / "reports"
SITE = ROOT / "site"
ARTIFACTS = ROOT / "outputs" / "pipeline"

TARGET = "trip_duration"
SEED = 42

# time-based split: train on January-May 2016, hold out every June trip
TEST_FROM = "2016-06-01"
# early stopping uses the last two weeks of the training period, never June
VALID_FROM = "2016-05-17"

# cleaning thresholds, chosen from the profile in notebooks/eda.ipynb
MIN_SECONDS = 60
MAX_SECONDS = 3 * 3600
MAX_KMH = 100.0
MIN_KM = 0.01
NYC_BBOX = {"lat_min": 40.49, "lat_max": 40.92, "lon_min": -74.27, "lon_max": -73.68}
MAX_PASSENGERS = 6
