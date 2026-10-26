"""Load the Kaggle trip file and apply the cleaning rules, keeping a count of what each rule removes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tripdur import config
from tripdur.features import haversine_km


def load(path=config.RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["pickup_datetime", "dropoff_datetime"])
    return df.sort_values("pickup_datetime", kind="stable").reset_index(drop=True)


def rule_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """One boolean mask per rule, True where the row breaks it. Rows can break several rules."""
    b = config.NYC_BBOX
    km = haversine_km(df.pickup_latitude, df.pickup_longitude, df.dropoff_latitude, df.dropoff_longitude)
    kmh = km / (df[config.TARGET] / 3600)
    inside = lambda lat, lon: lat.between(b["lat_min"], b["lat_max"]) & lon.between(b["lon_min"], b["lon_max"])
    return {
        "duration_under_1_min": df[config.TARGET] < config.MIN_SECONDS,
        "duration_over_3_hours": df[config.TARGET] > config.MAX_SECONDS,
        "outside_nyc": ~(inside(df.pickup_latitude, df.pickup_longitude) & inside(df.dropoff_latitude, df.dropoff_longitude)),
        "no_movement": pd.Series(km < config.MIN_KM, index=df.index),
        "speed_over_100_kmh": pd.Series(np.asarray(kmh) > config.MAX_KMH, index=df.index),
        "passenger_count_invalid": ~df.passenger_count.between(1, config.MAX_PASSENGERS),
    }


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    masks = rule_masks(df)
    bad = np.logical_or.reduce([m.to_numpy() for m in masks.values()])
    report = {
        "rows_in": int(len(df)),
        "rows_kept": int((~bad).sum()),
        "rows_removed": int(bad.sum()),
        "removed_by_rule": {k: int(m.sum()) for k, m in masks.items()},
    }
    assert report["rows_kept"] + report["rows_removed"] == report["rows_in"]
    return df.loc[~bad].reset_index(drop=True), report


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = df.pickup_datetime >= config.TEST_FROM
    return df.loc[~test].reset_index(drop=True), df.loc[test].reset_index(drop=True)
