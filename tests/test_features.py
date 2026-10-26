import numpy as np
import pandas as pd
import pytest

from tripdur import features as F
from tripdur.features import FEATURES, FeatureBuilder


def trips(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "vendor_id": rng.integers(1, 3, n),
        "passenger_count": rng.integers(1, 7, n),
        "pickup_datetime": pd.Timestamp("2016-03-01") + pd.to_timedelta(rng.integers(0, 90 * 86400, n), unit="s"),
        "pickup_latitude": rng.uniform(40.70, 40.80, n), "pickup_longitude": rng.uniform(-74.02, -73.93, n),
        "dropoff_latitude": rng.uniform(40.70, 40.80, n), "dropoff_longitude": rng.uniform(-74.02, -73.93, n),
    })


def test_haversine_matches_a_known_distance():
    # JFK to Times Square: 0.1167 deg north (12.98 km) and 0.2074 deg west (17.50 km at this latitude),
    # so about 21.8 km in a straight line
    km = F.haversine_km(40.6413, -73.7781, 40.7580, -73.9855)
    assert km == pytest.approx(21.8, abs=0.1)


def test_grid_axes_follow_manhattan_avenues():
    # one km along the avenue direction (29 degrees east of north) is all avenue, no street
    x, y = np.sin(F.GRID), np.cos(F.GRID)
    avenue, street = F.to_grid(np.array([x]), np.array([y]))
    assert avenue[0] == pytest.approx(1.0) and street[0] == pytest.approx(0.0, abs=1e-12)


def test_calendar_and_airport_features():
    df = pd.DataFrame({
        "vendor_id": [1], "passenger_count": [2], "pickup_datetime": [pd.Timestamp("2016-05-30 17:45:00")],
        "pickup_latitude": [40.6413], "pickup_longitude": [-73.7781],  # JFK
        "dropoff_latitude": [40.7580], "dropoff_longitude": [-73.9855],
    })
    f = F.base_features(df).iloc[0]
    assert f.hour == 17.75 and f.weekday == 0 and f.is_holiday == 1 and f.is_weekend == 0
    assert f.pickup_airport == 1 and f.dropoff_airport == 0
    assert f.day_of_year == 151


def test_test_targets_never_reach_the_features():
    train, test = trips(seed=1), trips(seed=2)
    fb = FeatureBuilder()
    fb.fit_transform(train, np.random.default_rng(3).normal(6, 0.5, len(train)))
    a = fb.transform(test.assign(trip_duration=1))
    b = fb.transform(test.assign(trip_duration=10_000))
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == FEATURES


def test_training_rows_get_out_of_fold_encodings():
    # one extreme target must not show up in its own row's encoding
    train = trips(n=500, seed=4)
    y = np.full(len(train), 6.0)
    y[0] = 60.0
    fb = FeatureBuilder()
    X = fb.fit_transform(train, y)
    assert X.te_pair.iloc[0] == pytest.approx(6.0)
    # while the full-data table used for new trips does include it
    assert fb.transform(train.iloc[[0]]).te_pair.iloc[0] > 6.0


def test_params_round_trip():
    train = trips(seed=5)
    fb = FeatureBuilder()
    fb.fit_transform(train, np.random.default_rng(6).normal(6, 0.5, len(train)))
    again = FeatureBuilder.from_params(fb.to_params())
    pd.testing.assert_frame_equal(fb.transform(train), again.transform(train))
