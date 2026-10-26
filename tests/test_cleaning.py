import pandas as pd

from tripdur import data

GOOD = {"id": "ok", "vendor_id": 1, "passenger_count": 1, "pickup_datetime": pd.Timestamp("2016-06-02 08:00"),
        "dropoff_datetime": pd.Timestamp("2016-06-02 08:15"), "pickup_latitude": 40.75, "pickup_longitude": -73.99,
        "dropoff_latitude": 40.76, "dropoff_longitude": -73.97, "store_and_fwd_flag": "N", "trip_duration": 900}


def frame(*changes):
    return pd.DataFrame([{**GOOD, **c} for c in changes])


def test_each_rule_catches_its_case():
    df = frame({}, {"trip_duration": 30}, {"trip_duration": 4 * 3600}, {"pickup_latitude": 41.5},
               {"dropoff_latitude": 40.75, "dropoff_longitude": -73.99},
               {"dropoff_latitude": 40.95 - 0.04, "dropoff_longitude": -73.70, "trip_duration": 120},
               {"passenger_count": 0})
    masks = data.rule_masks(df)
    assert masks["duration_under_1_min"].tolist() == [False, True, False, False, False, False, False]
    assert masks["duration_over_3_hours"].tolist()[2]
    assert masks["outside_nyc"].tolist()[3]
    assert masks["no_movement"].tolist()[4]
    assert masks["speed_over_100_kmh"].tolist()[5]
    assert masks["passenger_count_invalid"].tolist()[6]
    assert not any(m.iloc[0] for m in masks.values())


def test_clean_conserves_rows():
    df = frame({}, {"trip_duration": 30}, {"passenger_count": 9}, {})
    kept, report = data.clean(df)
    assert len(kept) == 2 and report["rows_removed"] == 2
    assert report["rows_kept"] + report["rows_removed"] == report["rows_in"]


def test_split_is_by_time():
    df = frame({"pickup_datetime": pd.Timestamp("2016-05-31 23:59")}, {"pickup_datetime": pd.Timestamp("2016-06-01 00:00")})
    train, test = data.split(df)
    assert len(train) == 1 and len(test) == 1
    assert train.pickup_datetime.max() < test.pickup_datetime.min()
