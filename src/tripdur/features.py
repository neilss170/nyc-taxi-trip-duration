"""Model features, computable from what a rider knows before the trip: two points, a pickup time,
passengers and vendor. site/predict.js mirrors every formula here; tests/test_parity.py holds them equal.

Fitted state (location clusters and target encodings) is learnt from training rows only and exported
to JSON with the model, so the browser applies exactly the same transformation.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import KFold

EARTH_KM = 6371.0088
# local flat projection centred on midtown: km per degree of latitude / longitude at 40.75 N
LAT0, LON0 = 40.75, -73.97
KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON = 111.320 * math.cos(math.radians(LAT0))
# Manhattan's avenues run about 29 degrees east of true north
GRID = math.radians(29.0)
GRID_SIN, GRID_COS = math.sin(GRID), math.cos(GRID)
AIRPORTS = {"jfk": (40.6413, -73.7781), "lga": (40.7769, -73.8740), "ewr": (40.6895, -74.1745)}
AIRPORT_KM = 2.0
HOLIDAYS_2016 = ["2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30"]
N_CLUSTERS = 40
SMOOTHING = 20.0

FEATURES = [
    "vendor_id", "passenger_count",
    "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude",
    "pickup_avenue_km", "pickup_street_km", "dropoff_avenue_km", "dropoff_street_km",
    "haversine_km", "grid_km", "avenue_km", "street_km", "bearing_deg",
    "hour", "weekday", "day_of_year", "is_weekend", "is_holiday",
    "pickup_airport", "dropoff_airport",
    "te_pickup", "te_dropoff", "te_pair",
]


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))


def project(lat, lon):
    """Kilometres east (x) and north (y) of midtown."""
    return (np.asarray(lon) - LON0) * KM_PER_DEG_LON, (np.asarray(lat) - LAT0) * KM_PER_DEG_LAT


def to_grid(x, y):
    """Kilometres along the avenues and along the streets."""
    return x * GRID_SIN + y * GRID_COS, x * GRID_COS - y * GRID_SIN


def nearest_airport(lat, lon):
    """1 = JFK, 2 = LGA, 3 = EWR when within AIRPORT_KM, else 0."""
    out = np.zeros(len(lat), dtype=np.int64)
    for code, (alat, alon) in enumerate(AIRPORTS.values(), start=1):
        near = haversine_km(lat, lon, alat, alon) < AIRPORT_KM
        out = np.where((out == 0) & near, code, out)
    return out


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Everything that needs no fitting."""
    plat, plon = df.pickup_latitude.to_numpy(float), df.pickup_longitude.to_numpy(float)
    dlat, dlon = df.dropoff_latitude.to_numpy(float), df.dropoff_longitude.to_numpy(float)
    px, py = project(plat, plon)
    dx, dy = project(dlat, dlon)
    pa, ps = to_grid(px, py)
    da, ds = to_grid(dx, dy)
    ts = df.pickup_datetime
    out = pd.DataFrame({
        "vendor_id": df.vendor_id.to_numpy(float),
        "passenger_count": df.passenger_count.to_numpy(float),
        "pickup_latitude": plat, "pickup_longitude": plon, "dropoff_latitude": dlat, "dropoff_longitude": dlon,
        "pickup_avenue_km": pa, "pickup_street_km": ps, "dropoff_avenue_km": da, "dropoff_street_km": ds,
        "haversine_km": haversine_km(plat, plon, dlat, dlon),
        "grid_km": np.abs(da - pa) + np.abs(ds - ps),
        "avenue_km": da - pa,
        "street_km": ds - ps,
        "bearing_deg": np.degrees(np.arctan2(dx - px, dy - py)),
        "hour": (ts.dt.hour + ts.dt.minute / 60).to_numpy(float),
        "weekday": ts.dt.weekday.to_numpy(float),
        "day_of_year": ts.dt.dayofyear.to_numpy(float),
        "is_weekend": (ts.dt.weekday >= 5).to_numpy(float),
        "is_holiday": ts.dt.strftime("%Y-%m-%d").isin(HOLIDAYS_2016).to_numpy(float),
        "pickup_airport": nearest_airport(plat, plon).astype(float),
        "dropoff_airport": nearest_airport(dlat, dlon).astype(float),
    }, index=df.index)
    return out


class FeatureBuilder:
    """Location clusters plus smoothed mean log-duration per pickup cluster, dropoff cluster and cluster pair."""

    def __init__(self):
        self.centroids = None
        self.global_mean = None
        self.tables = None

    # ---- clusters
    def _assign(self, lat, lon):
        x, y = project(lat, lon)
        pts = np.column_stack([x, y])
        out = np.empty(len(pts), dtype=np.int64)
        for i in range(0, len(pts), 100_000):
            d = ((pts[i:i + 100_000, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
            out[i:i + 100_000] = d.argmin(axis=1)
        return out

    def _keys(self, df):
        p = self._assign(df.pickup_latitude.to_numpy(float), df.pickup_longitude.to_numpy(float))
        d = self._assign(df.dropoff_latitude.to_numpy(float), df.dropoff_longitude.to_numpy(float))
        return {"te_pickup": p, "te_dropoff": d, "te_pair": p * N_CLUSTERS + d}

    # ---- target encodings
    @staticmethod
    def _table(keys, y, global_mean, size):
        s = np.bincount(keys, weights=y, minlength=size)
        n = np.bincount(keys, minlength=size)
        return (s + SMOOTHING * global_mean) / (n + SMOOTHING)

    def fit_transform(self, df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Fit on training rows. Training rows get out-of-fold encodings so no row sees its own target."""
        x, yk = [], []
        for lat, lon in (("pickup_latitude", "pickup_longitude"), ("dropoff_latitude", "dropoff_longitude")):
            px, py = project(df[lat].to_numpy(float), df[lon].to_numpy(float))
            x.append(px); yk.append(py)
        pts = np.column_stack([np.concatenate(x), np.concatenate(yk)])
        km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42, batch_size=4096, n_init=3).fit(pts)
        self.centroids = km.cluster_centers_
        self.global_mean = float(np.mean(y))
        keys = self._keys(df)
        sizes = {"te_pickup": N_CLUSTERS, "te_dropoff": N_CLUSTERS, "te_pair": N_CLUSTERS ** 2}
        self.tables = {k: self._table(keys[k], y, self.global_mean, sizes[k]) for k in keys}

        out = base_features(df)
        oof = {k: np.empty(len(df)) for k in keys}
        for fit_idx, enc_idx in KFold(5, shuffle=True, random_state=42).split(df):
            gm = float(np.mean(y[fit_idx]))
            for k in keys:
                oof[k][enc_idx] = self._table(keys[k][fit_idx], y[fit_idx], gm, sizes[k])[keys[k][enc_idx]]
        for k in keys:
            out[k] = oof[k]
        return out[FEATURES]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = base_features(df)
        for k, idx in self._keys(df).items():
            out[k] = self.tables[k][idx]
        return out[FEATURES]

    def to_params(self) -> dict:
        return {
            "lat0": LAT0, "lon0": LON0, "km_per_deg_lat": KM_PER_DEG_LAT, "km_per_deg_lon": KM_PER_DEG_LON,
            "grid_sin": GRID_SIN, "grid_cos": GRID_COS, "earth_km": EARTH_KM, "airport_km": AIRPORT_KM,
            "airports": [list(v) for v in AIRPORTS.values()], "holidays": HOLIDAYS_2016,
            "n_clusters": N_CLUSTERS, "centroids": self.centroids.tolist(), "global_mean": self.global_mean,
            "tables": {k: v.tolist() for k, v in self.tables.items()}, "features": FEATURES,
        }

    @classmethod
    def from_params(cls, p: dict) -> "FeatureBuilder":
        fb = cls()
        fb.centroids = np.asarray(p["centroids"])
        fb.global_mean = p["global_mean"]
        fb.tables = {k: np.asarray(v) for k, v in p["tables"].items()}
        return fb
