"""Small JSON extracts for the static dashboard in site/, built from the full trip file."""

from __future__ import annotations

import base64
import json

import numpy as np
import pandas as pd

from tripdur import config, data
from tripdur.features import KM_PER_DEG_LAT, KM_PER_DEG_LON, LAT0, LON0, N_CLUSTERS, FeatureBuilder, haversine_km, project

# map window: Manhattan, Brooklyn, Queens and both NYC airports; cells of 75 m
MAP = {"lat_min": 40.60, "lat_max": 40.88, "lon_min": -74.05, "lon_max": -73.74, "cell_km": 0.075}
HOUR_BANDS = [0, 6, 10, 16, 20, 24]
BAND_NAMES = ["Night (0-6)", "Morning (6-10)", "Midday (10-16)", "Evening (16-20)", "Late (20-24)"]


def _write(name, obj):
    (config.SITE / "data").mkdir(parents=True, exist_ok=True)
    (config.SITE / "data" / f"{name}.json").write_text(json.dumps(obj, separators=(",", ":")))


def grid_shape():
    x0, y0 = project(MAP["lat_min"], MAP["lon_min"])
    x1, y1 = project(MAP["lat_max"], MAP["lon_max"])
    return int(np.ceil((x1 - x0) / MAP["cell_km"])), int(np.ceil((y1 - y0) / MAP["cell_km"])), float(x0), float(y0)


def density(lat, lon):
    w, h, x0, y0 = grid_shape()
    x, y = project(lat, lon)
    cx, cy = ((x - x0) / MAP["cell_km"]).astype(int), ((y - y0) / MAP["cell_km"]).astype(int)
    ok = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
    counts = np.bincount(cy[ok] * w + cx[ok], minlength=w * h)
    return counts, cy * w + cx, ok


def run() -> dict:
    raw = data.load()
    df, cleaning = data.clean(raw)
    minutes = df.trip_duration / 60
    km = haversine_km(df.pickup_latitude, df.pickup_longitude, df.dropoff_latitude, df.dropoff_longitude)
    kmh = km / (df.trip_duration / 3600)

    _write("overview", {
        "rows_raw": int(len(raw)), "rows_clean": int(len(df)),
        "first_day": f"{df.pickup_datetime.min():%Y-%m-%d}", "last_day": f"{df.pickup_datetime.max():%Y-%m-%d}",
        "median_minutes": round(float(minutes.median()), 2), "median_km": round(float(np.median(km)), 2),
        "median_kmh": round(float(np.median(kmh)), 1),
        "vendor_share": {str(k): int(v) for k, v in df.vendor_id.value_counts().sort_index().items()},
        "passengers": {str(k): int(v) for k, v in df.passenger_count.value_counts().sort_index().items()},
        "cleaning": cleaning,
    })

    day = df.groupby(df.pickup_datetime.dt.date)
    _write("daily", [{"date": str(k), "trips": int(len(g)), "median_minutes": round(float(g.trip_duration.median() / 60), 2)}
                     for k, g in day])

    edges = list(range(0, 62, 2))
    hist = np.histogram(minutes.clip(upper=61), bins=edges + [1e9])[0]
    _write("duration_hist", [{"from": a, "to": (edges + [None])[i + 1], "trips": int(c)} for i, (a, c) in enumerate(zip(edges, hist))])

    hw = pd.DataFrame({"weekday": df.pickup_datetime.dt.weekday, "hour": df.pickup_datetime.dt.hour, "kmh": kmh, "min": minutes})
    _write("hour_weekday", [{"weekday": int(w), "hour": int(h), "trips": int(len(g)),
                             "median_kmh": round(float(g.kmh.median()), 2), "median_minutes": round(float(g["min"].median()), 2)}
                            for (w, h), g in hw.groupby(["weekday", "hour"])])

    w, h, x0, y0 = grid_shape()
    pick, pick_cell, pick_ok = density(df.pickup_latitude, df.pickup_longitude)
    drop, _, _ = density(df.dropoff_latitude, df.dropoff_longitude)
    speed_sum = np.bincount(pick_cell[pick_ok], weights=np.asarray(kmh)[pick_ok], minlength=w * h)
    speed = np.where(pick >= 10, speed_sum / np.maximum(pick, 1), -1)
    # binary layers keep the fine grid small: counts as little-endian uint16, speed as uint8 half-km/h steps (255 = no data)
    b64 = lambda a: base64.b64encode(a.tobytes()).decode("ascii")
    assert pick.max() < 65536 and drop.max() < 65536
    speed_u8 = np.where(speed < 0, 255, np.clip(np.round(speed * 2), 0, 254)).astype("<u1")
    _write("grid", {**MAP, "width": w, "height": h, "x0": x0, "y0": y0, "lat0": LAT0, "lon0": LON0,
                    "km_per_deg_lat": KM_PER_DEG_LAT, "km_per_deg_lon": KM_PER_DEG_LON,
                    "encoding": "base64: pickups/dropoffs uint16 LE, pickup_kmh uint8 = km/h * 2 (255 = fewer than 10 trips)",
                    "pickups": b64(pick.astype("<u2")), "dropoffs": b64(drop.astype("<u2")), "pickup_kmh": b64(speed_u8)})

    # "similar past trips" for the predictor: same pickup cluster, dropoff cluster and time-of-day band
    model = json.loads((config.SITE / "model" / "model.json").read_text())
    fb = FeatureBuilder.from_params(model["params"])
    keys = fb._keys(df)
    band = np.searchsorted(HOUR_BANDS, df.pickup_datetime.dt.hour.to_numpy(), side="right") - 1
    key = keys["te_pair"] * len(BAND_NAMES) + band
    grp = pd.DataFrame({"key": key, "min": minutes.to_numpy()}).groupby("key")["min"]
    size = N_CLUSTERS * N_CLUSTERS * len(BAND_NAMES)
    med, cnt = np.full(size, -1.0), np.zeros(size, dtype=int)
    med[grp.median().index] = grp.median().round(1).to_numpy()
    cnt[grp.size().index] = grp.size().to_numpy()
    _write("similar", {"hour_bands": HOUR_BANDS, "band_names": BAND_NAMES, "median_minutes": med.tolist(), "trips": cnt.tolist()})

    metrics = json.loads((config.REPORTS / "pipeline_metrics.json").read_text())
    notebook = config.REPORTS / "model_metrics.json"
    if notebook.exists():
        nb = json.loads(notebook.read_text())
        metrics["notebook_best"] = {**nb["models"][0], "split": nb["split"]}
    _write("metrics", metrics)
    return {"grid": [w, h], "files": sorted(p.name for p in (config.SITE / "data").glob("*.json"))}
