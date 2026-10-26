"""Clean, split by time, build features, compare models, export the winner for the browser."""

from __future__ import annotations

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tripdur import config, data, export
from tripdur.features import FEATURES, FeatureBuilder

LGB_PARAMS = {
    "objective": "regression", "learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 50,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "seed": config.SEED, "deterministic": True, "force_row_wise": True, "verbose": -1,
}


def scores(y_log, p_log) -> dict:
    true_s, pred_s = np.expm1(y_log), np.expm1(p_log)
    return {
        "rmsle": round(float(np.sqrt(np.mean((y_log - p_log) ** 2))), 4),
        "r2_log": round(float(r2_score(y_log, p_log)), 4),
        "mae_minutes": round(float(mean_absolute_error(true_s, pred_s) / 60), 3),
        "median_abs_error_minutes": round(float(np.median(np.abs(true_s - pred_s)) / 60), 3),
    }


def fit_lgb(X, y, inner):
    """Pick the number of rounds on the last two weeks of training, then refit on all of it."""
    probe = lgb.train(LGB_PARAMS, lgb.Dataset(X[inner], y[inner]), num_boost_round=5000,
                      valid_sets=[lgb.Dataset(X[~inner], y[~inner])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    rounds = int(round(probe.best_iteration * 1.1))
    return lgb.train(LGB_PARAMS, lgb.Dataset(X, y), num_boost_round=rounds), rounds


def run() -> dict:
    started = time.perf_counter()
    raw = data.load()
    df, cleaning = data.clean(raw)
    train, test = data.split(df)
    y_tr = np.log1p(train[config.TARGET].to_numpy(float))
    y_te = np.log1p(test[config.TARGET].to_numpy(float))

    fb = FeatureBuilder()
    X_tr = fb.fit_transform(train, y_tr)
    X_te = fb.transform(test)

    results, preds = [], {}
    others = {
        "Baseline (training mean)": DummyRegressor(),
        "Ridge regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        # the best configuration from notebooks/model.ipynb, on the new features
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, random_state=config.SEED),
    }
    for name, model in others.items():
        t = time.perf_counter()
        model.fit(X_tr, y_tr)
        preds[name] = model.predict(X_te)
        results.append({"model": name, **scores(y_te, preds[name]), "fit_seconds": round(time.perf_counter() - t, 1)})

    t = time.perf_counter()
    inner = (train.pickup_datetime < config.VALID_FROM).to_numpy()
    booster, rounds = fit_lgb(X_tr.to_numpy(), y_tr, inner)
    preds["LightGBM"] = booster.predict(X_te.to_numpy())
    results.append({"model": "LightGBM", **scores(y_te, preds["LightGBM"]), "fit_seconds": round(time.perf_counter() - t, 1)})
    results.sort(key=lambda r: r["rmsle"])

    # every June row, including the ones cleaning removes (how the Kaggle metric counts)
    raw_june = raw.loc[raw.pickup_datetime >= config.TEST_FROM].reset_index(drop=True)
    all_june = scores(np.log1p(raw_june[config.TARGET].to_numpy(float)), booster.predict(fb.transform(raw_june).to_numpy()))

    # the same pipeline on a random 80/20 split, to compare with notebooks/model.ipynb
    order = np.random.default_rng(config.SEED).permutation(len(df))
    cut = int(len(df) * 0.8)
    ra, rb = df.iloc[order[:cut]].reset_index(drop=True), df.iloc[order[cut:]].reset_index(drop=True)
    ya, yb = np.log1p(ra[config.TARGET].to_numpy(float)), np.log1p(rb[config.TARGET].to_numpy(float))
    fb_r = FeatureBuilder()
    Xa = fb_r.fit_transform(ra, ya)
    random_split = scores(yb, lgb.train(LGB_PARAMS, lgb.Dataset(Xa.to_numpy(), ya), num_boost_round=rounds)
                          .predict(fb_r.transform(rb).to_numpy()))

    p = preds["LightGBM"]
    true_min, abs_err = np.expm1(y_te) / 60, np.abs(np.expm1(y_te) - np.expm1(p)) / 60
    band = pd.cut(true_min, [0, 5, 10, 20, 40, 60, np.inf], labels=["<5", "5-10", "10-20", "20-40", "40-60", "60+"])
    frame = pd.DataFrame({"band": band, "hour": test.pickup_datetime.dt.hour, "abs": abs_err, "sq": (y_te - p) ** 2})
    by_band = [{"band": str(k), "trips": int(len(g)), "mae_minutes": round(float(g["abs"].mean()), 3),
                "rmsle": round(float(np.sqrt(g["sq"].mean())), 4)} for k, g in frame.groupby("band", observed=True)]
    by_hour = [{"hour": int(k), "trips": int(len(g)), "mae_minutes": round(float(g["abs"].mean()), 3),
                "rmsle": round(float(np.sqrt(g["sq"].mean())), 4)} for k, g in frame.groupby("hour")]
    gain = booster.feature_importance("gain")
    importance = sorted(({"feature": f, "gain_share": round(float(g / gain.sum()), 4)} for f, g in zip(FEATURES, gain)),
                        key=lambda d: -d["gain_share"])

    # browser model: the exact booster above plus the fitted feature state
    model = {**export.compact(booster), "params": fb.to_params(), "rounds": rounds}
    assert model["feature_names"] == [f"Column_{i}" for i in range(len(FEATURES))] or model["feature_names"] == FEATURES
    model["feature_names"] = FEATURES
    (config.SITE / "model").mkdir(parents=True, exist_ok=True)
    (config.SITE / "model" / "model.json").write_text(json.dumps(model, separators=(",", ":")))

    # parity fixture: raw inputs and the Python prediction for 300 June trips
    sample = test.sample(300, random_state=config.SEED)
    cols = ["pickup_datetime", "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude",
            "passenger_count", "vendor_id"]
    rows = sample[cols].assign(pickup_datetime=sample.pickup_datetime.dt.strftime("%Y-%m-%d %H:%M:%S"))
    fixture = {"rows": rows.to_dict(orient="records"),
               "features": X_te.loc[sample.index].to_numpy().tolist(),
               "prediction_log": booster.predict(X_te.loc[sample.index].to_numpy()).tolist()}
    (config.ROOT / "tests" / "fixtures").mkdir(parents=True, exist_ok=True)
    (config.ROOT / "tests" / "fixtures" / "parity.json").write_text(json.dumps(fixture))

    best = results[0]
    baseline = next(r for r in results if r["model"].startswith("Baseline"))
    report = {
        "split": f"time-based: train {train.pickup_datetime.min():%Y-%m-%d} to {train.pickup_datetime.max():%Y-%m-%d}, "
                 f"test {test.pickup_datetime.min():%Y-%m-%d} to {test.pickup_datetime.max():%Y-%m-%d}",
        "target": "log1p(trip_duration_seconds); RMSE on it is RMSLE",
        "cleaning": cleaning,
        "train_rows": int(len(train)), "test_rows": int(len(test)), "features": len(FEATURES),
        "lightgbm_rounds": rounds,
        "models": results,
        "best_model": best["model"],
        "best_rmsle_reduction_vs_baseline_pct": round((1 - best["rmsle"] / baseline["rmsle"]) * 100, 1),
        "best_on_all_june_rows_including_dirty": all_june,
        "best_on_random_80_20_split": random_split,
        "error_by_trip_length": by_band,
        "error_by_pickup_hour": by_hour,
        "feature_importance_gain": importance,
        "model_json_bytes": (config.SITE / "model" / "model.json").stat().st_size,
        "seconds": round(time.perf_counter() - started, 1),
    }
    config.REPORTS.mkdir(exist_ok=True)
    (config.REPORTS / "pipeline_metrics.json").write_text(json.dumps(report, indent=2))
    return report
