from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models.estimators import WindEstimator
from src.models.features import MODEL_FEATURES, enrich, split_data
from src.utils.common import load_config, resolve_path, setup_logging, sha256_file, utc, write_csv, write_json

LOG = logging.getLogger(__name__)


def scores(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"mae": float(np.mean(np.abs(actual - pred))),
            "rmse": float(np.sqrt(np.mean((actual - pred) ** 2))),
            "bias": float(np.mean(pred - actual))}


def radii(frame: pd.DataFrame, pred: np.ndarray, coverage: float) -> dict[str, float]:
    x = frame[["turbine_id", "lead_hours", "target_power_norm"]].copy()
    x["bucket"] = np.where(x.lead_hours <= 24, "01-24", "25-48")
    x["error"] = np.abs(x.target_power_norm - pred)
    result = {}
    for (turbine, bucket), subset in x.groupby(["turbine_id", "bucket"]):
        n = len(subset)
        q = min(1, math.ceil((n + 1) * coverage) / n)
        result[f"{turbine}/{bucket}"] = float(subset.error.quantile(q, interpolation="higher"))
    return result


def apply_intervals(frame: pd.DataFrame, pred: np.ndarray,
                    calibration: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    radius = np.array([calibration[f"{t}/{'01-24' if h <= 24 else '25-48'}"]
                       for t, h in zip(frame.turbine_id, frame.lead_hours)])
    return np.maximum(0, pred - radius), np.minimum(1, pred + radius)


def detailed_metrics(frame: pd.DataFrame, pred: np.ndarray, name: str,
                     lower: np.ndarray | None = None, upper: np.ndarray | None = None) -> list[dict[str, Any]]:
    x = frame[["turbine_id", "lead_hours", "target_power_norm"]].copy()
    x["prediction"] = pred
    x["bucket"] = np.where(x.lead_hours <= 24, "01-24", "25-48")
    if lower is not None:
        x["covered"] = (x.target_power_norm >= lower) & (x.target_power_norm <= upper)
        x["interval_width"] = upper - lower
    records = []
    groups = [(('ALL', 'ALL'), x), *list(x.groupby(["turbine_id", "bucket"]))]
    for (turbine, bucket), sub in groups:
        row = {"model": name, "turbine_id": turbine, "horizon": bucket,
               "n": len(sub), **scores(sub.target_power_norm.to_numpy(), sub.prediction.to_numpy())}
        if lower is not None:
            row.update(coverage=float(sub.covered.mean()), interval_width=float(sub.interval_width.mean()))
        records.append(row)
    return records


def train(cfg: dict[str, Any]) -> dict[str, Any]:
    settings = cfg["training"]
    source = resolve_path(cfg, settings["dataset"])
    x = enrich(pd.read_csv(source))
    fit_end = utc(settings["fit_end_exclusive"])
    val_end = utc(settings["validation_end_exclusive"])
    test_end = utc(settings["test_end_exclusive"])
    fit, val, test = split_data(x, fit_end, val_end, test_end)
    root = cfg["_root"]
    output = root / "artifacts"
    output.mkdir(exist_ok=True)
    (root / "reports").mkdir(exist_ok=True)
    specs = [{"name": "historical_mean", "kind": "mean"}, {"name": "weather_power_curve", "kind": "curve"}]
    for depth in settings["depths"]:
        for loss in ["RMSE", "MAE"]:
            specs.append({"name": f"catboost_d{depth}_{loss.lower()}", "kind": "catboost", "depth": depth,
                          "loss": loss, "iterations": settings["iterations"],
                          "early_stopping_rounds": settings["early_stopping_rounds"]})
    seed, threads = settings["random_seed"], settings["thread_count"]
    candidates: dict[str, WindEstimator] = {}
    predictions: dict[str, np.ndarray] = {}
    validation_rows = []
    for spec in specs:
        LOG.info("Fitting %s on %d rows, validating %d", spec["name"], len(fit), len(val))
        model = WindEstimator(spec, seed, threads).fit(fit, val)
        pred = model.predict(val)
        candidates[spec["name"]] = model
        predictions[spec["name"]] = pred
        validation_rows.append({"model": spec["name"], "iterations": model.iterations,
                                **scores(val.target_power_norm.to_numpy(), pred)})
    ranked = sorted(validation_rows, key=lambda r: (r["mae"], r["rmse"]))
    best = ranked[0]["model"]
    LOG.info("Winner selected only on December: %s MAE=%.6f", best, ranked[0]["mae"])
    selected_spec = candidates[best].spec.copy()
    selected_spec["iterations"] = candidates[best].iterations
    calibration_dec = radii(val, predictions[best], settings["interval_coverage"])
    # Models evaluated on January may use only labels available at Dec 31 23:00.
    jan_asof = val_end - pd.Timedelta(hours=1)
    jan_fit = x[x.target_power_norm.notna() & (x.valid_time + pd.Timedelta(hours=1) <= jan_asof)]
    jan_model = WindEstimator(selected_spec, seed, threads).fit(jan_fit)
    jan_pred = jan_model.predict(test)
    lower, upper = apply_intervals(test, jan_pred, calibration_dec)
    jan_rows = detailed_metrics(test, jan_pred, "selected_model", lower, upper)
    for spec in specs[:2]:
        baseline = WindEstimator(spec, seed, threads).fit(jan_fit)
        jan_rows += detailed_metrics(test, baseline.predict(test), spec["name"])
    # Frozen persistence: exactly the same no-new-telemetry protocol as February.
    persistence = {}
    for turbine in cfg["turbines"]:
        history = pd.read_csv(resolve_path(cfg, cfg["paths"]["processed"]) / f"{turbine['id']}_hourly.csv")
        history["valid_time"] = pd.to_datetime(history.valid_time, utc=True)
        past = history[history.power_norm.notna() & (history.valid_time + pd.Timedelta(hours=1) <= jan_asof)]
        persistence[turbine["id"]] = float(past.sort_values("valid_time").iloc[-1].power_norm)
    jan_rows += detailed_metrics(test, test.turbine_id.map(persistence).to_numpy(), "frozen_persistence")
    january = test[["turbine_id", "issue_time", "run_time", "valid_time", "lead_hours", "target_power_norm"]].copy()
    january["prediction"] = jan_pred
    january["lower"] = lower
    january["upper"] = upper
    write_csv(root / "reports/january_predictions.csv", january)
    write_csv(root / "reports/validation_leaderboard.csv", pd.DataFrame(validation_rows).sort_values("mae"))
    write_csv(root / "reports/january_metrics.csv", pd.DataFrame(jan_rows))
    final_asof = test_end - pd.Timedelta(hours=1)
    final_fit = x[x.target_power_norm.notna() & (x.valid_time + pd.Timedelta(hours=1) <= final_asof)]
    final_model = WindEstimator(selected_spec, seed, threads).fit(final_fit)
    calibration_mask = test.valid_time + pd.Timedelta(hours=1) <= final_asof
    calibration_jan = radii(test[calibration_mask], jan_pred[calibration_mask.to_numpy()], settings["interval_coverage"])
    # Save a weather curve as a separate diagnostic/fallback model.
    fallback = WindEstimator(specs[1], seed, threads).fit(final_fit)
    bundle = {"model": final_model, "fallback_model": fallback, "calibration": calibration_jan,
              "features": MODEL_FEATURES, "training_asof": final_asof.isoformat(),
              "weather_model": cfg["weather"]["model"], "selected_spec": selected_spec}
    joblib.dump(bundle, output / "forecast_bundle.joblib", compress=3)
    if final_model.cat is not None:
        final_model.cat.save_model(str(output / "catboost.cbm"))
        importance = pd.DataFrame({"feature": MODEL_FEATURES, "importance": final_model.cat.feature_importances_})
        write_csv(root / "reports/feature_importance.csv", importance.sort_values("importance", ascending=False))
    report = {"selected_model": best, "selection_metric": "December MAE", "selected_spec": selected_spec,
              "rows": {"fit": len(fit), "validation": len(val), "january_test": len(test), "final_fit": len(final_fit)},
              "boundaries": {"fit": str(fit_end), "validation": str(val_end), "test": str(test_end)},
              "january_metrics": jan_rows, "validation_leaderboard": ranked,
              "target_interval_coverage": settings["interval_coverage"], "calibration_january": calibration_jan,
              "dataset_sha256": sha256_file(source), "model_sha256": sha256_file(output / "forecast_bundle.joblib"),
              "final_training_asof": str(final_asof), "provenance_verified": bool(x.availability_verified.all()),
              "interval_note": "Empirical residual bands; coverage measured out of time, not guaranteed under drift."}
    write_json(root / "reports/training_report.json", report)
    LOG.info("January selected MAE %.6f; final model saved", jan_rows[0]["mae"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Select on December, test January, fit frozen February model")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    try:
        cfg = load_config(args.config)
        setup_logging(cfg)
        train(cfg)
        return 0
    except Exception:
        LOG.exception("Training failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
