from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.build_features import FEATURE_COLUMNS

PHYSICS_COLUMNS = ["wind100_cubed", "wind10_cubed", "density_adjusted_wind",
                   "wind_shear_ratio", "wind_speed_difference", "wind100_u", "wind100_v",
                   "forecast_wind_delta", "forecast_wind_mean3", "dayofyear_sin", "dayofyear_cos"]
MODEL_FEATURES = FEATURE_COLUMNS + PHYSICS_COLUMNS
TIME_COLUMNS = ["valid_time", "issue_time", "run_time", "available_at"]


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    """Only information in the selected weather release enters model features."""
    x = frame.copy()
    for column in TIME_COLUMNS:
        x[column] = pd.to_datetime(x[column], utc=True)
    x["turbine_id"] = x["turbine_id"].astype(str)
    x = x.sort_values(["turbine_id", "issue_time", "valid_time"]).reset_index(drop=True)
    x["wind100_cubed"] = x["wind_speed_100m"] ** 3
    x["wind10_cubed"] = x["wind_speed_10m"] ** 3
    x["density_adjusted_wind"] = x["wind_speed_100m"] * (x["air_density"] / 1.225) ** (1 / 3)
    x["wind_shear_ratio"] = x["wind_speed_100m"] / (x["wind_speed_10m"] + 0.5)
    x["wind_speed_difference"] = x["wind_speed_100m"] - x["wind_speed_10m"]
    radians = np.deg2rad(x["wind_direction_100m"])
    x["wind100_u"] = -x["wind_speed_100m"] * np.sin(radians)
    x["wind100_v"] = -x["wind_speed_100m"] * np.cos(radians)
    groups = x.groupby(["turbine_id", "issue_time"])["wind_speed_100m"]
    x["forecast_wind_delta"] = groups.diff().fillna(0)
    x["forecast_wind_mean3"] = groups.transform(lambda s: s.rolling(3, min_periods=1).mean())
    angle = 2 * np.pi * (x["valid_time"].dt.dayofyear - 1) / 365.25
    x["dayofyear_sin"], x["dayofyear_cos"] = np.sin(angle), np.cos(angle)
    if not np.isfinite(x[[c for c in MODEL_FEATURES if c != "turbine_id"]]).all().all():
        raise ValueError("Non-finite predictor in feature allowlist")
    return x


def split_data(frame: pd.DataFrame, fit_boundary: pd.Timestamp,
               validation_boundary: pd.Timestamp, test_boundary: pd.Timestamp
               ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Purged split: a model can only use labels available at its first issue."""
    fit_asof = fit_boundary - pd.Timedelta(hours=1)
    test_asof = validation_boundary - pd.Timedelta(hours=1)
    known = frame[frame.target_power_norm.notna()].copy()
    train = known[known.valid_time + pd.Timedelta(hours=1) <= fit_asof]
    validation = known[(known.valid_time >= fit_boundary) &
                       (known.valid_time + pd.Timedelta(hours=1) <= test_asof) &
                       (known.issue_time >= fit_asof)]
    test = known[(known.valid_time >= validation_boundary) &
                 (known.valid_time < test_boundary) & (known.issue_time >= test_asof)]
    for left, right in [(train, validation), (train, test), (validation, test)]:
        if set(left.valid_time) & set(right.valid_time):
            raise ValueError("Target hours overlap between temporal splits")
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("One temporal split is empty; collect historical weather first")
    return train, validation, test
