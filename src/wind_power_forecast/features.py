from __future__ import annotations

import math

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "wind_speed",
    "temperature",
    "hour_sin",
    "hour_cos",
    "dayofyear_sin",
    "dayofyear_cos",
    "month_sin",
    "month_cos",
]


def add_time_features(df: pd.DataFrame, timestamp_column: str = "timestamp") -> pd.DataFrame:
    result = df.copy()
    timestamp = pd.to_datetime(result[timestamp_column], errors="raise")
    if timestamp.isna().any():
        raise ValueError("Feature timestamps must not be missing.")
    result[timestamp_column] = timestamp

    hour_angle = 2 * math.pi * (timestamp.dt.hour + timestamp.dt.minute / 60) / 24
    days_in_year = np.where(timestamp.dt.is_leap_year, 366, 365)
    day_angle = 2 * math.pi * (timestamp.dt.dayofyear - 1) / days_in_year
    month_angle = 2 * math.pi * timestamp.dt.month / 12

    result["hour_sin"] = np.sin(hour_angle)
    result["hour_cos"] = np.cos(hour_angle)
    result["dayofyear_sin"] = np.sin(day_angle)
    result["dayofyear_cos"] = np.cos(day_angle)
    result["month_sin"] = np.sin(month_angle)
    result["month_cos"] = np.cos(month_angle)
    return result
