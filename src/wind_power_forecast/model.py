from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, add_time_features


@dataclass(frozen=True)
class ModelReport:
    """MAEs are in normalized-power units; validation is not NWP forecast skill."""

    model_name: str
    training_rows: int
    validation_rows: int
    validation_mae: float | None
    baseline_mae: float | None
    baseline_name: str
    candidate_mae: dict[str, float]
    validation_kind: str
    validation_training_rows: int
    training_started_at: str
    training_ended_at: str
    validation_training_ended_at: str
    validation_started_at: str
    validation_ended_at: str
    training_mean_power: float
    dropped_training_rows: int
    refit_on_all_history: bool
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class WindPowerCurveRegressor:
    """Learn a monotone wind-to-power curve without assuming rated wind speed."""

    def fit(self, features: pd.DataFrame, target: pd.Series):
        from sklearn.isotonic import IsotonicRegression

        self.curve_ = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        self.curve_.fit(features["wind_speed"], target)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.curve_.predict(features["wind_speed"])


def _training_data(hourly: pd.DataFrame) -> pd.DataFrame:
    required = ["timestamp", "wind_speed", "temperature", "normalized_power"]
    missing = set(required).difference(hourly.columns)
    if missing:
        raise ValueError(f"Training data is missing columns: {', '.join(sorted(missing))}")

    data = hourly[required].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    # A duplicate hour could otherwise appear on both sides of the time split.
    if data["timestamp"].dropna().duplicated().any():
        raise ValueError("Training timestamps must be unique; aggregate duplicate hours first.")
    for column in required[1:]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    valid = (
        data["timestamp"].notna()
        & np.isfinite(data[required[1:]]).all(axis=1)
        & data["wind_speed"].ge(0)
        & data["normalized_power"].between(0, 1)
    )
    return add_time_features(data.loc[valid].sort_values("timestamp").reset_index(drop=True))


def train_power_model(hourly: pd.DataFrame):
    """Select on the last historical block, then refit on all permitted history.

    The caller must restrict rows to information available at the issue time and
    exclude the held-out competition period. This function never reads labels
    or weather outside the supplied frame.
    """
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error
    from threadpoolctl import threadpool_limits

    data = _training_data(hourly)
    if len(data) < 48:
        raise ValueError("Need at least 48 valid hourly rows to train the forecasting model.")

    validation_rows = min(24 * 30, max(24, len(data) // 5))
    train = data.iloc[:-validation_rows]
    validation = data.iloc[-validation_rows:]
    baseline_name = "training_mean_normalized_power"
    candidates = {
        "isotonic_wind_power_curve": WindPowerCurveRegressor(),
        "sklearn.HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.04,
            l2_regularization=0.05,
            early_stopping=False,
            random_state=42,
        ),
        baseline_name: DummyRegressor(strategy="mean"),
    }
    scores: dict[str, float] = {}
    # Explicitly cap OpenMP: otherwise small hourly datasets can spend more time
    # coordinating threads than fitting, especially during repeated issue runs.
    with threadpool_limits(limits=1, user_api="openmp"):
        for name, candidate in candidates.items():
            candidate.fit(train[FEATURE_COLUMNS], train["normalized_power"])
            predicted = predict_power(candidate, validation)
            scores[name] = float(mean_absolute_error(validation["normalized_power"], predicted))

        selected_name = min(scores, key=scores.get)
        model = candidates[selected_name]
        model.fit(data[FEATURE_COLUMNS], data["normalized_power"])

    report = ModelReport(
        model_name=selected_name,
        training_rows=len(data),
        validation_rows=len(validation),
        validation_mae=scores[selected_name],
        baseline_mae=scores[baseline_name],
        baseline_name=baseline_name,
        candidate_mae=scores,
        validation_kind="chronological_observed_weather_holdout",
        validation_training_rows=len(train),
        training_started_at=data["timestamp"].iloc[0].isoformat(),
        training_ended_at=data["timestamp"].iloc[-1].isoformat(),
        validation_training_ended_at=train["timestamp"].iloc[-1].isoformat(),
        validation_started_at=validation["timestamp"].iloc[0].isoformat(),
        validation_ended_at=validation["timestamp"].iloc[-1].isoformat(),
        training_mean_power=float(data["normalized_power"].mean()),
        dropped_training_rows=int(len(hourly) - len(data)),
        refit_on_all_history=True,
        limitations=(
            (
                "Validation uses measured wind and temperature, not archived weather forecasts; "
                "it does not measure 24-48 hour operational forecasting skill."
            ),
            (
                "The same chronological holdout selects the model, so its MAE is a selection "
                "diagnostic rather than an independent test score."
            ),
            (
                "NWP wind height and grid location can differ from turbine measurements; "
                "no historical NWP-to-turbine bias calibration is fitted."
            ),
            (
                "The monotone power curve cannot represent high-wind cut-out or unobserved "
                "curtailment and outages."
            ),
        ),
    )
    return model, report


def predict_power(model, features: pd.DataFrame) -> pd.Series:
    from threadpoolctl import threadpool_limits

    required = {"timestamp", "wind_speed", "temperature"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Prediction data is missing columns: {', '.join(sorted(missing))}")
    matrix = add_time_features(features)
    for column in ("wind_speed", "temperature"):
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
    if not np.isfinite(matrix[FEATURE_COLUMNS]).all().all() or matrix["wind_speed"].lt(0).any():
        raise ValueError("Prediction features must be finite, with nonnegative wind speed.")
    if matrix.empty:
        return pd.Series(index=features.index, dtype=float, name="normalized_power")
    with threadpool_limits(limits=1, user_api="openmp"):
        prediction = np.asarray(model.predict(matrix[FEATURE_COLUMNS]), dtype=float)
    if prediction.ndim != 1 or len(prediction) != len(features) or not np.isfinite(prediction).all():
        raise ValueError("Model must return one finite normalized-power prediction per hour.")
    return pd.Series(
        prediction.clip(0, 1), index=features.index, name="normalized_power"
    )
