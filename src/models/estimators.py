from __future__ import annotations

from typing import Any

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.models.features import MODEL_FEATURES


class WindEstimator:
    """Serializable turbine model: mean, isotonic curve, or CPU CatBoost."""

    def __init__(self, spec: dict[str, Any], seed: int = 42, threads: int = 4) -> None:
        self.spec = spec.copy()
        self.seed = seed
        self.threads = threads
        self.means: dict[str, float] = {}
        self.curves: dict[str, IsotonicRegression] = {}
        self.cat: CatBoostRegressor | None = None
        self.iterations = int(spec.get("iterations", 600))

    def fit(self, frame: pd.DataFrame, validation: pd.DataFrame | None = None) -> WindEstimator:
        if frame.target_power_norm.isna().any() or not frame.target_power_norm.between(0, 1).all():
            raise ValueError("Model fit requires valid normalized targets")
        self.means = frame.groupby("turbine_id").target_power_norm.mean().to_dict()
        kind = self.spec["kind"]
        if kind == "curve":
            for turbine, subset in frame.groupby("turbine_id"):
                self.curves[turbine] = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(
                    subset.density_adjusted_wind, subset.target_power_norm)
        elif kind == "catboost":
            self.cat = CatBoostRegressor(
                iterations=self.iterations, depth=int(self.spec["depth"]),
                loss_function=self.spec.get("loss", "RMSE"), eval_metric="MAE",
                learning_rate=0.04, l2_leaf_reg=8, random_seed=self.seed,
                thread_count=self.threads, allow_writing_files=False, verbose=False)
            weight = 1 / frame.groupby(["turbine_id", "valid_time"]).target_power_norm.transform("size")
            extra: dict[str, Any] = {}
            if validation is not None:
                extra = {"eval_set": (validation[MODEL_FEATURES], validation.target_power_norm),
                         "early_stopping_rounds": int(self.spec.get("early_stopping_rounds", 80)),
                         "use_best_model": True}
            self.cat.fit(frame[MODEL_FEATURES], frame.target_power_norm,
                         cat_features=["turbine_id"], sample_weight=weight, **extra)
            self.iterations = self.cat.tree_count_
        elif kind != "mean":
            raise ValueError(f"Unknown estimator {kind}")
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if set(frame.turbine_id) - set(self.means):
            raise ValueError("Unseen turbine")
        if self.spec["kind"] == "catboost":
            if self.cat is None:
                raise RuntimeError("Estimator has not been fitted")
            pred = self.cat.predict(frame[MODEL_FEATURES])
        elif self.spec["kind"] == "curve":
            pred = np.array([self.curves[t].predict([v])[0] for t, v in
                             zip(frame.turbine_id, frame.density_adjusted_wind)])
        else:
            pred = frame.turbine_id.map(self.means).to_numpy()
        return np.clip(pred, 0, 1)
