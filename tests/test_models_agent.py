from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.agent.run import ForecastAgent
from src.agent.planner import decide
from src.data.build_features import add_features
from src.data.fetch_weather import WeatherUnavailableError
from src.models.features import enrich, split_data
from src.models.estimators import WindEstimator
from src.models.train import apply_intervals
from src.utils.common import load_config, utc


def sample_weather(issue: pd.Timestamp, times: pd.DatetimeIndex, turbine: str) -> pd.DataFrame:
    n = len(times)
    x = pd.DataFrame({"valid_time": times, "turbine_id": turbine, "issue_time": issue,
        "run_time": issue.floor("D") + pd.Timedelta(hours=12),
        "available_at": issue.floor("D") + pd.Timedelta(hours=20),
        "wind_speed_10m": np.linspace(2, 8, n), "wind_speed_100m": np.linspace(3, 10, n),
        "wind_direction_10m": 30.0, "wind_direction_100m": 40.0,
        "temperature_2m": 5.0, "surface_pressure": 930.0, "air_density": 1.16,
        "availability_verified": False, "availability_basis": "test", "weather_sha256": "test-digest",
        "lead_hours": np.arange(1, n + 1), "weather_lead_hours": np.arange(12, n + 12)})
    return x


class StubClient:
    def __init__(self, first_failure: bool = False) -> None:
        self.first_failure = first_failure
        self.calls = 0

    def fetch(self, turbine: dict[str, Any], issue: pd.Timestamp, times: pd.DatetimeIndex) -> pd.DataFrame:
        self.calls += 1
        if self.first_failure and self.calls == 1:
            raise WeatherUnavailableError("Synthetic provider outage")
        # Earlier requested issue must represent an earlier available run.
        x = sample_weather(issue, times, turbine["id"])
        x["run_time"] = (issue - pd.Timedelta(hours=8)).floor("6h")
        x["available_at"] = x.run_time + pd.Timedelta(hours=8)
        return x

    def close(self) -> None:
        return None


class ModelAgentTests(unittest.TestCase):
    def test_split_removes_overlapping_target_hours_and_future_labels(self) -> None:
        fit_end, val_end, test_end = map(utc, ["2025-12-01T00:00Z", "2026-01-01T00:00Z", "2026-02-01T00:00Z"])
        rows = []
        for day in pd.date_range("2025-11-25", "2026-01-31", tz="UTC"):
            issue = day + pd.Timedelta(hours=23)
            for target in pd.date_range(issue + pd.Timedelta(hours=1), periods=48, freq="h"):
                rows.append({"issue_time": issue, "valid_time": target, "target_power_norm": 0.5})
        train, val, test = split_data(pd.DataFrame(rows), fit_end, val_end, test_end)
        self.assertLessEqual((train.valid_time + pd.Timedelta(hours=1)).max(), val.issue_time.min())
        self.assertFalse(set(val.valid_time) & set(test.valid_time))
        self.assertLessEqual((val.valid_time + pd.Timedelta(hours=1)).max(), test.issue_time.min())
        self.assertGreaterEqual(test.issue_time.min(), val_end - pd.Timedelta(hours=1))

    def test_agent_fallback_and_idempotence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            cfg = load_config(Path(__file__).resolve().parents[1] / "config/config.yaml")
            cfg["_root"] = Path(temp)
            (Path(temp) / "artifacts").mkdir()
            issue = utc("2026-01-31T23:00Z")
            times = pd.date_range(issue + pd.Timedelta(hours=1), periods=48, freq="h")
            frame = enrich(add_features(pd.concat([sample_weather(issue, times, t) for t in ["T1", "T2"]])))
            frame["target_power_norm"] = 0.18603061746927146
            model = WindEstimator({"kind": "mean"}).fit(frame)
            calibration = {f"{t}/{b}": 0.1 for t in ["T1", "T2"] for b in ["01-24", "25-48"]}
            joblib.dump({"model": model, "calibration": calibration, "training_asof": str(issue),
                         "weather_model": "ecmwf_ifs"}, Path(temp) / "artifacts/forecast_bundle.joblib")
            stub = StubClient(first_failure=True)
            with patch("src.agent.run.WeatherClient", return_value=stub):
                agent = ForecastAgent(cfg)
                first = agent.run(issue)
                self.assertEqual(len(first), 96)
                self.assertEqual(first.fallback_age_hours.max(), 6)
                self.assertTrue(first.power_norm.between(0, 1).all())
                self.assertTrue((first.lower_80 <= first.power_norm).all())
                # Same run/input successfully fetched next time and then cached idempotently.
                second = agent.run(issue)
                third = agent.run(issue)
                self.assertEqual(second.forecast_id.iloc[0], third.forecast_id.iloc[0])
                np.testing.assert_array_equal(second.power_norm.to_numpy(), third.power_norm.to_numpy())
                with sqlite3.connect(agent.database) as db:
                    events = [r[0] for r in db.execute("SELECT state FROM events")]
                self.assertIn("REPLAN", events)
                self.assertIn("UNCHANGED", events)
                with self.assertRaisesRegex(ValueError, "not available"):
                    agent.run(issue - pd.Timedelta(days=1))
                agent.close()

    def test_llm_key_is_explicit(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                decide({"weather_failed": False}, use_llm=True)
        self.assertEqual(decide({"weather_failed": True})["action"], "retry_older")

    def test_integrity_and_internal_failures_halt_without_older_run(self) -> None:
        class FailingClient:
            def __init__(self, error: Exception) -> None:
                self.error = error
                self.calls = 0

            def fetch(self, *args: Any) -> pd.DataFrame:
                self.calls += 1
                raise self.error

            def close(self) -> None:
                return None

        for error in (ValueError("Corrupt weather cache entry"),
                      ValueError("Unexpected weather unit"),
                      ValueError("Weather unavailable at issue"),
                      RuntimeError("Unexpected internal failure")):
            with self.subTest(error=str(error)), tempfile.TemporaryDirectory() as temp:
                cfg = load_config(Path(__file__).resolve().parents[1] / "config/config.yaml")
                cfg["_root"] = Path(temp)
                (Path(temp) / "artifacts").mkdir()
                issue = utc("2026-01-31T23:00Z")
                # This fixture is created locally. No supplied/archive joblib
                # is loaded, and forecasting must stop before touching a model.
                joblib.dump({"model": None, "calibration": {}, "training_asof": str(issue),
                             "weather_model": "ecmwf_ifs"}, Path(temp) / "artifacts/forecast_bundle.joblib")
                client = FailingClient(error)
                with patch("src.agent.run.WeatherClient", return_value=client):
                    agent = ForecastAgent(cfg)
                    try:
                        with self.assertRaises(type(error)):
                            agent.run(issue)
                        self.assertEqual(client.calls, 1)
                        with sqlite3.connect(agent.database) as db:
                            states = [row[0] for row in db.execute("SELECT state FROM events")]
                            published = db.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
                        self.assertIn("FAILED", states)
                        self.assertNotIn("REPLAN", states)
                        self.assertEqual(published, 0)
                    finally:
                        agent.close()


if __name__ == "__main__":
    unittest.main()
