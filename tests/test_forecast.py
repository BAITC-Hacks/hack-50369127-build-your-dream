import json
import sys
import unittest
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wind_power_forecast.evaluation import compile_submission, evaluate_forecasts
from wind_power_forecast.forecast import ForecastAgent, forecast_window
from wind_power_forecast.settings import (
    DataSettings,
    OutputSettings,
    Settings,
    TurbineSettings,
    WeatherSettings,
)


@dataclass
class ConstantModel:
    value: float

    def predict(self, features):
        return np.full(len(features), self.value)


@dataclass
class StubReport:
    training_mean_power: float

    def to_dict(self):
        return {"training_mean_power": self.training_mean_power}


class ForecastAgentTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = Settings(
            project_timezone="Asia/Almaty",
            data=DataSettings(
                turbine_1_csv=self.root / "turbine_1.csv",
                timestamp_column="timestamp",
                wind_speed_column="wind_speed",
                target_column="normalized_power",
                temperature_column="temperature",
                expected_step_minutes=10,
                hourly_output_csv=self.root / "hourly.csv",
            ),
            weather=WeatherSettings(
                endpoint="https://example.invalid/forecast",
                model="test_model",
                forecast_hours=168,
                timezone="UTC",
                daily_calculation_hour_utc=18,
                model_latency_hours=8,
                wind_speed_variable="wind_speed_100m",
                temperature_variable="temperature_2m",
                hourly_variables=("wind_speed_100m", "temperature_2m"),
            ),
            outputs=OutputSettings(
                forecast_dir=self.root / "forecasts",
                weather_dir=self.root / "weather",
                run_log_dir=self.root / "reports" / "runs",
            ),
            turbines=tuple(
                TurbineSettings(
                    id=f"turbine_{i}", latitude=43.6, longitude=78.5,
                    data_csv=self.root / f"turbine_{i}.csv", capacity_mw=None,
                    model_source="self", weight=weight,
                )
                for i, weight in ((1, 1.0), (2, 3.0))
            ),
        )
        for i, january_power in ((1, 0.2), (2, 0.8)):
            timestamps = pd.date_range("2026-01-29", "2026-02-03", freq="10min", inclusive="left")
            pd.DataFrame({
                "timestamp": timestamps,
                "wind_speed": 5.0,
                "temperature": 10.0,
                # Deliberately different February labels reveal accidental reuse.
                "normalized_power": np.where(
                    timestamps.month == 1, january_power, 1.0 if i == 1 else 0.0
                ),
            }).to_csv(self.root / f"turbine_{i}.csv", index=False)
        self.trained = []
        self.weather_version = "original"
        training_patch = patch(
            "wind_power_forecast.forecast.train_power_model", side_effect=self.train_stub
        )
        weather_patch = patch(
            "wind_power_forecast.forecast.get_weather_for_forecast", side_effect=self.weather_stub
        )
        self.train_mock = training_patch.start()
        self.weather_mock = weather_patch.start()
        self.addCleanup(training_patch.stop)
        self.addCleanup(weather_patch.stop)

    def train_stub(self, hourly):
        self.trained.append(hourly.copy())
        mean = float(hourly["normalized_power"].mean())
        return ConstantModel(mean), StubReport(mean)

    def weather_stub(self, weather, turbine, issued_at, targets, cache_dir, **kwargs):
        issued = pd.Timestamp(issued_at).tz_convert("UTC")
        run = issued.floor("D") + pd.Timedelta(hours=6)
        return pd.DataFrame({
            "timestamp": targets.tz_convert("UTC"),
            "wind_speed": 5.0,
            "temperature": 10.0,
        }), {
            "run": run.isoformat(),
            "payload_sha256": f"{turbine.id}-{self.weather_version}",
            "request_url": f"https://example.invalid/{turbine.id}",
            "assumed_available_at": (run + pd.Timedelta(hours=8)).isoformat(),
        }

    def test_separate_turbines_exclude_unfinished_hour_and_all_february_labels(self):
        agent = ForecastAgent(self.settings)
        january = agent.run(date(2026, 1, 31))
        self.assertEqual(len(self.trained), 2)
        for training, expected_power in zip(self.trained, (0.2, 0.8)):
            self.assertEqual(len(training), 71)
            self.assertEqual(training["timestamp"].max().hour, 22)
            self.assertEqual(training["available_at"].max().hour, 23)
            np.testing.assert_allclose(training["normalized_power"], expected_power)
        self.assertEqual(january.checks["training_cutoff"], "2026-01-31T23:00:00+05:00")

        february = agent.run(date(2026, 2, 2))
        self.assertEqual(len(self.trained), 4)
        for training, expected_power in zip(self.trained[2:], (0.2, 0.8)):
            self.assertEqual(len(training), 72)
            self.assertTrue((training["timestamp"].dt.month == 1).all())
            self.assertEqual(training["timestamp"].max().hour, 23)
            np.testing.assert_allclose(training["normalized_power"], expected_power)
        self.assertEqual(february.checks["training_cutoff"], "2026-02-01T00:00:00+05:00")
        self.assertFalse(february.checks["test_labels_used_in_training"])

    def test_strict_mode_rejects_unverified_hindcast_before_training_or_publish(self):
        settings = replace(self.settings, weather=replace(self.settings.weather, require_as_issued=True))
        with self.assertRaisesRegex(ValueError, "original operational forecasts"):
            ForecastAgent(settings).run(date(2026, 1, 31))
        self.assertEqual(len(self.trained), 0)
        self.assertFalse(settings.outputs.forecast_dir.exists())

    def test_48_exact_future_hours_and_configured_weight_aggregation(self):
        report = ForecastAgent(self.settings).run(date(2026, 1, 31))
        output = pd.read_csv(report.forecast_csv)
        timestamps = pd.to_datetime(output["target_time"], utc=True)
        expected = pd.date_range("2026-01-31T19:00Z", periods=48, freq="h")
        self.assertEqual(timestamps.tolist(), expected.tolist())
        self.assertEqual(output["forecast_hour"].tolist(), list(range(1, 49)))
        np.testing.assert_allclose(output["lead_hours"], np.arange(1, 49))
        np.testing.assert_allclose(output["wind_farm_normalized_power"], 0.65)
        self.assertNotIn("wind_farm_power_mw", output)
        self.assertEqual(report.checks["aggregation"], "configured_weights_assumed")

    def test_known_capacity_aggregation_and_hourly_energy(self):
        settings = replace(self.settings, turbines=tuple(
            replace(turbine, capacity_mw=capacity)
            for turbine, capacity in zip(self.settings.turbines, (3.0, 1.0))
        ))
        report = ForecastAgent(settings).run(date(2026, 1, 31))
        output = pd.read_csv(report.forecast_csv)
        np.testing.assert_allclose(output["wind_farm_normalized_power"], 0.35)
        np.testing.assert_allclose(output["wind_farm_power_mw"], 1.4)
        np.testing.assert_allclose(output["wind_farm_energy_mwh"], 1.4)
        self.assertEqual(report.checks["aggregation"], "capacity_weighted")

    def test_unchanged_reuses_outputs_and_updated_weather_recalculates(self):
        agent = ForecastAgent(self.settings, refresh=True)
        first = agent.run(date(2026, 1, 31))
        original_bytes = Path(first.forecast_csv).read_bytes()
        repeated = agent.run(date(2026, 1, 31))
        self.assertEqual(repeated.status, "unchanged")
        self.assertEqual(first.fingerprint, repeated.fingerprint)
        self.assertEqual(original_bytes, Path(repeated.forecast_csv).read_bytes())
        self.assertEqual(self.train_mock.call_count, 2)

        self.weather_version = "revised"
        updated = agent.run(date(2026, 1, 31))
        self.assertEqual(updated.status, "calculated")
        self.assertNotEqual(first.fingerprint, updated.fingerprint)
        self.assertEqual(self.train_mock.call_count, 2)
        revisions = list((self.settings.outputs.forecast_dir / "revisions").glob("*.csv"))
        self.assertEqual(len(revisions), 2)
        events = [json.loads(line)["action"] for line in (
            self.settings.outputs.run_log_dir / "agent_events.jsonl"
        ).read_text(encoding="utf-8").splitlines()]
        self.assertIn("reuse_unchanged", events)
        self.assertEqual(events.count("publish"), 2)

    def test_test_label_changes_do_not_change_forecast_or_retrain(self):
        agent = ForecastAgent(self.settings)
        first = agent.run(date(2026, 2, 2))
        source = self.settings.turbines[0].data_csv
        changed = pd.read_csv(source)
        february = pd.to_datetime(changed["timestamp"]).dt.month == 2
        changed.loc[february, "normalized_power"] = 0.12345
        changed.to_csv(source, index=False)
        repeated = agent.run(date(2026, 2, 2))
        self.assertEqual(first.fingerprint, repeated.fingerprint)
        self.assertEqual(repeated.status, "unchanged")
        self.assertEqual(self.train_mock.call_count, 2)

    def test_modified_output_is_rebuilt_even_when_inputs_are_unchanged(self):
        agent = ForecastAgent(self.settings)
        first = agent.run(date(2026, 1, 31))
        output_path = Path(first.forecast_csv)
        original = output_path.read_bytes()
        output_path.write_text("corrupt forecast", encoding="utf-8")
        repeated = agent.run(date(2026, 1, 31))
        self.assertEqual(repeated.status, "calculated")
        self.assertEqual(output_path.read_bytes(), original)

    def test_duplicate_weather_hour_fails_before_training_or_publication(self):
        def duplicated(*args, **kwargs):
            frame, metadata = self.weather_stub(*args, **kwargs)
            frame.loc[47, "timestamp"] = frame.loc[46, "timestamp"]
            return frame, metadata

        self.weather_mock.side_effect = duplicated
        with self.assertRaisesRegex(ValueError, "exact forecast grid"):
            ForecastAgent(self.settings).run(date(2026, 1, 31))
        self.assertEqual(self.train_mock.call_count, 0)
        self.assertFalse((self.settings.outputs.forecast_dir / "forecast_2026-01-31.csv").exists())
        self.assertTrue((self.settings.outputs.run_log_dir / "failed_2026-01-31.json").exists())

    def write_test_forecasts(self):
        self.settings.outputs.forecast_dir.mkdir(parents=True, exist_ok=True)
        for offset in range(28):
            day = date(2026, 1, 31) + timedelta(days=offset)
            issued_at, targets = forecast_window(self.settings, day)
            frame = pd.DataFrame({
                "calculation_date": str(day),
                "issued_at": issued_at.tz_convert("UTC").isoformat(),
                "target_time": targets,
                "forecast_hour": np.arange(1, 49),
                "turbine_1_normalized_power": 0.2,
                "turbine_2_normalized_power": 0.8,
                "wind_farm_normalized_power": 0.65,
                "turbine_1_mean_baseline": 0.3,
                "turbine_2_mean_baseline": 0.7,
                "turbine_1_persistence_baseline": 0.1,
                "turbine_2_persistence_baseline": 0.9,
            })
            frame.to_csv(self.settings.outputs.forecast_dir / f"forecast_{day}.csv", index=False)

    def test_submission_has_exactly_672_unique_day_ahead_hours(self):
        self.write_test_forecasts()
        submission, all_forecasts = compile_submission(self.settings)
        self.assertEqual(len(submission), 672)
        self.assertTrue(submission["target_time"].is_unique)
        self.assertEqual(submission["horizon_day"].unique().tolist(), [1])
        self.assertEqual(len(all_forecasts[all_forecasts["horizon_day"] == 2]), 648)
        self.assertTrue((self.settings.outputs.forecast_dir / "submission.csv").exists())

    def test_incomplete_test_period_is_rejected(self):
        self.write_test_forecasts()
        (self.settings.outputs.forecast_dir / "forecast_2026-02-14.csv").unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete test period"):
            compile_submission(self.settings)

    def test_no_actuals_has_explicit_status_for_both_horizons(self):
        self.write_test_forecasts()
        report = evaluate_forecasts(self.settings, actual_dir=self.root / "missing_actuals")
        self.assertEqual(report["status"], "unavailable_no_test_actuals")
        self.assertEqual(report["expected_hours"], 672)
        for horizon in ("day_ahead", "second_day"):
            for turbine in self.settings.turbines:
                self.assertEqual(report[horizon][turbine.id], {"hours": 0, "status": "no_actuals"})

    def test_evaluation_scores_only_complete_observed_hours(self):
        self.write_test_forecasts()
        actual_dir = self.root / "actuals"
        actual_dir.mkdir()
        timestamps = pd.date_range("2026-02-01", periods=11, freq="10min")
        pd.DataFrame({
            "timestamp": timestamps, "wind_speed": 5.0, "temperature": 10.0,
            "normalized_power": 0.5,
        }).to_csv(actual_dir / "turbine_1.csv", index=False)
        report = evaluate_forecasts(self.settings, actual_dir=actual_dir)
        first = report["day_ahead"]["turbine_1"]
        self.assertEqual(first["hours"], 1)
        self.assertAlmostEqual(first["mae"], 0.3)
        self.assertAlmostEqual(first["mean_baseline"]["mae"], 0.2)
        self.assertAlmostEqual(first["persistence_baseline"]["mae"], 0.4)

    def test_out_of_range_predictions_are_rejected_before_evaluation(self):
        self.write_test_forecasts()
        path = self.settings.outputs.forecast_dir / "forecast_2026-01-31.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "turbine_1_normalized_power"] = 1.5
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "prediction"):
            compile_submission(self.settings)

    def test_second_day_nonfinite_prediction_is_rejected(self):
        self.write_test_forecasts()
        path = self.settings.outputs.forecast_dir / "forecast_2026-01-31.csv"
        frame = pd.read_csv(path)
        frame.loc[24, "turbine_1_normalized_power"] = np.nan
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "prediction"):
            compile_submission(self.settings)

    def test_duplicate_second_day_issue_target_is_rejected(self):
        self.write_test_forecasts()
        path = self.settings.outputs.forecast_dir / "forecast_2026-01-31.csv"
        frame = pd.read_csv(path)
        pd.concat([frame, frame.iloc[[24]]]).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            compile_submission(self.settings)

    def test_calculation_date_must_agree_with_actual_local_issue_date(self):
        self.write_test_forecasts()
        path = self.settings.outputs.forecast_dir / "forecast_2026-01-31.csv"
        frame = pd.read_csv(path)
        frame["issued_at"] = "2026-01-30T18:00:00+00:00"
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "issue"):
            compile_submission(self.settings)

    def test_existing_run_manifest_checksum_mismatch_is_rejected(self):
        self.write_test_forecasts()
        self.settings.outputs.run_log_dir.mkdir(parents=True, exist_ok=True)
        (self.settings.outputs.run_log_dir / "run_2026-01-31.json").write_text(
            json.dumps({"checks": {"forecast_sha256": "0" * 64}}), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "checksum|hash|integrity"):
            compile_submission(self.settings)


if __name__ == "__main__":
    unittest.main()
