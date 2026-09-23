import sys
import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wind_power_forecast.evaluation import evaluate_forecasts
from wind_power_forecast.settings import (
    DataSettings,
    OutputSettings,
    Settings,
    TurbineSettings,
    WeatherSettings,
)


class FarmEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = Settings(
            project_timezone="UTC",
            data=DataSettings(
                turbine_1_csv=self.root / "turbine_1.csv",
                timestamp_column="timestamp", wind_speed_column="wind",
                target_column="power", temperature_column="temperature",
                expected_step_minutes=10, hourly_output_csv=self.root / "unused.csv",
            ),
            weather=WeatherSettings(
                endpoint="https://example.invalid", model="test", forecast_hours=48,
                timezone="UTC", daily_calculation_hour_utc=23, model_latency_hours=8,
                wind_speed_variable="wind", temperature_variable="temperature",
                hourly_variables=("wind", "temperature"),
            ),
            outputs=OutputSettings(
                forecast_dir=self.root / "forecasts", weather_dir=self.root / "weather",
                run_log_dir=self.root / "reports" / "runs",
            ),
            turbines=tuple(
                TurbineSettings(
                    id=f"turbine_{i}", latitude=43.6, longitude=78.5,
                    data_csv=self.root / f"turbine_{i}.csv", capacity_mw=None,
                    model_source="self", weight=weight,
                ) for i, weight in ((1, 1.0), (2, 3.0))
            ),
            test_start_date=date(2026, 2, 1), test_end_date=date(2026, 2, 1),
        )

    def write_forecasts(self):
        self.settings.outputs.forecast_dir.mkdir()
        has_capacity = all(t.capacity_mw is not None for t in self.settings.turbines)
        weights = [
            t.capacity_mw if has_capacity else t.weight for t in self.settings.turbines
        ]
        for day, predictions in (("2026-01-31", (0.4, 0.6)), ("2026-01-30", (0.2, 0.4))):
            frame = pd.DataFrame({
                "calculation_date": day,
                "issued_at": f"{day}T23:00:00Z",
                "target_time": pd.date_range("2026-02-01", periods=24, freq="h", tz="UTC"),
                "turbine_1_normalized_power": predictions[0],
                "turbine_2_normalized_power": predictions[1],
                "wind_farm_normalized_power": sum(
                    value * weight for value, weight in zip(predictions, weights)
                ) / sum(weights),
                "turbine_1_mean_baseline": 0.1, "turbine_2_mean_baseline": 0.3,
                "turbine_1_persistence_baseline": 0.3,
                "turbine_2_persistence_baseline": 0.9,
            })
            frame.to_csv(self.settings.outputs.forecast_dir / f"forecast_{day}.csv", index=False)

    def write_actual(self, turbine: int, hours: list[tuple[int, float, int]]):
        rows = []
        for hour, power, samples in hours:
            for minute in range(0, samples * 10, 10):
                rows.append({
                    "timestamp": f"2026-02-01 {hour:02d}:{minute:02d}:00",
                    "wind": 5, "power": power, "temperature": 10,
                })
        pd.DataFrame(rows).to_csv(self.root / f"turbine_{turbine}.csv", index=False)

    def test_configured_weights_score_farm_and_both_baselines_by_horizon(self):
        self.write_forecasts()
        self.write_actual(1, [(0, 0.2, 6), (1, 0.2, 6)])
        self.write_actual(2, [(0, 0.8, 6), (1, 0.8, 6)])
        report = evaluate_forecasts(self.settings)
        self.assertEqual(report["farm_aggregation"]["basis"], "configured_weights_assumed")
        self.assertEqual(report["farm_aggregation"]["normalized_weights"], {
            "turbine_1": 0.25, "turbine_2": 0.75,
        })
        for label, mae in (("day_ahead", 0.1), ("second_day", 0.3)):
            farm = report[label]["wind_farm_normalized_power"]
            self.assertEqual(farm["hours"], 2)
            self.assertAlmostEqual(farm["mae"], mae)
            self.assertAlmostEqual(farm["rmse"], mae)
            self.assertAlmostEqual(farm["bias"], -mae)
            self.assertAlmostEqual(farm["mean_baseline"]["mae"], 0.4)
            self.assertAlmostEqual(farm["persistence_baseline"]["mae"], 0.1)
            self.assertEqual(farm["mean_baseline"]["hours"], 2)

    def test_known_capacities_override_configured_weights(self):
        self.settings = replace(self.settings, turbines=tuple(
            replace(turbine, capacity_mw=capacity)
            for turbine, capacity in zip(self.settings.turbines, (3.0, 1.0))
        ))
        self.write_forecasts()
        self.write_actual(1, [(0, 0.2, 6)])
        self.write_actual(2, [(0, 0.8, 6)])
        report = evaluate_forecasts(self.settings)
        self.assertEqual(report["farm_aggregation"]["basis"], "capacity_weighted")
        self.assertEqual(report["farm_aggregation"]["normalized_weights"], {
            "turbine_1": 0.75, "turbine_2": 0.25,
        })
        first = report["day_ahead"]["wind_farm_normalized_power"]
        self.assertAlmostEqual(first["bias"], 0.1)
        self.assertAlmostEqual(first["mean_baseline"]["mae"], 0.2)
        self.assertAlmostEqual(first["persistence_baseline"]["mae"], 0.1)
        second = report["second_day"]["wind_farm_normalized_power"]
        self.assertAlmostEqual(second["bias"], -0.1)

    def test_only_common_complete_target_hours_are_scored(self):
        self.write_forecasts()
        self.write_actual(1, [(0, 0.2, 6), (1, 0.4, 6)])
        self.write_actual(2, [(0, 0.8, 5), (1, 0.8, 6), (2, 0.1, 6)])
        report = evaluate_forecasts(self.settings)
        self.assertEqual(report["day_ahead"]["turbine_1"]["hours"], 2)
        self.assertEqual(report["day_ahead"]["turbine_2"]["hours"], 2)
        for label, error in (("day_ahead", 0.15), ("second_day", 0.35)):
            farm = report[label]["wind_farm_normalized_power"]
            self.assertEqual(farm["hours"], 1)
            self.assertAlmostEqual(farm["mae"], error)
            self.assertEqual(farm["mean_baseline"]["hours"], 1)

    def test_missing_turbine_file_does_not_create_partial_farm_metrics(self):
        self.write_forecasts()
        self.write_actual(1, [(0, 0.2, 6)])
        report = evaluate_forecasts(self.settings)
        self.assertEqual(report["status"], "evaluated")
        self.assertEqual(report["day_ahead"]["turbine_1"]["hours"], 1)
        for label in ("day_ahead", "second_day"):
            self.assertEqual(report[label]["wind_farm_normalized_power"], {
                "hours": 0, "status": "no_actuals",
            })

    def test_disjoint_observed_hours_do_not_create_farm_metrics(self):
        self.write_forecasts()
        self.write_actual(1, [(0, 0.2, 6)])
        self.write_actual(2, [(1, 0.8, 6)])
        report = evaluate_forecasts(self.settings)
        for label in ("day_ahead", "second_day"):
            self.assertEqual(report[label]["wind_farm_normalized_power"], {
                "hours": 0, "status": "no_actuals",
            })

    def test_missing_baseline_component_is_not_replaced_with_zero(self):
        self.write_forecasts()
        self.write_actual(1, [(0, 0.2, 6)])
        self.write_actual(2, [(0, 0.8, 6)])
        for path in self.settings.outputs.forecast_dir.glob("*.csv"):
            frame = pd.read_csv(path).drop(columns="turbine_2_persistence_baseline")
            frame.to_csv(path, index=False)
        report = evaluate_forecasts(self.settings)
        for label in ("day_ahead", "second_day"):
            farm = report[label]["wind_farm_normalized_power"]
            self.assertEqual(farm["hours"], 1)
            self.assertEqual(farm["persistence_baseline"], {
                "hours": 0, "status": "incomplete_baseline",
            })
            self.assertAlmostEqual(farm["mean_baseline"]["mae"], 0.4)


if __name__ == "__main__":
    unittest.main()
