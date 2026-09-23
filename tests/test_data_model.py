from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from windagent.data import iso_utc, load_history
from windagent.model import effective_wind, predict, train_models


UTC = timezone.utc


class DataTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "history.csv"

    def write(self, content):
        self.path.write_text(content, encoding="utf-8-sig")
        return self.path

    def test_russian_columns_decimal_comma_mean_and_available_time(self):
        self.write(
            "Статистическое время;Турбина;Средняя скорость ветра, м/с;"
            "Средняя температура окружающей среды, °C;Нормализированная активная мощность на стороне линии\n"
            "2026-01-31T23:00:00+05:00;1;4,0;-10,0;0,2\n"
            "2026-01-31T23:30:00+05:00;1;6,0;-8,0;0,4\n"
            "2026-01-31T23:30:00+05:00;1;6,0;-8,0;0,4\n"
        )
        rows, report = load_history(self.path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["timestamp"], datetime(2026, 1, 31, 18, tzinfo=UTC))
        self.assertEqual(rows[0]["available_at"], datetime(2026, 1, 31, 18, 30, tzinfo=UTC))
        self.assertEqual(rows[0]["sample_count"], 2)
        self.assertAlmostEqual(rows[0]["power"], 0.3)
        self.assertEqual(rows[0]["wind_speed"], 5)
        self.assertEqual(report["identical_duplicates_removed"], 1)
        self.assertEqual(report["aggregation_rows_collapsed"], 1)

    def test_conflicting_duplicate_fails_with_line(self):
        self.write("timestamp,turbine_id,wind_speed,temperature,power\n"
                   "2026-01-01T00:00Z,1,4,0,0.2\n2026-01-01T00:00Z,1,4,0,0.3\n")
        with self.assertRaisesRegex(ValueError, "строка 3.*дубликат"):
            load_history(self.path)

    def test_nan_missing_and_invalid_ranges_are_not_clipped(self):
        for wind, temperature, power in (("nan", "0", ".2"), ("4", "inf", ".2"),
                                         ("-1", "0", ".2"), ("4", "-274", ".2"),
                                         ("4", "0", "1.1"), ("4", "0", "-.1"),
                                         ("", "0", ".2")):
            with self.subTest(wind=wind, temperature=temperature, power=power):
                self.write(f"timestamp,turbine_id,wind_speed,temperature,power\n"
                           f"2026-01-01T00:00Z,1,{wind},{temperature},{power}\n")
                with self.assertRaisesRegex(ValueError, "строка 2"):
                    load_history(self.path)

    def test_naive_timezone_is_explicit_and_gaps_are_not_imputed(self):
        self.write("timestamp,turbine_id,wind_speed,temperature,power\n"
                   "2026-01-01 05:00,1,4,0,20\n2026-01-01 07:00,1,5,0,30\n")
        with self.assertRaisesRegex(ValueError, "часового пояса"):
            load_history(self.path, timezone_offset_hours=None, power_scale=100)
        rows, report = load_history(self.path, timezone_offset_hours=5, power_scale=100)
        self.assertEqual(rows[0]["timestamp"], datetime(2026, 1, 1, tzinfo=UTC))
        self.assertEqual(rows[0]["power"], .2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(report["turbines"]["1"]["missing_hours"], 1)
        self.assertEqual(report["naive_timestamp_rows"], 2)

    def test_mapping_and_missing_columns(self):
        self.write("when,unit,w,t,p\n2026-01-01T00:00Z,2,4,0,0.2\n")
        with self.assertRaisesRegex(ValueError, "timestamp"):
            load_history(self.path)
        rows, _ = load_history(self.path, column_mapping={"timestamp": "when", "turbine_id": "unit",
                                                         "wind_speed": "w", "temperature": "t", "power": "p"})
        self.assertEqual(rows[0]["turbine_id"], "2")


def sample_rows(count=240, turbines=("1", "2")):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for index in range(count):
        for turbine_id in turbines:
            wind = 2 + ((index * 7) % 24) / 2
            temperature = -10 + index % 25
            corrected = effective_wind(wind, temperature)
            power = min(1.0, max(0.0, (corrected - 3) / 9))
            if turbine_id == "2":
                power *= .8
            rows.append({"timestamp": start + timedelta(hours=index), "turbine_id": turbine_id,
                         "wind_speed": wind, "temperature": temperature, "power": power})
    return rows


class ModelTests(unittest.TestCase):
    def test_future_values_cannot_change_fitted_model_or_holdout(self):
        rows = sample_rows()
        cutoff = rows[359]["timestamp"]
        first = train_models(rows, cutoff)
        changed = [dict(row) for row in rows]
        for row in changed:
            if row["timestamp"] > cutoff:
                row.update(wind_speed=float("nan"), temperature=float("inf"), power=-100)
        second = train_models(changed, cutoff)
        self.assertEqual(first, second)
        for diagnostic in first["diagnostics"].values():
            self.assertLess(diagnostic["fit_end"], diagnostic["holdout_start"])
            self.assertLess(diagnostic["fit_available_end"], diagnostic["holdout_start"])
            self.assertLessEqual(diagnostic["holdout_end"], iso_utc(cutoff))
        self.assertEqual(set(first["per_turbine"]), {"1", "2"})
        json.dumps(first, allow_nan=False)

    def test_subhourly_future_measurement_excludes_entire_aggregate(self):
        rows = sample_rows(80, ("1",))
        cutoff = rows[-1]["timestamp"]
        rows[-1]["available_at"] = cutoff + timedelta(minutes=30)
        trained = train_models(rows, cutoff)
        expected = train_models(rows[:-1], cutoff)
        self.assertEqual(trained["per_turbine"], expected["per_turbine"])
        self.assertEqual(trained["excluded_future_rows"], 1)
        self.assertEqual(trained["training_end"], iso_utc(cutoff - timedelta(hours=1)))

    def test_learned_curve_improves_constant_and_forecasts_stay_bounded(self):
        rows = sample_rows()
        cutoff = rows[-1]["timestamp"]
        models = train_models(rows, cutoff)
        weather = [{"turbine_id": turbine_id, "wind_speed": wind, "temperature": -10,
                    "valid_time": cutoff + timedelta(hours=index + 1)}
                   for index, wind in enumerate((0, 4, 7, 10, 15, 99)) for turbine_id in ("1", "2")]
        forecast = predict(models, weather)
        for turbine_id, diagnostic in models["diagnostics"].items():
            self.assertLess(diagnostic["selected"]["mae"], diagnostic["constant_baseline"]["mae"] / 2)
        for row in forecast:
            self.assertTrue(0 <= row["lower"] <= row["power_pred"] <= row["upper"] <= 1)
        self.assertTrue(forecast[-1]["wind_outside_training_range"])
        self.assertGreater(forecast[6]["power_pred"], forecast[7]["power_pred"])
        self.assertFalse(models["interval"]["calibrated"])

    def test_constant_zero_power_is_finite_and_flagged(self):
        rows = sample_rows(96, ("1",))
        for row in rows:
            row["power"] = 0
        cutoff = rows[-1]["timestamp"]
        models = train_models(rows, cutoff)
        self.assertEqual(models["per_turbine"]["1"]["kind"], "constant")
        self.assertEqual(models["diagnostics"]["1"]["selected"]["rmse"], 0)
        self.assertTrue(any("постоянная мощность" in warning for warning in models["warnings"]))
        self.assertEqual(predict(models, [{"turbine_id": "1", "valid_time": cutoff + timedelta(hours=1),
                                          "wind_speed": 5, "temperature": -20}])[0]["power_pred"], 0)

    def test_temperature_density_and_invalid_prediction(self):
        self.assertGreater(effective_wind(8, -20), effective_wind(8, 30))
        self.assertAlmostEqual(effective_wind(8, 15), 8)
        rows = sample_rows(96, ("1",))
        cutoff = rows[-1]["timestamp"]
        models = train_models(rows, cutoff)
        for invalid in (float("nan"), -101, 71):
            with self.assertRaises(ValueError):
                predict(models, [{"turbine_id": "1", "valid_time": cutoff + timedelta(hours=1),
                                  "wind_speed": 5, "temperature": invalid}])
        with self.assertRaisesRegex(ValueError, "позже"):
            predict(models, [{"turbine_id": "1", "valid_time": cutoff, "wind_speed": 5, "temperature": 0}])
        with self.assertRaisesRegex(ValueError, "турбины"):
            predict(models, [{"turbine_id": "3", "valid_time": cutoff + timedelta(hours=1),
                              "wind_speed": 5, "temperature": 0}])

    def test_insufficient_history_and_naive_cutoff_fail(self):
        rows = sample_rows(47, ("1",))
        with self.assertRaisesRegex(ValueError, "минимум 48"):
            train_models(rows, rows[-1]["timestamp"])
        with self.assertRaisesRegex(ValueError, "часовым поясом"):
            train_models(rows, datetime(2026, 1, 1))


if __name__ == "__main__":
    unittest.main()
