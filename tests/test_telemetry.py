from __future__ import annotations

import csv
import hashlib
import io
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from windagent.data import load_history
from windagent.model import train_models
from windagent.telemetry import history_to_csv, load_turbine_files


UTC = timezone.utc
HEADERS = ["ID", "Статистическое время", "Средняя скорость ветра(m/s)",
           "Нормализованная активная мощность", "Средняя температура окружающей среды(°C)"]


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def source(self, name="turbine.csv", hours=2, convention="start", omitted=(), power="0"):
        path = self.folder / name
        start = datetime(2026, 1, 1)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(HEADERS)
            for index in range(hours * 6):
                if index in omitted:
                    continue
                at = start + timedelta(minutes=10 * (index + int(convention == "end")))
                # Match the original export: hours have no leading zero.
                stamp = at.strftime("%Y-%m-%d ") + str(at.hour) + at.strftime(":%M:%S")
                writer.writerow([index + 1, stamp, 5 + index % 6, power, -10 + index % 6])
        return path

    def test_source_id_is_not_turbine_and_zero_power_is_preserved(self):
        one = self.source("one.csv", power="0")
        two = self.source("two.csv", power="0.5")
        rows, report = load_turbine_files({"1": one, "2": two})
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["turbine_id"] for row in rows}, {"1", "2"})
        self.assertEqual([row["power"] for row in rows if row["turbine_id"] == "1"], [0, 0])
        self.assertTrue(all(row["sample_count"] == 6 for row in rows))
        self.assertEqual(report["turbines"]["1"]["source_sha256"], hashlib.sha256(one.read_bytes()).hexdigest())
        self.assertEqual(report["turbines"]["1"]["raw_rows"], 12)
        self.assertEqual(report["turbines"]["1"]["raw_first_timestamp"], "2026-01-01 0:00:00")
        self.assertEqual(rows[0]["wind_speed"], 7.5)

    def test_partial_hour_is_excluded_without_imputation(self):
        rows, report = load_turbine_files({"1": self.source(hours=3, omitted=(7, 8))})
        self.assertEqual([row["timestamp"].hour for row in rows], [0, 2])
        turbine = report["turbines"]["1"]
        self.assertEqual(turbine["partial_hours_excluded"], 1)
        self.assertEqual(turbine["missing_10minute_slots"], 2)
        self.assertEqual(turbine["rejected_hours"][0]["reasons"], ["incomplete_hour"])
        with self.assertRaisesRegex(ValueError, "min_samples=6"):
            load_turbine_files({"1": self.source()}, min_samples=5)

    def test_interval_end_convention_matches_start_and_hour_end_availability(self):
        start_rows, _ = load_turbine_files({"1": self.source("start.csv", convention="start")})
        end_rows, _ = load_turbine_files({"1": self.source("end.csv", convention="end")}, interval_convention="end")
        self.assertEqual(start_rows, end_rows)
        self.assertTrue(all(row["available_at"] == row["timestamp"] + timedelta(hours=1) for row in start_rows))

    def test_timezone_is_an_explicit_assumption_not_a_guess(self):
        source = self.source()
        zero, report_zero = load_turbine_files({"1": source})
        five, report_five = load_turbine_files({"1": source}, timezone_offset_hours=5)
        self.assertEqual(zero[0]["timestamp"], datetime(2026, 1, 1, tzinfo=UTC))
        self.assertEqual(five[0]["timestamp"], zero[0]["timestamp"] - timedelta(hours=5))
        self.assertEqual(report_zero["timezone_offset_hours"], 0)
        self.assertEqual(report_five["timezone_offset_hours"], 5)
        self.assertFalse(report_zero["timezone_confirmed"])
        self.assertFalse(report_five["interval_convention_confirmed"])
        with self.assertRaises(ValueError):
            load_turbine_files({"1": source}, timezone_offset_hours=None)

    def test_canonical_roundtrip_preserves_availability_and_sample_count(self):
        original, _ = load_turbine_files({"1": self.source(hours=3, power="0.25")})
        canonical = self.folder / "canonical.csv"
        canonical.write_text(history_to_csv(original), encoding="utf-8")
        loaded, report = load_history(canonical, timezone_offset_hours=None)
        self.assertEqual(original, loaded)
        self.assertTrue(report["explicit_availability"])
        self.assertTrue(report["explicit_sample_count"])
        self.assertEqual(report["naive_timestamp_rows"], 0)

    def test_exact_cutoff_requires_completed_hour_even_after_roundtrip(self):
        rows, _ = load_turbine_files({"1": self.source(hours=50)})
        canonical = self.folder / "canonical.csv"
        canonical.write_text(history_to_csv(rows), encoding="utf-8")
        rows, _ = load_history(canonical)
        cutoff = rows[-1]["timestamp"]
        at_start = train_models(rows, cutoff)
        at_end = train_models(rows, cutoff + timedelta(hours=1))
        self.assertEqual(at_start["training_rows"], 49)
        self.assertEqual(at_end["training_rows"], 50)
        self.assertEqual(at_start["excluded_future_rows"], 1)
        self.assertEqual(at_start["training_available_end"], cutoff.isoformat().replace("+00:00", "Z"))

    def test_bad_target_excludes_entire_hour_and_is_not_clipped(self):
        path = self.source(hours=2, power="0.2")
        records = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"))))
        records[2][3] = "1.01"
        with path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(records)
        rows, report = load_turbine_files({"1": path})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["timestamp"].hour, 1)
        self.assertEqual(report["turbines"]["1"]["rejection_counts"], {"target_out_of_range": 1})
        self.assertEqual(report["turbines"]["1"]["invalid_hours_excluded"], 1)
        self.assertEqual(report["turbines"]["1"]["rejected_rows"][0]["line"], 3)
        self.assertEqual(report["turbines"]["1"]["partial_hours_excluded"], 0)

    def test_generic_loader_validates_metadata_and_accepts_source_column_aliases(self):
        path = self.folder / "mapped.csv"
        path.write_text(
            "Статистическое время,turbine_id,Средняя скорость ветра(m/s),Нормализованная активная мощность,Средняя температура окружающей среды(°C),available_at,sample_count\n"
            "2026-01-01 0:00:00,1,5,0,-10,2026-01-01 1:00:00,6\n", encoding="utf-8")
        rows, _ = load_history(path, timezone_offset_hours=0)
        self.assertEqual(rows[0]["sample_count"], 6)
        self.assertEqual(rows[0]["available_at"].hour, 1)
        valid = history_to_csv(rows)
        for bad in (valid.replace(",6\r\n", ",0\r\n"),
                    valid.replace(",6\r\n", ",1.5\r\n"),
                    valid.replace("2026-01-01T01:00:00Z", "2025-12-31T23:00:00Z")):
            path.write_text(bad, encoding="utf-8", newline="")
            with self.assertRaises(ValueError):
                load_history(path)


if __name__ == "__main__":
    unittest.main()
