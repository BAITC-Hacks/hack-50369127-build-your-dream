import sys
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wind_power_forecast.data import (
    audit_and_resample_hourly,
    make_hourly_measurements,
    read_measurements,
    write_audit_report,
)
from wind_power_forecast.settings import DataSettings


class MeasurementDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = DataSettings(
            turbine_1_csv=self.root / "turbine_1.csv",
            timestamp_column="time", wind_speed_column="wind", target_column="power",
            temperature_column="temperature", expected_step_minutes=10,
            hourly_output_csv=self.root / "hourly.csv",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write(self, rows: list[list], path: Path | None = None, sep: str = ",") -> None:
        pd.DataFrame(rows, columns=["time", "wind", "power", "temperature"]).to_csv(
            path or self.settings.turbine_1_csv, index=False, sep=sep, encoding="utf-8-sig"
        )

    def full_hour(self, hour: str, power: float = 0.5) -> list[list]:
        return [[str(t), 5, power, 10] for t in pd.date_range(hour, periods=6, freq="10min")]

    def test_iso_month_is_not_interpreted_as_day(self) -> None:
        self.write([
            ["2023-03-11 0:00:00", 5, 0.2, 10],
            ["12.03.2023 00:00:00", 6, 0.3, 11],
        ])
        result = read_measurements(self.settings, "UTC")
        self.assertEqual(result["time"].dt.strftime("%Y-%m-%d").tolist(), [
            "2023-03-11", "2023-03-12"
        ])

    def test_decimal_comma_missing_values_and_nonfinite_numbers(self) -> None:
        self.write([
            ["2026-01-31 00:00", "6,75", "0,39", "−2,5"],
            ["2026-01-31 00:10", "inf", "-", ""],
        ], sep=";")
        result = read_measurements(self.settings, "UTC")
        self.assertEqual(result.loc[0, "wind"], 6.75)
        self.assertEqual(result.loc[0, "power"], 0.39)
        self.assertEqual(result.loc[0, "temperature"], -2.5)
        self.assertTrue(result.loc[1, ["wind", "power", "temperature"]].isna().all())

    def test_missing_timestamp_fails(self) -> None:
        self.write([[None, 5, 0.2, 10]])
        with self.assertRaisesRegex(ValueError, "Missing measurement timestamp"):
            read_measurements(self.settings, "UTC")

    def test_mixed_aware_and_naive_fails(self) -> None:
        self.write([
            ["2026-01-31T00:00:00Z", 5, 0.2, 10],
            ["2026-01-31 00:10:00", 5, 0.2, 10],
        ])
        with self.assertRaisesRegex(ValueError, "Mixed timezone"):
            read_measurements(self.settings, "UTC")

    def test_explicit_offsets_disambiguate_clock_change(self) -> None:
        self.write([
            ["2024-02-29T23:00:00+06:00", 5, 0.2, 10],
            ["2024-02-29T23:00:00+05:00", 6, 0.3, 11],
        ])
        result = read_measurements(self.settings, "Asia/Almaty")
        self.assertEqual(len(result), 2)
        self.assertEqual(result["time"].diff().iloc[1], pd.Timedelta(hours=1))
        self.assertEqual(result.attrs["ambiguous_or_nonexistent_timestamps_removed"], 0)

    def test_kazakhstan_transition_is_excluded_and_audited(self) -> None:
        rows = (
            self.full_hour("2024-02-29 22:00")
            + self.full_hour("2024-02-29 23:00")
            + self.full_hour("2024-03-01 00:00")
        )
        self.write(rows)
        audit = audit_and_resample_hourly(self.settings, "Asia/Almaty")
        self.assertEqual(audit.ambiguous_or_nonexistent_timestamps_removed, 6)
        self.assertEqual(audit.rows, 12)
        self.assertEqual(audit.hourly_rows, 4)
        self.assertEqual(audit.hours_with_usable_target, 2)
        self.assertEqual(audit.hours_without_readings, 2)

    def test_duplicate_rows_do_not_reweight_the_hour(self) -> None:
        rows = self.full_hour("2026-01-31 00:00")
        rows[0][2] = 0
        self.write(rows + [rows[0]])
        result = make_hourly_measurements(self.settings, "UTC", write_output=False)
        self.assertAlmostEqual(result.loc[0, "normalized_power"], 2.5 / 6)
        self.assertEqual(result.loc[0, "samples_present"], 6)
        self.assertEqual(result.attrs["exact_duplicates_removed"], 1)
        rows.append([rows[0][0], 5, 0.7, 10])
        self.write(rows)
        with self.assertRaisesRegex(ValueError, "Conflicting measurements"):
            read_measurements(self.settings, "UTC")

    def test_coverage_gaps_and_out_of_range_values_are_not_imputed(self) -> None:
        rows = self.full_hour("2026-01-31 00:00")
        rows += self.full_hour("2026-01-31 02:00")[:5]
        rows += self.full_hour("2026-01-31 03:00")
        rows[-1][2] = 1.2
        self.write(rows)
        result = make_hourly_measurements(self.settings, "UTC", write_output=False)
        self.assertEqual(result["samples_present"].tolist(), [6, 0, 5, 6])
        self.assertEqual(result["target_usable"].tolist(), [True, False, False, False])
        self.assertTrue(result.loc[1:3, "normalized_power"].isna().all())
        self.assertEqual(result.loc[3, "normalized_power_valid_samples"], 5)
        self.assertAlmostEqual(result.loc[2, "target_coverage"], 5 / 6)

    def test_cutoff_excludes_unfinished_hour_and_future_samples(self) -> None:
        self.write(self.full_hour("2026-01-31 00:00") + self.full_hour("2026-01-31 01:00"))
        result = make_hourly_measurements(
            self.settings, "UTC", end_exclusive="2026-01-31T01:30:00Z",
            min_valid_samples=1, write_output=False,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "available_at"], pd.Timestamp("2026-01-31T01:00Z"))
        self.assertFalse(self.settings.hourly_output_csv.exists())

    def test_source_override_uses_second_turbine(self) -> None:
        self.write(self.full_hour("2026-01-31 00:00", power=0.2))
        second = self.root / "turbine_2.csv"
        self.write(self.full_hour("2026-01-31 00:00", power=0.8), path=second)
        result = make_hourly_measurements(
            self.settings, "UTC", source_csv=second, write_output=False
        )
        self.assertAlmostEqual(result.loc[0, "normalized_power"], 0.8)

    def test_empty_date_window_returns_empty_typed_frame(self) -> None:
        self.write(self.full_hour("2026-01-31 00:00"))
        result = make_hourly_measurements(
            self.settings, "UTC", start="2026-02-01", end_exclusive="2026-03-01",
            write_output=False,
        )
        self.assertTrue(result.empty)
        self.assertIn("normalized_power", result.columns)
        self.assertEqual(str(result["timestamp"].dt.tz), "UTC")

    def test_off_grid_readings_fail(self) -> None:
        self.write([["2026-01-31 00:01", 5, 0.2, 10]])
        with self.assertRaisesRegex(ValueError, "10-minute grid"):
            read_measurements(self.settings, "UTC")

    def test_audit_writes_standard_json_with_all_targets_missing(self) -> None:
        self.write([["2026-01-31 00:00", 5, None, 10]])
        audit = audit_and_resample_hourly(self.settings, "UTC")
        report = self.root / "audit.json"
        write_audit_report(audit, report)
        self.assertIsNone(audit.target_min)
        self.assertNotIn("NaN", report.read_text(encoding="utf-8"))
        self.assertEqual(audit.hours_with_usable_target, 0)

    def test_invalid_interval_size_fails(self) -> None:
        self.write([["2026-01-31 00:00", 5, 0.2, 10]])
        with self.assertRaisesRegex(ValueError, "divisor of 60"):
            read_measurements(replace(self.settings, expected_step_minutes=7), "UTC")


if __name__ == "__main__":
    unittest.main()
