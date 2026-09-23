import sys
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wind_power_forecast.settings import load_settings
from wind_power_forecast.weather import build_single_run_url, select_run_for_calculation_date


class SettingsWeatherTest(unittest.TestCase):
    def test_load_settings_and_build_single_run_url(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            config_dir.mkdir()
            config_path = config_dir / "settings.toml"
            config_path.write_text(
                """
[project]
timezone = "UTC"

[data]
turbine_1_csv = "data/raw/turbine_1.csv"
timestamp_column = "timestamp"
wind_speed_column = "wind"
target_column = "normalized_active_power"
temperature_column = "temperature"
expected_step_minutes = 10
hourly_output_csv = "data/processed/turbine_1_hourly.csv"

[weather]
provider = "open-meteo-single-runs"
endpoint = "https://single-runs-api.open-meteo.com/v1/forecast"
model = "ecmwf_ifs"
forecast_hours = 48
timezone = "UTC"
daily_calculation_hour_utc = 23
model_latency_hours = 6
wind_speed_variable = "wind_speed_100m"
temperature_variable = "temperature_2m"
hourly_variables = ["wind_speed_100m", "temperature_80m"]

[outputs]
forecast_dir = "data/processed/forecasts"
weather_dir = "data/external/weather"
run_log_dir = "reports/runs"

[[turbines]]
id = "turbine_1"
latitude = 43.25
longitude = 76.95
data_csv = "data/raw/turbine_1.csv"
capacity_mw = 1.0
model_source = "self"
""",
                encoding="utf-8",
            )

            settings = load_settings(config_path)
            self.assertEqual(settings.project_timezone, "UTC")
            self.assertEqual(settings.data.turbine_1_csv, root / "data/raw/turbine_1.csv")
            self.assertEqual(settings.turbines[0].id, "turbine_1")

            url = build_single_run_url(
                settings.weather,
                settings.turbines[0],
                "2026-02-01T00:00",
            )
            parsed = urlparse(url)
            query = parse_qs(parsed.query)
            self.assertEqual(parsed.netloc, "single-runs-api.open-meteo.com")
            self.assertEqual(query["models"], ["ecmwf_ifs"])
            self.assertEqual(query["run"], ["2026-02-01T00:00"])
            self.assertEqual(query["forecast_hours"], ["48"])
            self.assertEqual(query["hourly"], ["wind_speed_100m,temperature_80m"])
            self.assertEqual(query["wind_speed_unit"], ["ms"])

    def test_select_run_with_latency(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            config_dir.mkdir()
            config_path = config_dir / "settings.toml"
            config_path.write_text(
                """
[project]
timezone = "UTC"

[data]
turbine_1_csv = "data/raw/turbine_1.csv"
timestamp_column = "timestamp"
wind_speed_column = "wind"
target_column = "normalized_active_power"
temperature_column = "temperature"
expected_step_minutes = 10
hourly_output_csv = "data/processed/turbine_1_hourly.csv"

[weather]
provider = "open-meteo-single-runs"
endpoint = "https://single-runs-api.open-meteo.com/v1/forecast"
model = "ecmwf_ifs"
forecast_hours = 48
timezone = "UTC"
daily_calculation_hour_utc = 23
model_latency_hours = 6
wind_speed_variable = "wind_speed_100m"
temperature_variable = "temperature_2m"
hourly_variables = ["wind_speed_100m", "temperature_2m"]

[outputs]
forecast_dir = "data/processed/forecasts"
weather_dir = "data/external/weather"
run_log_dir = "reports/runs"

[[turbines]]
id = "turbine_1"
latitude = 43.25
longitude = 76.95
data_csv = "data/raw/turbine_1.csv"
capacity_mw = 1.0
model_source = "self"
""",
                encoding="utf-8",
            )
            settings = load_settings(config_path)
            self.assertEqual(
                select_run_for_calculation_date(settings.weather, date(2026, 1, 31)),
                "2026-01-31T12:00",
            )


if __name__ == "__main__":
    unittest.main()
