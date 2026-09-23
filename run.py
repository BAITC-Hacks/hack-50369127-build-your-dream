from __future__ import annotations

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from wind_power_forecast.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["forecast-day", *sys.argv[1:]]))
