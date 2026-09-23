"""Run web-app tests separately from optional research dependencies."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    suite = unittest.TestSuite()
    for filename in ("test_agent.py", "test_data_model.py", "test_weather.py",
                     "test_telemetry.py", "test_project.py"):
        suite.addTests(unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern=filename))
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
