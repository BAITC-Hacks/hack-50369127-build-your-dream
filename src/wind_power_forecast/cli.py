from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

from .settings import load_settings


def _default_config() -> Path:
    local = Path("config/settings.toml")
    return local if local.exists() else Path("config/settings.example.toml")


def _print(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def audit_data(args) -> int:
    from .data import audit_and_resample_hourly, write_audit_report

    settings = load_settings(args.config)
    reports = {}
    for turbine in settings.turbines:
        if turbine.data_csv is None:
            raise ValueError(f"No data_csv for {turbine.id}")
        data = replace(
            settings.data,
            turbine_1_csv=turbine.data_csv,
            hourly_output_csv=settings.data.hourly_output_csv.parent / f"{turbine.id}_hourly.csv",
        )
        report = audit_and_resample_hourly(data, settings.project_timezone)
        write_audit_report(report, args.output.parent / f"{turbine.id}_audit.json")
        reports[turbine.id] = report.to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    _print(reports)
    return 0


def check_weather_run(args) -> int:
    from .weather import check_single_run

    settings = load_settings(args.config)
    turbines = {t.id: t for t in settings.turbines}
    if args.turbine not in turbines:
        raise ValueError(f"Unknown turbine: {args.turbine}")
    result = check_single_run(settings.weather, turbines[args.turbine], args.run, args.output_dir)
    _print(result.to_dict())
    return 0 if result.ok else 2


def import_weather_cache(args) -> int:
    from .weather import import_weather_archive

    settings = load_settings(args.config)
    _print(import_weather_archive(args.archive, settings.outputs.weather_dir))
    return 0


def forecast_day(args) -> int:
    from .forecast import run_daily_forecast

    result = run_daily_forecast(
        load_settings(args.config),
        date.fromisoformat(args.date),
        offline=args.offline,
        refresh=args.refresh,
    )
    _print(result.to_dict())
    return 0


def forecast_range(args) -> int:
    from .evaluation import compile_submission
    from .forecast import ForecastAgent

    settings = load_settings(args.config)
    first = (
        date.fromisoformat(args.start_date)
        if args.start_date
        else settings.test_start_date - timedelta(days=1)
    )
    last = date.fromisoformat(args.end_date) if args.end_date else settings.test_end_date
    if last < first:
        raise ValueError("--end-date must follow --start-date")
    agent = ForecastAgent(settings, offline=args.offline, refresh=args.refresh)
    summary = []
    current = first
    while current <= last:
        report = agent.run(current)
        summary.append(
            {
                "date": str(current),
                "rows": report.rows,
                "status": report.status,
                "weather_run": report.weather_run,
            }
        )
        print(f"{current}: {report.status}, {report.rows} hours", file=sys.stderr, flush=True)
        current += timedelta(days=1)
    if first <= settings.test_start_date - timedelta(
        days=1
    ) and last >= settings.test_end_date - timedelta(days=1):
        submission, all_forecasts = compile_submission(settings)
        _print(
            {
                "runs": summary,
                "submission_hours": len(submission),
                "all_forecasts_test_hours": len(all_forecasts),
                "submission": str(settings.outputs.forecast_dir / "submission.csv"),
            }
        )
    else:
        _print({"runs": summary})
    return 0


def evaluate(args) -> int:
    from .evaluation import evaluate_forecasts

    _print(evaluate_forecasts(load_settings(args.config), args.actual_dir))
    return 0


def run_agent(args) -> int:
    from .forecast import ForecastAgent

    if args.cycles < 1 or args.interval_seconds < 1:
        raise ValueError("Agent cycles and polling interval must be positive")
    agent = ForecastAgent(load_settings(args.config), offline=args.offline, refresh=args.refresh)
    day = date.fromisoformat(args.date)
    for iteration in range(args.cycles):
        report = agent.run(day)
        _print(
            {
                "cycle": iteration + 1,
                "status": report.status,
                "fingerprint": report.fingerprint,
                "forecast_csv": report.forecast_csv,
            }
        )
        if iteration + 1 < args.cycles:
            time.sleep(args.interval_seconds)
    return 0


def validate_history(args) -> int:
    """Frozen pre-test evaluation with historical NWP, isolated from February outputs."""
    from .evaluation import evaluate_forecasts
    from .forecast import ForecastAgent

    settings = load_settings(args.config)
    first, last = date.fromisoformat(args.start_date), date.fromisoformat(args.end_date)
    if first > last or last > settings.training_end_date:
        raise ValueError("Historical validation must finish within the available training period")
    output_root = (
        settings.outputs.run_log_dir.parent / "historical_validation" / settings.weather.provider
    )
    validation = replace(
        settings,
        training_end_date=first - timedelta(days=1),
        test_start_date=first,
        test_end_date=last,
        outputs=replace(
            settings.outputs,
            forecast_dir=output_root / "forecasts",
            run_log_dir=output_root / "runs",
        ),
    )
    agent = ForecastAgent(validation, offline=args.offline, refresh=args.refresh)
    current = first - timedelta(days=1)
    while current < last:
        result = agent.run(current)
        print(f"validation {current}: {result.status}", file=sys.stderr, flush=True)
        current += timedelta(days=1)
    _print(evaluate_forecasts(validation))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproducible wind-power forecasting agent")
    commands = parser.add_subparsers(required=True)
    audit = commands.add_parser("audit-data", help="Audit and aggregate both telemetry CSVs")
    audit.add_argument("--config", type=Path, default=_default_config())
    audit.add_argument("--output", type=Path, default=Path("reports/data_audit.json"))
    audit.set_defaults(func=audit_data)
    weather = commands.add_parser("check-weather-run", help="Fetch an exact archived NWP run")
    weather.add_argument("--config", type=Path, default=_default_config())
    weather.add_argument("--run", required=True, help="UTC initialization YYYY-MM-DDTHH:MM")
    weather.add_argument("--turbine", default="turbine_1")
    weather.add_argument("--output-dir", type=Path, default=Path("data/external/weather"))
    weather.set_defaults(func=check_weather_run)
    importer = commands.add_parser(
        "import-weather-cache", help="Validate and import the supplied ZIP weather cache"
    )
    importer.add_argument("--config", type=Path, default=_default_config())
    importer.add_argument("--archive", type=Path, required=True)
    importer.set_defaults(func=import_weather_cache)
    for name, handler in (
        ("forecast-day", forecast_day),
        ("forecast-range", forecast_range),
        ("agent", run_agent),
    ):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=_default_config())
        command.add_argument(
            "--offline", action="store_true", help="Use validated cached runs only"
        )
        command.add_argument(
            "--refresh", action="store_true", help="Retrieve runs again and compare input hashes"
        )
        if name == "forecast-range":
            command.add_argument("--start-date", help="Default: day before test_start_date")
            command.add_argument("--end-date", help="Default: test_end_date, inclusive")
        else:
            command.add_argument(
                "--date", required=True, help="Calculation date in project timezone"
            )
        if name == "agent":
            command.add_argument("--cycles", type=int, default=1)
            command.add_argument("--interval-seconds", type=float, default=60)
        command.set_defaults(func=handler)
    evaluation = commands.add_parser(
        "evaluate", help="Score frozen outputs if actual test targets exist"
    )
    evaluation.add_argument("--config", type=Path, default=_default_config())
    evaluation.add_argument(
        "--actual-dir", type=Path, help="Optional directory of turbine_1.csv/turbine_2.csv actuals"
    )
    evaluation.set_defaults(func=evaluate)
    validation = commands.add_parser(
        "validate-history", help="Evaluate archived-weather forecasts before February"
    )
    validation.add_argument("--config", type=Path, default=_default_config())
    validation.add_argument("--start-date", default="2026-01-01")
    validation.add_argument("--end-date", default="2026-01-31")
    validation.add_argument("--offline", action="store_true")
    validation.add_argument("--refresh", action="store_true")
    validation.set_defaults(func=validate_history)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, OSError, RuntimeError, ModuleNotFoundError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
