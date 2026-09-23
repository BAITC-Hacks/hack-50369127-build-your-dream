from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .settings import DataSettings


@dataclass(frozen=True)
class DataAudit:
    source_csv: str
    rows: int
    columns: list[str]
    started_at: str
    ended_at: str
    missing_values: dict[str, int]
    duplicate_timestamps: int
    most_common_step_minutes: float | None
    expected_step_minutes: int
    unexpected_step_count: int
    target_min: float | None
    target_max: float | None
    target_out_of_range_count: int
    wind_speed_min: float | None
    wind_speed_max: float | None
    temperature_min: float | None
    temperature_max: float | None
    hourly_rows: int
    hourly_output_csv: str
    input_rows: int = 0
    exact_duplicates_removed: int = 0
    ambiguous_or_nonexistent_timestamps_removed: int = 0
    source_timezone: str = ""
    timestamp_convention: str = "interval_start"
    timezone_resolution_policy: str = "exclude_ambiguous_and_nonexistent_local_times"
    missing_slots: int = 0
    hours_with_usable_target: int = 0
    hours_without_readings: int = 0
    incomplete_hours: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _boundary(value: str | pd.Timestamp, timezone: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("Time boundary must not be missing")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    return timestamp.tz_convert(timezone)


def _parse_timestamps(values: pd.Series, timezone: str) -> tuple[pd.Series, int]:
    strings = values.astype("string").str.strip()
    if strings.isna().any() or strings.eq("").any():
        raise ValueError("Missing measurement timestamp")
    aware = strings.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", case=False, regex=True)
    if aware.any() and not aware.all():
        raise ValueError("Mixed timezone-aware and naive measurement timestamps")
    # ISO dates remain year-first if the source also contains DD.MM.YYYY dates.
    iso = strings.str.match(r"^\d{4}-\d{2}-\d{2}(?:[ T]|$)")
    if aware.all():
        parsed = pd.to_datetime(strings, format="mixed", utc=True, errors="raise")
        return parsed.dt.tz_convert(timezone), 0
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    parsed.loc[iso] = pd.to_datetime(strings[iso], format="mixed", errors="raise")
    parsed.loc[~iso] = pd.to_datetime(
        strings[~iso], format="mixed", dayfirst=True, errors="raise"
    )
    if parsed.isna().any():
        raise ValueError("Missing measurement timestamp")
    # Kazakhstan's UTC+6 -> UTC+5 change repeats a local hour. Without offsets
    # its fold is unknown, so exclude and report instead of inventing a time.
    localized = parsed.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
    return localized, int(localized.isna().sum())


def _numeric(values: pd.Series) -> pd.Series:
    strings = values.astype("string").str.strip().str.replace(r"\s+", "", regex=True)
    strings = strings.str.replace(",", ".", regex=False).str.replace("−", "-", regex=False)
    numeric = pd.to_numeric(strings, errors="coerce").astype(float)
    return numeric.where(~numeric.isin([float("inf"), float("-inf")]))


def read_measurements(
    settings: DataSettings,
    timezone: str,
    *,
    source_csv: Path | str | None = None,
    start: str | pd.Timestamp | None = None,
    end_exclusive: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Read one turbine, optionally filtering interval starts to [start, end).

    Naive times use the configured source timezone. Numeric missing/invalid
    values remain missing; there is no interpolation. Exact duplicates ignore
    unrelated ID columns; conflicting observations for one instant fail.
    """
    path = Path(source_csv) if source_csv is not None else settings.turbine_1_csv
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}. Update the data path in the config.")
    step = settings.expected_step_minutes
    if step <= 0 or 60 % step:
        raise ValueError("expected_step_minutes must be a positive divisor of 60")
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", dtype=str)
    df.columns = [str(column).strip() for column in df.columns]
    ts_col = settings.timestamp_column
    numeric_columns = [
        settings.wind_speed_column, settings.target_column, settings.temperature_column
    ]
    missing_columns = {ts_col, *numeric_columns}.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in CSV: {', '.join(sorted(missing_columns))}")
    if df.empty:
        raise ValueError(f"Measurement CSV is empty: {path}")
    input_rows = len(df)
    df[ts_col], unresolved = _parse_timestamps(df[ts_col], timezone)
    df = df.dropna(subset=[ts_col]).copy()
    for column in numeric_columns:
        df[column] = _numeric(df[column])
    grid = df[ts_col].dt
    aligned = (
        grid.minute.mod(step).eq(0) & grid.second.eq(0)
        & grid.microsecond.eq(0) & grid.nanosecond.eq(0)
    )
    if not aligned.all():
        raise ValueError(f"Measurement timestamps must be on the {step}-minute grid")
    duplicates = int(df[ts_col].duplicated().sum())
    exact_duplicates = int(df.duplicated(subset=[ts_col, *numeric_columns]).sum())
    df = df.drop_duplicates(subset=[ts_col, *numeric_columns])
    if df[ts_col].duplicated().any():
        raise ValueError("Conflicting measurements at the same timestamp")
    if start is not None:
        df = df[df[ts_col] >= _boundary(start, timezone)]
    if end_exclusive is not None:
        df = df[df[ts_col] < _boundary(end_exclusive, timezone)]
    if (
        start is not None and end_exclusive is not None
        and _boundary(start, timezone) >= _boundary(end_exclusive, timezone)
    ):
        raise ValueError("start must precede end_exclusive")
    df = df.sort_values(ts_col).reset_index(drop=True)
    df.attrs.update(
        source_csv=str(path), input_rows=input_rows,
        duplicate_timestamps=duplicates, exact_duplicates_removed=exact_duplicates,
        ambiguous_or_nonexistent_timestamps_removed=unresolved,
        source_timezone=timezone, timestamp_convention="interval_start",
        timezone_resolution_policy="exclude_ambiguous_and_nonexistent_local_times",
    )
    return df


def _resample_hourly(
    df: pd.DataFrame,
    settings: DataSettings,
    timezone: str,
    *,
    min_valid_samples: int | None,
    start: str | pd.Timestamp | None,
    end_exclusive: str | pd.Timestamp | None,
) -> pd.DataFrame:
    expected = 60 // settings.expected_step_minutes
    minimum = expected if min_valid_samples is None else min_valid_samples
    if not isinstance(minimum, int) or isinstance(minimum, bool) or not 1 <= minimum <= expected:
        raise ValueError(f"min_valid_samples must be an integer between 1 and {expected}")
    mapping = {
        settings.wind_speed_column: "wind_speed",
        settings.target_column: "normalized_power",
        settings.temperature_column: "temperature",
    }
    x = df.set_index(settings.timestamp_column)[list(mapping)].rename(columns=mapping)
    bounds = {"wind_speed": (0, 75), "normalized_power": (0, 1), "temperature": (-80, 65)}
    for column, limits in bounds.items():
        x[column] = x[column].where(x[column].between(*limits))
    grouped = x.resample("1h", label="left", closed="left")
    counts = grouped.count()
    hourly = grouped.mean().where(counts >= minimum)
    hourly["samples_present"] = grouped.size()
    for column in mapping.values():
        hourly[f"{column}_valid_samples"] = counts[column]
    hourly["target_coverage"] = counts["normalized_power"] / expected
    hourly["target_usable"] = hourly["normalized_power"].notna()
    hourly["hour_complete"] = counts.eq(expected).all(axis=1)
    hourly = hourly.reset_index().rename(columns={settings.timestamp_column: "timestamp"})
    # An interval-average target is available only after its entire hour ends.
    hourly["available_at"] = hourly["timestamp"] + pd.Timedelta(hours=1)
    if start is not None:
        hourly = hourly[hourly["timestamp"] >= _boundary(start, timezone)]
    if end_exclusive is not None:
        hourly = hourly[hourly["available_at"] <= _boundary(end_exclusive, timezone)]
    hourly = hourly.reset_index(drop=True)
    hourly.attrs.update(df.attrs, min_valid_samples=minimum)
    return hourly


def make_hourly_measurements(
    settings: DataSettings,
    timezone: str,
    *,
    source_csv: Path | str | None = None,
    start: str | pd.Timestamp | None = None,
    end_exclusive: str | pd.Timestamp | None = None,
    min_valid_samples: int | None = None,
    write_output: bool = True,
) -> pd.DataFrame:
    """Aggregate complete hours, retaining gaps and explicit coverage columns.

    All expected readings are required independently for each mean by default.
    A partially observed last hour is excluded until its end when a cutoff is
    given. Use write_output=False for multiple turbines or temporary views.
    """
    df = read_measurements(
        settings, timezone, source_csv=source_csv, start=start, end_exclusive=end_exclusive
    )
    hourly = _resample_hourly(
        df, settings, timezone, min_valid_samples=min_valid_samples,
        start=start, end_exclusive=end_exclusive,
    )
    if write_output:
        settings.hourly_output_csv.parent.mkdir(parents=True, exist_ok=True)
        hourly.to_csv(settings.hourly_output_csv, index=False)
    return hourly


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def audit_and_resample_hourly(
    settings: DataSettings,
    timezone: str,
    *,
    source_csv: Path | str | None = None,
) -> DataAudit:
    df = read_measurements(settings, timezone, source_csv=source_csv)
    if df.empty:
        raise ValueError("No measurements remain after timestamp validation")
    ts_col = settings.timestamp_column
    step_minutes = df[ts_col].diff().dropna().dt.total_seconds().div(60)
    hourly = _resample_hourly(
        df, settings, timezone, min_valid_samples=None, start=None, end_exclusive=None
    )
    settings.hourly_output_csv.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(settings.hourly_output_csv, index=False)
    target = df[settings.target_column]
    wind_speed = df[settings.wind_speed_column]
    temperature = df[settings.temperature_column]
    expected_slots = int(
        (df[ts_col].max() - df[ts_col].min())
        / pd.Timedelta(minutes=settings.expected_step_minutes)
    ) + 1
    return DataAudit(
        source_csv=df.attrs["source_csv"],
        rows=len(df), columns=list(df.columns),
        started_at=df[ts_col].min().isoformat(), ended_at=df[ts_col].max().isoformat(),
        missing_values={column: int(value) for column, value in df.isna().sum().items()},
        duplicate_timestamps=df.attrs["duplicate_timestamps"],
        most_common_step_minutes=(float(step_minutes.mode().iloc[0]) if len(step_minutes) else None),
        expected_step_minutes=settings.expected_step_minutes,
        unexpected_step_count=int((step_minutes != settings.expected_step_minutes).sum()),
        target_min=_finite_or_none(target.min()), target_max=_finite_or_none(target.max()),
        target_out_of_range_count=int(((target < 0) | (target > 1)).sum()),
        wind_speed_min=_finite_or_none(wind_speed.min()),
        wind_speed_max=_finite_or_none(wind_speed.max()),
        temperature_min=_finite_or_none(temperature.min()),
        temperature_max=_finite_or_none(temperature.max()),
        hourly_rows=len(hourly), hourly_output_csv=str(settings.hourly_output_csv),
        input_rows=df.attrs["input_rows"],
        exact_duplicates_removed=df.attrs["exact_duplicates_removed"],
        ambiguous_or_nonexistent_timestamps_removed=(
            df.attrs["ambiguous_or_nonexistent_timestamps_removed"]
        ),
        source_timezone=timezone, missing_slots=expected_slots - len(df),
        hours_with_usable_target=int(hourly["target_usable"].sum()),
        hours_without_readings=int(hourly["samples_present"].eq(0).sum()),
        incomplete_hours=int((~hourly["hour_complete"]).sum()),
    )


def write_audit_report(audit: DataAudit, output_path: Path) -> None:
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit.to_dict(), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
