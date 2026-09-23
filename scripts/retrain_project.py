"""Rebuild and train in a new directory; never load the supplied joblib.

Run this with the separate Python 3.12 research environment. The active app
state and imported project remain read-only. The only model serialization
performed by the inspected train() function is writing a freshly fitted model.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from windagent.common import digest, read_json, save_json


REQUIRED_INPUTS = (
    "data/raw/turbine_1.csv", "data/raw/turbine_2.csv", "data/cache/weather.sqlite",
    "reports/training_report.json", "reports/january_predictions.csv",
)
RESEARCH_ISSUE_START = "2024-12-31"
RESEARCH_ISSUE_END = "2026-02-28"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def default_workspace() -> Path:
    """Find the existing default session without constructing/writing an app."""
    candidates = (
        REPOSITORY / "runtime",
        Path(tempfile.gettempdir()) / "wind-agent" / digest(str(REPOSITORY))[:12],
    )
    for candidate in candidates:
        if (candidate / "state.json").is_file():
            return candidate
    raise ValueError("Не найдена активная сессия Wind Agent; укажите --workspace с state.json.")


def verified_inputs(workspace: Path) -> tuple[dict, dict[str, dict]]:
    state_path = workspace / "state.json"
    state = read_json(state_path)
    info = state.get("project")
    if not info or state.get("dataset", {}).get("kind") != "project_archive":
        raise ValueError("В указанной сессии должен быть импортирован проверенный ZIP-проект.")
    folder = Path(info["folder"])
    result = {}
    for asset in REQUIRED_INPUTS:
        expected = info.get("asset_hashes", {}).get(asset)
        path = folder / asset
        if not isinstance(expected, str) or len(expected) != 64 or not path.is_file():
            raise ValueError("Отсутствует исходный файл или его SHA-256: " + asset)
        actual = sha256(path)
        if actual != expected:
            raise ValueError("Исходный файл не совпадает с проверенным архивом: " + asset)
        result[asset] = {"path": str(path), "expected_sha256": expected, "actual_sha256": actual}
    # Copying the main file of a database with a live journal is not a stable
    # snapshot. The imported archive is expected to be an immutable database.
    cache = Path(result["data/cache/weather.sqlite"]["path"])
    for suffix in ("-wal", "-journal"):
        sidecar = Path(str(cache) + suffix)
        if sidecar.exists() and sidecar.stat().st_size:
            raise ValueError("У исходного SQLite есть активный журнал; сначала нужен согласованный снимок: " + str(sidecar))
    return info, result


def dependency_versions() -> dict[str, str | None]:
    result = {}
    for package in ("numpy", "pandas", "PyYAML", "requests", "catboost", "scikit-learn", "scipy", "joblib"):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    return result


def compare_metrics(own: list[dict], archived: list[dict]) -> list[dict]:
    key = lambda item: (item["model"], item["turbine_id"], item["horizon"])
    left, right = {key(row): row for row in own}, {key(row): row for row in archived}
    result = []
    for group in sorted(set(left) | set(right)):
        a, b = left.get(group), right.get(group)
        row = {"model": group[0], "turbine_id": group[1], "horizon": group[2],
               "retrained_n": a.get("n") if a else None, "archive_n": b.get("n") if b else None}
        for field in ("mae", "rmse", "bias", "coverage", "interval_width"):
            av, bv = a.get(field) if a else None, b.get(field) if b else None
            row["retrained_" + field], row["archive_" + field] = av, bv
            row["delta_" + field] = av - bv if av is not None and bv is not None else None
        result.append(row)
    return result


def compare_predictions(own: Path, archived: Path) -> dict:
    def load(path):
        rows = {}
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                # Preserve identities while normalizing equivalent UTC strings.
                issue = datetime.fromisoformat(row["issue_time"].replace("Z", "+00:00"))
                valid = datetime.fromisoformat(row["valid_time"].replace("Z", "+00:00"))
                if issue.tzinfo is None or valid.tzinfo is None:
                    raise ValueError("В январском CSV отсутствует часовой пояс.")
                identity = (row["turbine_id"], issue, valid)
                if identity in rows:
                    raise ValueError("Повтор январского прогнозного часа: " + str(identity))
                values = (float(row["target_power_norm"]), float(row["prediction"]))
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("Нечисловое значение в январском CSV.")
                rows[identity] = values
        return rows
    a, b = load(own), load(archived)
    shared = a.keys() & b.keys()
    return {
        "retrained_rows": len(a), "archive_rows": len(b), "matched_rows": len(shared),
        "only_retrained_rows": len(a.keys() - b.keys()), "only_archive_rows": len(b.keys() - a.keys()),
        "max_absolute_target_difference": max((abs(a[key][0] - b[key][0]) for key in shared), default=None),
        "max_absolute_prediction_difference": max((abs(a[key][1] - b[key][1]) for key in shared), default=None),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Независимое обучение в новой папке без загрузки исходного joblib")
    parser.add_argument("--workspace", type=Path, help="Активная папка Wind Agent с state.json (только чтение)")
    parser.add_argument("--output", type=Path,
                        default=Path(tempfile.gettempdir()) / "wind-agent" / "retraining",
                        help="Родительская папка для нового уникального эксперимента")
    args = parser.parse_args(argv)
    summary = {"status": "starting", "started_at": now(), "stage": "preflight"}
    run_folder = None
    started = time.monotonic()

    def checkpoint(stage, message, **updates):
        summary.update(stage=stage, elapsed_seconds=round(time.monotonic() - started, 3), **updates)
        if run_folder is not None:
            save_json(run_folder / "reproduction_summary.json", summary)
        print(f"[{stage}] {message}", flush=True)

    try:
        workspace = (args.workspace or default_workspace()).resolve()
        checkpoint("preflight", "Проверяю SHA-256 исходных CSV, кеша и архивных отчётов.")
        info, inputs = verified_inputs(workspace)
        state_hash_before = sha256(workspace / "state.json")
        versions = dependency_versions()
        missing = [name for name, version in versions.items() if version is None]
        if missing:
            raise ValueError("Для исследовательского окружения нужны requirements-models.txt; отсутствуют: " + ", ".join(missing))
        # These are the inspected local modules. They define functions without
        # loading any supplied model; train() writes its own freshly fit bundle.
        import pandas as pd
        import yaml
        from src.data.build_features import build
        from src.models.features import MODEL_FEATURES, enrich, split_data
        from src.models.train import train
        from src.utils.common import load_config, setup_logging, utc
        from windagent.project import FEATURE_NAMES

        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        run_folder = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-"), dir=output))
        cache_copy = run_folder / "data/cache/weather.sqlite"
        cache_copy.parent.mkdir(parents=True, exist_ok=True)
        source_cache = Path(inputs["data/cache/weather.sqlite"]["path"])
        shutil.copyfile(source_cache, cache_copy)
        copy_hash = sha256(cache_copy)
        if copy_hash != inputs["data/cache/weather.sqlite"]["expected_sha256"]:
            raise ValueError("Скопированный кеш не совпадает с проверенным источником.")

        template = REPOSITORY / "config/config.yaml"
        cfg = deepcopy(load_config(template))
        cfg["_root"] = run_folder
        cfg["paths"].update(processed="data/processed", features="data/features",
                            cache="data/cache/weather.sqlite", logs="logs")
        cfg["dataset"].update(issue_start_date=RESEARCH_ISSUE_START, issue_end_date=RESEARCH_ISSUE_END)
        tag = RESEARCH_ISSUE_START.replace("-", "") + "_" + RESEARCH_ISSUE_END.replace("-", "")
        cfg["training"]["dataset"] = f"data/features/dataset_{tag}.csv"
        for turbine in cfg["turbines"]:
            if turbine["id"] not in {"T1", "T2"}:
                raise ValueError("Ожидались идентификаторы T1/T2 в исследовательской конфигурации.")
            asset = f"data/raw/turbine_{turbine['id'][1:]}.csv"
            turbine["telemetry_file"] = inputs[asset]["path"]
        cfg_path = run_folder / "config/config.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(yaml.safe_dump({key: value for key, value in cfg.items() if key != "_root"},
                                          allow_unicode=True, sort_keys=False), encoding="utf-8")
        archived_report = read_json(Path(inputs["reports/training_report.json"]["path"]))
        code_files = [*sorted((REPOSITORY / "src").rglob("*.py")), Path(__file__).resolve()]
        checkpoint("prepared", f"Новая папка: {run_folder}", status="running",
                   run_folder=str(run_folder), source_workspace=str(workspace), sources=inputs,
                   source_archive_sha256=info.get("archive_sha256"),
                   initial_cache_copy_sha256=copy_hash, config_path=str(cfg_path),
                   config_template_sha256=sha256(template),
                   code_sha256={str(path.relative_to(REPOSITORY)): sha256(path) for path in code_files},
                   environment={"python": sys.version, "executable": sys.executable,
                                "platform": platform.platform(), "packages": versions},
                   assumptions={"telemetry_timezone": cfg["telemetry"]["source_timezone"],
                                "timestamp_convention": cfg["telemetry"]["timestamp_convention"],
                                "weather_availability_policy": cfg["weather"]["availability_policy"],
                                "weather_delay_hours": cfg["weather"]["availability_delay_hours"]},
                   io_policy={"supplied_joblib": "never_loaded", "active_state": "read_only",
                              "raw_csv": "read_only", "weather_cache": "private_copy"},
                   active_state_sha256_before=state_hash_before)
        setup_logging(cfg)
        checkpoint("build", "Собираю признаки offline из копии кеша; пропускаю только недоступные выпуски.")
        manifest = build(cfg, offline=True, allow_missing=True)
        feature_path = run_folder / manifest["dataset"]
        frame = enrich(pd.read_csv(feature_path))
        fit_end = utc(cfg["training"]["fit_end_exclusive"])
        val_end = utc(cfg["training"]["validation_end_exclusive"])
        test_end = utc(cfg["training"]["test_end_exclusive"])
        fit, validation, january = split_data(frame, fit_end, val_end, test_end)
        jan_asof, final_asof = val_end - pd.Timedelta(hours=1), test_end - pd.Timedelta(hours=1)
        january_fit = frame[frame.target_power_norm.notna() & (frame.valid_time + pd.Timedelta(hours=1) <= jan_asof)]
        final_fit = frame[frame.target_power_norm.notna() & (frame.valid_time + pd.Timedelta(hours=1) <= final_asof)]
        final_calibration = january[january.valid_time + pd.Timedelta(hours=1) <= final_asof]

        def temporal_rows(subset):
            return {"rows": len(subset), "distinct_turbine_target_hours": len(subset[["turbine_id", "valid_time"]].drop_duplicates()),
                    "first_issue": subset.issue_time.min().isoformat(), "last_issue": subset.issue_time.max().isoformat(),
                    "first_target": subset.valid_time.min().isoformat(), "last_target": subset.valid_time.max().isoformat(),
                    "latest_label_available_at": (subset.valid_time + pd.Timedelta(hours=1)).max().isoformat()}

        temporal = {name: temporal_rows(part) for name, part in (
            ("fit", fit), ("validation", validation), ("january_test", january),
            ("january_fit", january_fit), ("final_fit", final_fit), ("final_calibration", final_calibration))}
        checks = {"fit_available_by_first_validation_issue": bool((fit.valid_time + pd.Timedelta(hours=1)).max() <= validation.issue_time.min()),
                  "validation_available_by_first_january_issue": bool((validation.valid_time + pd.Timedelta(hours=1)).max() <= january.issue_time.min()),
                  "january_fit_available_by_cutoff": bool((january_fit.valid_time + pd.Timedelta(hours=1)).max() <= jan_asof),
                  "final_fit_available_by_cutoff": bool((final_fit.valid_time + pd.Timedelta(hours=1)).max() <= final_asof)}
        if not all(checks.values()):
            raise ValueError("Нарушены временные границы обучающих выборок.")
        checkpoint("train", "Обучаю кандидатов на истории, выбираю на декабре, независимо оцениваю январь.",
                   build={key: manifest[key] for key in ("rows", "issues", "requested_issues", "excluded_issues", "labels_present", "availability_verified")},
                   feature_dataset=str(feature_path), feature_dataset_sha256=sha256(feature_path),
                   temporal_rows=temporal, temporal_boundary_checks=checks,
                   january_training_cutoff=jan_asof.isoformat(), final_training_cutoff=final_asof.isoformat())
        own = train(cfg)
        for name, count in own["rows"].items():
            if count != temporal[name]["rows"]:
                raise ValueError("Число строк в отчёте обучения отличается от проверенного набора: " + name)
        native = run_folder / "artifacts/catboost.cbm"
        native_details = {"exists": native.is_file(), "path": str(native),
                          "sha256": sha256(native) if native.is_file() else None,
                          "feature_contract_matches_app": MODEL_FEATURES == FEATURE_NAMES}
        if native.is_file():
            # Prepare a reviewable, forecast-only request for the native worker.
            # Creating this JSON does not run a supplied joblib or switch models.
            preview_issue = utc(cfg["dataset"]["evaluation_start"]) - pd.Timedelta(hours=1)
            preview = frame[frame.issue_time == preview_issue].sort_values(["turbine_id", "valid_time"])
            if len(preview):
                features = [[str(row[0]), *[float(value) for value in row[1:]]]
                            for row in preview[MODEL_FEATURES].itertuples(index=False, name=None)]
                request_path = run_folder / "native_preview_request.json"
                save_json(request_path, {"model_path": str(native), "feature_names": MODEL_FEATURES, "features": features})
                native_details.update(preview_request=str(request_path), preview_issue=preview_issue.isoformat(), preview_rows=len(features))
        unchanged = {asset: sha256(Path(item["path"])) == item["expected_sha256"] for asset, item in inputs.items()}
        if not all(unchanged.values()):
            raise ValueError("Исходные файлы изменились во время обучения; результат требует проверки.")
        state_hash_after = sha256(workspace / "state.json")
        checkpoint("complete", "Независимое обучение завершено; активная модель не заменена.",
                   status="complete", finished_at=now(), selected_model=own["selected_model"], selected_spec=own["selected_spec"],
                   archive_selected_model=archived_report["selected_model"], archive_selected_spec=archived_report["selected_spec"],
                   own_january_metrics=own["january_metrics"],
                   january_metrics_comparison=compare_metrics(own["january_metrics"], archived_report["january_metrics"]),
                   january_predictions_comparison=compare_predictions(run_folder / "reports/january_predictions.csv",
                                                                      Path(inputs["reports/january_predictions.csv"]["path"])),
                   native_cbm=native_details, sources_unchanged=unchanged,
                   active_state_sha256_after=state_hash_after,
                   active_state_content_unchanged=state_hash_after == state_hash_before,
                   fresh_training_report=str(run_folder / "reports/training_report.json"))
        print(str(run_folder / "reproduction_summary.json"), flush=True)
        return 0
    except Exception as exc:
        summary.update(status="failed", finished_at=now(), error_type=type(exc).__name__, error=str(exc),
                       elapsed_seconds=round(time.monotonic() - started, 3))
        if run_folder is not None:
            save_json(run_folder / "reproduction_summary.json", summary)
        print(f"Ошибка независимого обучения: {exc}", file=sys.stderr, flush=True)
        if run_folder is not None:
            print(f"Диагностика сохранена: {run_folder / 'reproduction_summary.json'}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
