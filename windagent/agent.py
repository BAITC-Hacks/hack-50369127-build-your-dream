"""Rule-driven forecasting agent: observe, validate, plan, predict, analyse, persist.

No LLM/API key is needed. Decisions and input versions are recorded explicitly.
"""
from datetime import timedelta
import csv
import io
import math
from pathlib import Path
import sys
import tempfile

from .common import ROOT, digest, iso, load_config, now, read_json, save_json, utc
from .data import load_history
from .demo import WARNING as DEMO_WARNING, fetch_demo_weather, history_csv
from .model import predict, train_models


class ForecastAgent:
    def __init__(self, workspace=None, config=None, weather_provider=None):
        self.workspace = Path(workspace or ROOT / "runtime")
        try:
            self.workspace.mkdir(parents=True, exist_ok=True)
        except OSError:
            if workspace is not None:
                raise
            # Some Windows installations protect Documents from Python writes.
            # Temp is user-writable; a stable per-repository folder preserves sessions.
            self.workspace = Path(tempfile.gettempdir()) / "wind-agent" / digest(str(ROOT))[:12]
            self.workspace.mkdir(parents=True, exist_ok=True)
            print(f"Папка проекта недоступна для записи; результаты: {self.workspace}", file=sys.stderr)
        self.config = config or load_config()
        self.weather_provider = weather_provider
        self.state_path = self.workspace / "state.json"
        self.state = read_json(self.state_path) if self.state_path.exists() else {
            "dataset": None, "model": None, "last_run": None, "backtest": None}
        if self.state.get("scenario_config"):
            self.config = self.state["scenario_config"]

    def snapshot(self):
        return {**self.state, "config": self.config, "workspace": str(self.workspace)}

    def _save(self):
        save_json(self.state_path, self.state)

    def import_csv(self, text, filename="history.csv", timezone_offset_hours=5, power_scale=1, demo=False):
        if not text.strip():
            raise ValueError("CSV-файл пуст.")
        # Content-addressed storage preserves previous imports and never trusts a client path.
        identifier = digest({"csv": text, "offset": timezone_offset_hours, "scale": power_scale})
        path = self.workspace / "imports" / (identifier + ".csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="")
        rows, report = load_history(path, timezone_offset_hours=timezone_offset_hours, power_scale=power_scale)
        expected = {t["id"] for t in self.config["turbines"]}
        received = {r["turbine_id"] for r in rows}
        if received != expected:
            raise ValueError(f"CSV должен содержать турбины {', '.join(sorted(expected))}; получено: {', '.join(sorted(received))}.")
        model = train_models(rows, utc(self.config["training_cutoff"]))
        report["excluded_after_cutoff"] = model["excluded_future_rows"]
        report["training_rows"] = model["training_rows"]
        if model["excluded_future_rows"]:
            report["warnings"].append(f"Из обучения исключено {model['excluded_future_rows']} строк, недоступных к отсечке {self.config['training_cutoff']}.")
        self.state.update({"dataset": {"source": Path(filename).name, "demo": demo,
            "rows": len(rows), "turbines": sorted(received), "report": report, "sha256": identifier,
            "path": str(path), "timezone_offset_hours": timezone_offset_hours, "power_scale": power_scale,
            "imported_at": now()}, "model": model, "last_run": None, "backtest": None})
        self._save()
        return self.snapshot()

    def demo(self):
        if self.state["dataset"] and not self.state["dataset"]["demo"]:
            raise ValueError("Уже загружена реальная история. Для отдельного демо запустите CLI с --workspace runtime-demo.")
        return self.import_csv(history_csv(self.config), "Учебная история (синтетическая)", demo=True)

    def import_turbines(self, paths, timezone_offset_hours=0, interval_convention="start"):
        from .telemetry import history_to_csv, load_turbine_files
        rows, report = load_turbine_files(paths, timezone_offset_hours, interval_convention)
        self.import_csv(history_to_csv(rows), "Две турбины · реальные CSV", timezone_offset_hours=timezone_offset_hours)
        self.state["dataset"]["report"]["source_audit"] = report
        self.state["dataset"]["report"]["warnings"].extend(report.get("warnings", []))
        self._save()
        return self.snapshot()

    def import_project(self, archive_path, external_turbines=None):
        from .project import install_archive, WARNINGS
        from .telemetry import history_to_csv, load_turbine_files
        info = install_archive(archive_path, self.workspace / "projects", external_turbines)
        folder = Path(info["folder"])
        paths = {t: folder / f"data/raw/turbine_{t}.csv" for t in ("1", "2")}
        rows, report = load_turbine_files(paths, timezone_offset_hours=0, interval_convention="start")
        text = history_to_csv(rows)
        source_path = self.workspace / "imports" / (digest(text) + ".csv")
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(text, encoding="utf-8", newline="")
        training = read_json(folder / "reports/training_report.json")
        config = {**load_config(), "timezone_offset_hours": 0, "availability_lag_hours": 8,
            "training_cutoff": "2026-01-31T23:00:00+00:00", "test_start": "2026-02-01T00:00:00+00:00",
            "test_end": "2026-03-01T00:00:00+00:00"}
        # Record the previous session before replacing only the active selection.
        if self.state.get("dataset"):
            save_json(self.workspace / "sessions" / (digest(self.state)[:20] + ".json"), self.state)
        model = {"kind": "catboost_archive", "name": "CatBoost · depth 4 · MAE", "training_rows": training["rows"]["final_fit"],
            "training_cutoff": info["training_cutoff"], "training_end": info["training_cutoff"], "diagnostics": {},
            "january_metrics": training["january_metrics"], "imported_report": True, "independently_verified_metrics": True,
            "january_audit": info["january_audit"], "validation": "Выбор модели на декабре, независимые прогнозы января; февральские факты отсутствуют.",
            "warnings": WARNINGS.copy(), "interval": {"description": WARNINGS[-1], "calibrated": True, "coverage_target": .8}}
        report["warnings"] = list(dict.fromkeys(report.get("warnings", []) + WARNINGS))
        dataset = {"source": "HackAlemAI · turbine 1 + turbine 2", "demo": False, "kind": "project_archive",
            "rows": len(rows), "turbines": ["1", "2"], "report": report, "path": str(source_path),
            "timezone_offset_hours": 0, "power_scale": 1, "sha256": digest(text), "imported_at": now()}
        self.config = config
        self.state = {"dataset": dataset, "model": model, "project": info, "scenario_config": config,
                      "last_run": None, "backtest": None}
        self._save()
        return self.snapshot()

    def _models_at(self, as_of):
        dataset = self.state["dataset"]
        if not dataset:
            raise ValueError("Сначала загрузите CSV с историей ВЭС или создайте учебное демо.")
        cutoff = min(as_of, utc(self.config["training_cutoff"]))
        rows, _ = load_history(Path(dataset["path"]), timezone_offset_hours=dataset["timezone_offset_hours"],
                               power_scale=dataset["power_scale"])
        # Always revalidate the actual file: external input edits must invalidate model/run caches.
        training = [r for r in rows if utc(r.get("available_at", r["timestamp"])) <= cutoff
                    and utc(r["timestamp"]) <= cutoff]
        fingerprint = digest({"rows": training, "cutoff": iso(cutoff), "version": 1})
        model_path = self.workspace / "models" / (fingerprint + ".json")
        if model_path.exists():
            model = read_json(model_path)
        else:
            model = train_models(training, cutoff)
            save_json(model_path, model)
        return model, fingerprint, training

    def forecast(self, as_of, hours=48, mode="archive", refresh=False):
        as_of = utc(as_of)
        if as_of.minute or as_of.second or as_of.microsecond:
            raise ValueError("Момент расчёта должен приходиться на начало часа.")
        if hours not in (24, 48):
            raise ValueError("Горизонт прогноза должен быть 24 или 48 часов.")
        if mode not in ("archive", "demo", "project"):
            raise ValueError("Режим должен быть archive, demo или project.")
        if not self.state["dataset"]:
            raise ValueError("Сначала загрузите историю ВЭС.")
        if mode == "project":
            if self.state["dataset"].get("kind") != "project_archive":
                raise ValueError("Сначала импортируйте предоставленный ZIP-проект.")
            try:
                return self._project_forecast(as_of, hours, refresh)
            except (ValueError, RuntimeError, OSError) as exc:
                save_json(self.workspace / "last_error.json", {"as_of": iso(as_of), "events": [
                    {"stage": "error", "status": "error", "message": str(exc), "time": now()}]})
                raise
        if self.state["dataset"].get("kind") == "project_archive":
            raise ValueError("Для импортированной CatBoost-модели выберите источник «Кеш проекта · ECMWF».")
        if self.state["dataset"]["demo"] != (mode == "demo"):
            raise ValueError("Учебная история работает только с учебной погодой; для архива загрузите реальную историю.")
        events = []

        def event(stage, message, status="done"):
            events.append({"stage": stage, "status": status, "message": message, "time": now()})

        event("plan", f"Горизонт {hours} ч; отсечение истории не позднее {self.config['training_cutoff']}.")
        try:
            model, model_hash, training = self._models_at(as_of)
            event("prepare", f"Проверена история: {len(training)} доступных часовых записей.")
            if mode == "demo":
                provider = fetch_demo_weather
            elif self.weather_provider is not None:
                provider = self.weather_provider
            else:
                from .weather import fetch_weather
                provider = fetch_weather
            weather = provider(as_of=as_of, targets=self.config["turbines"], hours=hours,
                cache_dir=self.workspace / "weather", refresh=refresh,
                availability_lag_hours=self.config["availability_lag_hours"])
            self._validate_weather(weather, as_of, hours)
            event("weather", "Получены полные погодные ряды; время доступности проверено по метаданным источника.")
            # Retrieval time is not a change in the forecast. Only actual weather content matters.
            weather_version = [{k: p.get(k) for k in ("turbine_id", "source", "model", "run_time",
                               "available_at", "sha256", "availability_basis")} for p in weather["provenance"]]
            identifier = digest({"as_of": iso(as_of), "hours": hours, "mode": mode, "model": model_hash,
                                 "weather": weather_version, "rows": weather["rows"]})[:24]
            run_path = self.workspace / "runs" / (identifier + ".json")
            if run_path.exists():
                result = read_json(run_path)
                event("reuse", "Входные данные не изменились: возвращён сохранённый расчёт.")
                result = {**result, "reused": True, "events": events + result["events"][-1:]}
            else:
                forecast_rows = predict(model, weather["rows"])
                event("predict", f"Модель рассчитала {len(forecast_rows)} значений мощности.")
                analysis, warnings = self._analyse(forecast_rows, training)
                for turbine_id, fitted in model["per_turbine"].items():
                    age = (as_of - utc(fitted["persistence_timestamp"])).total_seconds() / 3600
                    if age > 48:
                        warnings.append(f"Турбина {turbine_id}: последняя известная мощность к выпуску устарела на {age:.0f} ч; базовый persistence заморожен вместе с январской историей.")
                warnings = list(dict.fromkeys(weather.get("warnings", []) + model.get("warnings", []) + warnings))
                if mode == "demo" and DEMO_WARNING not in warnings:
                    warnings.insert(0, DEMO_WARNING)
                event("analyse", f"Проверены границы мощности и полнота горизонта; предупреждений: {len(warnings)}.")
                event("save", "Почасовой прогноз и сведения об источниках сохранены.")
                result = {"id": identifier, "as_of": iso(as_of), "hours": hours, "mode": mode,
                    "created_at": now(), "rows": forecast_rows, "provenance": weather["provenance"],
                    "model_sha256": model_hash, "model_training_cutoff": iso(min(as_of, utc(self.config["training_cutoff"]))),
                    "warnings": warnings, "analysis": analysis, "events": events, "reused": False,
                    "eligibility": {"competition_ready": False, "reason":
                        "Учебная синтетика." if mode == "demo" else
                        "Требуется подтверждение, что архив содержит исходные оперативные прогнозы и время их публикации; также проверьте часовой пояс и высоту ветра."}}
                save_json(run_path, result)
            self.state["last_run"] = result
            self.state["model"] = model
            self._save()
            return result
        except (ValueError, RuntimeError, OSError) as exc:
            event("error", str(exc), "error")
            save_json(self.workspace / "last_error.json", {"as_of": iso(as_of), "events": events})
            raise

    def _project_forecast(self, as_of, hours, refresh):
        from .project import ProjectEngine, model_python
        engine = ProjectEngine(self.state["project"], self.config["turbines"])
        if as_of < utc(engine.report["final_training_asof"]):
            raise ValueError("Импортированная модель обучена позже выбранного выпуска. Ранние выпуски требуют отдельного обучения.")
        source_path = Path(self.state["dataset"]["path"])
        with source_path.open(encoding="utf-8", newline="") as source_file:
            source_text = source_file.read()
        if digest(source_text) != self.state["dataset"]["sha256"]:
            raise ValueError("Изменена история импортированного проекта. Повторно импортируйте проверенный архив.")
        events = []

        def event(stage, message):
            events.append({"stage": stage, "status": "done", "message": message, "time": now()})

        event("plan", "Проектный сценарий: 23:00 UTC, полные 10-минутные интервалы, модель заморожена 31 января.")
        weather = engine.weather(as_of, hours)
        self._validate_weather(weather, as_of, hours)
        event("weather", "Исходные ответы из SQLite-кеша перепроверены по SHA-256; сохранены выпуск и предполагаемая доступность.")
        intended = "recomputed" if model_python() else "saved_archive"
        signature = {"model_sha256": self.state["project"]["asset_hashes"]["artifacts/catboost.cbm"],
                     "calibration_sha256": self.state["project"]["asset_hashes"]["reports/training_report.json"],
                     "archive_sha256": self.state["project"]["archive_sha256"],
                     "dataset_sha256": self.state["dataset"]["sha256"],
                     "weather": [{k: p[k] for k in ("turbine_id", "run_time", "sha256")} for p in weather["provenance"]],
                     "as_of": iso(as_of), "hours": hours, "execution": intended, "adapter_version": 2}
        identifier = digest(signature)[:24]
        run_path = self.workspace / "runs" / (identifier + ".json")
        event("prepare", "Сформированы только признаки архивного погодного прогноза; измерения целевого часа не используются.")
        if run_path.exists() and not refresh:
            result = read_json(run_path)
            result["reused"] = True
            event("reuse", "Версии модели и погоды не изменились: использован сохранённый расчёт.")
            result["events"] = events + result["events"][-1:]
        else:
            predictions, execution, comparison = engine.predict(as_of, weather)
            event("predict", "Выполнен новый расчёт native CatBoost на проверенном погодном кеше." if execution == "recomputed" else
                  "Открыт сохранённый расчёт из предоставленного архива; новый расчёт модели не выполнялся.")
            source, _ = load_history(self.state["dataset"]["path"], 0, 1)
            available = [row for row in source if utc(row["available_at"]) <= min(as_of, utc(self.config["training_cutoff"]))]
            last = {}
            for row in available:
                last[row["turbine_id"]] = row["power"]
            for row in predictions:
                row["persistence_pred"] = last[row["turbine_id"]]
                if not 0 <= row["lower"] <= row["power_pred"] <= row["upper"] <= 1:
                    raise ValueError("Импортированная модель вернула некорректную полосу мощности.")
            powers = [row["power_pred"] for row in predictions]
            analysis = {"complete": True, "row_count": len(predictions), "mean_power": sum(powers) / len(powers),
                "min_power": min(powers), "max_power": max(powers), "unit": "normalized_power",
                "archive_reproduction_max_error": comparison, "aggregation": "equal_weight_mean_not_farm_energy"}
            event("analyse", "Покрытие и границы мощности проверены." + (f" Отклонение от исходного расчёта: {comparison:.3g}." if comparison is not None else ""))
            event("save", "Прогноз, версия модели и происхождение погодного кеша сохранены.")
            result = {"id": identifier, "as_of": iso(as_of), "hours": hours, "mode": "project", "created_at": now(),
                "rows": predictions, "execution": execution, "engine": {"kind": "catboost", "execution": execution},
                "model_sha256": signature["model_sha256"], "model_training_cutoff": self.config["training_cutoff"],
                "provenance": weather["provenance"], "warnings": weather["warnings"], "analysis": analysis,
                "events": events, "reused": False, "eligibility": {"competition_ready": False,
                    "reason": "Нужно подтвердить часовой пояс телеметрии, смысл интервала и историческую доступность погодных выпусков."}}
            save_json(run_path, result)
        self.state["last_run"] = result
        self._save()
        return result

    def _validate_weather(self, weather, as_of, hours):
        expected = {(t["id"], as_of + timedelta(hours=h)) for t in self.config["turbines"] for h in range(1, hours + 1)}
        seen = set()
        for row in weather["rows"]:
            key = (row["turbine_id"], utc(row["valid_time"]))
            if key in seen:
                raise ValueError("Источник погоды вернул повторяющийся прогнозный час.")
            seen.add(key)
            if not (math.isfinite(row["wind_speed"]) and 0 <= row["wind_speed"] <= 100
                    and math.isfinite(row["temperature"]) and -100 <= row["temperature"] <= 70):
                raise ValueError("Источник погоды вернул недопустимые значения.")
        if seen != expected:
            raise ValueError("Погода не покрывает весь прогнозный горизонт для всех турбин. Расчёт остановлен.")
        provenance = weather.get("provenance", [])
        if len(provenance) != len(self.config["turbines"]) or {p["turbine_id"] for p in provenance} != {t["id"] for t in self.config["turbines"]}:
            raise ValueError("Отсутствуют сведения об источнике погоды для одной из турбин.")
        for p in provenance:
            if utc(p["run_time"]) > utc(p["available_at"]) or utc(p["available_at"]) > as_of:
                raise ValueError("Погода стала доступна после момента расчёта: утечка будущего запрещена.")
            if not p.get("sha256") or not p.get("availability_basis"):
                raise ValueError("Источник погоды не указал версию и основание времени доступности.")

    def _analyse(self, rows, training):
        warnings = []
        outside = 0
        ranges = {}
        for turbine in self.config["turbines"]:
            winds = [r["wind_speed"] for r in training if r["turbine_id"] == turbine["id"]]
            ranges[turbine["id"]] = (min(winds), max(winds))
        for row in rows:
            if not all(math.isfinite(row[k]) and 0 <= row[k] <= 1 for k in ("power_pred", "lower", "upper")):
                raise ValueError("Модель вернула мощность вне допустимого диапазона.")
            if not row["lower"] <= row["power_pred"] <= row["upper"]:
                raise ValueError("Модель вернула некорректные границы полосы ошибки.")
            low, high = ranges[row["turbine_id"]]
            outside += int(row.get("wind_outside_training_range", not low <= row["wind_speed"] <= high))
        if outside:
            warnings.append(f"В {outside} строках ветер выходит за диапазон обучения: результат менее надёжен.")
        powers = [r["power_pred"] for r in rows]
        return {"mean_power": sum(powers) / len(powers), "min_power": min(powers), "max_power": max(powers),
                "out_of_range_hours": outside, "complete": True, "row_count": len(rows),
                "unit": "normalized_power", "aggregation": "equal_weight_mean_not_farm_energy"}, warnings

    def backtest(self, mode="archive", hours=48):
        start, end = utc(self.config["test_start"]), utc(self.config["test_end"])
        issue = start - timedelta(hours=1)
        result_rows, runs = [], []
        # Includes 28 February issue, whose targets lie in March; keep its audit but not its targets in February export.
        while issue < end:
            result = self.forecast(issue, hours, mode)
            runs.append({"id": result["id"], "as_of": result["as_of"], "hours": hours,
                         "execution": result.get("execution", "recomputed")})
            for row in result["rows"]:
                valid = utc(row["valid_time"])
                if start <= valid < end:
                    result_rows.append(self._export_row(result, row))
            issue += timedelta(days=1)
        coverage = {}
        for t in self.config["turbines"]:
            coverage[t["id"]] = len({r["valid_time"] for r in result_rows if r["turbine_id"] == t["id"]})
        expected = int((end - start).total_seconds() / 3600)
        if any(v != expected for v in coverage.values()):
            raise ValueError("Ретропрогноз не покрывает тестовый месяц полностью.")
        report = {"runs": len(runs), "rows": len(result_rows), "coverage_hours": min(coverage.values()),
            "coverage_by_turbine": coverage, "expected_hours": expected, "mode": mode, "hours": hours,
            "created_at": now(), "issue_runs": runs,
            "warnings": ["Ошибки за февраль не вычислены: необходимы отдельные фактические значения для оценки.",
                "Прогнозы с разных дат выпуска сохранены отдельно. CSV submission содержит только первые 24 ч каждого выпуска.",
                DEMO_WARNING if mode == "demo" else "Доступность и оперативное происхождение архива требуют подтверждения."],
            "competition_ready": False}
        executions = {run["execution"] for run in runs}
        report["execution"] = next(iter(executions)) if len(executions) == 1 else "mixed"
        report["execution_counts"] = {value: sum(run["execution"] == value for run in runs) for value in executions}
        save_json(self.workspace / "backtest_rows.json", result_rows)
        save_json(self.workspace / "backtest_report.json", report)
        self.state["backtest"] = report
        self._save()
        return report

    def _export_row(self, run, row):
        provenance = next(p for p in run["provenance"] if p["turbine_id"] == row["turbine_id"])
        return {"forecast_origin": run["as_of"], "valid_time": row["valid_time"], "turbine_id": row["turbine_id"],
            "lead_hours": int((utc(row["valid_time"]) - utc(run["as_of"])).total_seconds() / 3600),
            "power_pred": row["power_pred"], "lower": row["lower"], "upper": row["upper"],
            "persistence_pred": row.get("persistence_pred"), "wind_speed": row["wind_speed"], "temperature": row["temperature"],
            "weather_run": provenance["run_time"], "weather_available_at": provenance["available_at"],
            "availability_basis": provenance["availability_basis"], "weather_sha256": provenance["sha256"],
            "model_sha256": run["model_sha256"], "mode": run["mode"],
            "execution": run.get("execution", "recomputed"), "competition_ready": False}

    def export(self, kind="forecast"):
        if kind == "forecast":
            run = self.state["last_run"]
            if not run:
                raise ValueError("Сначала выполните прогноз.")
            rows = [self._export_row(run, row) for row in run["rows"]]
        elif kind in ("backtest", "submission"):
            if not self.state["backtest"]:
                raise ValueError("Сначала выполните ретропрогноз февраля.")
            rows = read_json(self.workspace / "backtest_rows.json")
            if kind == "submission":
                rows = [r for r in rows if 1 <= r["lead_hours"] <= 24]
        else:
            raise ValueError("Неизвестный вид экспорта.")
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        return stream.getvalue()
