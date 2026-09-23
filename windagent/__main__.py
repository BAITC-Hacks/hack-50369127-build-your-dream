import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from .agent import ForecastAgent
from .common import encoded, load_config, save_json
from .evaluate import evaluate
from .server import serve


def main(argv=None):
    parser = argparse.ArgumentParser(description="Wind Agent — почасовой прогноз ВЭС")
    parser.add_argument("--workspace", type=Path, help="Папка результатов (по умолчанию runtime)")
    parser.add_argument("--config", type=Path, help="Путь к config.json")
    sub = parser.add_subparsers(dest="command", required=True)
    web = sub.add_parser("serve", help="Запустить локальную веб-панель")
    web.add_argument("--port", type=int, default=8080)
    demo = sub.add_parser("demo", help="Создать синтетическую историю и выполнить учебный прогноз")
    demo.add_argument("--backtest", action="store_true", help="Также выполнить весь февраль")
    ingest = sub.add_parser("import", help="Загрузить историю CSV и обучить модели")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--timezone-offset", type=float, default=5)
    ingest.add_argument("--power-scale", type=float, default=1)
    for command in ("forecast", "backtest", "watch"):
        p = sub.add_parser(command)
        p.add_argument("--mode", choices=("archive", "demo"), default="archive")
        p.add_argument("--hours", type=int, choices=(24, 48), default=48)
        if command != "backtest":
            p.add_argument("--as-of", default="2026-01-31T23:00:00+05:00")
        if command == "forecast":
            p.add_argument("--refresh", action="store_true")
        if command == "watch":
            p.add_argument("--interval", type=int, default=300)
            p.add_argument("--iterations", type=int, default=0, help="0 — до Ctrl+C")
            p.add_argument("--history", type=Path, help="Повторно импортировать CSV при изменении")
            p.add_argument("--timezone-offset", type=float, default=5)
            p.add_argument("--power-scale", type=float, default=1)
    out = sub.add_parser("export")
    out.add_argument("--kind", choices=("forecast", "backtest", "submission"), default="forecast")
    out.add_argument("--output", type=Path, required=True)
    score = sub.add_parser("evaluate", help="Оценить сохранённый CSV по отдельно предоставленным фактам")
    score.add_argument("--forecast", type=Path, required=True)
    score.add_argument("--actuals", type=Path, required=True)
    score.add_argument("--timezone-offset", type=float, default=5)
    score.add_argument("--power-scale", type=float, default=1)
    args = parser.parse_args(argv)
    try:
        agent = ForecastAgent(args.workspace, load_config(args.config))
        result = None
        if args.command == "serve":
            serve(agent, args.port)
        elif args.command == "demo":
            agent.demo()
            result = agent.forecast(agent.config["training_cutoff"], 48, "demo")
            if args.backtest:
                result = agent.backtest("demo", 48)
        elif args.command == "import":
            result = agent.import_csv(args.path.read_text(encoding="utf-8-sig"), args.path.name,
                                      args.timezone_offset, args.power_scale)
        elif args.command == "forecast":
            result = agent.forecast(args.as_of, args.hours, args.mode, args.refresh)
        elif args.command == "backtest":
            result = agent.backtest(args.mode, args.hours)
        elif args.command == "export":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(agent.export(args.kind), encoding="utf-8-sig", newline="")
            result = {"saved": str(args.output)}
        elif args.command == "evaluate":
            result = evaluate(args.forecast, args.actuals, args.timezone_offset, args.power_scale)
        elif args.command == "watch":
            if args.interval < 30 or args.iterations < 0:
                raise ValueError("Интервал проверки не меньше 30 секунд; число повторов неотрицательное.")
            previous_hash = None
            index = 0
            while args.iterations == 0 or index < args.iterations:
                try:
                    if args.history:
                        contents = args.history.read_bytes()
                        current_hash = hashlib.sha256(contents).hexdigest()
                        if current_hash != previous_hash:
                            agent.import_csv(contents.decode("utf-8-sig"), args.history.name,
                                             args.timezone_offset, args.power_scale)
                            previous_hash = current_hash
                    run = agent.forecast(args.as_of, args.hours, args.mode, refresh=True)
                    print(json.dumps({"id": run["id"], "as_of": run["as_of"], "reused": run["reused"]}), flush=True)
                except (ValueError, RuntimeError, OSError) as exc:
                    print(f"Ошибка цикла: {exc}", file=sys.stderr, flush=True)
                index += 1
                if args.iterations == 0 or index < args.iterations:
                    time.sleep(args.interval)
        if result is not None:
            # Full records live in runtime; concise CLI status remains readable for beginners.
            summary = {k: v for k, v in result.items() if k not in ("config", "rows", "model", "last_run", "events", "provenance", "issue_runs")}
            if isinstance(result.get("rows"), list):
                summary["row_count"] = len(result["rows"])
            elif "rows" in result:
                summary["rows"] = result["rows"]
            print(encoded(summary).decode("utf-8"))
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
