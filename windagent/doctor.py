"""Local environment diagnosis without network calls or model deserialization."""
import json
import subprocess
import sys

from .project import model_python


def diagnose(agent):
    backend = {"ready": False, "python": None, "catboost_version": None}
    try:
        runner = model_python()
        backend["python"] = runner
        if runner:
            probe = subprocess.run([runner, "-B", "-c",
                "import catboost,json; print(json.dumps({'version':catboost.__version__}))"],
                capture_output=True, timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if probe.returncode:
                backend["error"] = probe.stderr.decode("utf-8", errors="replace")[-1200:]
            else:
                backend["catboost_version"] = json.loads(probe.stdout)["version"]
                backend["ready"] = True
        else:
            backend["error"] = "Среда CatBoost не найдена. Инструкция установки — README.md."
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        backend["error"] = str(exc)
    dataset = agent.state.get("dataset") or {}
    imported = dataset.get("kind") == "project_archive"
    return {"server_python": sys.version.split()[0], "workspace": str(agent.workspace),
        "dataset_kind": dataset.get("kind"), "dataset_rows": dataset.get("rows", 0),
        "timezone_offset_hours": agent.config["timezone_offset_hours"], "model_backend": backend,
        "can_recompute_project": imported and backend["ready"],
        "note": "Это проверка среды. Для проверки модели и входов выполните forecast --mode project --refresh.",
        "next_command": "python -B -m windagent forecast --mode project --hours 48 --refresh" if imported else
                        'python -B -m windagent import-project "путь к wind-forecast-hackathon.zip"'}
