from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the complete wind forecasting experiment")
    parser.add_argument("--online", action="store_true", help="Allow downloads; default uses included cache")
    parser.add_argument("--train", action="store_true", help="Rebuild training data and refit models")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    offline = [] if args.online else ["--offline"]
    commands = []
    if args.train:
        commands.append(["src.data.build_features", "--start", "2024-12-31", "--end", "2026-02-28", "--allow-missing", *offline])
        commands.append(["src.models.train"])
    commands.extend([["src.agent.run", *offline], ["src.report"]])
    for command in commands:
        subprocess.run([sys.executable, "-m", *command], cwd=root, check=True)


if __name__ == "__main__":
    main()
