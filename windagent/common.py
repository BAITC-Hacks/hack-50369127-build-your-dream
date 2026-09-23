"""Small shared helpers; timestamps are always timezone aware."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

UTC = timezone.utc
ROOT = Path(__file__).resolve().parent.parent


def utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Укажите время ISO 8601 с часовым поясом, например 2026-01-31T23:00:00+05:00.")
    return value.astimezone(UTC)


def iso(value):
    return utc(value).isoformat()


def now():
    return datetime.now(UTC).isoformat()


def json_default(value):
    if isinstance(value, datetime):
        return iso(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, default=json_default).encode("utf-8")


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded(value))
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_config(path=None):
    config = read_json(path or ROOT / "config.json")
    if len({item["id"] for item in config["turbines"]}) != len(config["turbines"]):
        raise ValueError("Идентификаторы турбин в config.json должны быть уникальными.")
    for item in config["turbines"]:
        if not (-90 <= item["latitude"] <= 90 and -180 <= item["longitude"] <= 180):
            raise ValueError("Некорректные координаты турбины.")
    return config

