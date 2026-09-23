"""Optional process-local ecCodes DLL discovery for Windows Python 3.14.

The Python bindings use CFFI's C ABI and support an external native library.
The bootstrap extracts only the native DLLs from a checksum-pinned ECMWF wheel;
no CPython extension from a different interpreter version is imported.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

_DLL_DIRECTORY_HANDLES: dict[str, object] = {}


def prepare_eccodes_native(native_root: Path | None = None) -> Path | None:
    """Enable the optional workspace DLL bundle before importing ``eccodes``.

    A normal ecCodes installation remains usable without this bundle. Environment
    changes affect only this Python process, never system/user configuration.
    """
    if os.name != "nt":
        return None
    if native_root is None:
        native_root = Path(__file__).resolve().parents[2] / ".cache" / "eccodes-native"
    native_root = Path(native_root).resolve()
    manifest_path = native_root / "manifest.json"
    if not manifest_path.is_file():
        return None
    library_dir = native_root / "lib"
    if str(library_dir) in _DLL_DIRECTORY_HANDLES:
        return native_root
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise RuntimeError("Unsupported workspace ecCodes native manifest")
    entries = manifest.get("libraries", [])
    if not entries or not any(entry.get("name") == "libeccodes.dll" for entry in entries):
        raise RuntimeError("Workspace ecCodes bundle is missing libeccodes.dll")
    for entry in entries:
        name = entry["name"]
        if Path(name).name != name or not name.lower().endswith(".dll"):
            raise RuntimeError("Invalid DLL name in workspace ecCodes manifest")
        path = library_dir / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Workspace ecCodes DLL checksum mismatch: {name}")
    handle = os.add_dll_directory(str(library_dir))
    _DLL_DIRECTORY_HANDLES[str(library_dir)] = handle
    os.environ["ECCODES_PYTHON_USE_FINDLIBS"] = "1"
    # findlibs explicitly documents ECCODES_DIR/lib/libeccodes.dll discovery.
    os.environ["ECCODES_DIR"] = str(native_root)
    return native_root
