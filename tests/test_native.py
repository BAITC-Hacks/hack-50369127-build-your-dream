import hashlib
import importlib.util
import json
import os
import struct
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from wind_power_forecast import native

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap_eccodes_windows.py"
SPEC = importlib.util.spec_from_file_location("eccodes_bootstrap", SCRIPT)
bootstrap_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap_module)


def pe_image(dependency):
    """Minimal PE import table, sufficient to verify ABI-dependency rejection."""
    blob = bytearray(1024)
    blob[:2] = b"MZ"
    struct.pack_into("<I", blob, 0x3C, 64)
    blob[64:68] = b"PE\x00\x00"
    struct.pack_into("<HH", blob, 68, 0x8664, 1)
    struct.pack_into("<H", blob, 84, 240)
    struct.pack_into("<H", blob, 88, 0x20B)
    struct.pack_into("<I", blob, 208, 0x1000)
    struct.pack_into("<IIII", blob, 336, 512, 0x1000, 512, 512)
    struct.pack_into("<IIIII", blob, 512, 0, 0, 0, 0x1050, 0)
    name = dependency.encode("ascii") + b"\x00"
    blob[592 : 592 + len(name)] = name
    return bytes(blob)


def test_rejects_cross_version_cpython_dll_dependency():
    with pytest.raises(ValueError, match="CPython runtime"):
        bootstrap_module.dll_imports(pe_image("python313.dll"))
    assert bootstrap_module.dll_imports(pe_image("KERNEL32.dll")) == ["KERNEL32.dll"]


def test_bootstrap_rejects_unverified_wheel_before_extracting(tmp_path):
    wheel = tmp_path / bootstrap_module.WHEEL_NAME
    wheel.write_bytes(b"unverified")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="SHA256"):
        bootstrap_module.bootstrap(wheel, output)
    assert not output.exists()


def test_bootstrap_keeps_memfs_dependency_but_never_extracts_cross_version_pyd(
    tmp_path, monkeypatch
):
    wheel = tmp_path / bootstrap_module.WHEEL_NAME
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("eccodes/eccodes.dll", pe_image("eccodes_memfs.dll"))
        archive.writestr("eccodes/eccodes_memfs.dll", pe_image("KERNEL32.dll"))
        archive.writestr("eccodes/_eccodes.cp313-win_amd64.pyd", b"must never load")
    monkeypatch.setattr(
        bootstrap_module, "WHEEL_SHA256", hashlib.sha256(wheel.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(bootstrap_module, "PROJECT_ROOT", tmp_path)
    output = tmp_path / "bundle"
    manifest = bootstrap_module.bootstrap(wheel, output)
    assert {item["name"] for item in manifest["libraries"]} == {
        "eccodes.dll",
        "eccodes_memfs.dll",
        "libeccodes.dll",
    }
    assert not list(output.rglob("*.pyd"))
    assert manifest["python_extensions_extracted"] is False


@pytest.mark.skipif(os.name != "nt", reason="Windows DLL search helper")
def test_native_loader_rejects_modified_dll_without_loading(tmp_path):
    library_dir = tmp_path / "lib"
    library_dir.mkdir()
    (library_dir / "libeccodes.dll").write_bytes(b"modified")
    manifest = {
        "schema_version": 1,
        "libraries": [
            {"name": "libeccodes.dll", "sha256": hashlib.sha256(b"original").hexdigest()}
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (
        patch("wind_power_forecast.native.os.add_dll_directory") as load,
        pytest.raises(RuntimeError, match="checksum mismatch"),
    ):
        native.prepare_eccodes_native(tmp_path)
    load.assert_not_called()
