"""Prepare external native ecCodes DLLs without installing a cross-version wheel.

Download the official binary wheel (do not install it into Python 3.14):
python -m pip download eccodes==2.48.0 --platform win_amd64 --python-version 313 \
    --implementation cp --abi cp313 --only-binary=:all: --no-deps \
    --dest .cache/eccodes-wheels

Then run this script with --wheel PATH. Only native DLLs and licensing documents
are read; Python files and all .pyd extensions are ignored. The official PyPI
release SHA256 is pinned below. No network access is performed by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import zipfile
from pathlib import Path, PurePosixPath

WHEEL_NAME = "eccodes-2.48.0-cp313-cp313-win_amd64.whl"
WHEEL_SHA256 = "764834c8e01580e8a9977876da5e4c988fb2e9fa2d111f4047e97bcd1aab7e75"
SOURCE_URL = "https://pypi.org/project/eccodes/2.48.0/#files"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def dll_imports(blob: bytes) -> list[str]:
    """Read the PE import table to reject any native DLL tied to a CPython ABI."""
    if blob[:2] != b"MZ":
        raise ValueError("Native library is not a Windows PE image")
    pe_offset = struct.unpack_from("<I", blob, 0x3C)[0]
    if blob[pe_offset : pe_offset + 4] != b"PE\x00\x00":
        raise ValueError("Native library has an invalid PE signature")
    machine, section_count = struct.unpack_from("<HH", blob, pe_offset + 4)
    optional_size = struct.unpack_from("<H", blob, pe_offset + 20)[0]
    optional = pe_offset + 24
    if machine != 0x8664 or struct.unpack_from("<H", blob, optional)[0] != 0x20B:
        raise ValueError("Only native AMD64 PE32+ libraries are supported")
    sections = []
    for index in range(section_count):
        offset = optional + optional_size + index * 40
        virtual_size, virtual_address, raw_size, raw_offset = struct.unpack_from(
            "<IIII", blob, offset + 8
        )
        sections.append((virtual_address, max(virtual_size, raw_size), raw_offset))

    def file_offset(rva):
        for address, length, raw_offset in sections:
            if address <= rva < address + length:
                return raw_offset + rva - address
        raise ValueError("PE import table refers to an invalid section")

    import_rva = struct.unpack_from("<I", blob, optional + 120)[0]
    if not import_rva:
        return []
    imports = []
    offset = file_offset(import_rva)
    while True:
        descriptor = struct.unpack_from("<IIIII", blob, offset)
        if not any(descriptor):
            break
        name_offset = file_offset(descriptor[3])
        name_end = blob.index(b"\x00", name_offset)
        name = blob[name_offset:name_end].decode("ascii")
        if name.lower().startswith("python") and name.lower().endswith(".dll"):
            raise ValueError(f"A native library depends on the CPython runtime {name}")
        imports.append(name)
        offset += 20
    return imports


def bootstrap(wheel: Path, output: Path) -> dict:
    wheel = wheel.resolve()
    output = output.resolve()
    if wheel.name != WHEEL_NAME:
        raise ValueError(f"Expected the verified official wheel {WHEEL_NAME}")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if digest != WHEEL_SHA256:
        raise ValueError("Wheel SHA256 differs from the pinned official PyPI release")
    if not output.is_relative_to(PROJECT_ROOT):
        raise ValueError("The native DLL output must stay inside this workspace")
    library_dir = output / "lib"
    libraries = {}
    licenses = {}
    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = PurePosixPath(info.filename).name
            if name.lower().endswith(".dll"):
                if name.lower() in {key.lower() for key in libraries}:
                    raise ValueError(f"Duplicate native library basename in wheel: {name}")
                blob = archive.read(info)
                libraries[name] = (blob, info.filename, dll_imports(blob))
            elif ".dist-info/licenses/" in info.filename:
                licenses[name] = archive.read(info)
    eccodes_names = [
        name
        for name in libraries
        if name.lower() in ("eccodes.dll", "libeccodes.dll")
        or name.lower().startswith(("eccodes-", "libeccodes-"))
    ]
    if len(eccodes_names) != 1:
        raise ValueError(f"Expected exactly one native ecCodes DLL, found {eccodes_names}")
    # findlibs expects an unversioned basename. Keep the vendor filename as well
    # because dependency import tables may refer to it.
    native_name = eccodes_names[0]
    if native_name != "libeccodes.dll":
        libraries["libeccodes.dll"] = libraries[native_name]
    library_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, (blob, member, imports) in sorted(libraries.items()):
        path = library_dir / name
        path.write_bytes(blob)
        entries.append(
            {
                "name": name,
                "source_member": member,
                "sha256": hashlib.sha256(blob).hexdigest(),
                "imports": imports,
            }
        )
    if licenses:
        license_dir = output / "licenses"
        license_dir.mkdir(parents=True, exist_ok=True)
        for name, blob in licenses.items():
            (license_dir / name).write_bytes(blob)
    manifest = {
        "schema_version": 1,
        "source_wheel": wheel.name,
        "source_sha256": digest,
        "source_url": SOURCE_URL,
        "libraries": entries,
        "python_extensions_extracted": False,
        "native_library_path": str(library_dir / "libeccodes.dll"),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def selfcheck(output: Path) -> dict:
    if os.name != "nt":
        raise RuntimeError("The native library self-check requires Windows")
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from wind_power_forecast.native import prepare_eccodes_native

    prepare_eccodes_native(output)
    import eccodes

    handle = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    try:
        eccodes.codes_set(handle, "dataDate", 20260201)
        eccodes.codes_set(handle, "dataTime", 600)
        encoded = eccodes.codes_get_message(handle)
    finally:
        eccodes.codes_release(handle)
    decoded = eccodes.codes_new_from_message(encoded)
    try:
        check = {
            "native_version": eccodes.codes_get_api_version(),
            "edition": eccodes.codes_get(decoded, "edition"),
            "dataDate": eccodes.codes_get(decoded, "dataDate"),
            "dataTime": eccodes.codes_get(decoded, "dataTime"),
            "value_count": len(eccodes.codes_get_values(decoded)),
            "definitions_path": eccodes.codes_definition_path(),
            "samples_path": eccodes.codes_samples_path(),
        }
        if check["edition"] != 2 or check["dataDate"] != 20260201 or check["dataTime"] != 600:
            raise RuntimeError("Native ecCodes sample round-trip validation failed")
        return check
    finally:
        eccodes.codes_release(decoded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / ".cache" / "eccodes-native")
    parser.add_argument("--skip-selfcheck", action="store_true")
    args = parser.parse_args()
    manifest = bootstrap(args.wheel, args.output)
    if not args.skip_selfcheck:
        manifest["selfcheck"] = selfcheck(args.output)
        (args.output / "selfcheck.json").write_text(
            json.dumps(manifest["selfcheck"], indent=2), encoding="utf-8"
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
