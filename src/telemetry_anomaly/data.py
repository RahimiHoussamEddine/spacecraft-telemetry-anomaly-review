"""Acquire and hash-validate the pinned, small OPSSAT-AD v2 data files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = ROOT / "data/source_manifest.json"
RAW_DIR = ROOT / "data/raw"
REPORTS_DIR = ROOT / "reports"


def _hashes(path: Path) -> dict[str, Any]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            size += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
    return {"bytes": size, "md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def _download(url: str, destination: Path, expected_md5: str) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = _hashes(destination)
        if actual["md5"] != expected_md5:
            raise ValueError(f"Existing file hash mismatch; refusing to replace {destination}")
        return {**actual, "reused": True}

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "portfolio-dataset-client/1.0"})
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            with (
                urllib.request.urlopen(request, timeout=90) as response,
                temporary.open("wb") as file,
            ):
                if response.status != 200:
                    raise ValueError(f"Expected HTTP 200, received {response.status}")
                while chunk := response.read(1024 * 1024):
                    file.write(chunk)
            actual = _hashes(temporary)
            if actual["md5"] != expected_md5:
                raise ValueError(
                    f"Pinned MD5 mismatch for {destination.name}: {actual['md5']} != {expected_md5}"
                )
            os.replace(temporary, destination)
            return {**actual, "reused": False}
        except (OSError, urllib.error.URLError, TimeoutError, ValueError) as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt == 0:
                continue
    raise RuntimeError(f"Could not download {destination.name}: {last_error}")


def fetch() -> dict[str, Any]:
    manifest = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    results: dict[str, Any] = {
        "doi": manifest["doi"],
        "version": manifest["version"],
        "license": manifest["license"],
        "files": {},
    }
    for name, info in manifest["recorded_files"].items():
        result = _download(info["url"], RAW_DIR / name, info["expected_md5"])
        results["files"][name] = result
        print(f"{name}: {result['bytes']:,} bytes; MD5 verified; SHA-256 {result['sha256']}")
    REPORTS_DIR.mkdir(exist_ok=True)
    output = REPORTS_DIR / "data_manifest.json"
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def validate_local() -> dict[str, Any]:
    manifest = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    report_path = REPORTS_DIR / "data_manifest.json"
    if not report_path.exists():
        raise FileNotFoundError("Run `make fetch` before validating local data")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for name, info in manifest["recorded_files"].items():
        path = RAW_DIR / name
        actual = _hashes(path)
        if actual["md5"] != info["expected_md5"]:
            raise ValueError(f"Local source hash mismatch for {name}")
        if actual["sha256"] != report["files"][name]["sha256"]:
            raise ValueError(f"Local SHA-256 changed for {name}")
        print(f"{name}: {actual['bytes']:,} bytes; MD5 and SHA-256 verified")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_local()
    else:
        fetch()


if __name__ == "__main__":
    main()
