#!/usr/bin/env python3
"""Audit the explicit source-only publication set, without reading data/outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = (
    "README.md", "METHOD.md", "LICENSE.md", "THIRD_PARTY_NOTICES.md",
    "RELEASE_SCOPE.md", "pyproject.toml", ".gitignore", ".env.example",
)
DIRECTORIES = {"src": {".py"}, "scripts": {".py"}, "tests": {".py"}, "configs": {".yaml"}}
FORBIDDEN = (
    re.compile("/" + "home/" + r"(?!user(?:/|\b))[^/\s]+/"),
    re.compile("/" + "media/" + r"[^\s]+"),
    re.compile("gh" + r"[opusr]_[A-Za-z0-9]{30,}"),
    re.compile("github" + r"_pat_[A-Za-z0-9_]{30,}"),
    re.compile("AKIA" + r"[0-9A-Z]{16}"),
    re.compile("-----BEGIN " + r"(?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(r"(?i)(?:api_key|access_token|password)\s*[:=]\s*[\"'][A-Za-z0-9_+/=-]{12,}[\"']"),
)


def collect() -> list[dict]:
    selected = [ROOT / name for name in ROOT_FILES]
    for directory, suffixes in DIRECTORIES.items():
        for path in sorted((ROOT / directory).rglob("*")):
            if "__pycache__" in path.parts or path.is_dir():
                continue
            if path.is_symlink() or path.suffix not in suffixes:
                raise ValueError(f"Unexpected file in source directory: {path.relative_to(ROOT)}")
            selected.append(path)
    manifest = []
    for path in selected:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Not a regular source file: {path.relative_to(ROOT)}")
        raw = path.read_bytes()
        if len(raw) > 512_000 or b"\0" in raw:
            raise ValueError(f"Oversized/binary file: {path.relative_to(ROOT)}")
        content = raw.decode("utf-8")
        if any(pattern.search(content) for pattern in FORBIDDEN):
            raise ValueError(f"Potential private material: {path.relative_to(ROOT)}")
        manifest.append({"path": path.relative_to(ROOT).as_posix(),
                         "sha256": hashlib.sha256(raw).hexdigest(),
                         "bytes": len(raw), "content": content})
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit path/hash/size manifest")
    args = parser.parse_args()
    manifest = collect()
    if args.json:
        print(json.dumps([{key: value for key, value in item.items() if key != "content"}
                          for item in manifest], indent=2))
    else:
        print(f"PASS: {len(manifest)} source/config/documentation files; "
              f"{sum(item['bytes'] for item in manifest):,} bytes.")
        print("Only this allowlisted set may be published. This scan is not a guarantee "
              "against every possible secret; manually review changes too.")


if __name__ == "__main__":
    main()
