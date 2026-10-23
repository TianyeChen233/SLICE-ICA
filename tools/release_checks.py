#!/usr/bin/env python3
"""Offline checks for the current source tree and public-release contents."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "docs/files.sha256.json"
SKIP = {".git", "__pycache__", ".ruff_cache", ".pytest_cache", ".venv", "outputs", "local_data"}
SECRET = re.compile(
    r"sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"
)
PRIVATE_SERVER = re.compile(
    r"/home/[A-Za-z0-9_.-]+/|(?<!\\d)(?:10\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}|192\\.168\\.\\d{1,3}\\.\\d{1,3}|172\\.(?:1[6-9]|2\\d|3[01])\\.\\d{1,3}\\.\\d{1,3})(?!\\d)"
)
RAW_SUFFIXES = (".nii", ".nii.gz", ".pkl", ".pickle", ".pt", ".pth", ".ckpt", ".npz", ".npy")


def main():
    failures, actual_files = [], {}
    required = [
        "README.md",
        "RIGHTS.md",
        "requirements.txt",
        "docs/REPRODUCING.md",
        "docs/references.bib",
        MANIFEST,
        "docs/DATA_ACCESS.md",
        "data/results/manuscript_reported_results.json",
    ]
    for name in required:
        if not (ROOT / name).is_file():
            failures.append(f"Required file missing: {name}")
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(part in SKIP for part in relative.parts):
            continue
        name = relative.as_posix()
        if name != MANIFEST:
            actual_files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if (
            path.name.endswith(RAW_SUFFIXES)
            or path.name == ".env"
            or path.suffix in {".pem", ".key"}
        ):
            failures.append(f"Excluded data/credential file: {name}")
            continue
        if path.stat().st_size > 10 * 1024 * 1024:
            failures.append(f"Unexpected large file: {name}")
        try:
            content = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            failures.append(f"Unexpected binary file: {name}")
            continue
        if SECRET.search(content):
            failures.append(f"Credential signature: {name}")
        if PRIVATE_SERVER.search(content) and name != "tools/release_checks.py":
            failures.append(f"Private server path/address: {name}")
        try:
            if path.suffix == ".py":
                ast.parse(content, filename=name)
            elif path.suffix == ".json":
                json.loads(content)
        except (SyntaxError, ValueError):
            failures.append(f"Invalid source/JSON: {name}")
    if (ROOT / MANIFEST).is_file():
        expected = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))["files"]
        if actual_files != expected:
            changed = sorted(
                name
                for name in set(actual_files) | set(expected)
                if actual_files.get(name) != expected.get(name)
            )
            failures.extend(f"Current file checksum mismatch: {name}" for name in changed)
    if failures:
        print("\\n".join(failures))
        raise SystemExit(1)
    print(
        f"PASS: {len(actual_files)} current-file checksums, syntax and bounded publication checks."
    )
    print("Scope: software checks, not end-to-end scientific-result verification.")


if __name__ == "__main__":
    main()
