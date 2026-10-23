#!/usr/bin/env python3
"""Aggregate complete run-level metrics; never reconstruct or pool a confusion matrix."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

REQUIRED = ("f1", "accuracy", "precision")
ALIASES = {"acc": "accuracy", "pre": "precision", "rec": "recall", "model": "configuration", "mode": "configuration"}


def summarize(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for index, original in enumerate(rows, 1):
        row = {ALIASES.get(str(k).strip().lower(), str(k).strip().lower()): v for k, v in original.items()}
        group = str(row.get("configuration", "")).strip()
        if not group:
            raise ValueError(f"Row {index}: missing configuration")
        for key in REQUIRED:
            try:
                value = float(row[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Row {index}: missing or invalid {key}") from exc
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Row {index}: {key} must be finite and in [0,1]")
            row[key] = value
        if row.get("recall") not in (None, ""):
            value = float(row["recall"])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Row {index}: invalid recall")
            row["recall"] = value
        groups[group].append(row)
    if not groups:
        raise ValueError("No run rows supplied")
    result = {"aggregation": "unweighted arithmetic mean of per-run metrics", "configurations": {}}
    for group, records in sorted(groups.items()):
        metrics = list(REQUIRED)
        if all(r.get("recall") not in (None, "") for r in records):
            metrics.append("recall")
        summary = {"n_runs": len(records), "seeds": [r.get("seed") for r in records], "metrics": {}}
        for key in metrics:
            values = [r[key] for r in records]
            summary["metrics"][key] = {
                "mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            }
        result["configurations"][group] = summary
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as stream:
        result = summarize(list(csv.DictReader(stream)))
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
