"""Check manuscript-to-summary transcription and printed seed arithmetic, not experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(path):
    source = path.read_text(encoding="utf-8-sig")
    text = re.sub(r"\\textbf\{([^{}]+)\}", r"\1", source)
    numeric_rows = []
    for line in text.splitlines():
        cells = line.split("&")
        values = []
        for cell in cells:
            cleaned = cell.strip().rstrip("\\").strip()
            if re.fullmatch(r"0(?:\.\d+)?|1(?:\.0+)?", cleaned):
                values.append(float(cleaned))
        if len(values) >= 3:
            numeric_rows.append(values)
    reported = json.loads((ROOT / "data/results/manuscript_reported_results.json").read_text())
    expected = [
        list(row[k] for k in ("f1", "accuracy", "precision"))
        for row in reported["primary_results"].values()
    ]
    expected += [
        list(row[k] for k in ("mean_f1", "mean_accuracy", "mean_precision"))
        for row in reported["component_ablations"]["rows"]
    ]
    expected += [
        list(row[k] for k in ("f1", "accuracy", "precision", "recall"))
        for row in reported["backbone_comparison"]["rows"]
    ]
    missing = [
        row for row in expected if not any(observed[: len(row)] == row for observed in numeric_rows)
    ]
    if missing:
        raise ValueError(f"Summary rows missing from manuscript: {missing}")
    series = reported["separate_hybrid_seed_repeats"]
    if round(statistics.mean(series["f1"]), 3) != series["mean_f1_rounded"]:
        raise ValueError("Printed seed mean mismatch")
    if round(statistics.stdev(series["f1"]), 3) != series["sample_sd_rounded"]:
        raise ValueError("Printed seed sample-SD mismatch")
    return {
        "manuscript_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "classification_rows_matched": len(expected),
        "seed_arithmetic": "consistent",
        "verification_scope": "transcription and seed arithmetic only; source experiments not rerun",
        "author_confirmation": "User confirms manuscript data are correct; values are preserved, not regenerated",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.manuscript), indent=2))


if __name__ == "__main__":
    main()
