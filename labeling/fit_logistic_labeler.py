"""Explicit training-only logistic fit and validation-only calibration; never auto-run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .logistic import fit_logreg, metrics, predict_prob
    from .reference_context import META_FIELDS, split_sets, strict_json
except ImportError:
    from logistic import fit_logreg, metrics, predict_prob
    from reference_context import META_FIELDS, split_sets, strict_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-labeled-csv", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.39,
        help="Predeclared signal threshold; never chosen on test",
    )
    args = parser.parse_args()
    if args.output.exists() or not 0 <= args.threshold <= 1:
        parser.error("New output path and valid predeclared threshold required")
    manifest = strict_json(args.split_manifest)
    partitions = split_sets(manifest)
    aliases = manifest.get("participant_map", {})
    with args.aligned_labeled_csv.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        features = [k for k in reader.fieldnames or [] if k.lower() not in META_FIELDS]
        rows = list(reader)
    selected = {s: [] for s in ("train", "val")}
    people = {s: set() for s in selected}
    for row in rows:
        person = aliases.get(
            f"{row['dataset']}:{row['group_id']}", f"{row['dataset']}:{row['group_id']}"
        )
        for split in selected:
            if person in partitions[split] and row["is_labeled"] == "1":
                selected[split].append(row)
                people[split].add(person)
    arrays = {}
    for split, subset in selected.items():
        if not subset:
            raise ValueError(f"Empty approved {split} labeler input")
        x = np.asarray([[float(row[k]) for k in features] for row in subset])
        y = np.asarray([float(row["label"]) for row in subset])
        if (
            not np.all(np.isfinite(x))
            or not set(y.tolist()) <= {0.0, 1.0}
            or len(set(y.tolist())) != 2
        ):
            raise ValueError("Finite features and both binary signal-label classes required")
        arrays[split] = x, y.astype(int)
    model = fit_logreg(*arrays["train"])
    xv, yv = arrays["val"]
    artifact = {k: v.tolist() if isinstance(v, np.ndarray) else float(v) for k, v in model.items()}
    artifact.update(
        {
            "feature_names": features,
            "fitted_subjects": sorted(people["train"]),
            "calibration_subjects": sorted(people["val"]),
            "selection_partition": "validation",
            "predeclared_threshold": args.threshold,
            "validation_metric": metrics(yv, predict_prob(model, xv), args.threshold),
            "split_manifest_sha256": hashlib.sha256(args.split_manifest.read_bytes()).hexdigest(),
            "implementation": "2026-10-23-safe-v1",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(artifact, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
