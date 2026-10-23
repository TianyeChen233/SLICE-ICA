"""Select balanced training-only demonstrations from an explicitly identified FIX source."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

try:
    from .reference_context import ReferenceContext, strict_json
except ImportError:
    from reference_context import ReferenceContext, strict_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-labeled-csv", type=Path, required=True)
    parser.add_argument("--logistic-model", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--reference-dataset", required=True)
    parser.add_argument("--source-description", required=True)
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.per_class < 1:
        parser.error("New output and positive per-class count required")
    manifest = strict_json(args.split_manifest)
    model = strict_json(args.logistic_model)
    aliases = manifest.get("participant_map", {})
    train = {aliases.get(x, x) for x in manifest["train_subjects"]}
    names, candidates = model["feature_names"], {0: [], 1: []}
    with args.aligned_labeled_csv.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            identity = f"{row['dataset']}:{row['group_id']}"
            person = aliases.get(identity, identity)
            if (
                row["dataset"] != args.reference_dataset
                or person not in train
                or row["is_labeled"] != "1"
            ):
                continue
            label = float(row["label"])
            if label not in (0.0, 1.0):
                raise ValueError("Invalid expert signal label")
            candidates[int(label)].append(
                {
                    "participant": person,
                    "session": row.get("session") or "not_recorded_in_source_table",
                    "ic_index": int(row["ic_index"]),
                    "label_signal": int(label),
                    "feature_values_in_order": [float(row[k]) for k in names],
                }
            )
    rng, selected = random.Random(args.seed), []
    for label in (0, 1):
        if len(candidates[label]) < args.per_class:
            raise ValueError("Insufficient approved training-only reference components")
        selected.extend(rng.sample(candidates[label], args.per_class))
    bundle = {
        "feature_names": names,
        "references": selected,
        "source_description": args.source_description,
        "selection_seed": args.seed,
        "per_class": args.per_class,
    }
    ReferenceContext(bundle, manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(bundle, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
