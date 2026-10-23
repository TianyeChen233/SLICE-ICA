"""Attach *new grounded-run* provenance to matching trusted PKLs, in a new directory."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl-root", type=Path, required=True, help="Trusted local PKLs only")
    parser.add_argument("--label-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        parser.error("A new output directory is required; never overwrite source records")
    config = json.loads((args.label_run / "run_config.json").read_text(encoding="utf-8"))
    if (
        config.get("implementation") != "2026-10-23-grounded-v1"
        or config.get("run_kind") != "grounded_labeling"
    ):
        raise ValueError("Only newly generated grounded-run provenance is accepted")
    source_files = sorted(args.pkl_root.rglob("*.pkl"))
    if not source_files:
        raise ValueError("No trusted PKLs found")
    cache = {}
    for source in source_files:
        with source.open("rb") as stream:
            record = pickle.load(stream)
        subject, session, ic = (
            str(record["subject"]),
            str(record["session"]),
            int(record["ic_index"]),
        )
        key = subject, session
        if key not in cache:
            folder = args.label_run / subject / session
            reasons = [
                json.loads(line)
                for line in (folder / "reasons.jsonl").read_text().splitlines()
                if line
            ]
            features = [
                json.loads(line)
                for line in (folder / "ml_predictions.jsonl").read_text().splitlines()
                if line
            ]
            cache[key] = ({row["ic"]: row for row in reasons}, {row["ic"]: row for row in features})
        reasons, features = cache[key]
        row, inputs = reasons[ic], features[ic]
        # Prevent attaching new provenance to an unrelated/misaligned old record.
        observed = np.asarray(record["features"], dtype=np.float32).reshape(-1)
        expected = np.asarray(inputs["feature_values_in_order"], dtype=np.float32)
        if observed.shape != expected.shape or not np.array_equal(observed, expected):
            raise ValueError("PKL descriptors do not match the new grounded request")
        if float(record["label"]) != 1 - row["final_label_signal"]:
            raise ValueError("PKL noise target differs from the newly generated signal label")
        record["ml_prob_signal"] = row["confidence_probability"]
        record["upstream_training_participants"] = row["upstream_training_participants"]
        record["label_implementation"] = config["implementation"]
        record["reference_context_sha256"] = config["reference_context_sha256"]
        target = args.output_root / source.relative_to(args.pkl_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            pickle.dump(record, stream, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
