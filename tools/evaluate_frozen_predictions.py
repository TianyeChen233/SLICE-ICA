"""Report fixed-threshold noise-positive metrics per dataset and pooled; no fitting."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def metrics(rows, threshold):
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for row in rows:
        gold, probability = row["gold_noise"], row["p_noise"]
        if (
            type(gold) is not int
            or gold not in (0, 1)
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError("Binary noise labels and finite probabilities required")
        prediction = probability >= threshold
        key = ("tp" if prediction else "fn") if gold else ("fp" if prediction else "tn")
        counts[key] += 1
    n = len(rows)
    if not n:
        raise ValueError("Empty evaluation set")
    tp, tn, fp, fn = [counts[k] for k in ("tp", "tn", "fp", "fn")]
    return {
        "n": n,
        "n_noise": tp + fn,
        "n_signal": tn + fp,
        **counts,
        "accuracy": (tp + tn) / n,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def summarize(inputs):
    threshold, artifact = None, None
    seen, combined, datasets = set(), [], {}
    for dataset, document in inputs:
        if dataset in datasets or document.get("threshold_source") != "validation":
            raise ValueError(
                "Unique dataset names and frozen validation-selected thresholds required"
            )
        current, digest = float(document["threshold"]), document["artifact_sha256"]
        if not math.isfinite(current) or not 0 <= current <= 1 or not digest:
            raise ValueError("Invalid frozen artifact/threshold")
        if threshold is not None and (current != threshold or digest != artifact):
            raise ValueError("Pooled inputs must use the same frozen model and threshold")
        threshold, artifact = current, digest
        rows = document["predictions"]
        for row in rows:
            identity = row["participant"], row["session"], row["ic"]
            if identity in seen:
                raise ValueError("Duplicate component across evaluation inputs")
            seen.add(identity)
        datasets[dataset] = metrics(rows, threshold)
        combined.extend(rows)
    return {
        "positive_class": "noise",
        "threshold_source": "validation",
        "threshold": threshold,
        "artifact_sha256": artifact,
        "per_dataset": datasets,
        "pooled": metrics(combined, threshold),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="DATASET=predictions.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    documents = []
    for item in args.input:
        name, path = item.split("=", 1)
        documents.append((name, json.loads(Path(path).read_text(encoding="utf-8"))))
    result = summarize(documents)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
