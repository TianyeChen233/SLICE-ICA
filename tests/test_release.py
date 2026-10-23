from pathlib import Path
import csv
import importlib.util
import math
import statistics
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aggregation = load_module("summarize_runs", "tools/summarize_runs.py")
alignment = load_module("feature_aligner", "labeling/feature_aligner.py")


class ReleaseTests(unittest.TestCase):
    def test_mean_and_sample_sd(self):
        with (ROOT / "examples/synthetic_repeated_metrics.csv").open(newline="") as stream:
            result = aggregation.summarize(list(csv.DictReader(stream)))
        group = result["configurations"]["synthetic_demo"]
        self.assertEqual(group["n_runs"], 2)
        self.assertAlmostEqual(group["metrics"]["f1"]["mean"], 0.85)
        self.assertAlmostEqual(group["metrics"]["f1"]["sample_sd"], statistics.stdev([0.8, 0.9]))

    def test_missing_metric_is_rejected(self):
        with self.assertRaises(ValueError):
            aggregation.summarize([{"configuration": "demo", "f1": 0.8, "accuracy": 0.8}])

    def test_nonfinite_metric_is_rejected(self):
        with self.assertRaises(ValueError):
            aggregation.summarize([{"configuration": "demo", "f1": math.nan, "accuracy": 0.8, "precision": 0.8}])

    def test_missing_recall_is_not_partially_averaged(self):
        rows = [
            {"configuration": "demo", "f1": 0.8, "accuracy": 0.8, "precision": 0.8, "recall": 0.8},
            {"configuration": "demo", "f1": 0.9, "accuracy": 0.9, "precision": 0.9},
        ]
        result = aggregation.summarize(rows)
        self.assertNotIn("recall", result["configurations"]["demo"]["metrics"])

    def test_FIX_export_label_convention(self):
        labels, _ = alignment.parse_hand_labels_noise(ROOT / "examples/hand_labels_noise.txt")
        self.assertEqual(labels, {1: 1, 2: 0})


if __name__ == "__main__":
    unittest.main()
