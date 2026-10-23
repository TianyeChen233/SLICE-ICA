"""Synthetic-only regression tests; no training, participant input, or API calls."""

from __future__ import annotations

import ast
import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from classification.protocol import cross_fit, group_folds, validate_manifest, validate_records
from labeling.grounded_labeler import call_api, logistic_probability
from labeling.reference_context import ReferenceContext, fuse_probability, validate_response


def manifest():
    return {
        "identity_scope": "global_participant",
        "train_subjects": ["FIX:a", "FIX:b", "HCP:c"],
        "val_subjects": ["FIX:v"],
        "test_subjects": ["FIX:t"],
        "external_subjects": ["MSC:e"],
    }


def references():
    return {
        "feature_names": ["descriptor_a", "descriptor_b"],
        "references": [
            {
                "participant": "FIX:a",
                "session": "s",
                "ic_index": 1,
                "label_signal": 1,
                "feature_values_in_order": [0.1, 0.2],
            },
            {
                "participant": "FIX:b",
                "session": "s",
                "ic_index": 2,
                "label_signal": 0,
                "feature_values_in_order": [0.7, 0.8],
            },
        ],
    }


def target():
    return {
        "ic": 1,
        "ml_prob_signal": 0.5,
        "ml_label_signal": 1,
        "feature_values_in_order": [0.3, 0.4],
    }


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.context = ReferenceContext(references(), manifest())

    def test_reference_pairs_really_appear_in_every_payload(self):
        for person in ["FIX:t", "MSC:e", "HCP:c"]:
            payload = self.context.payload(person, [target()])
            self.assertEqual(
                [r["ground_truth_label_signal"] for r in payload["fix_reference_examples"]], [1, 0]
            )
            self.assertNotIn("ground_truth_label_signal", payload["items"][0])

    def test_test_reference_rejected(self):
        bundle = references()
        bundle["references"][0]["participant"] = "FIX:t"
        with self.assertRaises(ValueError):
            ReferenceContext(bundle, manifest())

    def test_same_participant_reference_rejected(self):
        with self.assertRaises(ValueError):
            self.context.payload("FIX:a", [target()])

    def test_unregistered_target_rejected(self):
        with self.assertRaises(ValueError):
            self.context.payload("unregistered", [target()])

    def test_target_gold_field_rejected(self):
        with self.assertRaises(ValueError):
            self.context.payload("MSC:e", [{**target(), "label_signal": 0}])

    def test_label_cannot_be_a_feature(self):
        bundle = references()
        bundle["feature_names"][0] = "label"
        with self.assertRaises(ValueError):
            ReferenceContext(bundle, manifest())

    def test_nonfinite_and_boolean_features_rejected(self):
        for value in [float("nan"), float("inf"), True]:
            item = target()
            item["feature_values_in_order"][0] = value
            with self.assertRaises(ValueError):
                self.context.payload("MSC:e", [item])

    def test_response_validation(self):
        row = {"ic": 1, "p_signal": 0.7, "override_ml": False, "reason": "synthetic"}
        self.assertEqual(validate_response([row], [target()])[0]["p_signal"], 0.7)
        for bad in [
            {**row, "ic": 2},
            {**row, "override_ml": "false"},
            {**row, "p_signal": float("nan")},
            {**row, "p_signal": True},
        ]:
            with self.assertRaises(ValueError):
                validate_response([bad], [target()])

    def test_fusion_matches_manuscript(self):
        self.assertAlmostEqual(fuse_probability(0.5, 0.8, False), 0.56)
        self.assertEqual(fuse_probability(0.5, 0.8, True), 0.8)

    def test_mock_api_has_reference_context_and_no_target_label(self):
        payload = self.context.payload("MSC:e", [target()])
        captured = []

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *unused):
                return False

            def read(self):
                content = json.dumps(
                    [{"ic": 1, "p_signal": 0.6, "override_ml": False, "reason": "synthetic"}]
                )
                return json.dumps({"choices": [{"message": {"content": content}}]}).encode()

        def fake_urlopen(req, timeout):
            captured.append(json.loads(req.data))
            return Response()

        with patch("labeling.grounded_labeler.request.urlopen", fake_urlopen):
            reply = call_api(
                "https://example.invalid/v1",
                "synthetic-not-a-secret",
                "demo",
                payload,
                timeout=1,
                attempts=1,
            )
        sent = json.loads(captured[0]["messages"][1]["content"])
        self.assertEqual(len(sent["fix_reference_examples"]), 2)
        self.assertEqual(set(sent["items"][0]), set(target()))
        self.assertEqual(reply[0]["ic"], 1)

    def test_logistic_extreme_values_do_not_overflow(self):
        import numpy as np

        model = {"mu": np.zeros(2), "std": np.ones(2), "w": np.ones(2), "b": 0}
        self.assertEqual(logistic_probability(model, [1000, 1000]), 1.0)
        self.assertEqual(logistic_probability(model, [-1000, -1000]), 0.0)


def records():
    return [
        {
            "subject": f"person_{i}",
            "session": "s",
            "ic_index": 1,
            "path": f"synthetic_{i}",
            "bucket": split,
            "is_pseudo": False,
        }
        for i, split in enumerate(["train", "train", "train", "val", "test"])
    ]


class SplitTests(unittest.TestCase):
    def test_overlapping_splits_rejected(self):
        m = manifest()
        m["test_subjects"] = ["FIX:a"]
        with self.assertRaises(ValueError):
            validate_manifest(m)

    def test_session_alias_overlap_rejected(self):
        m = manifest()
        m["participant_map"] = {"FIX:a": "same_person", "FIX:t": "same_person"}
        with self.assertRaises(ValueError):
            validate_manifest(m)

    def test_local_group_identity_not_silently_accepted(self):
        m = manifest()
        m.pop("identity_scope")
        with self.assertRaises(ValueError):
            validate_manifest(m)

    def test_record_partition_guards(self):
        validate_records(records(), [])
        for modified in [
            records() + [records()[0]],
            [{**r, "subject": "person_0"} for r in records()],
        ]:
            with self.assertRaises(ValueError):
                validate_records(modified, [])

    def test_duplicate_input_in_other_person_rejected(self):
        rows = records()
        rows[0]["content_hash"] = rows[-1]["content_hash"] = "same_synthetic_content"
        with self.assertRaises(ValueError):
            validate_records(rows, [])

    def test_pseudo_heldout_and_unknown_upstream_rejected(self):
        for split, upstream in [("test", ["person_0"]), ("train", ["person_4"]), ("train", None)]:
            pseudo = {
                "subject": "person_0",
                "session": "other",
                "ic_index": 2,
                "path": "pseudo",
                "bucket": split,
                "is_pseudo": True,
                "upstream_training_participants": upstream,
            }
            with self.assertRaises(ValueError):
                validate_records(records(), [pseudo])

    def test_group_fold_keeps_sessions_together(self):
        rows = records()[:3] + [{**records()[0], "session": "other"}]
        seen = []
        for train, held in group_folds(rows, 3, 42):
            self.assertFalse(
                {rows[i]["subject"] for i in train} & {rows[i]["subject"] for i in held}
            )
            seen.extend(held)
        self.assertEqual(sorted(seen), list(range(4)))

    def test_callbacks_never_select_on_oof_or_test(self):
        training = records()[:3]
        calls = []

        def fit(pool, selection, tag):
            calls.append((copy.deepcopy(pool), copy.deepcopy(selection), tag))
            return tag

        def infer(model, held):
            return [r["subject"] for r in held]

        blocks, model = cross_fit(training, [], [records()[3]], fit=fit, infer=infer, n_folds=3)
        for pool, selection, tag in calls[:-1]:
            held = blocks[int(tag[-1])][1]
            self.assertEqual(selection, [])
            self.assertFalse(set(held) & {r["subject"] for r in pool})
        self.assertEqual(model, "final")
        self.assertEqual(calls[-1][1], [records()[3]])
        self.assertNotIn("person_4", repr(calls))

    def test_upstream_weak_label_leak_excluded_from_oof(self):
        pseudo = {
            "subject": "auxiliary",
            "upstream_training_participants": ["person_0", "person_1", "person_2"],
        }
        calls = []
        cross_fit(
            records()[:3],
            [pseudo],
            [records()[3]],
            fit=lambda pool, selection, tag: calls.append((pool, tag)),
            infer=lambda model, held: None,
            n_folds=3,
        )
        for pool, tag in calls[:-1]:
            self.assertNotIn(pseudo, pool)
        self.assertIn(pseudo, calls[-1][0])

    def test_frozen_inference_has_no_training_or_threshold_calls(self):
        path = ROOT / "classification/scripts/run_hybrid_ablation_trial.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "collect_frozen_probs"
        )
        forbidden = {
            "fit",
            "evaluate",
            "find_best_threshold",
            "backward",
            "step",
            "load_state_dict",
        }
        names = {
            n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
            for n in ast.walk(function)
            if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))
        }
        self.assertFalse(names & forbidden)
        self.assertFalse(
            any(
                isinstance(n, ast.Name) and n.id == "target_records"
                for n in ast.walk(
                    next(
                        n
                        for n in tree.body
                        if isinstance(n, ast.FunctionDef) and n.name == "fit_deep_model"
                    )
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
