"""Validated training-only FIX demonstrations for stateless chat-completion calls."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

META_FIELDS = {
    "label",
    "label_signal",
    "ground_truth",
    "dataset",
    "group_id",
    "subject",
    "session",
    "ic_index",
    "is_labeled",
}


def strict_json(path):
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Expected a finite numeric value, not a boolean/string")
    return float(value)


def split_sets(manifest):
    if manifest.get("identity_scope") != "global_participant":
        raise ValueError("Participant identifiers must be explicitly globally scoped")
    aliases = manifest.get("participant_map", {})
    if not isinstance(aliases, dict) or any(
        not isinstance(v, str) or not v.strip() for v in aliases.values()
    ):
        raise ValueError("Invalid canonical participant mapping")
    sets = {}
    for split in ("train", "val", "test", "external"):
        values = manifest.get(f"{split}_subjects", [])
        if not isinstance(values, list) or any(
            not isinstance(v, str) or not v.strip() for v in values
        ):
            raise ValueError("Invalid participant manifest")
        if len(values) != len(set(values)):
            raise ValueError("Duplicate split identifiers")
        sets[split] = {aliases.get(v, v) for v in values}
    if not all(sets[s] for s in ("train", "val", "test")):
        raise ValueError("Explicit nonempty train, val, test sets are required")
    for i, left in enumerate(sets):
        for right in list(sets)[i + 1 :]:
            if sets[left] & sets[right]:
                raise ValueError("Reference/calibration/test participant overlap")
    return sets


class ReferenceContext:
    """No demonstration is drawn from a held-out participant or the target person."""

    def __init__(self, bundle, manifest):
        self.partitions = split_sets(manifest)
        self.aliases = manifest.get("participant_map", {})
        self.feature_names = bundle["feature_names"]
        if (
            not isinstance(self.feature_names, list)
            or not self.feature_names
            or len(self.feature_names) != len(set(self.feature_names))
            or any(
                not isinstance(k, str) or not k.strip() or k.lower() in META_FIELDS
                for k in self.feature_names
            )
        ):
            raise ValueError(
                "Feature names must be unique descriptors, never labels/identity fields"
            )
        self.references = bundle["references"]
        if not self.references:
            raise ValueError("FIX reference demonstrations are required")
        seen, classes = set(), set()
        for example in self.references:
            person = self.person(example["participant"])
            if person not in self.partitions["train"]:
                raise ValueError("Every demonstration must belong to the approved training split")
            key = (person, example["session"], example["ic_index"])
            if key in seen or not isinstance(key[2], int) or isinstance(key[2], bool) or key[2] < 1:
                raise ValueError("Duplicate/invalid reference component")
            seen.add(key)
            label = example["label_signal"]
            if type(label) is not int or label not in (0, 1):
                raise ValueError("Reference labels must use signal=1, noise=0")
            classes.add(label)
            self.vector(example["feature_values_in_order"])
        if classes != {0, 1}:
            raise ValueError("Reference context must contain both expert-annotated classes")
        encoded = json.dumps(bundle, sort_keys=True, allow_nan=False).encode("utf-8")
        self.sha256 = hashlib.sha256(encoded).hexdigest()

    def person(self, value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Global participant ID is required")
        return self.aliases.get(value, value)

    def vector(self, values):
        if not isinstance(values, list) or len(values) != len(self.feature_names):
            raise ValueError("Feature vector does not match reference feature order")
        return [finite_number(x) for x in values]

    def for_target(self, participant):
        person = self.person(participant)
        if person not in set().union(*self.partitions.values()):
            raise ValueError("Target participant is absent from the approved manifest")
        if any(self.person(r["participant"]) == person for r in self.references):
            raise ValueError("Target participant also occurs in the demonstration context")
        return [
            {
                "reference_id": f"FIX-reference-{i + 1}",
                "feature_values_in_order": self.vector(r["feature_values_in_order"]),
                "ground_truth_label_signal": r["label_signal"],
            }
            for i, r in enumerate(self.references)
        ]

    def payload(self, participant, items):
        safe_items = []
        allowed = {"ic", "ml_prob_signal", "ml_label_signal", "feature_values_in_order"}
        for item in items:
            if set(item) != allowed:
                raise ValueError(
                    "Unexpected target field; target ground-truth labels are forbidden"
                )
            probability = finite_number(item["ml_prob_signal"])
            if (
                not 0 <= probability <= 1
                or type(item["ml_label_signal"]) is not int
                or item["ml_label_signal"] not in (0, 1)
            ):
                raise ValueError("Invalid logistic prediction")
            if not isinstance(item["ic"], int) or isinstance(item["ic"], bool) or item["ic"] < 1:
                raise ValueError("Target IC index must be positive")
            safe_items.append(
                {**item, "feature_values_in_order": self.vector(item["feature_values_in_order"])}
            )
        if not safe_items or len({x["ic"] for x in safe_items}) != len(safe_items):
            raise ValueError("Empty batch or repeated target ICs")
        return {
            "label_definition": {"0": "noise", "1": "signal"},
            "feature_names_in_order": self.feature_names,
            "fix_reference_examples": self.for_target(participant),
            "items": safe_items,
        }


def validate_response(response, items):
    if not isinstance(response, list) or len(response) != len(items):
        raise ValueError("Response length/order must match the request")
    result = []
    for row, item in zip(response, items):
        if not isinstance(row, dict) or type(row.get("ic")) is not int or row["ic"] != item["ic"]:
            raise ValueError("Response IC order/identity mismatch")
        if type(row.get("override_ml")) is not bool or not isinstance(row.get("reason"), str):
            raise ValueError("Invalid override/reason types")
        probability = min(1.0, max(0.0, finite_number(row["p_signal"])))
        result.append({**row, "p_signal": probability, "reason": row["reason"][:1000]})
    return result


def fuse_probability(ml, llm, override):
    if type(override) is not bool:
        raise ValueError("Override must be a JSON boolean")
    return llm if override else 0.8 * ml + 0.2 * llm
