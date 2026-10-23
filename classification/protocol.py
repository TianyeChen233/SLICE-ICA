"""Fail-closed split guards and group out-of-fold planning (no model dependencies)."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

PROTOCOL_VERSION = "2026-10-24-partition-v1"


def validate_manifest(manifest: Mapping) -> dict[str, set[str]]:
    """IDs must identify people globally; map repeat session IDs via participant_map."""
    aliases = manifest.get("participant_map", {})
    if not isinstance(aliases, dict):
        raise ValueError("participant_map must be an object")
    if manifest.get("identity_scope") != "global_participant":
        raise ValueError(
            "Declare identity_scope=global_participant; session-local group IDs are not sufficient"
        )
    partitions = {}
    for split in ("train", "val", "test", "external"):
        values = manifest.get(f"{split}_subjects", [])
        if not isinstance(values, list) or any(
            not isinstance(x, str) or not x.strip() for x in values
        ):
            raise ValueError(f"Invalid {split} participant identifiers")
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate identifiers in {split}")
        canonical = [aliases.get(x, x) for x in values]
        if any(not isinstance(x, str) or not x.strip() for x in canonical):
            raise ValueError("Empty canonical participant ID")
        partitions[split] = set(canonical)
    if not all(partitions[s] for s in ("train", "val", "test")):
        raise ValueError("Nonempty train, val and test participant sets are required")
    names = list(partitions)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            if partitions[left] & partitions[right]:
                raise ValueError(f"Participant overlap: {left}/{right}")
    return partitions


def validate_records(expert: Sequence[dict], pseudo: Sequence[dict]) -> None:
    """Reject identity, file, or image/time-series duplication across held-out splits."""
    group_splits, identities, paths, contents = {}, {}, {}, {}
    counts = {s: 0 for s in ("train", "val", "test")}
    for record in [*expert, *pseudo]:
        group = record.get("participant", record.get("subject"))
        split = record.get("bucket")
        if not group or split not in counts:
            raise ValueError("Every record needs a canonical participant and split")
        if record.get("is_pseudo") and split != "train":
            raise ValueError("Pseudo labels may enter training only")
        if record.get("is_pseudo"):
            upstream = record.get("upstream_training_participants")
            train_people = {
                r.get("participant", r.get("subject")) for r in expert if r.get("bucket") == "train"
            }
            if not isinstance(upstream, list) or not upstream or not set(upstream) <= train_people:
                raise ValueError("Missing/held-out pseudo-labeler training/reference provenance")
        if group in group_splits and group_splits[group] != split:
            raise ValueError("Participant crosses splits")
        group_splits[group] = split
        if not record.get("is_pseudo"):
            counts[split] += 1
        identity = (group, record.get("session", ""), record.get("ic_index"))
        if identity[-1] is None or int(identity[-1]) < 1:
            raise ValueError("Positive IC index is required for duplicate checks")
        if identity in identities:
            raise ValueError("Duplicate component identity, including expert/pseudo overlap")
        identities[identity] = split
        for value, seen in [(str(record["path"]), paths), (record.get("content_hash"), contents)]:
            if value is None:
                continue
            if value in seen and seen[value] != split:
                raise ValueError("Identical input occurs across splits")
            seen[value] = split
    if not all(counts.values()):
        raise ValueError("Empty expert partition after input filtering")


def group_folds(records: Sequence[dict], n_folds: int, seed: int):
    """Yield train/held index lists; all sessions of a person remain together."""
    groups = sorted({r.get("participant") or r["subject"] for r in records})
    if not 2 <= n_folds <= len(groups):
        raise ValueError("OOF folds must be between 2 and the number of training participants")
    random.Random(seed).shuffle(groups)
    for fold in range(n_folds):
        held = set(groups[fold::n_folds])
        train_idx, held_idx = [], []
        for index, record in enumerate(records):
            group = record.get("participant") or record["subject"]
            (held_idx if group in held else train_idx).append(index)
        yield train_idx, held_idx


def cross_fit(
    expert_train,
    pseudo_train,
    validation,
    *,
    fit,
    infer,
    n_folds=5,
    seed=42,
    final_tag="final",
):
    """Callback orchestration: no test argument can reach fit/model selection.

    Each OOF fit uses fixed, predeclared epochs (empty selection set). A final
    model may use validation for early stopping. Neither receives held OOF labels.
    """
    blocks = []
    for fold, (train_idx, held_idx) in enumerate(group_folds(expert_train, n_folds, seed)):
        held = [expert_train[i] for i in held_idx]
        held_people = {r.get("participant") or r["subject"] for r in held}
        pool = [expert_train[i] for i in train_idx]
        for record in pseudo_train:
            upstream = record.get("upstream_training_participants")
            if not isinstance(upstream, list) or not upstream:
                raise ValueError("Pseudo-label provenance is required for nested OOF isolation")
            if (record.get("participant") or record["subject"]) not in held_people and not (
                set(upstream) & held_people
            ):
                pool.append(record)
        model = fit(pool, [], f"oof_{fold}")
        blocks.append((held_idx, infer(model, held)))
    final = fit([*expert_train, *pseudo_train], validation, final_tag)
    return blocks, final


def partition_cross_fit(
    expert_train,
    pseudo_train,
    validation,
    *,
    fit,
    infer,
    n_folds=5,
    seed=42,
):
    """Generate train OOF features and separately fitted validation/test encoders.

    Both partition-specific encoders fit the same permitted training pool and
    use validation only for checkpoint selection. There is intentionally no
    test argument: neither test inputs nor test labels can reach a fit callback.
    The training stack remains OOF, rather than in-sample training predictions.
    """

    def person(record):
        return record.get("participant") or record["subject"]

    training_people = {person(r) for r in [*expert_train, *pseudo_train]}
    reference_people = {person(r) for r in expert_train}
    selection_people = {person(r) for r in validation}
    if not training_people or not selection_people or training_people & selection_people:
        raise ValueError("Nonempty, participant-disjoint training/validation sets required")
    if any(r.get("bucket", "train") != "train" for r in [*expert_train, *pseudo_train]):
        raise ValueError("Only training records may enter encoder fitting")
    if any(r.get("bucket", "val") != "val" for r in validation):
        raise ValueError("Checkpoint selection accepts validation records only")
    for record in pseudo_train:
        upstream = record.get("upstream_training_participants")
        if not isinstance(upstream, list) or not upstream or not set(upstream) <= reference_people:
            raise ValueError("Training-only labeler/reference provenance is required")
    blocks, validation_model = cross_fit(
        expert_train,
        pseudo_train,
        validation,
        fit=fit,
        infer=infer,
        n_folds=n_folds,
        seed=seed,
        final_tag="validation",
    )
    test_model = fit([*expert_train, *pseudo_train], validation, "test")
    return blocks, {"validation": validation_model, "test": test_model}
