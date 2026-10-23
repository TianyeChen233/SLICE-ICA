"""Frozen ICA classification; no fitting or threshold search."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb

CLASSIFICATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLASSIFICATION))
from hcp_architecture import build_hcp_classifier
from inference_io import (
    build_batch_tensors,
    collate_batch,
    load_pkl_sample,
    to_stack,
)
from protocol import PROTOCOL_VERSION, validate_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--pkl-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument(
        "--gold-labels-confirmed",
        action="store_true",
        help="Confirm PKL labels are expert gold, not automatic labels/placeholders",
    )
    args = parser.parse_args()
    if args.output.exists() or args.batch_size < 1:
        parser.error("New output path and positive batch size required")
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    if (
        artifact.get("protocol_version") != PROTOCOL_VERSION
        or artifact.get("selection_partition") != "validation"
    ):
        parser.error("A validation-selected artifact with matching protocol metadata is required")
    threshold = float(artifact["threshold_from_val"])
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid frozen noise threshold")
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    partitions = validate_manifest(manifest)
    aliases = manifest.get("participant_map", {})
    files = sorted(args.pkl_root.rglob("*.pkl"))
    if not files:
        raise ValueError("No input PKL files")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = None
    if artifact.get("deep_checkpoints"):
        if (
            artifact.get("stack_encoder_protocol")
            != "participant_oof_train_separate_validation_test"
        ):
            raise ValueError("Missing partition-specific encoder protocol")
        checkpoint = Path(artifact["deep_checkpoints"]["frozen_test"])
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            payload.get("protocol_version") != PROTOCOL_VERSION
            or payload.get("selection_partition") != "validation"
            or payload.get("stack_partition") != "test"
        ):
            raise ValueError("Not a frozen test/external encoder selected using validation")
        if (
            not payload.get("training_participants")
            or not set(payload["training_participants"]) <= partitions["train"]
            or not payload.get("selection_participants")
            or not set(payload["selection_participants"]) <= partitions["val"]
        ):
            raise ValueError(
                "Checkpoint provenance does not match approved training/validation participants"
            )
        model = build_hcp_classifier(payload["image_backbone"], payload["temporal_backbone"])
        model.load_state_dict(payload["model"])
        model.to(device).eval()
    booster = None
    if artifact.get("xgb_model"):
        booster = xgb.XGBClassifier()
        booster.load_model(artifact["xgb_model"])
    if artifact["ablation"] in {"stack_embedding", "stack_embedding_pca185"}:
        raise ValueError("Unsupported embedding diagnostic")
    results, identities = [], set()
    for start in range(0, len(files), args.batch_size):
        samples = [load_pkl_sample(p) for p in files[start : start + args.batch_size]]
        image, tc, length, features, labels, metadata = collate_batch(samples)
        for meta in metadata:
            person = aliases.get(meta["subject"], meta["subject"])
            if person not in partitions["test"] | partitions["external"]:
                raise ValueError(
                    "Held-out inference input is absent from approved test/external participants"
                )
            identity = (person, meta["session"], meta["ic_index"])
            if identity in identities:
                raise ValueError("Duplicate held-out component")
            identities.add(identity)
        if model is not None:
            batch = build_batch_tensors(image, tc, length, features, device)
            batch.pop("label", None)
            if artifact["ablation"] in {"deep_image_only", "no_tc_no_manual"}:
                batch["time_course"].zero_()
            if artifact["ablation"] in {"deep_tc_only", "no_ic_no_manual"}:
                batch["image"].zero_()
            with torch.no_grad():
                probability = torch.sigmoid(model(batch)["predict"].view(-1)).cpu().numpy()
        if booster is not None:
            if model is None:
                stack = features
            else:
                stack = to_stack(probability, features)
                if artifact["ablation"] in {
                    "stack_no_manual",
                    "no_tc_no_manual",
                    "no_ic_no_manual",
                }:
                    stack = stack[:, :2]
                elif artifact["ablation"] == "stack_no_logit":
                    stack = np.concatenate([stack[:, :1], features], axis=1)
            probability = booster.predict_proba(stack)[:, 1]
        for meta, gold, score in zip(metadata, labels, probability):
            if gold not in (0, 1) or not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid label/probability")
            results.append(
                {
                    "participant": aliases.get(meta["subject"], meta["subject"]),
                    "session": meta["session"],
                    "ic": meta["ic_index"],
                    "gold_noise": int(gold) if args.gold_labels_confirmed else None,
                    "p_noise": float(score),
                    "predicted_noise": int(score >= threshold),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "protocol_version": PROTOCOL_VERSION,
                "threshold_source": "validation",
                "threshold": threshold,
                "artifact_sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
                "predictions": results,
            },
            stream,
            indent=2,
            allow_nan=False,
        )


if __name__ == "__main__":
    main()
