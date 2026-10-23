#!/usr/bin/env python3
"""
Single-trial hybrid pipeline runner for ablation studies.
Participant-group cross-fitting with validation-only model selection.

Environment:
  HCP_EXP_DIR       (required) experiment root; must contain audit/split_manifest.json
  HCP_ABLATION      (required) one of:
                      full, no_stack_xgb, stack_no_manual, stack_no_logit,
                      deep_image_only, deep_tc_only, manual_xgb_only,
                      no_tc_no_manual, no_ic_no_manual,
                      stack_embedding, stack_embedding_pca185
  HCP_PROJECT_ROOT  default classification
  HCP_TRUE_DIR      default $PROJECT_ROOT/data/expert
  HCP_PSEUDO_DIR    pseudo flat / cache
  HCP_CUDA_DEVICE   default 0  (within visible GPUs)
  HCP_PARAMS_JSON   optional path to JSON with same shape as final_best_config.json "params";
                    default: read HCP_B0_EXP_DIR/audit/final_best_config.json
  HCP_IMAGE_BACKBONE   spatial encoder: convnextv2 (default), resnet18, swin_t, efficientnet_b0, cnn
  HCP_TEMPORAL_BACKBONE  sequence encoder: lstm (default), gru, mamba, transformer

Participant OOF stacking and validation-only checkpoint selection.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(os.environ.get("HCP_PROJECT_ROOT", "classification"))
EXP_DIR = Path(os.environ["HCP_EXP_DIR"])
ABLATION = os.environ["HCP_ABLATION"].strip()

TRUE_DIR = Path(os.environ.get("HCP_TRUE_DIR", str(PROJECT_ROOT / "data" / "expert")))
_PSEUDO_FLAT_DEFAULT = PROJECT_ROOT / "data" / "pseudo"
_PSEUDO_FALLBACK = PROJECT_ROOT / "data" / "pseudo"
PSEUDO_DIR = Path(
    os.environ.get(
        "HCP_PSEUDO_DIR",
        str(_PSEUDO_FLAT_DEFAULT if _PSEUDO_FLAT_DEFAULT.is_dir() else _PSEUDO_FALLBACK),
    )
)

B0_EXP_DEFAULT = PROJECT_ROOT / "experiments/baseline"
B0_CONFIG_DEFAULT = B0_EXP_DEFAULT / "audit" / "final_best_config.json"

sys_path = str(PROJECT_ROOT)
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
from hcp_architecture import build_hcp_classifier, resolve_deep_arch_tags
from protocol import PROTOCOL_VERSION, cross_fit, validate_manifest, validate_records

DEFAULT_SEED = 42  # overridable via HCP_SEED in main()
IMAGE_SIZE = (64, 64)
BATCH_SIZE = int(os.environ.get("HCP_BATCH_SIZE", "96"))
EPOCHS = int(os.environ.get("HCP_EPOCHS", "6"))
PATIENCE = int(os.environ.get("HCP_PATIENCE", "2"))

LOG_DIR = EXP_DIR / "logs"
AUDIT_DIR = EXP_DIR / "audit"
MODEL_DIR = EXP_DIR / "models"
STACK_DIR = EXP_DIR / "stacking"
PRED_DIR = EXP_DIR / "predictions"
for d in (LOG_DIR, AUDIT_DIR, MODEL_DIR, STACK_DIR, PRED_DIR):
    d.mkdir(parents=True, exist_ok=True)

RUN_LOG = LOG_DIR / "ablation_worker.log"

EMB_STACK_ABLATIONS = frozenset({"stack_embedding", "stack_embedding_pca185"})


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def apply_ablation_to_batch(bb: Dict) -> Dict:
    """bb: tensors on device (training/eval)."""
    if ABLATION in ("deep_image_only", "no_tc_no_manual"):
        bb["time_course"].zero_()
    elif ABLATION in ("deep_tc_only", "no_ic_no_manual"):
        bb["image"].zero_()
    return bb


def _infer_ic_index(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("ica_"):
        return -1
    head = stem.split("__", 1)[0]
    if not head.startswith("ica_"):
        return -1
    tail = head[len("ica_") :]
    try:
        return int(tail)
    except Exception:
        return -1


def parse_pkl(path: Path):
    with path.open("rb") as f:
        o = pickle.load(f)
    img = np.asarray(o["image"], dtype=np.float32)
    tc = np.asarray(o["tc"], dtype=np.float32).reshape(-1)
    feat = np.asarray(o["features"], dtype=np.float32).reshape(-1)
    raw_label = float(o["label"])
    if raw_label not in (0.0, 1.0) or not np.all(np.isfinite(feat)):
        raise ValueError("Binary labels and finite FIX descriptors are required")
    y = int(raw_label)
    p = float(o.get("ml_prob_signal", 0.5))
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("Invalid pseudo-label confidence probability")
    subj = str(o.get("subject", path.parent.name))
    sess = str(o.get("session", "na"))
    raw_ic = o.get("ic_index", None)
    ic = int(raw_ic) if raw_ic is not None else _infer_ic_index(path)
    return img, tc, feat, y, p, subj, sess, ic


def resize_image(img: np.ndarray) -> np.ndarray:
    x = img.mean(axis=2)
    t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=IMAGE_SIZE, mode="bilinear", align_corners=False).squeeze(0)
    t = (t - t.mean()) / (t.std() + 1e-8)
    return t.numpy()


class MixDataset(Dataset):
    def __init__(self, records: List[Dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img, tc, feat, y, _, subj, sess, ic = parse_pkl(r["path"])
        img = resize_image(img)
        tc = (tc - tc.mean()) / (tc.std() + 1e-8)
        return {
            "image": torch.from_numpy(img.astype(np.float32)),
            "time_course": torch.from_numpy(tc.astype(np.float32)).unsqueeze(-1),
            "seq_len": torch.tensor(len(tc), dtype=torch.long),
            "label": torch.tensor(float(y), dtype=torch.float32),
            "weight": torch.tensor(float(r["weight"]), dtype=torch.float32),
            "feature": torch.from_numpy(feat.astype(np.float32)),
            "meta": {"subject": subj, "session": sess, "ic_index": ic},
        }

    @staticmethod
    def collate_fn(batch):
        max_len = max(x["time_course"].shape[0] for x in batch)
        tc = torch.zeros(len(batch), max_len, 1, dtype=torch.float32)
        for i, x in enumerate(batch):
            tc[i, : x["time_course"].shape[0]] = x["time_course"]
        return {
            "image": torch.stack([x["image"] for x in batch]),
            "time_course": tc,
            "seq_len": torch.stack([x["seq_len"] for x in batch]),
            "label": torch.stack([x["label"] for x in batch]),
            "weight": torch.stack([x["weight"] for x in batch]),
            "feature": torch.stack([x["feature"] for x in batch]),
            "meta": [x["meta"] for x in batch],
        }


def loader_kwargs():
    raw = os.environ.get("HCP_NUM_WORKERS", "").strip()
    if raw:
        n = max(0, int(raw))
    else:
        n = min(8, max(2, (os.cpu_count() or 8) // 4))
    kw = {"num_workers": n, "pin_memory": bool(torch.cuda.is_available())}
    if n > 0:
        kw["prefetch_factor"] = 2
        kw["persistent_workers"] = os.environ.get("HCP_PERSISTENT_WORKERS", "0") == "1"
    return kw


def find_best_threshold(y, p):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.2, 0.9, 71):
        f1 = f1_score(y, (p >= t).astype(int))
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t


def evaluate(y, p):
    t = find_best_threshold(y, p)
    pred = (p >= t).astype(int)
    return {
        "threshold": t,
        "f1": float(f1_score(y, pred)),
        "acc": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
    }


def load_records(split: str = "train_val_test"):
    split_manifest = json.loads((AUDIT_DIR / "split_manifest.json").read_text(encoding="utf-8"))
    partitions = validate_manifest(split_manifest)
    aliases = split_manifest.get("participant_map", {})
    train_sub, val_sub, test_sub = [partitions[s] for s in ("train", "val", "test")]

    true_records = []
    for p in sorted(TRUE_DIR.rglob("*.pkl")):
        img, tc, feat, y, _, subj, sess, ic = parse_pkl(p)
        person = aliases.get(subj, subj)
        if feat.shape[0] != 185:
            raise RuntimeError(f"true feature dim !=185: {p} -> {feat.shape[0]}")
        if person in train_sub:
            bucket = "train"
        elif person in val_sub:
            bucket = "val"
        elif person in test_sub:
            bucket = "test"
        else:
            continue
        true_records.append(
            {
                "path": p.resolve(),
                "subject": subj,
                "participant": person,
                "session": sess,
                "ic_index": ic,
                "content_hash": input_hash(img, tc),
                "label": y,
                "weight": 1.0,
                "bucket": bucket,
                "is_pseudo": False,
            }
        )

    pseudo_records = []
    for p in sorted(PSEUDO_DIR.rglob("*.pkl")):
        img, tc, feat, y, prob, subj, sess, ic = parse_pkl(p)
        person = aliases.get(subj, subj)
        if feat.shape[0] != 185:
            raise RuntimeError(f"pseudo feature dim !=185: {p} -> {feat.shape[0]}")
        if person not in train_sub:
            continue
        with p.open("rb") as stream:
            source_record = pickle.load(stream)
        upstream = source_record.get("upstream_training_participants")
        if not isinstance(upstream, list):
            raise ValueError("Pseudo PKL requires labeler/reference provenance")
        pseudo_records.append(
            {
                "path": p.resolve(),
                "subject": subj,
                "participant": person,
                "session": sess,
                "ic_index": ic,
                "content_hash": input_hash(img, tc),
                "label": y,
                "ml_prob_signal": prob,
                "is_pseudo": True,
                "bucket": "train",
                "upstream_training_participants": [aliases.get(x, x) for x in upstream],
            }
        )

    validate_records(true_records, pseudo_records)
    return true_records, pseudo_records


def input_hash(image, time_course):
    digest = hashlib.sha256()
    for array in (image, time_course):
        values = np.ascontiguousarray(array, dtype=np.float32)
        if not np.all(np.isfinite(values)):
            raise ValueError("Nonfinite image/time-course input")
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def build_weighted_pseudo(pseudo_records, params):
    keep = []
    for r in pseudo_records:
        conf = abs(float(r["ml_prob_signal"]) - 0.5) * 2
        if conf < params["pseudo_conf_threshold"]:
            continue
        w = params["pseudo_base_weight"] * (
            conf ** params["pseudo_gamma"] if params["pseudo_gamma"] > 0 else 1.0
        )
        rr = dict(r)
        rr["weight"] = float(w)
        keep.append(rr)
    if params["pseudo_sample_ratio"] < 1.0 and keep:
        n = max(1, int(len(keep) * params["pseudo_sample_ratio"]))
        keep = random.sample(keep, n)
    return keep


def fit_deep_model(
    train_records, validation_records, save_ckpt_path, *, image_backbone, temporal_backbone
):
    """Train on training records only; select checkpoints on validation only."""
    import copy

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_hcp_classifier(image_backbone, temporal_backbone).to(device)
    kwargs = loader_kwargs()
    train_loader = DataLoader(
        MixDataset(train_records),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=MixDataset.collate_fn,
        **kwargs,
    )
    validation_loader = None
    if validation_records:
        validation_loader = DataLoader(
            MixDataset(validation_records),
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=MixDataset.collate_fn,
            **kwargs,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    best_f1, bad_epochs, best_state, selected_epoch = -1.0, 0, None, 0
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            tensors = {k: v.to(device) for k, v in batch.items() if k != "meta"}
            tensors = apply_ablation_to_batch(tensors)
            optimizer.zero_grad(set_to_none=True)
            # Targets are never supplied to the network's forward path.
            inputs = {k: v for k, v in tensors.items() if k not in ("label", "weight")}
            logits = model(inputs)["predict"].view(-1)
            loss = (criterion(logits, tensors["label"]) * tensors["weight"]).mean()
            loss.backward()
            optimizer.step()
        selected_epoch = epoch + 1
        if validation_loader is None:
            continue  # OOF models use a fixed epoch budget, never held-fold labels.
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in validation_loader:
                inputs = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if k not in ("meta", "label", "weight")
                }
                inputs = apply_ablation_to_batch(inputs)
                ps.append(torch.sigmoid(model(inputs)["predict"].view(-1)).cpu().numpy())
                ys.append(batch["label"].numpy().astype(int))
        score = evaluate(np.concatenate(ys), np.concatenate(ps))["f1"]
        if score > best_f1:
            best_f1, bad_epochs = score, 0
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
        selected_epoch = best_epoch
    payload = {
        "model": model.state_dict(),
        "protocol_version": PROTOCOL_VERSION,
        "selection_partition": "validation" if validation_records else "fixed_epochs",
        "selected_epoch": selected_epoch,
        "ablation": ABLATION,
        "image_backbone": image_backbone,
        "temporal_backbone": temporal_backbone,
    }
    torch.save(payload, save_ckpt_path)
    model.eval()
    return model


def collect_frozen_probs(model, target_records):
    """Inference only: no threshold search, optimizer, or checkpoint selection."""
    device = next(model.parameters()).device
    loader = DataLoader(
        MixDataset(target_records),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=MixDataset.collate_fn,
        **loader_kwargs(),
    )
    probabilities, labels, features = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = {
                k: v.to(device) for k, v in batch.items() if k not in ("meta", "label", "weight")
            }
            inputs = apply_ablation_to_batch(inputs)
            probabilities.append(torch.sigmoid(model(inputs)["predict"].view(-1)).cpu().numpy())
            labels.append(batch["label"].numpy().astype(int))
            features.append(batch["feature"].numpy())
    return np.concatenate(probabilities), np.concatenate(labels), np.vstack(features)


def safe_stack_inputs(
    train_records,
    pseudo_records,
    validation_records,
    test_records,
    *,
    image_backbone,
    temporal_backbone,
):
    """Group OOF training features; one frozen final encoder for validation/test."""

    def fit(records, selection, tag):
        return fit_deep_model(
            records,
            selection,
            MODEL_DIR / f"{ABLATION}_{tag}.pth",
            image_backbone=image_backbone,
            temporal_backbone=temporal_backbone,
        )

    blocks, final_model = cross_fit(
        train_records,
        pseudo_records,
        validation_records,
        fit=fit,
        infer=collect_frozen_probs,
        n_folds=int(os.environ.get("HCP_OOF_FOLDS", "5")),
        seed=int(os.environ.get("HCP_SEED", str(DEFAULT_SEED))),
    )
    n = len(train_records)
    p_train, y_train = np.empty(n), np.empty(n, dtype=int)
    f_train = np.empty((n, 185), dtype=np.float32)
    for indices, (p, y, f) in blocks:
        p_train[indices], y_train[indices], f_train[indices] = p, y, f
    validation_output = collect_frozen_probs(final_model, validation_records)
    # Test labels are read for final reporting only, after every fit is complete.
    test_output = collect_frozen_probs(final_model, test_records)
    return (p_train, y_train, f_train), validation_output, test_output


def _encode_embeddings(model: nn.Module, bb: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    core = model.module if isinstance(model, nn.DataParallel) else model
    return core.encode_embeddings(bb)


def to_stack_full(p, f):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    return np.concatenate([p.reshape(-1, 1), lp, f], axis=1)


def to_stack_no_manual(p, _f):
    del _f
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    return np.concatenate([p.reshape(-1, 1), lp], axis=1)


def to_stack_no_logit(p, f):
    return np.concatenate([p.reshape(-1, 1), f], axis=1)


def to_stack_embedding(fs, ft, f):
    """XGB input: normalized spatial emb || temporal emb || 185-d manual features."""
    return np.concatenate([fs, ft, f], axis=1)


def to_stack_embedding_pca185(
    fs_tr, ft_tr, f_tr, fs_va, ft_va, f_va, fs_te, ft_te, f_te
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PCA, Dict]:
    """PCA-reduce concat(fs, ft) on train to manual_dim, then || manual -> 2*manual_dim (370)."""
    manual_dim = int(f_tr.shape[1])
    E_tr = np.concatenate([fs_tr, ft_tr], axis=1)
    E_va = np.concatenate([fs_va, ft_va], axis=1)
    E_te = np.concatenate([fs_te, ft_te], axis=1)
    pca = PCA(n_components=manual_dim, random_state=42)
    R_tr = pca.fit_transform(E_tr)
    R_va = pca.transform(E_va)
    R_te = pca.transform(E_te)
    Xtr = np.concatenate([R_tr, f_tr], axis=1)
    Xva = np.concatenate([R_va, f_va], axis=1)
    Xte = np.concatenate([R_te, f_te], axis=1)
    meta = {
        "stack_mode": "pca185(deep_emb)+manual",
        "deep_emb_raw_dim": int(E_tr.shape[1]),
        "deep_emb_pca_dim": manual_dim,
        "manual_dim": manual_dim,
        "xgb_input_dim": int(Xtr.shape[1]),
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
    log(
        f"pca185 stack | raw_emb={E_tr.shape[1]} -> {manual_dim}, "
        f"xgb_dim={Xtr.shape[1]}, evr_sum={meta['pca_explained_variance_ratio_sum']:.4f}"
    )
    return Xtr, Xva, Xte, pca, meta


def collect_manual_only(records: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    ys, fs = [], []
    for r in records:
        _, _, feat, y, _, _, _, _ = parse_pkl(r["path"])
        ys.append(int(y))
        fs.append(feat.astype(np.float32).reshape(-1))
    return np.asarray(ys, dtype=np.int64), np.vstack(fs)


def load_params() -> Dict:
    pj = os.environ.get("HCP_PARAMS_JSON", "").strip()
    if pj:
        src = Path(pj)
    else:
        b0 = Path(os.environ.get("HCP_B0_EXP_DIR", str(B0_EXP_DEFAULT)))
        src = b0 / "audit" / "final_best_config.json"
        if not src.is_file():
            src = B0_CONFIG_DEFAULT
    obj = json.loads(src.read_text(encoding="utf-8"))
    return obj["params"]


def run_trial_manual_xgb(params: Dict, true_records: List, pseudo_records: List) -> Dict:
    tr_true = [r for r in true_records if r["bucket"] == "train"]
    va_true = [r for r in true_records if r["bucket"] == "val"]
    te_true = [r for r in true_records if r["bucket"] == "test"]
    pseudo_w = []  # The manuscript's booster is fitted on expert-labeled rows only.
    train_records = tr_true

    y_tr, f_tr = collect_manual_only(train_records)
    y_va, f_va = collect_manual_only(va_true)
    y_te, f_te = collect_manual_only(te_true)

    np.save(STACK_DIR / "manual_only_y_train.npy", y_tr)
    np.save(STACK_DIR / "manual_only_X_train.npy", f_tr)
    np.save(STACK_DIR / "manual_only_y_val.npy", y_va)
    np.save(STACK_DIR / "manual_only_X_val.npy", f_va)
    np.save(STACK_DIR / "manual_only_y_test.npy", y_te)
    np.save(STACK_DIR / "manual_only_X_test.npy", f_te)

    xcfg = params["xgb"]
    model = xgb.XGBClassifier(
        max_depth=xcfg["max_depth"],
        n_estimators=xcfg["n_estimators"],
        learning_rate=xcfg["learning_rate"],
        subsample=xcfg["subsample"],
        colsample_bytree=xcfg["colsample_bytree"],
        reg_alpha=xcfg["reg_alpha"],
        reg_lambda=xcfg["reg_lambda"],
        min_child_weight=xcfg["min_child_weight"],
        gamma=xcfg["gamma"],
        eval_metric="logloss",
        tree_method="hist",
    )
    model.fit(f_tr, y_tr)
    pva = model.predict_proba(f_va)[:, 1]
    pte = model.predict_proba(f_te)[:, 1]
    val_metric = evaluate(y_va, pva)
    thr = val_metric["threshold"]
    test_pred = (pte >= thr).astype(int)
    test_metric = {
        "threshold_from_val": float(thr),
        "f1": float(f1_score(y_te, test_pred)),
        "acc": float(accuracy_score(y_te, test_pred)),
        "precision": float(precision_score(y_te, test_pred, zero_division=0)),
        "recall": float(recall_score(y_te, test_pred, zero_division=0)),
    }
    xgb_path = MODEL_DIR / "ablation_manual_xgb.json"
    model.save_model(str(xgb_path))
    artifact = {
        "protocol_version": PROTOCOL_VERSION,
        "selection_partition": "validation",
        "ablation": ABLATION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "xgb_model": str(xgb_path),
        "threshold_from_val": float(thr),
        "params": params,
        "val_metric": val_metric,
        "test_metric": test_metric,
        "n_train_true": len(tr_true),
        "n_pseudo_kept": len(pseudo_w),
    }
    (AUDIT_DIR / "ablation_artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (AUDIT_DIR / "ablation_result.json").write_text(
        json.dumps(
            {"val_metric": val_metric, "test_metric": test_metric}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    return artifact


def run_trial_main(params: Dict, true_records: List, pseudo_records: List) -> Dict:
    tr_true = [r for r in true_records if r["bucket"] == "train"]
    va_true = [r for r in true_records if r["bucket"] == "val"]
    te_true = [r for r in true_records if r["bucket"] == "test"]
    pseudo_w = build_weighted_pseudo(pseudo_records, params)
    ib_env = os.environ.get("HCP_IMAGE_BACKBONE", "convnextv2").strip()
    tb_env = os.environ.get("HCP_TEMPORAL_BACKBONE", "lstm").strip()
    deep_arch = resolve_deep_arch_tags(ib_env, tb_env)

    tag = f"ablation_{ABLATION}"
    if ABLATION in EMB_STACK_ABLATIONS:
        raise ValueError(
            "Raw embeddings from independently fitted OOF encoders are not aligned; this diagnostic is unsupported."
        )
    (p_tr, y_tr, f_tr), (p_va, y_va, f_va), (p_te, y_te, f_te) = safe_stack_inputs(
        tr_true,
        pseudo_w,
        va_true,
        te_true,
        image_backbone=ib_env,
        temporal_backbone=tb_env,
    )
    ckpt_final = MODEL_DIR / f"{ABLATION}_final.pth"

    np.save(STACK_DIR / "p_train.npy", p_tr)
    np.save(STACK_DIR / "y_train.npy", y_tr)
    np.save(STACK_DIR / "f_train.npy", f_tr)
    np.save(STACK_DIR / "p_val.npy", p_va)
    np.save(STACK_DIR / "y_val.npy", y_va)
    np.save(STACK_DIR / "f_val.npy", f_va)
    np.save(STACK_DIR / "p_test.npy", p_te)
    np.save(STACK_DIR / "y_test.npy", y_te)
    np.save(STACK_DIR / "f_test.npy", f_te)

    if ABLATION == "no_stack_xgb":
        val_metric = evaluate(y_va, p_va)
        thr = val_metric["threshold"]
        test_pred = (p_te >= thr).astype(int)
        test_metric = {
            "threshold_from_val": float(thr),
            "f1": float(f1_score(y_te, test_pred)),
            "acc": float(accuracy_score(y_te, test_pred)),
            "precision": float(precision_score(y_te, test_pred, zero_division=0)),
            "recall": float(recall_score(y_te, test_pred, zero_division=0)),
        }
        artifact = {
            "protocol_version": PROTOCOL_VERSION,
            "selection_partition": "validation",
            "ablation": ABLATION,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "deep_checkpoints": {
                "frozen_final": str(ckpt_final),
            },
            "xgb_model": None,
            "threshold_from_val": float(thr),
            "params": params,
            "val_metric": val_metric,
            "test_metric": test_metric,
            "n_train_true": len(tr_true),
            "n_pseudo_kept": len(pseudo_w),
            **deep_arch,
        }
        (AUDIT_DIR / "ablation_artifact.json").write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (AUDIT_DIR / "ablation_result.json").write_text(
            json.dumps(
                {"val_metric": val_metric, "test_metric": test_metric, **deep_arch},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return artifact

    if ABLATION in ("stack_no_manual", "no_tc_no_manual", "no_ic_no_manual"):
        stack_fn = to_stack_no_manual
    elif ABLATION == "stack_no_logit":
        stack_fn = to_stack_no_logit
    else:
        stack_fn = to_stack_full

    Xtr = stack_fn(p_tr, f_tr)
    Xva = stack_fn(p_va, f_va)
    Xte = stack_fn(p_te, f_te)
    stack_meta = {
        "protocol_version": PROTOCOL_VERSION,
        "stack_training_features": "participant_group_out_of_fold",
        "stack_mode": "p+logit+manual" if ABLATION == "full" else ABLATION,
        "xgb_input_dim": int(Xtr.shape[1]),
    }
    np.save(STACK_DIR / "X_stack_train.npy", Xtr)
    np.save(STACK_DIR / "X_stack_val.npy", Xva)
    np.save(STACK_DIR / "X_stack_test.npy", Xte)

    xcfg = params["xgb"]
    model = xgb.XGBClassifier(
        max_depth=xcfg["max_depth"],
        n_estimators=xcfg["n_estimators"],
        learning_rate=xcfg["learning_rate"],
        subsample=xcfg["subsample"],
        colsample_bytree=xcfg["colsample_bytree"],
        reg_alpha=xcfg["reg_alpha"],
        reg_lambda=xcfg["reg_lambda"],
        min_child_weight=xcfg["min_child_weight"],
        gamma=xcfg["gamma"],
        eval_metric="logloss",
        tree_method="hist",
    )
    model.fit(Xtr, y_tr)
    pva = model.predict_proba(Xva)[:, 1]
    pte = model.predict_proba(Xte)[:, 1]
    val_metric = evaluate(y_va, pva)
    thr = val_metric["threshold"]
    test_pred = (pte >= thr).astype(int)
    test_metric = {
        "threshold_from_val": float(thr),
        "f1": float(f1_score(y_te, test_pred)),
        "acc": float(accuracy_score(y_te, test_pred)),
        "precision": float(precision_score(y_te, test_pred, zero_division=0)),
        "recall": float(recall_score(y_te, test_pred, zero_division=0)),
    }
    xgb_path = MODEL_DIR / f"{tag}_xgb.json"
    model.save_model(str(xgb_path))
    artifact = {
        "protocol_version": PROTOCOL_VERSION,
        "selection_partition": "validation",
        "ablation": ABLATION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "deep_checkpoints": {
            "frozen_final": str(ckpt_final),
        },
        "xgb_model": str(xgb_path),
        "threshold_from_val": float(thr),
        "params": params,
        "val_metric": val_metric,
        "test_metric": test_metric,
        "n_train_true": len(tr_true),
        "n_pseudo_kept": len(pseudo_w),
        **deep_arch,
        **stack_meta,
    }
    (AUDIT_DIR / "ablation_artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (AUDIT_DIR / "ablation_result.json").write_text(
        json.dumps(
            {"val_metric": val_metric, "test_metric": test_metric, **deep_arch},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return artifact


def main():
    if EPOCHS < 1 or PATIENCE < 1:
        raise SystemExit("Epoch and patience budgets must be positive")
    if (AUDIT_DIR / "ablation_result.json").exists():
        raise SystemExit("Refusing to overwrite a completed experiment; choose a new HCP_EXP_DIR")
    seed = int(os.environ.get("HCP_SEED", str(DEFAULT_SEED)))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    valid = {
        "full",
        "no_stack_xgb",
        "stack_no_manual",
        "stack_no_logit",
        "deep_image_only",
        "no_tc_no_manual",
        "no_ic_no_manual",
        "deep_tc_only",
        "manual_xgb_only",
        "stack_embedding",
        "stack_embedding_pca185",
    }
    if ABLATION not in valid:
        raise SystemExit(f"HCP_ABLATION must be one of {sorted(valid)}, got {ABLATION!r}")
    if not (AUDIT_DIR / "split_manifest.json").is_file():
        raise SystemExit(f"missing {AUDIT_DIR / 'split_manifest.json'}")

    params = load_params()
    ib_env = os.environ.get("HCP_IMAGE_BACKBONE", "convnextv2").strip()
    tb_env = os.environ.get("HCP_TEMPORAL_BACKBONE", "lstm").strip()
    meta = {
        "protocol_version": PROTOCOL_VERSION,
        "stack_training_features": "participant_group_out_of_fold",
        "ablation": ABLATION,
        "seed": seed,
        "exp_dir": str(EXP_DIR),
        "project_root": str(PROJECT_ROOT),
        "true_dir": str(TRUE_DIR),
        "pseudo_dir": str(PSEUDO_DIR),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "params": params,
        "image_backbone_env": ib_env,
        "temporal_backbone_env": tb_env,
        **resolve_deep_arch_tags(ib_env, tb_env),
    }
    (AUDIT_DIR / "ablation_run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log(f"start ablation={ABLATION} exp_dir={EXP_DIR}")
    t0 = time.time()
    true_records, pseudo_records = load_records()
    meta["train_participants"] = sorted(
        {r["participant"] for r in true_records if r["bucket"] == "train"}
    )
    meta["validation_participants"] = sorted(
        {r["participant"] for r in true_records if r["bucket"] == "val"}
    )
    log(f"loaded true={len(true_records)} pseudo_train_pool={len(pseudo_records)}")

    if ABLATION == "manual_xgb_only":
        artifact = run_trial_manual_xgb(params, true_records, pseudo_records)
    else:
        artifact = run_trial_main(params, true_records, pseudo_records)

    meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
    meta["wall_seconds"] = time.time() - t0
    meta["artifact_summary"] = {
        k: artifact[k] for k in ("val_metric", "test_metric", "threshold_from_val") if k in artifact
    }
    (AUDIT_DIR / "ablation_run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(
        f"done ablation={ABLATION} wall_s={meta['wall_seconds']:.1f} test_f1={artifact['test_metric']['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
