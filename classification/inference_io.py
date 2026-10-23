"""Input preparation for frozen ICA classification."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

IMAGE_SIZE = (64, 64)


def resize_image(img: np.ndarray) -> np.ndarray:
    x = img.mean(axis=2)
    t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=IMAGE_SIZE, mode="bilinear", align_corners=False).squeeze(0)
    t = (t - t.mean()) / (t.std() + 1e-8)
    return t.numpy()


def build_batch_tensors(
    Ximg: np.ndarray,
    tc_pad: np.ndarray,
    seq_len: np.ndarray,
    Xfeat: np.ndarray,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        "image": torch.from_numpy(Ximg).to(device, non_blocking=True),
        "time_course": torch.from_numpy(tc_pad).to(device, non_blocking=True),
        "seq_len": torch.from_numpy(seq_len).to(device, non_blocking=True),
        "feature": torch.from_numpy(Xfeat).to(device, non_blocking=True),
        "label": torch.zeros((Ximg.shape[0],), device=device, dtype=torch.float32),
    }


def to_stack(p: np.ndarray, f: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    return np.concatenate([p.reshape(-1, 1), lp, f.astype(np.float64)], axis=1)


@torch.no_grad()
def deep_probs(model: torch.nn.Module, bb: Dict[str, torch.Tensor]) -> np.ndarray:
    model.eval()
    logits = model(bb)["predict"].view(-1)
    return torch.sigmoid(logits).detach().float().cpu().numpy()


def load_pkl_sample(path: Path):
    o = pickle.loads(path.read_bytes())
    img = o["image"].astype(np.float32)
    img = resize_image(img)
    tc = o["tc"].astype(np.float32).reshape(-1)
    tc = (tc - tc.mean()) / (tc.std() + 1e-8)
    feat = o["features"].astype(np.float32).reshape(-1)
    if feat.shape[0] != 185:
        raise ValueError(f"{path}: expected 185 features, got {feat.shape[0]}")
    y = int(o["label"])
    meta = {
        "path": str(path),
        "subject": str(o.get("subject", "")),
        "session": str(o.get("session", "")),
        "ic_index": int(o.get("ic_index", -1)),
    }
    return img, tc, feat, y, meta


def collate_batch(
    samples: List,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]:
    imgs, tcs, feats, ys, metas = zip(*samples)
    L = max(int(t.shape[0]) for t in tcs)
    B = len(samples)
    tc_pad = np.zeros((B, L, 1), dtype=np.float32)
    seq_len = np.zeros((B,), dtype=np.int64)
    for i, tc in enumerate(tcs):
        ll = int(tc.shape[0])
        seq_len[i] = ll
        tc_pad[i, :ll, 0] = tc
    Ximg = np.stack(imgs, axis=0)
    Xfeat = np.stack(feats, axis=0)
    y = np.asarray(ys, dtype=np.int64)
    return Ximg, tc_pad, seq_len, Xfeat, y, list(metas)
