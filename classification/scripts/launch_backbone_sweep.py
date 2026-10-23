#!/usr/bin/env python3
"""
Run hybrid backbone sweeps with a **fixed pool of concurrent workers** (default: one job per GPU).

Unlike batching all jobs on a subset until the slowest finishes, this uses a thread pool so fast
configs (e.g. LSTM on GPU0) do not sit idle while slow ones (e.g. Mamba) run on other GPUs.

Example:
  python launch_backbone_sweep.py \\
    --sweep-root classification/experiments/backbone_sweep \\
    --b0-exp classification/experiments/baseline \\
    --gpus 0,1,2,3 \\
    --ablation full \\
    --seed 42
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_IMAGES = ["convnextv2", "resnet18", "swin_t", "efficientnet_b0", "cnn"]
DEFAULT_TEMPORALS = ["lstm", "gru", "transformer", "mamba"]


def parse_gpus(s: str) -> list[int]:
    out = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        out.append(int(p))
    return out


def _run_one_job(
    *,
    gpu: int,
    img: str,
    tmp: str,
    exp_dir: Path,
    split_src: Path,
    python_exe: Path,
    worker: Path,
    project_root: Path,
    b0_exp: Path,
    ablation: str,
    seed: int,
) -> tuple[str, str, int, Path]:
    exp_dir.mkdir(parents=True, exist_ok=True)
    audit = exp_dir / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    shutil.copy2(split_src, audit / "split_manifest.json")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["HCP_CUDA_DEVICE"] = "0"
    env["HCP_PROJECT_ROOT"] = str(project_root)
    env["HCP_EXP_DIR"] = str(exp_dir)
    env["HCP_ABLATION"] = ablation
    env["HCP_SEED"] = str(seed)
    env["HCP_B0_EXP_DIR"] = str(b0_exp)
    env["HCP_IMAGE_BACKBONE"] = img
    env["HCP_TEMPORAL_BACKBONE"] = tmp

    log_path = exp_dir / "train.log"
    cmd = [str(python_exe), "-u", str(worker)]
    with open(log_path, "w", encoding="utf-8") as lf:
        rc = subprocess.call(cmd, env=env, cwd=str(project_root), stdout=lf, stderr=subprocess.STDOUT)
    return img, tmp, rc, log_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sweep-root",
        type=Path,
        default=Path("classification/experiments/backbone_sweep"),
    )
    ap.add_argument(
        "--b0-exp",
        type=Path,
        default=Path("classification/experiments/baseline"),
    )
    ap.add_argument("--gpus", type=str, default="0,1,2,3")
    ap.add_argument(
        "--max-concurrent",
        type=int,
        default=0,
        help="cap concurrent jobs (default: number of GPUs listed in --gpus)",
    )
    ap.add_argument("--ablation", type=str, default="full")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--project-root", type=Path, default=Path("classification"))
    ap.add_argument(
        "--python",
        type=Path,
        default=Path("python3"),
    )
    ap.add_argument(
        "--worker",
        type=Path,
        default=Path("classification/scripts/run_hybrid_ablation_trial.py"),
    )
    ap.add_argument("--images", type=str, default=",".join(DEFAULT_IMAGES), help="comma-separated")
    ap.add_argument("--temporals", type=str, default=",".join(DEFAULT_TEMPORALS), help="comma-separated")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--skip-if-result",
        action="store_true",
        help="skip jobs whose audit/ablation_result.json already exists (useful to fill missing combos)",
    )
    args = ap.parse_args()

    gpus = parse_gpus(args.gpus)
    if not gpus:
        raise SystemExit("no GPUs parsed from --gpus")

    max_workers = args.max_concurrent if args.max_concurrent > 0 else len(gpus)

    images = [x.strip() for x in args.images.split(",") if x.strip()]
    temporals = [x.strip() for x in args.temporals.split(",") if x.strip()]
    jobs = [(img, tmp) for img in images for tmp in temporals]
    if args.skip_if_result:
        kept = []
        for img, tmp in jobs:
            safe = f"{img}__{tmp}".replace("/", "_")
            res_path = args.sweep_root / safe / "audit" / "ablation_result.json"
            if res_path.is_file():
                print(f"[skip] {safe} already has {res_path.name}", flush=True)
                continue
            kept.append((img, tmp))
        jobs = kept
        if not jobs:
            print("nothing to run after --skip-if-result", flush=True)
            return

    split_src = args.b0_exp / "audit" / "split_manifest.json"
    if not split_src.is_file():
        raise SystemExit(f"missing split manifest: {split_src}")

    args.sweep_root.mkdir(parents=True, exist_ok=True)
    summary_lines: list[str] = []

    for ji, (img, tmp) in enumerate(jobs):
        gpu = gpus[ji % len(gpus)]
        safe = f"{img}__{tmp}".replace("/", "_")
        exp_dir = args.sweep_root / safe
        line = f"scheduled gpu={gpu} {safe} -> {exp_dir}"
        summary_lines.append(line)
        print(line, flush=True)

    (args.sweep_root / "sweep_manifest.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    if args.dry_run:
        print(f"wrote {args.sweep_root / 'sweep_manifest.txt'}", flush=True)
        return

    failures: list[tuple[str, str, int, Path]] = []

    def submit_payload(idx_img_tmp):
        idx, (img, tmp) = idx_img_tmp
        gpu = gpus[idx % len(gpus)]
        safe = f"{img}__{tmp}".replace("/", "_")
        exp_dir = args.sweep_root / safe
        return _run_one_job(
            gpu=gpu,
            img=img,
            tmp=tmp,
            exp_dir=exp_dir,
            split_src=split_src,
            python_exe=args.python,
            worker=args.worker,
            project_root=args.project_root,
            b0_exp=args.b0_exp,
            ablation=args.ablation.strip(),
            seed=args.seed,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(submit_payload, (i, j)): j for i, j in enumerate(jobs)}
        for fut in as_completed(futures):
            img, tmp, rc, log_path = fut.result()
            if rc != 0:
                failures.append((img, tmp, rc, log_path))
                print(f"[fail] rc={rc} {img} {tmp} log={log_path}", file=sys.stderr, flush=True)
            else:
                print(f"[ok] {img} {tmp}", flush=True)

    rows = []
    for img in images:
        for tmp in temporals:
            safe = f"{img}__{tmp}".replace("/", "_")
            res_path = args.sweep_root / safe / "audit" / "ablation_result.json"
            if res_path.is_file():
                rows.append(str(res_path))
    (args.sweep_root / "result_paths.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")

    if failures:
        raise SystemExit(f"{len(failures)} job(s) failed; see stderr and logs under {args.sweep_root}")

    print(f"wrote {args.sweep_root / 'result_paths.txt'} ({len(rows)} results)", flush=True)


if __name__ == "__main__":
    main()
