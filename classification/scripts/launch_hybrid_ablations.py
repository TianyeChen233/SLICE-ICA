#!/usr/bin/env python3
"""
Launch hybrid ablation workers with a strict job queue:
  - Each running task uses exactly one GPU (CUDA_VISIBLE_DEVICES set to one id).
  - At most len(gpus) tasks run concurrently; when one finishes, the next queued task starts.
  - Default --gpus 0,1,2,3 for a 4-GPU server (override if needed).

Example:
  python launch_hybrid_ablations.py \\
    --ablation-root classification/experiments/ablation_runs \\
    --b0-exp classification/experiments/baseline \\
    --conda-env data_convert

Uses an explicit baseline configuration and split manifest.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


DEFAULT_ABLATIONS = [
    "no_stack_xgb",
    "stack_no_manual",
    "stack_no_logit",
    "deep_image_only",
    "deep_tc_only",
    "manual_xgb_only",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation-root", type=Path, required=True)
    ap.add_argument("--b0-exp", type=Path, required=True, help="Existing hybrid experiment (split_manifest + final_best_config source)")
    ap.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3",
        help="Comma-separated physical GPU ids (one job per GPU slot; default four cards 0-3)",
    )
    ap.add_argument(
        "--conda-env",
        type=str,
        default="data_convert",
        help="Ignored if --python is set; kept for backward compatibility",
    )
    ap.add_argument(
        "--python",
        type=Path,
        default=Path("python3"),
        help="Interpreter for worker (avoid conda run in non-interactive ssh)",
    )
    ap.add_argument(
        "--worker",
        type=Path,
        default=Path("classification/scripts/run_hybrid_ablation_trial.py"),
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--include-full",
        action="store_true",
        help="Also schedule 'full' (duplicate B0 pipeline; expensive)",
    )
    args = ap.parse_args()

    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        print("no gpus", file=sys.stderr)
        sys.exit(1)

    ablations = list(DEFAULT_ABLATIONS)
    if args.include_full:
        ablations = ["full"] + ablations

    split_src = args.b0_exp / "audit" / "split_manifest.json"
    if not split_src.is_file():
        print(f"missing {split_src}", file=sys.stderr)
        sys.exit(1)

    args.ablation_root.mkdir(parents=True, exist_ok=True)
    readme = args.ablation_root / "README_ABLATION.txt"
    lines = [
        "B0 baseline (existing hybrid):",
        f"  {args.b0_exp.resolve()}",
        "",
        "Ablation dirs created under this root; each has audit/split_manifest.json copied from B0.",
        "Worker script:",
        f"  {args.worker}",
        "",
        "Per-job env: HCP_EXP_DIR, HCP_ABLATION, HCP_B0_EXP_DIR, CUDA_VISIBLE_DEVICES, HCP_CUDA_DEVICE=0",
        "",
        "Scheduler: single-GPU per job; up to N concurrent jobs where N=len(--gpus); queue fills slots as tasks finish.",
        "",
    ]
    readme.write_text("\n".join(lines), encoding="utf-8")

    def start_one(abl: str, gpu: str):
        exp_dir = args.ablation_root / f"B_{abl}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        audit = exp_dir / "audit"
        audit.mkdir(parents=True, exist_ok=True)
        shutil.copy2(split_src, audit / "split_manifest.json")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["HCP_CUDA_DEVICE"] = "0"
        env["HCP_EXP_DIR"] = str(exp_dir.resolve())
        env["HCP_ABLATION"] = abl
        env["HCP_B0_EXP_DIR"] = str(args.b0_exp.resolve())
        env["HCP_PROJECT_ROOT"] = str(Path("classification").resolve())

        cmd = [str(args.python), str(args.worker)]
        log_path = exp_dir / "logs" / "launcher_subprocess.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if args.dry_run:
            print(f"DRY gpu={gpu} HCP_ABLATION={abl} HCP_EXP_DIR={exp_dir}")
            return None
        lf = open(log_path, "w", encoding="utf-8")
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        print(f"started ablation={abl} gpu={gpu} pid={p.pid} log={log_path}")
        return (abl, p, lf)

    job_queue = deque((abl, gpus[i % len(gpus)]) for i, abl in enumerate(ablations))
    if args.dry_run:
        while job_queue:
            abl, gpu = job_queue.popleft()
            start_one(abl, gpu)
        return

    running: list = []
    rc = 0
    while job_queue or running:
        while job_queue and len(running) < len(gpus):
            abl, gpu = job_queue.popleft()
            h = start_one(abl, gpu)
            if h is not None:
                running.append(h)
        time.sleep(2)
        for item in list(running):
            abl, p, lf = item
            if p.poll() is not None:
                r = p.returncode
                lf.close()
                running.remove(item)
                print(f"finished ablation={abl} exit={r}")
                if r != 0:
                    rc = r
    sys.exit(rc)


if __name__ == "__main__":
    main()
