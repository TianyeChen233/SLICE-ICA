#!/usr/bin/env python3
"""
Repeat hybrid ablation worker N times with different HCP_SEED; report mean/variance of metrics.

Multi-GPU: use --gpus 0,1,2,3 to run up to 4 jobs in parallel (one GPU per job); when a job
finishes, the next queued job starts on that GPU slot (pre-assigned round-robin).

Example:
  python run_repeated_experiments.py \\
    --out-root classification/experiments/repeated_runs \\
    --b0-exp classification/experiments/baseline \\
    --gpus 0,1,2,3 \\
    --n-repeats 5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def prepare_exp_dir(exp_dir: Path, b0_exp: Path) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    audit = exp_dir / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    shutil.copy2(b0_exp / "audit" / "split_manifest.json", audit / "split_manifest.json")


def spawn_job(
    python: Path,
    worker: Path,
    exp_dir: Path,
    b0_exp: Path,
    mode: str,
    seed: int,
    gpu: str,
) -> Tuple[subprocess.Popen, Any, Dict[str, Any]]:
    prepare_exp_dir(exp_dir, b0_exp)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["HCP_CUDA_DEVICE"] = "0"
    env["HCP_EXP_DIR"] = str(exp_dir.resolve())
    env["HCP_ABLATION"] = mode
    env["HCP_B0_EXP_DIR"] = str(b0_exp.resolve())
    env["HCP_PROJECT_ROOT"] = str(Path("classification").resolve())
    env["HCP_SEED"] = str(seed)
    cmd = [str(python), str(worker)]
    logf_path = exp_dir / "logs" / "repeat_run.log"
    logf_path.parent.mkdir(parents=True, exist_ok=True)
    lf = open(logf_path, "w", encoding="utf-8")
    p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    rec: Dict[str, Any] = {
        "mode": mode,
        "seed": seed,
        "gpu": gpu,
        "exp_dir": str(exp_dir),
        "log": str(logf_path),
    }
    return p, lf, rec


def collect_result(exp_dir: Path) -> Optional[Dict[str, Any]]:
    p = exp_dir / "audit" / "ablation_result.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--b0-exp", type=Path, required=True)
    ap.add_argument(
        "--modes",
        type=str,
        default="full,no_stack_xgb,stack_no_manual,stack_no_logit,deep_image_only,deep_tc_only,manual_xgb_only",
    )
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3",
        help="Comma-separated GPU ids; one training job per GPU at a time (default: four cards)",
    )
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
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        print("no --gpus", file=sys.stderr)
        sys.exit(1)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    args.out_root.mkdir(parents=True, exist_ok=True)

    job_specs: List[Tuple[str, int, int, str]] = []
    j = 0
    for mode in modes:
        for i in range(args.n_repeats):
            seed = args.base_seed + i
            gpu = gpus[j % len(gpus)]
            job_specs.append((mode, i, seed, gpu))
            j += 1

    if args.dry_run:
        for mode, i, seed, gpu in job_specs:
            tag = f"{mode}__seed{seed}__r{i:02d}"
            print("DRY", mode, tag, "gpu", gpu)
        return

    job_queue = deque(job_specs)
    running: List[Dict[str, Any]] = []
    launcher_log = args.out_root / "repeat_launcher_events.log"
    launcher_log.parent.mkdir(parents=True, exist_ok=True)

    def log_event(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with launcher_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    while job_queue or running:
        while job_queue and len(running) < len(gpus):
            mode, i, seed, gpu = job_queue.popleft()
            tag = f"{mode}__seed{seed}__r{i:02d}"
            exp_dir = args.out_root / mode / tag
            p, lf, rec = spawn_job(
                args.python, args.worker, exp_dir, args.b0_exp, mode, seed, gpu
            )
            rec["repeat_idx"] = i
            rec["tag"] = tag
            running.append({"p": p, "lf": lf, "rec": rec})
            log_event(f"started mode={mode} seed={seed} gpu={gpu} pid={p.pid} dir={exp_dir}")

        time.sleep(3)
        for item in list(running):
            proc: subprocess.Popen = item["p"]
            if proc.poll() is None:
                continue
            rc = proc.returncode
            lf = item["lf"]
            lf.close()
            rec = item["rec"]
            running.remove(item)
            log_event(
                f"finished mode={rec['mode']} seed={rec['seed']} gpu={rec['gpu']} exit={rc}"
            )
            if rc != 0:
                log_event(f"ERROR see log {rec['log']}")

    # collect all results
    manifest_rows: List[Dict[str, Any]] = []
    for mode, i, seed, gpu in job_specs:
        tag = f"{mode}__seed{seed}__r{i:02d}"
        exp_dir = args.out_root / mode / tag
        res = collect_result(exp_dir)
        row: Dict[str, Any] = {
            "mode": mode,
            "repeat_idx": i,
            "seed": seed,
            "gpu": gpu,
            "exp_dir": str(exp_dir),
        }
        if res is None:
            row["error"] = "missing ablation_result.json"
            manifest_rows.append(row)
            continue
        row["test_f1"] = res["test_metric"]["f1"]
        row["test_acc"] = res["test_metric"]["acc"]
        row["val_f1"] = res["val_metric"]["f1"]
        row["val_acc"] = res["val_metric"]["acc"]
        manifest_rows.append(row)

    by_mode: Dict[str, List[float]] = {}
    by_mode_acc: Dict[str, List[float]] = {}
    for row in manifest_rows:
        if "test_f1" not in row:
            continue
        by_mode.setdefault(row["mode"], []).append(row["test_f1"])
        by_mode_acc.setdefault(row["mode"], []).append(row["test_acc"])

    summary = []
    for mode, xs in sorted(by_mode.items()):
        n = len(xs)
        m = statistics.mean(xs)
        pop_var = statistics.pvariance(xs) if n > 1 else 0.0
        pop_std = statistics.pstdev(xs) if n > 1 else 0.0
        sample_var = statistics.variance(xs) if n > 1 else 0.0
        sample_std = statistics.stdev(xs) if n > 1 else 0.0
        acc_m = statistics.mean(by_mode_acc[mode])
        summary.append(
            {
                "mode": mode,
                "n_repeats": n,
                "test_f1_mean": m,
                "test_f1_variance_pop": pop_var,
                "test_f1_std_pop": pop_std,
                "test_f1_variance_sample": sample_var,
                "test_f1_std_sample": sample_std,
                "test_acc_mean": acc_m,
            }
        )

    out_json = args.out_root / "REPEAT_stats_summary.json"
    out_csv = args.out_root / "REPEAT_stats_summary.csv"
    out_json.write_text(
        json.dumps(
            {"gpus": gpus, "per_run": manifest_rows, "aggregated": summary},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else [])
        if summary:
            w.writeheader()
            w.writerows(summary)
    log_event(f"wrote {out_json} and {out_csv}")


if __name__ == "__main__":
    main()
