#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from PIL import Image, ImageDraw

if __package__:
    from .summary_statistics import adjust_bh, paired_wilcoxon
else:
    from summary_statistics import adjust_bh, paired_wilcoxon


def benjamini_hochberg(pvals):
    return adjust_bh(pvals)


def find_motion_file(sess_dir: Path):
    mc = sess_dir / "mc"
    if mc.exists():
        cands = sorted(mc.glob("*.par")) + sorted(mc.glob("*.txt"))
        if cands:
            return cands[0]
    cands = sorted(sess_dir.glob("rp*.txt"))
    if cands:
        return cands[0]
    return None


def compute_fd(motion, T):
    if motion.ndim == 1:
        motion = motion.reshape(-1, 1)
    if motion.shape[0] != T:
        raise RuntimeError(f"motion rows {motion.shape[0]} != T {T}")
    d = np.diff(motion, axis=0)
    return np.sum(np.abs(d), axis=1)


def compute_dvars(vol4d, mask_bool):
    flat = vol4d.reshape(-1, vol4d.shape[3])
    idx = np.where(mask_bool.ravel())[0]
    if idx.size == 0:
        raise RuntimeError("empty mask voxels")
    X = flat[idx, :]
    dX = np.diff(X, axis=1)
    return np.sqrt(np.mean(dX**2, axis=0))


def compute_tsnr(vol4d, mask_bool):
    flat = vol4d.reshape(-1, vol4d.shape[3])
    idx = np.where(mask_bool.ravel())[0]
    X = flat[idx, :]
    mu = np.mean(X, axis=1)
    sd = np.std(X, axis=1)
    tsnr = np.abs(mu) / (sd + 1e-8)
    tsnr = tsnr[np.isfinite(tsnr)]
    if tsnr.size == 0:
        raise RuntimeError("no finite tSNR")
    return float(np.mean(tsnr))


def compute_global_var(vol4d, mask_bool):
    flat = vol4d.reshape(-1, vol4d.shape[3])
    idx = np.where(mask_bool.ravel())[0]
    X = flat[idx, :]
    gs = np.mean(X, axis=0)
    return float(np.var(gs))


def corr_safe(x, y):
    n = min(len(x), len(y))
    if n < 3:
        return np.nan
    x = np.asarray(x[:n], dtype=float)
    y = np.asarray(y[:n], dtype=float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def cohen_d_paired(after, before):
    d = np.asarray(after) - np.asarray(before)
    sd = np.std(d, ddof=1)
    if sd < 1e-12:
        return np.nan
    return float(np.mean(d) / sd)


def session_metrics(sess_dir: Path):
    before_f = sess_dir / "filtered_func_data.nii.gz"
    after_f = sess_dir / "filtered_func_data_regress.nii.gz"
    mask_f = sess_dir / "mask.nii.gz"
    motion_f = find_motion_file(sess_dir)

    if not before_f.exists() or not after_f.exists() or not mask_f.exists() or motion_f is None:
        missing = []
        if not before_f.exists():
            missing.append(str(before_f))
        if not after_f.exists():
            missing.append(str(after_f))
        if not mask_f.exists():
            missing.append(str(mask_f))
        if motion_f is None:
            missing.append("motion_file")
        raise FileNotFoundError("; ".join(missing))

    img_b = nib.load(str(before_f))
    img_a = nib.load(str(after_f))
    img_m = nib.load(str(mask_f))
    vol_b = img_b.get_fdata(dtype=np.float32)
    vol_a = img_a.get_fdata(dtype=np.float32)
    mask = img_m.get_fdata(dtype=np.float32) > 0

    if vol_b.ndim != 4 or vol_a.ndim != 4:
        raise RuntimeError("fMRI not 4D")
    if vol_b.shape != vol_a.shape:
        raise RuntimeError("before/after shape mismatch")
    if mask.shape != vol_b.shape[:3]:
        raise RuntimeError("mask shape mismatch")
    if not np.allclose(img_b.affine, img_a.affine, atol=1e-5):
        raise RuntimeError("before/after affine mismatch")
    if not np.isfinite(vol_b).all() or not np.isfinite(vol_a).all():
        raise RuntimeError("non-finite data")

    T = vol_b.shape[3]
    motion = np.loadtxt(str(motion_f), dtype=np.float32)
    fd = compute_fd(motion, T)
    dvars_b = compute_dvars(vol_b, mask)
    dvars_a = compute_dvars(vol_a, mask)

    return {
        "subject": sess_dir.parent.name,
        "session": sess_dir.name,
        "session_dir": str(sess_dir),
        "timepoints": int(T),
        "mask_voxels": int(mask.sum()),
        "tSNR_before": compute_tsnr(vol_b, mask),
        "tSNR_after": compute_tsnr(vol_a, mask),
        "DVARS_mean_before": float(np.mean(dvars_b)),
        "DVARS_mean_after": float(np.mean(dvars_a)),
        "FD_DVARS_corr_before": corr_safe(fd, dvars_b),
        "FD_DVARS_corr_after": corr_safe(fd, dvars_a),
        "GS_var_before": compute_global_var(vol_b, mask),
        "GS_var_after": compute_global_var(vol_a, mask),
    }


def paired_stats(df, bcol, acol, name):
    b = df[bcol].to_numpy(dtype=float)
    a = df[acol].to_numpy(dtype=float)
    m = np.isfinite(b) & np.isfinite(a)
    b = b[m]
    a = a[m]
    result = paired_wilcoxon(df[bcol].to_numpy(), df[acol].to_numpy())
    return {
        "metric": name,
        **result,
        "effect_size_cohen_dz": cohen_d_paired(a, b) if len(b) >= 3 else np.nan,
    }


def draw_box(ax_draw, vals, x_center, y_min, y_max, h, w=38, color=(80, 80, 180)):
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    lo, hi = np.percentile(v, [5, 95])

    def y_map(z):
        if y_max <= y_min:
            return 0
        return int((y_max - z) / (y_max - y_min) * h)

    x0 = x_center - w // 2
    x1 = x_center + w // 2
    yq1, yq3, ymed, ylo, yhi = map(y_map, [q1, q3, med, lo, hi])
    ax_draw.rectangle([x0, yq3, x1, yq1], outline=color, width=2)
    ax_draw.line([x0, ymed, x1, ymed], fill=(0, 0, 0), width=2)
    ax_draw.line([x_center, yq3, x_center, yhi], fill=color, width=2)
    ax_draw.line([x_center, yq1, x_center, ylo], fill=color, width=2)


def save_simple_boxplot(before, after, title, out_png):
    W, H = 900, 600
    L, T, R, B = 90, 80, 80, 100
    plot_w = W - L - R
    plot_h = H - T - B

    b = np.asarray(before, dtype=float)
    a = np.asarray(after, dtype=float)
    allv = np.concatenate([b[np.isfinite(b)], a[np.isfinite(a)]])
    if len(allv) == 0:
        return
    y_min, y_max = np.percentile(allv, [1, 99])
    if y_max <= y_min:
        y_max = y_min + 1.0

    img = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(img)

    dr.text((L, 20), title, fill=(0, 0, 0))
    dr.line([L, T + plot_h, L + plot_w, T + plot_h], fill=(0, 0, 0), width=2)
    dr.line([L, T, L, T + plot_h], fill=(0, 0, 0), width=2)

    x_before = L + int(plot_w * 0.33)
    x_after = L + int(plot_w * 0.67)

    draw_box(dr, b, x_before, y_min, y_max, plot_h)
    draw_box(dr, a, x_after, y_min, y_max, plot_h)

    rng = np.random.default_rng(42)
    for v in b[np.isfinite(b)]:
        y = T + int((y_max - v) / (y_max - y_min) * plot_h)
        x = x_before + int((rng.random() - 0.5) * 50)
        dr.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(100, 100, 220))
    for v in a[np.isfinite(a)]:
        y = T + int((y_max - v) / (y_max - y_min) * plot_h)
        x = x_after + int((rng.random() - 0.5) * 50)
        dr.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(220, 100, 100))

    dr.text((x_before - 25, T + plot_h + 30), "Before", fill=(0, 0, 0))
    dr.text((x_after - 18, T + plot_h + 30), "After", fill=(0, 0, 0))
    dr.text((20, T), f"{y_max:.3f}", fill=(80, 80, 80))
    dr.text((20, T + plot_h - 10), f"{y_min:.3f}", fill=(80, 80, 80))

    img.save(out_png)


def main():
    parser = argparse.ArgumentParser(description="Paired session QC using two-sided Wilcoxon tests")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    data_root, run_dir = args.data_root, args.out_dir
    if not data_root.is_dir():
        parser.error("Data root must exist")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(exist_ok=True)

    sessions = [
        ses
        for sub in sorted(data_root.iterdir())
        if sub.is_dir()
        for ses in sorted(sub.iterdir())
        if ses.is_dir() and ses.name.startswith("rfMRI_")
    ]

    debug = {"total_sessions_found": len(sessions), "passed": 0, "failed": 0, "errors": []}
    rows = []

    for i, sess in enumerate(sessions, 1):
        try:
            rows.append(session_metrics(sess))
            debug["passed"] += 1
        except Exception as e:
            debug["failed"] += 1
            debug["errors"].append({"session_dir": str(sess), "error": str(e)})
        if i % 25 == 0:
            print(
                f"[progress] {i}/{len(sessions)} passed={debug['passed']} failed={debug['failed']}",
                flush=True,
            )

    with (run_dir / "logs" / "debug_summary.json").open("w", encoding="utf-8") as f:
        json.dump(debug, f, ensure_ascii=False, indent=2)
    if not rows:
        raise RuntimeError("No valid session pairs; see logs/debug_summary.json")
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "session_metrics_before_after.csv", index=False)

    stats = [
        paired_stats(df, "tSNR_before", "tSNR_after", "tSNR"),
        paired_stats(df, "DVARS_mean_before", "DVARS_mean_after", "DVARS_mean"),
        paired_stats(df, "FD_DVARS_corr_before", "FD_DVARS_corr_after", "FD_DVARS_corr"),
        paired_stats(df, "GS_var_before", "GS_var_after", "Global_signal_variance"),
    ]
    st = pd.DataFrame(stats)
    st["fdr_q_value"] = benjamini_hochberg(st["p_value"].to_numpy(dtype=float))
    st.to_csv(run_dir / "group_paired_stats_effectsize_fdr.csv", index=False)

    save_simple_boxplot(
        df["tSNR_before"].to_numpy(),
        df["tSNR_after"].to_numpy(),
        "tSNR Before vs After",
        run_dir / "group_tsnr_box_raincloud.png",
    )
    save_simple_boxplot(
        df["DVARS_mean_before"].to_numpy(),
        df["DVARS_mean_after"].to_numpy(),
        "DVARS mean Before vs After",
        run_dir / "group_dvars_mean_box_raincloud.png",
    )
    save_simple_boxplot(
        df["FD_DVARS_corr_before"].to_numpy(),
        df["FD_DVARS_corr_after"].to_numpy(),
        "FD-DVARS corr Before vs After",
        run_dir / "group_fd_dvars_corr_box_raincloud.png",
    )
    save_simple_boxplot(
        df["GS_var_before"].to_numpy(),
        df["GS_var_after"].to_numpy(),
        "Global signal var Before vs After",
        run_dir / "group_global_signal_variance_box_raincloud.png",
    )

    report = {
        "run_dir": str(run_dir),
        "n_sessions_total": len(sessions),
        "n_sessions_passed": debug["passed"],
        "n_sessions_failed": debug["failed"],
    }
    report.update(
        {
            "paired_test": "wilcoxon_two_sided",
            "analysis_unit": "session",
            "multiple_testing": "BH across the four QC metrics",
            "tsnr_definition": "mean_voxel(abs(temporal_mean)/(temporal_std+1e-8))",
        }
    )
    with (run_dir / "run_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
