#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib

if __package__:
    from .summary_statistics import adjust_bh, paired_wilcoxon
else:
    from summary_statistics import adjust_bh, paired_wilcoxon


def benjamini_hochberg(pvals):
    return adjust_bh(pvals)


def cohen_d_paired(after, before):
    d = np.asarray(after) - np.asarray(before)
    sd = np.std(d, ddof=1)
    if sd < 1e-12:
        return np.nan
    return float(np.mean(d) / sd)


def paired_stats(df, bcol, acol, name):
    b = df[bcol].to_numpy(dtype=float)
    a = df[acol].to_numpy(dtype=float)
    m = np.isfinite(b) & np.isfinite(a)
    b = b[m]
    a = a[m]
    return {
        "metric": name,
        **paired_wilcoxon(df[bcol].to_numpy(), df[acol].to_numpy()),
        "effect_size_cohen_dz": cohen_d_paired(a, b) if len(b) >= 3 else np.nan,
    }


def compute_tsnr_variants(vol4d, mask):
    flat = vol4d.reshape(-1, vol4d.shape[3])
    idx = np.where(mask.ravel())[0]
    X = flat[idx, :]

    mu = np.mean(X, axis=1)
    sd = np.std(X, axis=1)
    tsnr_signed = mu / (sd + 1e-8)
    tsnr_absmean = np.abs(mu) / (sd + 1e-8)

    gs = np.mean(X, axis=0)
    X_addback = X + gs[None, :]
    mu_add = np.mean(X_addback, axis=1)
    sd_add = np.std(X_addback, axis=1)
    tsnr_addback = mu_add / (sd_add + 1e-8)

    out = {
        "tSNR_signed_mean": float(np.mean(tsnr_signed[np.isfinite(tsnr_signed)])),
        "tSNR_absmean_mean": float(np.mean(tsnr_absmean[np.isfinite(tsnr_absmean)])),
        "tSNR_addback_mean": float(np.mean(tsnr_addback[np.isfinite(tsnr_addback)])),
    }
    return out


def main():
    parser = argparse.ArgumentParser(
        description="tSNR definitions and paired two-sided Wilcoxon tests"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    data_root, out_dir = args.data_root, args.out_dir
    if not data_root.is_dir():
        parser.error("Data root must exist")
    out_dir.mkdir(parents=True, exist_ok=False)

    sessions = [
        ses
        for sub in sorted(data_root.iterdir())
        if sub.is_dir()
        for ses in sorted(sub.iterdir())
        if ses.is_dir() and ses.name.startswith("rfMRI_")
    ]

    rows = []
    errors = []
    for i, sess in enumerate(sessions, 1):
        try:
            b = nib.load(str(sess / "filtered_func_data.nii.gz")).get_fdata(dtype=np.float32)
            a = nib.load(str(sess / "filtered_func_data_regress.nii.gz")).get_fdata(
                dtype=np.float32
            )
            m = nib.load(str(sess / "mask.nii.gz")).get_fdata(dtype=np.float32) > 0

            if b.shape != a.shape or b.ndim != 4:
                raise RuntimeError("shape mismatch or non-4d")
            if m.shape != b.shape[:3]:
                raise RuntimeError("mask mismatch")

            vb = compute_tsnr_variants(b, m)
            va = compute_tsnr_variants(a, m)
            rows.append(
                {
                    "subject": sess.parent.name,
                    "session": sess.name,
                    "session_dir": str(sess),
                    "tSNR_signed_before": vb["tSNR_signed_mean"],
                    "tSNR_signed_after": va["tSNR_signed_mean"],
                    "tSNR_absmean_before": vb["tSNR_absmean_mean"],
                    "tSNR_absmean_after": va["tSNR_absmean_mean"],
                    "tSNR_addback_before": vb["tSNR_addback_mean"],
                    "tSNR_addback_after": va["tSNR_addback_mean"],
                }
            )
            if i % 25 == 0:
                print(f"[progress] {i}/{len(sessions)}", flush=True)
        except Exception as e:
            errors.append({"session_dir": str(sess), "error": str(e)})

    if not rows:
        (out_dir / "errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
        raise RuntimeError("No valid paired sessions; see errors.json")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "session_tsnr_recalc.csv", index=False)

    stats = [
        paired_stats(df, "tSNR_signed_before", "tSNR_signed_after", "tSNR_signed"),
        paired_stats(df, "tSNR_absmean_before", "tSNR_absmean_after", "tSNR_absmean"),
        paired_stats(df, "tSNR_addback_before", "tSNR_addback_after", "tSNR_addback"),
    ]
    st = pd.DataFrame(stats)
    st["fdr_q_value"] = benjamini_hochberg(st["p_value"].to_numpy(dtype=float))
    st.to_csv(out_dir / "tsnr_recalc_stats_fdr.csv", index=False)

    report = {
        "n_sessions_total": len(sessions),
        "n_sessions_ok": len(df),
        "n_sessions_error": len(errors),
        "error_examples": errors[:10],
        "output_dir": str(out_dir),
        "paired_test": "wilcoxon_two_sided",
        "analysis_unit": "session",
        "manuscript_tsnr": "tSNR_absmean",
        "other_tsnr_variants": "Diagnostic definitions only, not substitutes for the manuscript definition",
    }
    with (out_dir / "tsnr_recalc_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
