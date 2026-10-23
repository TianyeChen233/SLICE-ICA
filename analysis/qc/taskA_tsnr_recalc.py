#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import ttest_rel


def benjamini_hochberg(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


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
    b = b[m]; a = a[m]
    if len(b) < 3:
        return {'metric': name, 'n': int(len(b)), 'before_mean': np.nan, 'after_mean': np.nan,
                'mean_delta_after_minus_before': np.nan, 't_stat': np.nan, 'p_value': np.nan,
                'effect_size_cohen_dz': np.nan}
    t, p = ttest_rel(a, b, nan_policy='omit')
    return {
        'metric': name,
        'n': int(len(b)),
        'before_mean': float(np.mean(b)),
        'after_mean': float(np.mean(a)),
        'mean_delta_after_minus_before': float(np.mean(a - b)),
        't_stat': float(t),
        'p_value': float(p),
        'effect_size_cohen_dz': cohen_d_paired(a, b)
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
        'tSNR_signed_mean': float(np.mean(tsnr_signed[np.isfinite(tsnr_signed)])),
        'tSNR_absmean_mean': float(np.mean(tsnr_absmean[np.isfinite(tsnr_absmean)])),
        'tSNR_addback_mean': float(np.mean(tsnr_addback[np.isfinite(tsnr_addback)])),
    }
    return out


def main():
    data_root = Path('local_data/ica_extraction/data')
    base_dir = Path('local_data/ica_extraction/downstream_taskA_qc_parallel_20260430_1240')
    out_dir = base_dir / 'tsnr_recalc'
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions = [ses for sub in sorted(data_root.iterdir()) if sub.is_dir() for ses in sorted(sub.iterdir()) if ses.is_dir() and ses.name.startswith('rfMRI_')]

    rows = []
    errors = []
    for i, sess in enumerate(sessions, 1):
        try:
            b = nib.load(str(sess / 'filtered_func_data.nii.gz')).get_fdata(dtype=np.float32)
            a = nib.load(str(sess / 'filtered_func_data_regress.nii.gz')).get_fdata(dtype=np.float32)
            m = nib.load(str(sess / 'mask.nii.gz')).get_fdata(dtype=np.float32) > 0

            if b.shape != a.shape or b.ndim != 4:
                raise RuntimeError('shape mismatch or non-4d')
            if m.shape != b.shape[:3]:
                raise RuntimeError('mask mismatch')

            vb = compute_tsnr_variants(b, m)
            va = compute_tsnr_variants(a, m)
            rows.append({
                'subject': sess.parent.name,
                'session': sess.name,
                'session_dir': str(sess),
                'tSNR_signed_before': vb['tSNR_signed_mean'],
                'tSNR_signed_after': va['tSNR_signed_mean'],
                'tSNR_absmean_before': vb['tSNR_absmean_mean'],
                'tSNR_absmean_after': va['tSNR_absmean_mean'],
                'tSNR_addback_before': vb['tSNR_addback_mean'],
                'tSNR_addback_after': va['tSNR_addback_mean'],
            })
            if i % 25 == 0:
                print(f'[progress] {i}/{len(sessions)}', flush=True)
        except Exception as e:
            errors.append({'session_dir': str(sess), 'error': str(e)})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'session_tsnr_recalc.csv', index=False)

    stats = [
        paired_stats(df, 'tSNR_signed_before', 'tSNR_signed_after', 'tSNR_signed'),
        paired_stats(df, 'tSNR_absmean_before', 'tSNR_absmean_after', 'tSNR_absmean'),
        paired_stats(df, 'tSNR_addback_before', 'tSNR_addback_after', 'tSNR_addback'),
    ]
    st = pd.DataFrame(stats)
    st['fdr_q_value'] = benjamini_hochberg(st['p_value'].to_numpy(dtype=float))
    st.to_csv(out_dir / 'tsnr_recalc_stats_fdr.csv', index=False)

    report = {
        'n_sessions_total': len(sessions),
        'n_sessions_ok': len(df),
        'n_sessions_error': len(errors),
        'error_examples': errors[:10],
        'output_dir': str(out_dir)
    }
    with (out_dir / 'tsnr_recalc_report.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
