#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import ttest_rel

base = Path('local_data/ica_extraction')
data_root = base / 'data'
atlas_path = base / 'atlases' / 'Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz'
info_path = base / 'atlases' / 'Schaefer2018_200Parcels_7Networks_order_info.txt'


def parse_schaefer_order_info(info_path: Path) -> dict[int, str]:
    """Parse Schaefer order_info.txt which may switch between (name,idx) and (idx,name) blocks."""
    raw = info_path.read_text(encoding='utf-8', errors='ignore').splitlines()
    lines = [ln.strip() for ln in raw if ln.strip() and not ln.strip().startswith('#')]
    id_to_label: dict[int, str] = {}
    i = 0
    while i < len(lines) - 1:
        a, b = lines[i], lines[i + 1]
        if a.startswith('7Networks_') and (not b.startswith('7Networks_')):
            parts = b.split()
            if parts and parts[0].isdigit():
                id_to_label[int(parts[0])] = a
                i += 2
                continue
        if b.startswith('7Networks_') and (not a.startswith('7Networks_')):
            parts = a.split()
            if parts and parts[0].isdigit():
                id_to_label[int(parts[0])] = b
                i += 2
                continue
        i += 1
    return id_to_label


def prep_common():
    atlas_img = nib.load(str(atlas_path))
    atlas = atlas_img.get_fdata(dtype=np.float32).astype(np.int32)
    aff = atlas_img.affine
    roi_ids = np.sort(np.unique(atlas)[1:])
    R = len(roi_ids)

    labels: list[str]
    if info_path.exists():
        id_to_label = parse_schaefer_order_info(info_path)
        labels = [id_to_label.get(int(rid), f'ROI_{int(rid):03d}') for rid in roi_ids]
    else:
        labels = [f'ROI_{int(rid):03d}' for rid in roi_ids]

    nets = []
    for s in labels:
        p = s.split('_')
        nets.append(p[2] if len(p) >= 3 else 'Unknown')
    uniq_nets = sorted(set(nets))
    net_to_idx = {n: i for i, n in enumerate(uniq_nets)}
    roi_to_net_idx = np.array([net_to_idx[n] for n in nets], dtype=np.int32)

    roi_vox = []
    roi_centers = np.zeros((R, 3), dtype=np.float32)
    for i, rid in enumerate(roi_ids):
        idx = np.where(atlas.ravel() == rid)[0]
        roi_vox.append(idx)
        xyz = np.column_stack(np.where(atlas == rid))
        c = xyz.mean(axis=0)
        mni = aff @ np.array([c[0], c[1], c[2], 1.0])
        roi_centers[i] = mni[:3]

    sessions = [
        ses
        for sub in sorted(data_root.iterdir())
        if sub.is_dir()
        for ses in sorted(sub.iterdir())
        if ses.is_dir() and ses.name.startswith('rfMRI_')
    ]
    tri_i, tri_j = np.triu_indices(R, k=1)
    return atlas, sessions, roi_ids, labels, nets, uniq_nets, roi_to_net_idx, roi_vox, roi_centers, tri_i, tri_j


def worker(out_dir: Path, worker_id: int, num_workers: int):
    atlas, sessions, roi_ids, labels, nets, uniq_nets, roi_to_net_idx, roi_vox, roi_centers, tri_i, tri_j = prep_common()
    shard = [s for i, s in enumerate(sessions) if i % num_workers == (worker_id - 1)]
    R = len(roi_ids)
    K = len(uniq_nets)
    E = len(tri_i)

    n = len(shard)
    edge_b = np.zeros((n, E), dtype=np.float32)
    edge_a = np.zeros((n, E), dtype=np.float32)
    net_b = np.zeros((n, K, K), dtype=np.float32)
    net_a = np.zeros((n, K, K), dtype=np.float32)

    ok = 0
    errs = []
    for si, sess in enumerate(shard):
        try:
            b = nib.load(str(sess / 'filtered_func_data.nii.gz')).get_fdata(dtype=np.float32)
            a = nib.load(str(sess / 'filtered_func_data_regress.nii.gz')).get_fdata(dtype=np.float32)
            T = b.shape[3]
            fb = b.reshape(-1, T)
            fa = a.reshape(-1, T)
            tsb = np.zeros((R, T), dtype=np.float32)
            tsa = np.zeros((R, T), dtype=np.float32)
            for r in range(R):
                idx = roi_vox[r]
                tsb[r] = fb[idx, :].mean(axis=0)
                tsa[r] = fa[idx, :].mean(axis=0)
            tsb = (tsb - tsb.mean(axis=1, keepdims=True)) / (tsb.std(axis=1, keepdims=True) + 1e-8)
            tsa = (tsa - tsa.mean(axis=1, keepdims=True)) / (tsa.std(axis=1, keepdims=True) + 1e-8)
            cb = np.nan_to_num(np.corrcoef(tsb), nan=0.0, posinf=0.0, neginf=0.0)
            ca = np.nan_to_num(np.corrcoef(tsa), nan=0.0, posinf=0.0, neginf=0.0)
            edge_b[si] = cb[tri_i, tri_j]
            edge_a[si] = ca[tri_i, tri_j]
            for ii in range(K):
                ri = np.where(roi_to_net_idx == ii)[0]
                for jj in range(K):
                    rj = np.where(roi_to_net_idx == jj)[0]
                    net_b[si, ii, jj] = cb[np.ix_(ri, rj)].mean()
                    net_a[si, ii, jj] = ca[np.ix_(ri, rj)].mean()
            ok += 1
        except Exception as e:
            errs.append({'session_dir': str(sess), 'error': str(e)})
        if (si + 1) % 10 == 0 or (si + 1) == len(shard):
            print(
                json.dumps(
                    {'event': 'progress', 'worker': worker_id, 'done': si + 1, 'total': len(shard), 'ok': ok, 'err': len(errs)}
                ),
                flush=True,
            )

    np.savez_compressed(
        out_dir / f'worker{worker_id}_arrays.npz',
        edge_before=edge_b,
        edge_after=edge_a,
        net_before=net_b,
        net_after=net_a,
        network_names=np.array(uniq_nets, dtype=object),
    )
    pd.DataFrame({'session_dir': [str(s) for s in shard]}).to_csv(out_dir / f'worker{worker_id}_sessions.csv', index=False)
    (out_dir / 'logs').mkdir(parents=True, exist_ok=True)
    (out_dir / 'logs' / f'worker{worker_id}_errors.json').write_text(json.dumps(errs, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'event': 'end', 'worker': worker_id, 'ok': ok, 'err': len(errs)}), flush=True)


def merge(out_dir: Path, num_workers: int):
    atlas, sessions, roi_ids, labels, nets, uniq_nets, roi_to_net_idx, roi_vox, roi_centers, tri_i, tri_j = prep_common()
    R = len(roi_ids)
    E = len(tri_i)

    eb = []
    ea = []
    nb = []
    na = []
    network_names_saved = None
    for w in range(1, num_workers + 1):
        d = np.load(out_dir / f'worker{w}_arrays.npz', allow_pickle=True)
        eb.append(d['edge_before'])
        ea.append(d['edge_after'])
        nb.append(d['net_before'])
        na.append(d['net_after'])
        if network_names_saved is None and 'network_names' in d.files:
            network_names_saved = np.array(d['network_names'], dtype=object)
    eb = np.concatenate(eb, axis=0)
    ea = np.concatenate(ea, axis=0)
    nb = np.concatenate(nb, axis=0)
    na = np.concatenate(na, axis=0)

    net_before_mean = nb.mean(axis=0)
    net_after_mean = na.mean(axis=0)
    net_delta = net_after_mean - net_before_mean

    tvals, pvals = ttest_rel(ea, eb, axis=0, nan_policy='omit')
    tvals = np.nan_to_num(tvals, nan=0.0)
    pvals = np.nan_to_num(pvals, nan=1.0)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = np.empty(E)
    prev = 1.0
    for i in range(E - 1, -1, -1):
        prev = min(prev, ranked[i] * E / (i + 1))
        q[i] = prev
    qvals = np.empty(E)
    qvals[order] = np.clip(q, 0, 1)
    delta = ea.mean(axis=0) - eb.mean(axis=0)
    sig = qvals < 0.05

    rows = []
    for e in np.where(sig)[0]:
        i = int(tri_i[e])
        j = int(tri_j[e])
        rows.append(
            {
                'roi_i': i + 1,
                'roi_j': j + 1,
                'parcel_i': int(roi_ids[i]),
                'parcel_j': int(roi_ids[j]),
                'label_i': labels[i],
                'label_j': labels[j],
                'net_i': nets[i],
                'net_j': nets[j],
                'before_mean_r': float(eb[:, e].mean()),
                'after_mean_r': float(ea[:, e].mean()),
                'delta_r': float(delta[e]),
                't_stat': float(tvals[e]),
                'p_value': float(pvals[e]),
                'q_value': float(qvals[e]),
                'x_i': float(roi_centers[i, 0]),
                'y_i': float(roi_centers[i, 1]),
                'z_i': float(roi_centers[i, 2]),
                'x_j': float(roi_centers[j, 0]),
                'y_j': float(roi_centers[j, 1]),
                'z_j': float(roi_centers[j, 2]),
            }
        )
    edf = pd.DataFrame(rows)
    if len(edf):
        edf = edf.reindex(edf['delta_r'].abs().sort_values(ascending=False).index)
    edf.to_csv(out_dir / 'edge_significant_fdr05.csv', index=False)

    if network_names_saved is None:
        network_names_saved = np.array(uniq_nets, dtype=object)
    np.savez_compressed(
        out_dir / 'network_matrices_v1.npz',
        net_before_mean=net_before_mean.astype(np.float32),
        net_after_mean=net_after_mean.astype(np.float32),
        net_delta=net_delta.astype(np.float32),
        network_names=network_names_saved,
    )

    sample = sessions[0]
    b0 = nib.load(str(sample / 'filtered_func_data.nii.gz')).get_fdata(dtype=np.float32)
    m0 = nib.load(str(sample / 'mask.nii.gz')).get_fdata(dtype=np.float32) > 0
    mean0 = b0.mean(axis=3)
    zs = [int(np.argmax(m0.sum(axis=(0, 1)))), int(m0.shape[2] * 0.35), int(m0.shape[2] * 0.65)]
    zs = sorted(set([max(0, min(m0.shape[2] - 1, z)) for z in zs]))
    qd = {'n_slices': np.array([len(zs)], dtype=np.int32)}
    for k, z in enumerate(zs):
        qd[f'bg_{k}'] = mean0[:, :, z].astype(np.float32)
        qd[f'mask_{k}'] = m0[:, :, z].astype(np.uint8)
        qd[f'atlas_{k}'] = atlas[:, :, z].astype(np.int32)
        qd[f'z_{k}'] = np.array([z], dtype=np.int32)
    np.savez_compressed(out_dir / 'atlas_alignment_qc_v1.npz', **qd)

    summary = {
        'n_sessions_merged': int(eb.shape[0]),
        'n_rois': int(R),
        'n_edges_total': int(E),
        'n_edges_significant_fdr05': int(sig.sum()),
        'n_networks': int(len(network_names_saved)),
        'schaefer_info_parsed_ids': int(len(parse_schaefer_order_info(info_path))) if info_path.exists() else 0,
    }
    (out_dir / 'fc_figure_pack_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['worker', 'merge'], required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--worker-id', type=int, default=1)
    ap.add_argument('--num-workers', type=int, default=4)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == 'worker':
        worker(out_dir, args.worker_id, args.num_workers)
    else:
        merge(out_dir, args.num_workers)


if __name__ == '__main__':
    main()
