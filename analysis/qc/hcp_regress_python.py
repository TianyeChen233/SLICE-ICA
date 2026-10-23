#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_noise_idx(label_file: Path):
    txt = label_file.read_text(encoding='utf-8', errors='ignore')

    m = re.search(r'\[(.*?)\]', txt, flags=re.S)
    if m:
        arr = [int(x) for x in re.findall(r'\d+', m.group(1))]
        if arr:
            return sorted(set(arr))

    out = []
    for ln in txt.splitlines():
        l = ln.strip().lower()
        if 'unclassified noise' in l and 'true' in l:
            m2 = re.match(r'^(\d+)\s*,', ln.strip())
            if m2:
                out.append(int(m2.group(1)))
    return sorted(set(out))


def find_motion_file(sess_dir: Path):
    mc = sess_dir / 'mc'
    if mc.exists():
        cands = sorted(mc.glob('*.par')) + sorted(mc.glob('*.txt'))
        if cands:
            return cands[0]

    cands = sorted(sess_dir.glob('rp*.txt'))
    if cands:
        return cands[0]
    return None


def list_sessions(data_root: Path):
    sessions = []
    for sub in sorted(data_root.iterdir()):
        if not sub.is_dir():
            continue
        for ses in sorted(sub.iterdir()):
            if ses.is_dir() and ses.name.startswith('rfMRI_'):
                sessions.append(ses)
    return sessions


def pick_shard(sessions, worker_id: int, num_workers: int):
    # worker_id is 1-based
    target = worker_id - 1
    return [s for i, s in enumerate(sessions) if (i % num_workers) == target]


def regress_session(sess_dir: Path, suffix: str):
    func = sess_dir / 'filtered_func_data.nii.gz'
    maskf = sess_dir / 'mask.nii.gz'
    mixf = sess_dir / 'filtered_func_data.ica' / 'melodic_mix'
    labf = sess_dir / 'hand_labels_noise.txt'

    for p in (func, maskf, mixf, labf):
        if not p.exists():
            raise FileNotFoundError(str(p))

    img = nib.load(str(func))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise RuntimeError(f'bad fmri ndim={data.ndim}')
    T = data.shape[3]

    mask = nib.load(str(maskf)).get_fdata(dtype=np.float32) > 0
    if mask.shape != data.shape[:3]:
        raise RuntimeError(f'mask shape mismatch mask={mask.shape}, fmri={data.shape[:3]}')

    mix = np.loadtxt(str(mixf), dtype=np.float32)
    if mix.ndim == 1:
        mix = mix.reshape(-1, 1)
    if mix.shape[0] == T:
        mixTN = mix
    elif mix.shape[1] == T:
        mixTN = mix.T
    else:
        raise RuntimeError(f'time mismatch fmri T={T}, mix={mix.shape}')

    noise_idx = parse_noise_idx(labf)
    if not noise_idx:
        raise RuntimeError('empty noise ic list in hand_labels_noise.txt')

    n_ic = mixTN.shape[1]
    if min(noise_idx) < 1 or max(noise_idx) > n_ic:
        raise RuntimeError(f'noise ic out of range: min={min(noise_idx)} max={max(noise_idx)} n_ic={n_ic}')
    noise_idx0 = np.array(sorted(set(noise_idx)), dtype=np.int32) - 1

    motion_file = find_motion_file(sess_dir)
    if motion_file is None:
        raise RuntimeError('motion file missing (mc/*.par or rp*.txt)')

    motion = np.loadtxt(str(motion_file), dtype=np.float32)
    if motion.ndim == 1:
        motion = motion.reshape(-1, 1)
    if motion.shape[0] != T:
        raise RuntimeError(f'motion rows mismatch {motion.shape[0]} vs {T}')

    motion_dm = motion - motion.mean(axis=0, keepdims=True)
    X = np.concatenate([np.ones((T, 1), dtype=np.float32), motion_dm, mixTN[:, noise_idx0]], axis=1)
    if not np.isfinite(X).all():
        raise RuntimeError('non-finite in regressors X')

    pinvX = np.linalg.pinv(X)

    flat = data.reshape(-1, T)
    vox = np.where(mask.ravel())[0]
    out_flat = np.zeros_like(flat, dtype=np.float32)

    chunk = 25000
    for s in range(0, len(vox), chunk):
        idx = vox[s:s + chunk]
        Y = flat[idx, :]      # (V,T)
        B = pinvX @ Y.T       # (K,V)
        fit = (X @ B).T       # (V,T)
        out_flat[idx, :] = Y - fit

    out = out_flat.reshape(data.shape)

    out_file = sess_dir / f'filtered_func_data{suffix}.nii.gz'
    nib.save(nib.Nifti1Image(out, img.affine, img.header), str(out_file))

    regmat = sess_dir / f'regressors{suffix}.npz'
    np.savez_compressed(
        regmat,
        X=X.astype(np.float32),
        noise_idx=(noise_idx0 + 1).astype(np.int32),
        motion_file=str(motion_file),
    )

    return out_file, regmat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--suffix', default='_regress')
    ap.add_argument('--worker-id', type=int, default=1)
    ap.add_argument('--num-workers', type=int, default=1)
    ap.add_argument('--max-sessions', type=int, default=0, help='0 means all sessions in this shard')
    ap.add_argument('--log-file', required=True)
    args = ap.parse_args()

    if args.worker_id < 1 or args.worker_id > args.num_workers:
        raise ValueError('worker-id must be in [1, num-workers]')

    data_root = Path(args.data_root)
    all_sessions = list_sessions(data_root)
    shard_sessions = pick_shard(all_sessions, args.worker_id, args.num_workers)
    if args.max_sessions and args.max_sessions > 0:
        shard_sessions = shard_sessions[:args.max_sessions]

    logf = Path(args.log_file)
    logf.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        'pid': os.getpid(),
        'worker_id': args.worker_id,
        'num_workers': args.num_workers,
        'all_sessions': len(all_sessions),
        'shard_sessions': len(shard_sessions),
        'suffix': args.suffix,
    }
    print(json.dumps({'event': 'start', **summary}, ensure_ascii=False), flush=True)

    ok = 0
    fail = 0
    with logf.open('w', encoding='utf-8') as f:
        f.write(json.dumps({'event': 'start', **summary}, ensure_ascii=False) + '\n')
        for i, sess in enumerate(shard_sessions, 1):
            rec = {
                'event': 'session',
                'worker_id': args.worker_id,
                'num_workers': args.num_workers,
                'index_in_shard': i,
                'shard_total': len(shard_sessions),
                'session_dir': str(sess),
                'status': 'ok',
                'message': '',
                'output_file': '',
                'regressor_file': '',
            }
            try:
                out_file, reg_file = regress_session(sess, args.suffix)
                rec['output_file'] = str(out_file)
                rec['regressor_file'] = str(reg_file)
                ok += 1
            except Exception as e:
                rec['status'] = 'fail'
                rec['message'] = str(e)
                fail += 1
            line = json.dumps(rec, ensure_ascii=False)
            f.write(line + '\n')
            print(line, flush=True)

        end = {
            'event': 'end',
            'worker_id': args.worker_id,
            'num_workers': args.num_workers,
            'ok': ok,
            'fail': fail,
            'processed': ok + fail,
        }
        f.write(json.dumps(end, ensure_ascii=False) + '\n')
        print(json.dumps(end, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
