#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib

if __package__:
    from .summary_statistics import (
        adjust_bh,
        network_block_means,
        paired_wilcoxon,
        session_network_summaries,
    )
else:
    from summary_statistics import (
        adjust_bh,
        network_block_means,
        paired_wilcoxon,
        session_network_summaries,
    )

base = Path("local_data/ica_extraction")
data_root = base / "data"
atlas_path = base / "atlases" / "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
info_path = base / "atlases" / "Schaefer2018_200Parcels_7Networks_order_info.txt"


def parse_schaefer_order_info(info_path: Path) -> dict[int, str]:
    """Parse Schaefer order_info.txt which may switch between (name,idx) and (idx,name) blocks."""
    raw = info_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    lines = [ln.strip() for ln in raw if ln.strip() and not ln.strip().startswith("#")]
    id_to_label: dict[int, str] = {}
    i = 0
    while i < len(lines) - 1:
        a, b = lines[i], lines[i + 1]
        if a.startswith("7Networks_") and (not b.startswith("7Networks_")):
            parts = b.split()
            if parts and parts[0].isdigit():
                id_to_label[int(parts[0])] = a
                i += 2
                continue
        if b.startswith("7Networks_") and (not a.startswith("7Networks_")):
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
    roi_ids = np.sort(np.unique(atlas[atlas > 0]))
    R = len(roi_ids)

    labels: list[str]
    id_to_label = parse_schaefer_order_info(info_path)
    if any(int(rid) not in id_to_label for rid in roi_ids):
        raise ValueError("Atlas label information must cover every parcel")
    labels = [id_to_label[int(rid)] for rid in roi_ids]

    nets = []
    for s in labels:
        p = s.split("_")
        if len(p) < 3 or not s.startswith("7Networks_"):
            raise ValueError(f"Invalid Yeo-7 parcel label: {s}")
        nets.append(p[2])
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
        if ses.is_dir() and ses.name.startswith("rfMRI_")
    ]
    tri_i, tri_j = np.triu_indices(R, k=1)
    return (
        atlas,
        sessions,
        roi_ids,
        labels,
        nets,
        uniq_nets,
        roi_to_net_idx,
        roi_vox,
        roi_centers,
        tri_i,
        tri_j,
    )


def worker(out_dir: Path, worker_id: int, num_workers: int):
    (
        atlas,
        sessions,
        roi_ids,
        labels,
        nets,
        uniq_nets,
        roi_to_net_idx,
        roi_vox,
        roi_centers,
        tri_i,
        tri_j,
    ) = prep_common()
    shard = [s for i, s in enumerate(sessions) if i % num_workers == (worker_id - 1)]
    R = len(roi_ids)
    K = len(uniq_nets)
    E = len(tri_i)

    edge_b, edge_a, net_b, net_a, valid_sessions = [], [], [], [], []
    atlas_affine = nib.load(str(atlas_path)).affine
    arrays_path = out_dir / f"worker{worker_id}_arrays.npz"
    if arrays_path.exists() or (out_dir / f"worker{worker_id}_sessions.csv").exists():
        raise FileExistsError("Use a fresh output directory; worker outputs already exist")

    ok = 0
    errs = []
    for si, sess in enumerate(shard):
        try:
            before_img = nib.load(str(sess / "filtered_func_data.nii.gz"))
            after_img = nib.load(str(sess / "filtered_func_data_regress.nii.gz"))
            b = before_img.get_fdata(dtype=np.float32)
            a = after_img.get_fdata(dtype=np.float32)
            if b.ndim != 4 or b.shape != a.shape or b.shape[:3] != atlas.shape:
                raise ValueError("Paired 4D inputs must match each other and the atlas grid")
            if not np.allclose(before_img.affine, after_img.affine, atol=1e-5) or not np.allclose(
                before_img.affine, atlas_affine, atol=1e-5
            ):
                raise ValueError("Input/atlas affine mismatch; explicit resampling is required")
            if not np.isfinite(b).all() or not np.isfinite(a).all() or b.shape[3] < 3:
                raise ValueError("At least three finite timepoints required")
            T = b.shape[3]
            fb = b.reshape(-1, T)
            fa = a.reshape(-1, T)
            tsb = np.zeros((R, T), dtype=np.float32)
            tsa = np.zeros((R, T), dtype=np.float32)
            for r in range(R):
                idx = roi_vox[r]
                tsb[r] = fb[idx, :].mean(axis=0)
                tsa[r] = fa[idx, :].mean(axis=0)
            if np.any(tsb.std(axis=1) <= 1e-12) or np.any(tsa.std(axis=1) <= 1e-12):
                raise ValueError("Constant parcel time course: correlation is undefined")
            cb, ca = np.corrcoef(tsb), np.corrcoef(tsa)
            block_before = network_block_means(cb, roi_to_net_idx, K)
            block_after = network_block_means(ca, roi_to_net_idx, K)
            # Append only complete successful pairs; failed sessions are never zero-filled.
            edge_b.append(cb[tri_i, tri_j])
            edge_a.append(ca[tri_i, tri_j])
            net_b.append(block_before)
            net_a.append(block_after)
            valid_sessions.append(
                {"subject": sess.parent.name, "session": sess.name, "session_dir": str(sess)}
            )
            ok += 1
        except Exception as e:
            errs.append({"session_dir": str(sess), "error": str(e)})
        if (si + 1) % 10 == 0 or (si + 1) == len(shard):
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "worker": worker_id,
                        "done": si + 1,
                        "total": len(shard),
                        "ok": ok,
                        "err": len(errs),
                    }
                ),
                flush=True,
            )

    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs" / f"worker{worker_id}_errors.json").write_text(
        json.dumps(errs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        arrays_path,
        edge_before=np.asarray(edge_b, dtype=np.float32).reshape(ok, E),
        edge_after=np.asarray(edge_a, dtype=np.float32).reshape(ok, E),
        net_before=np.asarray(net_b, dtype=np.float32).reshape(ok, K, K),
        net_after=np.asarray(net_a, dtype=np.float32).reshape(ok, K, K),
        network_names=np.asarray(uniq_nets, dtype=str),
    )
    pd.DataFrame(valid_sessions, columns=["subject", "session", "session_dir"]).to_csv(
        out_dir / f"worker{worker_id}_sessions.csv", index=False
    )
    print(json.dumps({"event": "end", "worker": worker_id, "ok": ok, "err": len(errs)}), flush=True)


def merge(out_dir: Path, num_workers: int):
    (
        atlas,
        sessions,
        roi_ids,
        labels,
        nets,
        uniq_nets,
        roi_to_net_idx,
        roi_vox,
        roi_centers,
        tri_i,
        tri_j,
    ) = prep_common()
    R = len(roi_ids)
    E = len(tri_i)

    eb = []
    ea = []
    nb = []
    na = []
    if (out_dir / "fc_figure_pack_summary.json").exists():
        raise FileExistsError("Refusing to overwrite a completed FC summary")
    network_names_saved = None
    session_frames = []
    for w in range(1, num_workers + 1):
        session_frame = pd.read_csv(out_dir / f"worker{w}_sessions.csv", dtype=str)
        with np.load(out_dir / f"worker{w}_arrays.npz", allow_pickle=False) as d:
            expected_shapes = {
                "edge_before": (len(session_frame), E),
                "edge_after": (len(session_frame), E),
                "net_before": (len(session_frame), len(uniq_nets), len(uniq_nets)),
                "net_after": (len(session_frame), len(uniq_nets), len(uniq_nets)),
            }
            for key, shape in expected_shapes.items():
                if d[key].shape != shape or not np.isfinite(d[key]).all():
                    raise ValueError(f"Invalid/misaligned worker array: {w}/{key}")
            if d["network_names"].tolist() != uniq_nets:
                raise ValueError("Worker network order does not match the atlas")
            eb.append(d["edge_before"])
            ea.append(d["edge_after"])
            nb.append(d["net_before"])
            na.append(d["net_after"])
            network_names_saved = np.asarray(d["network_names"], dtype=str)
        session_frames.append(session_frame)
    eb = np.concatenate(eb, axis=0)
    ea = np.concatenate(ea, axis=0)
    nb = np.concatenate(nb, axis=0)
    na = np.concatenate(na, axis=0)
    session_frame = pd.concat(session_frames, ignore_index=True)
    if len(session_frame) < 3 or session_frame[["subject", "session"]].isna().any().any():
        raise ValueError("At least three identified, complete session pairs required")
    if session_frame.duplicated(["subject", "session"]).any():
        raise ValueError("Duplicate session across workers")

    net_before_mean = nb.mean(axis=0)
    net_after_mean = na.mean(axis=0)
    net_delta = net_after_mean - net_before_mean

    edge_stats = [paired_wilcoxon(eb[:, e], ea[:, e]) for e in range(E)]
    pvals = np.asarray([r["p_value"] for r in edge_stats])
    qvals = adjust_bh(pvals)
    delta = ea.mean(axis=0) - eb.mean(axis=0)
    sig = qvals < 0.05

    rows = []
    for e in np.where(sig)[0]:
        i = int(tri_i[e])
        j = int(tri_j[e])
        rows.append(
            {
                "roi_i": i + 1,
                "roi_j": j + 1,
                "parcel_i": int(roi_ids[i]),
                "parcel_j": int(roi_ids[j]),
                "label_i": labels[i],
                "label_j": labels[j],
                "net_i": nets[i],
                "net_j": nets[j],
                "before_mean_r": float(eb[:, e].mean()),
                "after_mean_r": float(ea[:, e].mean()),
                "delta_r": float(delta[e]),
                "wilcoxon_stat": float(edge_stats[e]["wilcoxon_stat"]),
                "p_value": float(pvals[e]),
                "q_value": float(qvals[e]),
                "x_i": float(roi_centers[i, 0]),
                "y_i": float(roi_centers[i, 1]),
                "z_i": float(roi_centers[i, 2]),
                "x_j": float(roi_centers[j, 0]),
                "y_j": float(roi_centers[j, 1]),
                "z_j": float(roi_centers[j, 2]),
            }
        )
    edge_columns = [
        "roi_i",
        "roi_j",
        "parcel_i",
        "parcel_j",
        "label_i",
        "label_j",
        "net_i",
        "net_j",
        "before_mean_r",
        "after_mean_r",
        "delta_r",
        "wilcoxon_stat",
        "p_value",
        "q_value",
        "x_i",
        "y_i",
        "z_i",
        "x_j",
        "y_j",
        "z_j",
    ]
    edf = pd.DataFrame(rows, columns=edge_columns)
    if len(edf):
        edf = edf.reindex(edf["delta_r"].abs().sort_values(ascending=False).index)
    edf.to_csv(out_dir / "edge_significant_fdr05.csv", index=False)

    np.savez_compressed(
        out_dir / "network_matrices_v1.npz",
        net_before_mean=net_before_mean.astype(np.float32),
        net_after_mean=net_after_mean.astype(np.float32),
        net_delta=net_delta.astype(np.float32),
        network_names=network_names_saved,
    )

    # Preserve session identities and the exact unit underlying group summaries.
    before_summaries, after_summaries = session_network_summaries(nb), session_network_summaries(na)
    network_summary_stats = []
    for metric in before_summaries:
        session_frame[f"{metric}_before"] = before_summaries[metric]
        session_frame[f"{metric}_after"] = after_summaries[metric]
        network_summary_stats.append(
            {"metric": metric, **paired_wilcoxon(before_summaries[metric], after_summaries[metric])}
        )
    network_stats_frame = pd.DataFrame(network_summary_stats)
    network_stats_frame["fdr_q_value"] = adjust_bh(network_stats_frame["p_value"])
    network_stats_frame.to_csv(out_dir / "network_summary_paired_stats.csv", index=False)
    for k, name in enumerate(uniq_nets):
        session_frame[f"within_{name}_before"] = nb[:, k, k]
        session_frame[f"within_{name}_after"] = na[:, k, k]
    session_frame.to_csv(out_dir / "session_network_summaries.csv", index=False)

    sample = Path(session_frame["session_dir"].iloc[0])
    b0 = nib.load(str(sample / "filtered_func_data.nii.gz")).get_fdata(dtype=np.float32)
    m0 = nib.load(str(sample / "mask.nii.gz")).get_fdata(dtype=np.float32) > 0
    mean0 = b0.mean(axis=3)
    zs = [int(np.argmax(m0.sum(axis=(0, 1)))), int(m0.shape[2] * 0.35), int(m0.shape[2] * 0.65)]
    zs = sorted(set([max(0, min(m0.shape[2] - 1, z)) for z in zs]))
    qd = {"n_slices": np.array([len(zs)], dtype=np.int32)}
    for k, z in enumerate(zs):
        qd[f"bg_{k}"] = mean0[:, :, z].astype(np.float32)
        qd[f"mask_{k}"] = m0[:, :, z].astype(np.uint8)
        qd[f"atlas_{k}"] = atlas[:, :, z].astype(np.int32)
        qd[f"z_{k}"] = np.array([z], dtype=np.int32)
    np.savez_compressed(out_dir / "atlas_alignment_qc_v1.npz", **qd)

    summary = {
        "n_sessions_merged": int(eb.shape[0]),
        "n_participants_merged": int(session_frame["subject"].nunique()),
        "n_rois": int(R),
        "n_edges_total": int(E),
        "n_edges_significant_fdr05": int(sig.sum()),
        "n_networks": int(len(network_names_saved)),
        "schaefer_info_parsed_ids": int(len(parse_schaefer_order_info(info_path)))
        if info_path.exists()
        else 0,
        "paired_test": "wilcoxon_two_sided",
        "analysis_unit": "session",
        "edge_multiple_testing": "BH across all unique parcel edges",
        "network_summary_multiple_testing": "BH across within, between and within/between ratio",
        "within_network_definition": "Mean unique parcel pairs, excluding parcel self-correlations",
        "session_summary_weighting": "Equal network weight within; equal network-pair weight between",
        "ratio_definition": "Session within mean divided by session between mean; zero denominator is undefined",
        "ratio_of_group_means_before": float(
            before_summaries["within"].mean() / before_summaries["between"].mean()
        )
        if abs(before_summaries["between"].mean()) > 1e-12
        else None,
        "ratio_of_group_means_after": float(
            after_summaries["within"].mean() / after_summaries["between"].mean()
        )
        if abs(after_summaries["between"].mean()) > 1e-12
        else None,
    }
    (out_dir / "fc_figure_pack_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    global data_root, atlas_path, info_path
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["worker", "merge"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--worker-id", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--atlas", type=Path, required=True)
    ap.add_argument("--atlas-info", type=Path, required=True)
    args = ap.parse_args()
    if args.num_workers < 1 or not 1 <= args.worker_id <= args.num_workers:
        ap.error("Positive worker count and worker ID within 1..num_workers required")
    data_root, atlas_path, info_path = args.data_root, args.atlas, args.atlas_info
    if not data_root.is_dir() or not atlas_path.is_file() or not info_path.is_file():
        ap.error("Data root, atlas and complete Yeo-7 atlas information must exist")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "worker":
        worker(out_dir, args.worker_id, args.num_workers)
    else:
        merge(out_dir, args.num_workers)


if __name__ == "__main__":
    main()
