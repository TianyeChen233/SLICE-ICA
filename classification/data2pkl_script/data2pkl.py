#!/usr/bin/env python3
import argparse
import csv
import json
import pickle
import random
import re
import shutil
import traceback
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ICA dataset into per-IC PKL files with auto-calibration."
    )
    parser.add_argument("--input-root", default="local_data/DATA", help="Input dataset root")
    parser.add_argument(
        "--output-root",
        default="classification/data/ALL_DATASETS_20260421",
        help="Output dataset directory",
    )
    parser.add_argument("--seed", type=int, default=20260421, help="Random seed")
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete output root before running",
    )
    return parser.parse_args()


def discover_subject_dirs(input_root: Path):
    subjects = []
    for p in input_root.rglob("*.ica"):
        if p.is_dir() and p.name != "filtered_func_data.ica":
            subjects.append(p)
    return sorted(subjects, key=lambda x: str(x).lower())


def parse_label_file(label_file: Path):
    text = label_file.read_text(encoding="utf-8", errors="ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    bracket_line = None
    for line in lines:
        if "[" in line and "]" in line:
            bracket_line = line
            break

    if bracket_line is not None:
        nums = re.findall(r"-?\d+", bracket_line)
        indices = sorted(set(int(n) for n in nums if int(n) > 0))
        max_idx = max(indices) if indices else 0
        return {
            "mode": "noise_indices",
            "noise_indices_1based": indices,
            "max_noise_index": max_idx,
        }

    tokens = re.findall(r"-?\d+", text)
    if not tokens:
        raise ValueError(f"Unable to parse labels from: {label_file}")

    dense = np.array([int(t) for t in tokens], dtype=np.int64)
    return {"mode": "dense", "labels": dense}


def load_features(feature_file: Path):
    arr = np.loadtxt(feature_file, delimiter=",")
    arr = np.asarray(arr)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    return arr


def load_tc(tc_file: Path):
    tc = np.loadtxt(tc_file)
    tc = np.asarray(tc)
    if tc.ndim == 0:
        tc = tc.reshape(1, 1)
    return tc


def get_image_shape(image_file: Path):
    shape = nib.load(str(image_file)).shape
    squeezed = tuple(int(x) for x in np.squeeze(np.zeros(shape)).shape) if len(shape) > 0 else tuple(shape)
    if len(squeezed) == 0:
        squeezed = (1,)
    return tuple(int(x) for x in squeezed)


def determine_anchor_n(label_info, features_raw, tc_raw, image_shape, subject_id, decisions):
    if len(image_shape) != 4:
        raise ValueError(f"[{subject_id}] image shape must be 4D after squeeze, got {image_shape}")

    if tc_raw.ndim == 1:
        tc_shape = (int(tc_raw.shape[0]),)
    elif tc_raw.ndim == 2:
        tc_shape = (int(tc_raw.shape[0]), int(tc_raw.shape[1]))
    else:
        raise ValueError(f"[{subject_id}] tc ndim={tc_raw.ndim} not supported")

    if features_raw.ndim == 1:
        feat_shape = (int(features_raw.shape[0]),)
    elif features_raw.ndim == 2:
        feat_shape = (int(features_raw.shape[0]), int(features_raw.shape[1]))
    else:
        raise ValueError(f"[{subject_id}] features ndim={features_raw.ndim} not supported")

    img_dims = set(image_shape)
    tc_dims = set(tc_shape)
    feat_dims = set(feat_shape)

    candidates = sorted(img_dims & tc_dims & feat_dims)

    if label_info["mode"] == "dense":
        dense_len = int(label_info["labels"].shape[0])
        candidates = [n for n in candidates if n == dense_len]
    else:
        max_noise = int(label_info.get("max_noise_index", 0))
        candidates = [n for n in candidates if n >= max_noise]

    decisions.append(
        f"[{subject_id}] anchor candidates from image/tc/features intersection: {candidates}; "
        f"image_shape={image_shape}, tc_shape={tc_shape}, features_shape={feat_shape}"
    )

    if not candidates:
        raise ValueError(
            f"[{subject_id}] no anchor N found. image={image_shape}, tc={tc_shape}, features={feat_shape}"
        )

    if len(candidates) == 1:
        n = candidates[0]
        decisions.append(f"[{subject_id}] anchor N={n} (unique candidate)")
        return n

    max_noise = int(label_info.get("max_noise_index", 0)) if label_info["mode"] == "noise_indices" else 0

    scored = []
    for n in candidates:
        score = 0.0
        if image_shape[-1] == n:
            score += 4.0
        if len(tc_shape) == 2 and tc_shape[1] == n:
            score += 2.0
        if tc_shape[0] == n:
            score += 1.0
        if len(feat_shape) == 2 and feat_shape[0] == n:
            score += 2.0
        if len(feat_shape) == 2 and feat_shape[1] == n:
            score += 1.0
        if label_info["mode"] == "dense" and int(label_info["labels"].shape[0]) == n:
            score += 5.0
        if label_info["mode"] == "noise_indices":
            score -= abs(n - max_noise) * 1e-3
        scored.append((score, n))

    scored_sorted = sorted(scored, key=lambda x: (-x[0], x[1]))
    chosen = scored_sorted[0][1]
    decisions.append(
        f"[{subject_id}] multiple anchor candidates {candidates}, scores={scored_sorted}, choose N={chosen}"
    )
    return chosen


def align_features(features_raw, n_anchor, subject_id, decisions):
    arr = np.asarray(features_raw)

    if arr.ndim == 1:
        if n_anchor == 1:
            arr = arr.reshape(1, -1)
            decisions.append(f"[{subject_id}] features 1D -> reshape (1,F)")
        elif arr.shape[0] == n_anchor:
            arr = arr.reshape(n_anchor, 1)
            decisions.append(f"[{subject_id}] features 1D -> reshape (N,1)")
        else:
            raise ValueError(f"[{subject_id}] features 1D len={arr.shape[0]} not matching N={n_anchor}")
    elif arr.ndim != 2:
        raise ValueError(f"[{subject_id}] features ndim={arr.ndim} not supported")

    r, c = int(arr.shape[0]), int(arr.shape[1])
    if r == n_anchor and c != n_anchor:
        decisions.append(f"[{subject_id}] features kept as (N,F): {arr.shape}")
        return arr
    if c == n_anchor and r != n_anchor:
        decisions.append(f"[{subject_id}] features transposed (F,N)->(N,F): {arr.shape} -> {(c, r)}")
        return arr.T
    if r == n_anchor and c == n_anchor:
        decisions.append(f"[{subject_id}] features ambiguous square {arr.shape}; kept as-is")
        return arr
    raise ValueError(f"[{subject_id}] features shape {arr.shape} cannot align to N={n_anchor}")


def align_tc(tc_raw, n_anchor, subject_id, decisions):
    tc = np.asarray(tc_raw)
    if tc.ndim == 1:
        if n_anchor == 1:
            tc = tc.reshape(1, -1)
            decisions.append(f"[{subject_id}] tc 1D -> reshape (1,T)")
        elif tc.shape[0] == n_anchor:
            tc = tc.reshape(n_anchor, 1)
            decisions.append(f"[{subject_id}] tc 1D -> reshape (N,1)")
        else:
            raise ValueError(f"[{subject_id}] tc 1D len={tc.shape[0]} cannot align to N={n_anchor}")
    elif tc.ndim != 2:
        raise ValueError(f"[{subject_id}] tc ndim={tc.ndim} not supported")

    r, c = int(tc.shape[0]), int(tc.shape[1])
    if r == n_anchor and c != n_anchor:
        decisions.append(f"[{subject_id}] tc kept as (N,T): {tc.shape}")
        return tc
    if c == n_anchor and r != n_anchor:
        decisions.append(f"[{subject_id}] tc transposed (T,N)->(N,T): {tc.shape} -> {(c, r)}")
        return tc.T
    if r == n_anchor and c == n_anchor:
        decisions.append(f"[{subject_id}] tc ambiguous square {tc.shape}; kept as-is")
        return tc
    raise ValueError(f"[{subject_id}] tc shape {tc.shape} cannot align to N={n_anchor}")


def load_and_align_image(image_file: Path, n_anchor, subject_id, decisions):
    data = np.asarray(nib.load(str(image_file)).get_fdata(dtype=np.float32))
    data = np.squeeze(data)

    if data.ndim != 4:
        raise ValueError(f"[{subject_id}] image ndim={data.ndim}, expected 4")

    matches = [ax for ax, dim in enumerate(data.shape) if int(dim) == n_anchor]
    if not matches:
        raise ValueError(
            f"[{subject_id}] no image dimension matches N={n_anchor}, image shape={data.shape}"
        )

    if len(matches) == 1:
        chosen = matches[0]
    else:
        chosen = 3 if 3 in matches else matches[-1]
        decisions.append(f"[{subject_id}] image multiple axes match N {matches}, choose axis {chosen}")

    if chosen != 3:
        aligned = np.moveaxis(data, chosen, 3)
        decisions.append(f"[{subject_id}] image reordered axis {chosen}->3: {data.shape} -> {aligned.shape}")
    else:
        aligned = data
        decisions.append(f"[{subject_id}] image kept as (X,Y,Z,N): {aligned.shape}")

    if int(aligned.shape[3]) != n_anchor:
        raise ValueError(f"[{subject_id}] aligned image last axis {aligned.shape[3]} != N={n_anchor}")
    return aligned


def align_labels(label_info, n_anchor, subject_id, decisions):
    if label_info["mode"] == "dense":
        labels = np.asarray(label_info["labels"]).reshape(-1)
        if int(labels.shape[0]) != n_anchor:
            raise ValueError(
                f"[{subject_id}] dense labels len={labels.shape[0]} != N={n_anchor}"
            )
        decisions.append(f"[{subject_id}] labels dense vector kept, len={labels.shape[0]}")
        return labels.astype(np.int64)

    indices = label_info["noise_indices_1based"]
    labels = np.zeros(n_anchor, dtype=np.int64)
    bad = [idx for idx in indices if idx < 1 or idx > n_anchor]
    if bad:
        raise ValueError(f"[{subject_id}] label index out of range for N={n_anchor}: {bad[:10]}")
    for idx in indices:
        labels[idx - 1] = 1
    decisions.append(
        f"[{subject_id}] labels expanded from noise-index list, noise_count={int(labels.sum())}, len={n_anchor}"
    )
    return labels


def stringify_shape(shape_tuple):
    return "x".join(str(int(x)) for x in shape_tuple)


def write_mapping_csv(mapping_rows, output_root: Path):
    mapping_file = output_root / "mapping.csv"
    fieldnames = [
        "new_subject_id",
        "ic_index",
        "new_pkl_path",
        "original_subject",
        "original_ic_index",
        "tc_source_file",
        "label_source_file",
        "feature_source_file",
        "image_source_file",
    ]
    with mapping_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mapping_rows)
    return mapping_file


def make_self_check_report(
    output_root: Path,
    image_shape_counter,
    tc_len_counter,
    feat_dim_counter,
    label_counter,
    calibration_counter,
    calibration_examples,
    alignment_ok_subjects,
    total_subjects,
    total_ics,
    total_pkls,
    errors,
    random_sample_checks,
):
    report = {
        "shape_statistics": {
            "image_shapes": dict(image_shape_counter),
            "tc_lengths": dict(tc_len_counter),
            "feature_dimensions": dict(feat_dim_counter),
        },
        "label_distribution": dict(label_counter),
        "alignment_validation": {
            "aligned_subjects": alignment_ok_subjects,
            "total_subjects": total_subjects,
            "all_aligned": alignment_ok_subjects == total_subjects and len(errors) == 0,
        },
        "random_sample_check": random_sample_checks,
        "error_report": errors,
        "summary": {
            "subjects": total_subjects,
            "total_ics": total_ics,
            "total_pkl_files": total_pkls,
        },
        "calibration_decisions": {
            "decision_counts": dict(calibration_counter),
            "examples": calibration_examples[:200],
        },
    }

    report_json = output_root / "self_check_report.json"
    report_txt = output_root / "self_check_report.txt"
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("SELF-CHECK REPORT")
    lines.append("=" * 80)
    lines.append(f"Subjects: {total_subjects}")
    lines.append(f"Total ICs: {total_ics}")
    lines.append(f"Total PKL files: {total_pkls}")
    lines.append("")
    lines.append("Shape statistics:")
    lines.append(f"- image shapes: {dict(image_shape_counter)}")
    lines.append(f"- tc lengths: {dict(tc_len_counter)}")
    lines.append(f"- feature dimensions: {dict(feat_dim_counter)}")
    lines.append("")
    lines.append("Label distribution:")
    lines.append(f"- {dict(label_counter)}")
    lines.append("")
    lines.append("Alignment validation:")
    lines.append(f"- aligned subjects: {alignment_ok_subjects}/{total_subjects}")
    lines.append(f"- all aligned: {alignment_ok_subjects == total_subjects and len(errors) == 0}")
    lines.append("")
    lines.append("Calibration decision counts:")
    lines.append(f"- {dict(calibration_counter)}")
    lines.append("")
    lines.append("Random sample check:")
    for item in random_sample_checks:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Error report:")
    if errors:
        for e in errors:
            lines.append(f"- {e}")
    else:
        lines.append("- No errors")

    report_txt.write_text("\n".join(lines), encoding="utf-8")
    return report_json, report_txt


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if args.clean_output and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    subjects = discover_subject_dirs(input_root)
    if not subjects:
        raise RuntimeError(f"No subject directories found under {input_root}")

    image_shape_counter = Counter()
    tc_len_counter = Counter()
    feat_dim_counter = Counter()
    label_counter = Counter()
    calibration_counter = Counter()
    calibration_examples = []
    errors = []
    mapping_rows = []
    alignment_ok_subjects = 0
    total_ics = 0
    total_pkls = 0

    for sub_idx, subject_dir in enumerate(subjects, start=1):
        new_subject_id = f"sub{sub_idx:03d}"
        out_sub_dir = output_root / new_subject_id
        out_sub_dir.mkdir(parents=True, exist_ok=True)

        subject_tag = str(subject_dir.relative_to(input_root)).replace("\\", "/")
        image_file = subject_dir / "filtered_func_data.ica" / "melodic_IC.nii.gz"
        tc_file = subject_dir / "filtered_func_data.ica" / "melodic_mix"
        label_file = subject_dir / "hand_labels_noise.txt"
        feature_file = subject_dir / "fix" / "features.csv"

        try:
            missing = [
                name
                for name, fp in [
                    ("image", image_file),
                    ("tc", tc_file),
                    ("label", label_file),
                    ("features", feature_file),
                ]
                if not fp.exists()
            ]
            if missing:
                raise FileNotFoundError(f"[{subject_tag}] missing files: {missing}")

            decisions = []
            label_info = parse_label_file(label_file)
            features_raw = load_features(feature_file)
            tc_raw = load_tc(tc_file)
            image_shape = get_image_shape(image_file)

            n_anchor = determine_anchor_n(
                label_info=label_info,
                features_raw=features_raw,
                tc_raw=tc_raw,
                image_shape=image_shape,
                subject_id=subject_tag,
                decisions=decisions,
            )

            features = align_features(features_raw, n_anchor, subject_tag, decisions)
            tc = align_tc(tc_raw, n_anchor, subject_tag, decisions)
            img_4d = load_and_align_image(image_file, n_anchor, subject_tag, decisions)
            labels = align_labels(label_info, n_anchor, subject_tag, decisions)

            if int(features.shape[0]) != n_anchor:
                raise ValueError(f"[{subject_tag}] features rows={features.shape[0]} != N={n_anchor}")
            if int(tc.shape[0]) != n_anchor:
                raise ValueError(f"[{subject_tag}] tc rows={tc.shape[0]} != N={n_anchor}")
            if int(img_4d.shape[3]) != n_anchor:
                raise ValueError(f"[{subject_tag}] image N={img_4d.shape[3]} != N={n_anchor}")
            if int(labels.shape[0]) != n_anchor:
                raise ValueError(f"[{subject_tag}] label len={labels.shape[0]} != N={n_anchor}")

            for d in decisions:
                calibration_examples.append(d)
                if "tc transposed" in d:
                    calibration_counter["tc_transposed"] += 1
                if "tc kept as" in d:
                    calibration_counter["tc_not_transposed"] += 1
                if "image reordered" in d:
                    calibration_counter["image_reordered"] += 1
                if "image kept as" in d:
                    calibration_counter["image_not_reordered"] += 1
                if "multiple axes match" in d:
                    calibration_counter["image_multi_match_decision"] += 1
                if "features transposed" in d:
                    calibration_counter["features_transposed"] += 1
                if "features kept as" in d:
                    calibration_counter["features_not_transposed"] += 1

            alignment_ok_subjects += 1
            total_ics += n_anchor

            for ic_idx in range(n_anchor):
                pkl_name = f"ica_{ic_idx + 1:03d}.pkl"
                pkl_path = out_sub_dir / pkl_name
                payload = {
                    "image": img_4d[..., ic_idx].astype(np.float32),
                    "tc": tc[ic_idx].astype(np.float32).reshape(-1),
                    "label": int(labels[ic_idx]),
                    "features": features[ic_idx].astype(np.float32).reshape(-1),
                }
                with pkl_path.open("wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

                image_shape_counter[stringify_shape(payload["image"].shape)] += 1
                tc_len_counter[int(payload["tc"].shape[0])] += 1
                feat_dim_counter[int(payload["features"].shape[0])] += 1
                label_counter[int(payload["label"])] += 1
                total_pkls += 1

                mapping_rows.append(
                    {
                        "new_subject_id": new_subject_id,
                        "ic_index": ic_idx + 1,
                        "new_pkl_path": str(pkl_path),
                        "original_subject": subject_tag,
                        "original_ic_index": ic_idx + 1,
                        "tc_source_file": str(tc_file),
                        "label_source_file": str(label_file),
                        "feature_source_file": str(feature_file),
                        "image_source_file": str(image_file),
                    }
                )

            print(f"[OK] {new_subject_id} <- {subject_tag}, N={n_anchor}", flush=True)

        except Exception as e:
            err = f"{e}\n{traceback.format_exc()}"
            errors.append(err)
            print(f"[ERROR] {subject_tag}: {e}", flush=True)

    mapping_file = write_mapping_csv(mapping_rows, output_root)

    sample_rows = random.sample(mapping_rows, k=min(3, len(mapping_rows))) if mapping_rows else []
    random_sample_checks = []
    for row in sample_rows:
        p = Path(row["new_pkl_path"])
        with p.open("rb") as f:
            obj = pickle.load(f)
        random_sample_checks.append(
            {
                "new_subject_id": row["new_subject_id"],
                "ic_index": row["ic_index"],
                "pkl_path": row["new_pkl_path"],
                "image_shape": tuple(obj["image"].shape),
                "tc_shape": tuple(obj["tc"].shape),
                "features_shape": tuple(obj["features"].shape),
                "label": int(obj["label"]),
            }
        )

    report_json, report_txt = make_self_check_report(
        output_root=output_root,
        image_shape_counter=image_shape_counter,
        tc_len_counter=tc_len_counter,
        feat_dim_counter=feat_dim_counter,
        label_counter=label_counter,
        calibration_counter=calibration_counter,
        calibration_examples=calibration_examples,
        alignment_ok_subjects=alignment_ok_subjects,
        total_subjects=len(subjects),
        total_ics=total_ics,
        total_pkls=total_pkls,
        errors=errors,
        random_sample_checks=random_sample_checks,
    )

    print("\n=== FINISHED ===", flush=True)
    print(f"Subjects discovered: {len(subjects)}", flush=True)
    print(f"Aligned subjects: {alignment_ok_subjects}", flush=True)
    print(f"Total ICs: {total_ics}", flush=True)
    print(f"Total PKLs: {total_pkls}", flush=True)
    print(f"Mapping CSV: {mapping_file}", flush=True)
    print(f"Self-check JSON: {report_json}", flush=True)
    print(f"Self-check TXT: {report_txt}", flush=True)
    print(f"Errors: {len(errors)}", flush=True)


if __name__ == "__main__":
    main()
