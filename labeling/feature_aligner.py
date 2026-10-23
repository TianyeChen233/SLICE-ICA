#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


def read_canonical_header(unlabeled_root: Path):
    sample = next(unlabeled_root.glob("*/*/features.csv"), None)
    if not sample:
        raise FileNotFoundError("No unlabeled features.csv found")
    with sample.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        hdr = next(csv.reader(f), [])
    if not hdr:
        raise RuntimeError(f"Empty header in {sample}")
    return hdr


def parse_hand_labels_noise(path: Path):
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    labels = {}
    raw_rows = []
    for ln in lines[1:]:
        if not re.match(r"^\d+\s*,", ln):
            continue
        parts = [x.strip() for x in ln.split(",")]
        ic = int(parts[0])
        cls = parts[1] if len(parts) > 1 else "Unknown"
        flag = parts[2].lower() if len(parts) > 2 else ""
        is_noise = ("noise" in cls.lower()) or (flag == "true")
        labels[ic] = 0 if is_noise else 1
        raw_rows.append((ic, cls, is_noise))
    return labels, raw_rows


def align_labeled(labeled_root: Path, canonical_header, output_csv: Path, output_json: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    files = 0
    stats = {"partial_files": 0, "full_files": 0}
    fields = ["dataset", "group_id", "ic_index", "label", "is_labeled"] + canonical_header

    with output_csv.open("w", encoding="utf-8", newline="") as fw:
        w = csv.DictWriter(fw, fieldnames=fields)
        w.writeheader()
        for fpath in sorted(labeled_root.glob("**/fix/features.csv")):
            group_id = str(fpath.parent.parent.relative_to(labeled_root))
            label_path = fpath.parent.parent / "hand_labels_noise.txt"
            if not label_path.exists():
                continue
            labels, _ = parse_hand_labels_noise(label_path)
            data_lines = [ln.strip() for ln in fpath.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
            files += 1
            if len(labels) == len(data_lines):
                stats["full_files"] += 1
            else:
                stats["partial_files"] += 1

            for i, ln in enumerate(data_lines, start=1):
                vals = [x.strip() for x in ln.split(",")]
                if len(vals) != len(canonical_header):
                    continue
                rec = {
                    "dataset": "labeled",
                    "group_id": group_id,
                    "ic_index": i,
                    "label": labels.get(i, ""),
                    "is_labeled": 1 if i in labels else 0,
                }
                for j, k in enumerate(canonical_header):
                    rec[k] = vals[j]
                w.writerow(rec)
                rows_written += 1

    info = {
        "files_processed": files,
        "rows_written": rows_written,
        **stats,
        "output_csv": str(output_csv),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def main():
    ap = argparse.ArgumentParser(description="Align labeled FIX features to unlabeled canonical columns")
    ap.add_argument("--labeled-root", required=True)
    ap.add_argument("--unlabeled-root", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    header = read_canonical_header(Path(args.unlabeled_root))
    info = align_labeled(
        labeled_root=Path(args.labeled_root),
        canonical_header=header,
        output_csv=Path(args.output_csv),
        output_json=Path(args.output_json),
    )
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
