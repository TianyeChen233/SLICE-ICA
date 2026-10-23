#!/usr/bin/env python3
"""
Flatten PSEUDO_PKL_CACHE into ALL_DATASETS-like layout:

  <dst>/<new_subject_id>/ica_<ic:03d>__<session_token>.pkl

Pseudo pickles already carry ic_index / session / subject (HCP numeric). We rewrite
pickle['subject'] to new_subject_id unless --no-rewrite-subject-field.

Mapping options (exactly one):
  1) --map hcp_to_new_subject.csv  (columns: hcp_subject,new_subject_id)
  2) --auto-map-train-order --split-manifest path/to/split_manifest.json
       pairs sorted(HCP ids in cache) with sorted(train_subjects) by numeric sub index.
       WARNING: only use if your preprocessing guarantees this 1:1 ordering.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _safe_session_token(session: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", session.strip()).strip("_")
    return s or "sess"


def load_csv_map(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("empty CSV")
        cols = [c.strip() for c in reader.fieldnames]
        lower = {c.lower(): c for c in cols}

        def pick(*names):
            for n in names:
                if n.lower() in lower:
                    return lower[n.lower()]
            return None

        k1 = pick("hcp_subject", "hcp_subject_id", "hcp_id", "subject_hcp", "s1200")
        k2 = pick("new_subject_id", "sub_id", "new_subject", "subject_new")
        if not k1 or not k2:
            raise ValueError(
                f"CSV must include columns like hcp_subject,new_subject_id; got {reader.fieldnames}"
            )
        for row in reader:
            h = str(row[k1]).strip()
            s = str(row[k2]).strip()
            if not h or not s:
                continue
            out[h] = s
    if not out:
        raise ValueError("mapping CSV produced zero rows")
    return out


def discover_hcp_subjects(src: Path) -> List[str]:
    if not src.is_dir():
        raise FileNotFoundError(str(src))
    subs = [p.name for p in src.iterdir() if p.is_dir()]
    # HCP numeric folders only (digits)
    subs = [s for s in subs if s.isdigit()]
    return sorted(subs, key=lambda x: int(x))


def auto_map_train_order(split_manifest: Path, hcp_subjects: List[str]) -> Dict[str, str]:
    sp = json.loads(split_manifest.read_text(encoding="utf-8"))
    train = list(sp.get("train_subjects") or [])
    train = sorted(train, key=lambda s: int(str(s).replace("sub", "").strip() or "0"))

    if len(train) < len(hcp_subjects):
        raise ValueError(
            f"train_subjects({len(train)}) < pseudo_hcp_subjects({len(hcp_subjects)}); cannot pair"
        )

    # Take the first N train subjects in sorted order to match N pseudo HCP ids.
    chosen = train[: len(hcp_subjects)]
    return {hcp: sub for hcp, sub in zip(hcp_subjects, chosen, strict=True)}


def iter_src_pkls(src: Path) -> Iterable[Tuple[str, str, Path]]:
    for p in sorted(src.rglob("*.pkl")):
        rel = p.relative_to(src)
        parts = rel.parts
        if len(parts) < 3:
            continue
        hcp_sub = parts[0]
        session = parts[1]
        if not hcp_sub.isdigit():
            continue
        yield hcp_sub, session, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--map", type=Path, default=None)
    ap.add_argument("--auto-map-train-order", action="store_true")
    ap.add_argument("--split-manifest", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-rewrite-subject-field", action="store_true")
    args = ap.parse_args()

    if bool(args.map) == bool(args.auto_map_train_order):
        raise SystemExit("choose exactly one of: --map CSV  OR  --auto-map-train-order")

    src: Path = args.src
    dst: Path = args.dst

    if args.map:
        mp = load_csv_map(args.map)
    else:
        if not args.split_manifest:
            raise SystemExit("--auto-map-train-order requires --split-manifest")
        hcp_subjects = discover_hcp_subjects(src)
        mp = auto_map_train_order(args.split_manifest, hcp_subjects)
        print(
            "AUTO_MAP_WARNING: paired sorted pseudo HCP folders with sorted(train_subjects)[:N]; "
            "verify this matches your cohort linkage."
        )
        print(f"pairs_preview={list(mp.items())[:5]} ... total_pairs={len(mp)}")

    n_total = 0
    n_written = 0
    n_skipped = 0
    missing_hcp = set()

    for hcp_sub, session, p in iter_src_pkls(src):
        n_total += 1
        new_sub = mp.get(hcp_sub)
        if new_sub is None:
            missing_hcp.add(hcp_sub)
            n_skipped += 1
            continue

        obj = pickle.loads(p.read_bytes())
        if not isinstance(obj, dict):
            raise TypeError(f"unexpected pickle root type {type(obj)} @ {p}")

        ic = int(obj.get("ic_index", -1))
        if ic < 0:
            stem = p.stem
            if stem.startswith("ica_"):
                try:
                    ic = int(stem.split("_")[-1])
                except Exception:
                    ic = -1
        if ic < 0:
            raise ValueError(f"cannot infer ic_index for {p}")

        if not args.no_rewrite_subject_field:
            obj["subject"] = str(new_sub)

        fname = f"ica_{ic:03d}__{_safe_session_token(session)}.pkl"
        out_path = dst / new_sub / fname
        if args.dry_run:
            print(f"DRY {p} -> {out_path}")
            n_written += 1
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        n_written += 1

    print(
        f"done | scanned={n_total} written={n_written} skipped_unmapped={n_skipped} dst={dst}"
    )
    if missing_hcp:
        print(
            "WARNING: unmapped HCP subjects present in cache (skipped all their pkls):",
            ",".join(sorted(missing_hcp, key=lambda x: int(x))[:50]),
            ("..." if len(missing_hcp) > 50 else ""),
        )
        print(f"unmapped_distinct_subjects={len(missing_hcp)}")


if __name__ == "__main__":
    main()
