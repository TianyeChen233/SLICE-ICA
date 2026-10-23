#!/usr/bin/env python3
import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional


SUBJECT_ZIP_RE = re.compile(r"^(\d+)_Rest3TRecommended\.zip$")


def _extract_meta_from_nifti(nifti_path: Optional[Path]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "TR": None,
        "shape": None,
        "meta_source": None,
        "meta_error": None,
    }
    if nifti_path is None:
        return meta

    try:
        import nibabel as nib  # type: ignore

        img = nib.load(str(nifti_path))
        hdr = img.header
        shape = tuple(int(x) for x in img.shape)
        zooms = hdr.get_zooms()
        tr = float(zooms[3]) if len(zooms) > 3 else None
        meta["TR"] = tr
        meta["shape"] = shape
        meta["meta_source"] = "nibabel"
        return meta
    except Exception as exc:
        meta["meta_error"] = f"nibabel failed: {exc}"

    try:
        import subprocess

        out = subprocess.check_output(
            ["fslhd", str(nifti_path)],
            text=True,
            stderr=subprocess.STDOUT,
        )
        vals: Dict[str, str] = {}
        for line in out.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                vals[parts[0]] = parts[1]
        dims = []
        for key in ("dim1", "dim2", "dim3", "dim4"):
            if key in vals:
                try:
                    dims.append(int(float(vals[key])))
                except Exception:
                    pass
        tr = None
        if "pixdim4" in vals:
            try:
                tr = float(vals["pixdim4"])
            except Exception:
                pass
        meta["TR"] = tr
        meta["shape"] = tuple(dims) if dims else None
        meta["meta_source"] = "fslhd"
    except Exception as exc:
        prev = meta.get("meta_error") or ""
        sep = " | " if prev else ""
        meta["meta_error"] = f"{prev}{sep}fslhd failed: {exc}"

    return meta


def _empty_summary(raw_root: Path) -> Dict[str, Any]:
    return {
        "raw_root": str(raw_root),
        "data_mode": None,
        "total_subjects_found": 0,
        "subjects": {},
        "aggregate": {
            "session_counts": {},
            "subjects_with_any_rest": 0,
            "subjects_missing_rest": [],
            "missing_fmri": 0,
            "missing_movement_regressors": 0,
            "missing_brain_mask": 0,
            "existing_ica_found": 0,
            "subjects_zipped": 0,
            "subjects_unzipped": 0,
            "subjects_with_both_zip_and_folder": 0,
        },
    }


def _scan_unzipped_subject(raw_root: Path, sid: str, summary: Dict[str, Any]) -> None:
    subj_root = raw_root / sid
    results_root = subj_root / "MNINonLinear" / "Results"
    subj_info: Dict[str, Any] = {"sessions": {}, "notes": [], "source": "directory"}

    if not results_root.exists():
        subj_info["notes"].append("Missing MNINonLinear/Results")
        summary["subjects"][sid] = subj_info
        summary["aggregate"]["subjects_missing_rest"].append(sid)
        return

    session_dirs = sorted(
        [p for p in results_root.iterdir() if p.is_dir() and p.name.startswith("rfMRI_REST")]
    )
    if not session_dirs:
        subj_info["notes"].append("No rfMRI_REST* sessions found")
        summary["aggregate"]["subjects_missing_rest"].append(sid)
    else:
        summary["aggregate"]["subjects_with_any_rest"] += 1

    for sdir in session_dirs:
        sname = sdir.name
        fmri_candidates = [sdir / f"{sname}.nii.gz", sdir / "rfMRI.nii.gz"]
        fmri_path = next((p for p in fmri_candidates if p.exists()), None)
        if fmri_path is None:
            gz_files = sorted(sdir.glob("*.nii.gz"))
            fmri_path = gz_files[0] if gz_files else None

        movement = sdir / "Movement_Regressors.txt"
        brain_mask_candidates = [
            sdir / "brainmask_fs.2.nii.gz",
            sdir / "brainmask_fs.nii.gz",
            sdir / "brainmask.nii.gz",
            sdir / "mask.nii.gz",
        ]
        brain_mask = next((p for p in brain_mask_candidates if p.exists()), None)

        existing_ica_candidates = [
            sdir / "ICA" / "melodic_IC.nii.gz",
            sdir / "melodic.ica" / "melodic_IC.nii.gz",
            sdir / "filtered_func_data.ica" / "melodic_IC.nii.gz",
        ]
        existing_ica = next((p for p in existing_ica_candidates if p.exists()), None)

        if fmri_path is None:
            summary["aggregate"]["missing_fmri"] += 1
        if not movement.exists():
            summary["aggregate"]["missing_movement_regressors"] += 1
        if brain_mask is None:
            summary["aggregate"]["missing_brain_mask"] += 1
        if existing_ica is not None:
            summary["aggregate"]["existing_ica_found"] += 1

        meta = _extract_meta_from_nifti(fmri_path)
        subj_info["sessions"][sname] = {
            "session_dir": str(sdir),
            "fmri_file": str(fmri_path) if fmri_path else None,
            "TR": meta["TR"],
            "shape": meta["shape"],
            "meta_source": meta["meta_source"],
            "meta_error": meta["meta_error"],
            "movement_regressors": str(movement) if movement.exists() else None,
            "brain_mask": str(brain_mask) if brain_mask else None,
            "existing_ica": str(existing_ica) if existing_ica else None,
        }

        counts = summary["aggregate"]["session_counts"]
        counts[sname] = counts.get(sname, 0) + 1

    summary["subjects"][sid] = subj_info


def _scan_zipped_subject(zip_path: Path, sid: str, summary: Dict[str, Any]) -> None:
    subj_info: Dict[str, Any] = {
        "sessions": {},
        "notes": ["Data is compressed in zip; TR/shape unavailable before extraction"],
        "source": "zip",
        "zip_file": str(zip_path),
    }
    sessions: Dict[str, Dict[str, Any]] = {}
    has_any_rest = False

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for name in names:
            if "MNINonLinear/Results/rfMRI_REST" not in name:
                continue
            parts = name.split("/")
            try:
                idx = parts.index("Results")
            except ValueError:
                continue
            if idx + 1 >= len(parts):
                continue
            sname = parts[idx + 1]
            if not sname.startswith("rfMRI_REST"):
                continue
            has_any_rest = True
            rec = sessions.setdefault(
                sname,
                {
                    "session_dir_in_zip": f"{sid}/MNINonLinear/Results/{sname}",
                    "fmri_file": None,
                    "TR": None,
                    "shape": None,
                    "meta_source": "zip_listing_only",
                    "meta_error": "Need extraction for NIfTI header parsing",
                    "movement_regressors": None,
                    "brain_mask": None,
                    "existing_ica": None,
                },
            )
            lower = name.lower()
            if lower.endswith(".nii.gz") and rec["fmri_file"] is None:
                rec["fmri_file"] = f"{zip_path}::{name}"
            if name.endswith("Movement_Regressors.txt"):
                rec["movement_regressors"] = f"{zip_path}::{name}"
            if (
                lower.endswith("brainmask_fs.2.nii.gz")
                or lower.endswith("brainmask_fs.nii.gz")
                or lower.endswith("brainmask.nii.gz")
                or lower.endswith("mask.nii.gz")
            ):
                rec["brain_mask"] = f"{zip_path}::{name}"
            if lower.endswith("ica/melodic_ic.nii.gz") or lower.endswith("melodic.ica/melodic_ic.nii.gz"):
                rec["existing_ica"] = f"{zip_path}::{name}"

    if not has_any_rest:
        summary["aggregate"]["subjects_missing_rest"].append(sid)
        subj_info["notes"].append("No rfMRI_REST* sessions found in zip")
    else:
        summary["aggregate"]["subjects_with_any_rest"] += 1

    for sname, rec in sessions.items():
        if rec["fmri_file"] is None:
            summary["aggregate"]["missing_fmri"] += 1
        if rec["movement_regressors"] is None:
            summary["aggregate"]["missing_movement_regressors"] += 1
        if rec["brain_mask"] is None:
            summary["aggregate"]["missing_brain_mask"] += 1
        if rec["existing_ica"] is not None:
            summary["aggregate"]["existing_ica_found"] += 1
        counts = summary["aggregate"]["session_counts"]
        counts[sname] = counts.get(sname, 0) + 1
        subj_info["sessions"][sname] = rec

    summary["subjects"][sid] = subj_info


def build_summary(raw_root: Path) -> Dict[str, Any]:
    summary = _empty_summary(raw_root)
    entries = list(raw_root.iterdir())
    unzipped_subjects = sorted([p.name for p in entries if p.is_dir() and p.name.isdigit()])
    zipped_subjects: Dict[str, Path] = {}
    for p in entries:
        if not p.is_file():
            continue
        m = SUBJECT_ZIP_RE.match(p.name)
        if m:
            zipped_subjects[m.group(1)] = p

    all_subject_ids = sorted(set(unzipped_subjects) | set(zipped_subjects.keys()))
    summary["total_subjects_found"] = len(all_subject_ids)
    summary["aggregate"]["subjects_unzipped"] = len(unzipped_subjects)
    summary["aggregate"]["subjects_zipped"] = len(zipped_subjects)
    summary["aggregate"]["subjects_with_both_zip_and_folder"] = len(
        set(unzipped_subjects) & set(zipped_subjects.keys())
    )
    if unzipped_subjects and zipped_subjects:
        summary["data_mode"] = "mixed"
    elif unzipped_subjects:
        summary["data_mode"] = "unzipped"
    elif zipped_subjects:
        summary["data_mode"] = "zipped"
    else:
        summary["data_mode"] = "empty_or_unrecognized"

    for sid in all_subject_ids:
        if sid in unzipped_subjects:
            _scan_unzipped_subject(raw_root, sid, summary)
        else:
            _scan_zipped_subject(zipped_subjects[sid], sid, summary)

    return summary


def print_preview(summary: Dict[str, Any], sample_subjects: int) -> None:
    agg = summary["aggregate"]
    print("PHASE1_PARSE_DONE")
    print(f"DATA_MODE={summary['data_mode']}")
    print(f"TOTAL_SUBJECTS={summary['total_subjects_found']}")
    print(f"SUBJECTS_WITH_REST={agg['subjects_with_any_rest']}")
    print(f"SUBJECTS_MISSING_REST={len(agg['subjects_missing_rest'])}")
    print(f"MISSING_FMRI={agg['missing_fmri']}")
    print(f"MISSING_MOVEMENT_REGRESSORS={agg['missing_movement_regressors']}")
    print(f"MISSING_BRAIN_MASK={agg['missing_brain_mask']}")
    print(f"EXISTING_ICA_FOUND={agg['existing_ica_found']}")
    print(f"SUBJECTS_ZIPPED={agg['subjects_zipped']}")
    print(f"SUBJECTS_UNZIPPED={agg['subjects_unzipped']}")
    print("SESSION_COUNTS=" + json.dumps(agg["session_counts"], sort_keys=True))

    shown = 0
    for sid in sorted(summary["subjects"].keys()):
        sessions = summary["subjects"][sid].get("sessions", {})
        if not sessions:
            continue
        print("")
        print(f"Subject: {sid}")
        print(f"  Source: {summary['subjects'][sid].get('source')}")
        print("  Sessions:")
        for sname in sorted(sessions.keys()):
            sinfo = sessions[sname]
            mv = "YES" if sinfo.get("movement_regressors") else "NO"
            bm = "YES" if sinfo.get("brain_mask") else "NO"
            ex = "YES" if sinfo.get("existing_ica") else "NO"
            print(f"    {sname}:")
            print(f"      file: {sinfo.get('fmri_file')}")
            print(f"      TR: {sinfo.get('TR')}")
            print(f"      shape: {sinfo.get('shape')}")
            print(f"      Movement_Regressors: {mv}")
            print(f"      brain_mask: {bm}")
            print(f"      existing_ica: {ex}")
        shown += 1
        if shown >= sample_subjects:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse HCP rfMRI_REST structure")
    parser.add_argument("--raw-root", required=True, help="Absolute path to raw HCP root")
    parser.add_argument("--output-json", required=True, help="Absolute path to summary JSON")
    parser.add_argument("--sample-subjects", type=int, default=3, help="Subjects to print as sample")
    args = parser.parse_args()

    raw_root = Path(args.raw_root).resolve()
    output_json = Path(args.output_json).resolve()
    if not raw_root.exists():
        raise SystemExit(f"RAW root not found: {raw_root}")

    summary = build_summary(raw_root)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print_preview(summary, args.sample_subjects)
    print(f"SUMMARY_JSON={output_json}")


if __name__ == "__main__":
    main()
