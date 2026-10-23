#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str, log_file: Path) -> None:
    line = f"[{now_ts()}] {msg}"
    print(line, flush=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_cmd(
    cmd: List[str],
    log_file: Path,
    check: bool = True,
    timeout_sec: Optional[float] = None,
) -> subprocess.CompletedProcess:
    """Run a subprocess; optional hard wall-clock cap via GNU timeout(1) when available."""
    cmd_str = " ".join(shlex.quote(x) for x in cmd)
    exec_cmd = list(cmd)
    used_timeout_bin = False
    if timeout_sec is not None and timeout_sec > 0:
        timeout_bin = shutil.which("timeout")
        if timeout_bin:
            grace = min(120, max(10, int(timeout_sec * 0.02)))
            exec_cmd = [timeout_bin, "-k", str(grace), str(int(timeout_sec))] + exec_cmd
            used_timeout_bin = True
            log(f"RUN [hard timeout {int(timeout_sec)}s]: {cmd_str}", log_file)
        else:
            log(f"RUN [subprocess timeout {timeout_sec}s; install coreutils timeout(1) for stronger kills]: {cmd_str}", log_file)
    else:
        log(f"RUN: {cmd_str}", log_file)
    sub_timeout = None if used_timeout_bin else (float(timeout_sec) if timeout_sec else None)
    try:
        proc = subprocess.run(exec_cmd, capture_output=True, text=True, timeout=sub_timeout)
    except subprocess.TimeoutExpired as exc:
        log(f"TIMEOUT after {timeout_sec}s: {cmd_str}", log_file)
        raise RuntimeError(f"Command timed out after {timeout_sec}s: {cmd_str}") from exc
    if proc.stdout:
        with log_file.open("a", encoding="utf-8") as f:
            f.write(proc.stdout + ("\n" if not proc.stdout.endswith("\n") else ""))
    if proc.stderr:
        with log_file.open("a", encoding="utf-8") as f:
            f.write(proc.stderr + ("\n" if not proc.stderr.endswith("\n") else ""))
    if check and proc.returncode != 0:
        if proc.returncode == 124:
            raise RuntimeError(f"Command timed out (exit 124 from timeout(1)): {cmd_str}")
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd_str}")
    return proc


def build_conda_cmd(conda_bin: str, conda_env: str, executable_and_args: List[str]) -> List[str]:
    if "/" in conda_env:
        prefix = conda_env
    else:
        base = str(Path(conda_bin).resolve().parent.parent)
        prefix = str(Path(base) / "envs" / conda_env)
    cmd_part = " ".join(shlex.quote(x) for x in executable_and_args)
    prep = (
        f"export FSLDIR={shlex.quote(prefix)}; "
        "source \"$FSLDIR/etc/fslconf/fsl.sh\"; "
        "export PATH=\"$FSLDIR/bin:$PATH\"; "
        "export FSLOUTPUTTYPE=NIFTI_GZ; "
    )
    inner = prep + cmd_part
    return [conda_bin, "run", "-n", conda_env, "bash", "-lc", inner]


def fslhd_vals(
    conda_bin: str,
    conda_env: str,
    nifti_file: Path,
    log_file: Path,
    timeout_sec: Optional[float] = 600.0,
) -> Dict[str, str]:
    cmd = build_conda_cmd(conda_bin, conda_env, ["fslhd", str(nifti_file)])
    proc = run_cmd(cmd, log_file, check=True, timeout_sec=timeout_sec)
    vals: Dict[str, str] = {}
    for line in proc.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            vals[parts[0]] = parts[1]
    return vals


def locate_zip(raw_root: Path, subject: str) -> Path:
    zip_path = raw_root / f"{subject}_Rest3TRecommended.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")
    return zip_path


def choose_fmri_member(names: List[str], subject: str, session: str) -> str:
    prefix = f"{subject}/MNINonLinear/Results/{session}/"
    cands = [n for n in names if n.startswith(prefix) and n.endswith(".nii.gz")]
    if not cands:
        raise FileNotFoundError(f"No NIfTI found in {prefix}")

    # Exclude obvious non-timeseries files.
    excludes = (
        "brainmask",
        "jacobian",
        "dropouts",
        "mean",
        "sbref",
        "real_reference",
    )
    filtered = [n for n in cands if not any(x in n.lower() for x in excludes)]
    if not filtered:
        filtered = cands

    priority = [
        f"{session}_hp2000_clean_rclean_tclean_vn.nii.gz",
        f"{session}_hp2000_clean.nii.gz",
        f"{session}_PhaseTwo_gdc_dc.nii.gz",
        f"{session}_PhaseOne_gdc_dc.nii.gz",
        f"{session}_gdc_dc.nii.gz",
        f"{session}.nii.gz",
    ]
    lower_map = {Path(n).name.lower(): n for n in filtered}
    for p in priority:
        if p.lower() in lower_map:
            return lower_map[p.lower()]

    # Fallback to alphabetic candidate; later we validate dim4 and may switch.
    return sorted(filtered, key=lambda x: x.lower())[0]


def optional_member(names: List[str], subject: str, session: str, filename: str) -> Optional[str]:
    target = f"{subject}/MNINonLinear/Results/{session}/{filename}"
    return target if target in names else None


def extract_needed_files(
    zip_path: Path,
    subject: str,
    session: str,
    input_dir: Path,
) -> Dict[str, Optional[Path]]:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        fmri_member = choose_fmri_member(names, subject, session)
        move_member = optional_member(names, subject, session, "Movement_Regressors.txt")

        brain_mask_member = None
        for fn in ("brainmask_fs.2.nii.gz", "brainmask_fs.nii.gz", "brainmask.nii.gz", "mask.nii.gz"):
            m = optional_member(names, subject, session, fn)
            if m:
                brain_mask_member = m
                break

        input_dir.mkdir(parents=True, exist_ok=True)
        out_fmri = input_dir / Path(fmri_member).name
        with zf.open(fmri_member) as src, out_fmri.open("wb") as dst:
            shutil.copyfileobj(src, dst)

        out_move = None
        if move_member:
            out_move = input_dir / Path(move_member).name
            with zf.open(move_member) as src, out_move.open("wb") as dst:
                shutil.copyfileobj(src, dst)

        out_mask = None
        if brain_mask_member:
            out_mask = input_dir / Path(brain_mask_member).name
            with zf.open(brain_mask_member) as src, out_mask.open("wb") as dst:
                shutil.copyfileobj(src, dst)

    return {
        "fmri_file": out_fmri,
        "movement_regressors": out_move,
        "brain_mask": out_mask,
    }


def extract_single_member(zip_path: Path, member: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / Path(member).name
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as src, out.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return out


def extract_optional_member(zip_path: Path, member: str, out_path: Path) -> bool:
    with zipfile.ZipFile(zip_path) as zf:
        if member not in zf.namelist():
            return False
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as src, out_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return True


def best_timeseries_from_zip(
    zip_path: Path,
    subject: str,
    session: str,
    input_dir: Path,
    conda_bin: str,
    conda_env: str,
    log_file: Path,
    fslhd_timeout_sec: Optional[float] = 600.0,
) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        prefix = f"{subject}/MNINonLinear/Results/{session}/"
        all_nii = [n for n in zf.namelist() if n.startswith(prefix) and n.endswith(".nii.gz")]
    if not all_nii:
        raise FileNotFoundError(f"No NIfTI files for {subject}/{session} in {zip_path}")

    excludes = ("brainmask", "jacobian", "dropouts", "mean", "sbref", "real_reference")
    candidates = [n for n in all_nii if not any(x in n.lower() for x in excludes)]
    if not candidates:
        candidates = all_nii

    best_member = None
    best_dim4 = -1
    best_file = None
    for member in candidates:
        tmp_file = extract_single_member(zip_path, member, input_dir)
        try:
            vals = fslhd_vals(conda_bin, conda_env, tmp_file, log_file, timeout_sec=fslhd_timeout_sec)
            dim4 = int(float(vals.get("dim4", "0") or 0))
        except Exception:
            dim4 = 0
        log(f"Candidate {tmp_file.name} dim4={dim4}", log_file)
        if dim4 > best_dim4:
            best_dim4 = dim4
            best_member = member
            best_file = tmp_file

    if best_file is None:
        raise RuntimeError(f"Unable to pick timeseries file for {subject}/{session}")
    if best_dim4 < 50:
        raise RuntimeError(f"Best candidate has suspiciously low dim4={best_dim4}: {best_member}")
    return best_file


def read_tr_shape(
    conda_bin: str,
    conda_env: str,
    nifti_file: Path,
    log_file: Path,
    fslhd_timeout_sec: Optional[float] = 600.0,
) -> Tuple[float, Tuple[int, int, int, int]]:
    vals = fslhd_vals(conda_bin, conda_env, nifti_file, log_file, timeout_sec=fslhd_timeout_sec)

    dims = []
    for key in ("dim1", "dim2", "dim3", "dim4"):
        if key not in vals:
            raise RuntimeError(f"Missing {key} in fslhd output for {nifti_file}")
        dims.append(int(float(vals[key])))
    tr = float(vals.get("pixdim4", "0") or 0)
    if tr <= 0:
        raise RuntimeError(f"Invalid TR parsed from fslhd: {tr}")
    return tr, (dims[0], dims[1], dims[2], dims[3])


def find_features_csv(melodic_ica_dir: Path) -> Optional[Path]:
    for p in melodic_ica_dir.rglob("features.csv"):
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _read_motion_txt(path: Path) -> List[List[float]]:
    rows: List[List[float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        vals: List[float] = []
        for p in parts:
            try:
                vals.append(float(p))
            except Exception:
                vals.append(0.0)
        rows.append(vals)
    return rows


def _write_motion_par(path: Path, rows: List[List[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(" ".join(f"{x:.8f}" for x in r) + "\n")


def run_pipeline(args: argparse.Namespace) -> Dict[str, object]:
    t0 = time.time()
    out_session_dir = Path(args.output_root) / args.subject / args.session
    log_dir = Path(args.log_root)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_session_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{args.subject}_{args.session}.log"

    result: Dict[str, object] = {
        "subject": args.subject,
        "session": args.session,
        "status": "failed",
        "attempt": args.attempt,
        "output_dir": str(out_session_dir),
        "log_file": str(log_file),
        "tr": None,
        "shape": None,
        "runtime_sec": None,
        "error": None,
    }

    try:
        if args.resume and (out_session_dir / "features.csv").exists():
            if (out_session_dir / "features.csv").stat().st_size > 0:
                result["status"] = "skipped_completed"
                result["runtime_sec"] = round(time.time() - t0, 2)
                return result

        log(f"=== START subject={args.subject} session={args.session} attempt={args.attempt} ===", log_file)
        zip_path = locate_zip(Path(args.raw_root), args.subject)
        log(f"ZIP: {zip_path}", log_file)

        input_dir = out_session_dir / "input"
        extracted = extract_needed_files(zip_path, args.subject, args.session, input_dir)
        fmri_file = best_timeseries_from_zip(
            zip_path=zip_path,
            subject=args.subject,
            session=args.session,
            input_dir=input_dir,
            conda_bin=args.conda_bin,
            conda_env=args.conda_env,
            log_file=log_file,
            fslhd_timeout_sec=args.fslhd_timeout_sec,
        )
        log(f"Selected fMRI: {fmri_file}", log_file)
        if extracted["movement_regressors"]:
            log(f"Movement regressors: {extracted['movement_regressors']}", log_file)
        else:
            log("Movement regressors: MISSING", log_file)
        if extracted["brain_mask"]:
            log(f"Brain mask: {extracted['brain_mask']}", log_file)
        else:
            log("Brain mask: MISSING", log_file)

        tr, shape = read_tr_shape(
            args.conda_bin,
            args.conda_env,
            fmri_file,
            log_file,
            fslhd_timeout_sec=args.fslhd_timeout_sec,
        )
        result["tr"] = tr
        result["shape"] = shape
        log(f"TR={tr}, shape={shape}", log_file)

        melodic_ica_dir = out_session_dir / "melodic.ica"
        melodic_ok = (
            melodic_ica_dir.exists()
            and (melodic_ica_dir / "melodic_IC.nii.gz").exists()
            and (melodic_ica_dir / "melodic_mix").exists()
        )
        if not melodic_ok:
            if melodic_ica_dir.exists():
                shutil.rmtree(melodic_ica_dir)
            melodic_cmd = [
                "melodic",
                "-i",
                str(fmri_file),
                "-o",
                str(melodic_ica_dir),
                "--nobet",
                "--bgthreshold=10",
                f"--tr={tr}",
            ]
            if args.with_report:
                melodic_cmd.append("--report")
            run_cmd(
                build_conda_cmd(args.conda_bin, args.conda_env, melodic_cmd),
                log_file,
                check=True,
                timeout_sec=args.melodic_timeout_sec,
            )
        else:
            log(f"Reusing existing melodic output at {melodic_ica_dir}", log_file)
        if not (melodic_ica_dir / "melodic_IC.nii.gz").exists():
            raise RuntimeError(f"melodic output invalid: missing {melodic_ica_dir / 'melodic_IC.nii.gz'}")
        # Build a FEAT-like directory tree required by pyfix.
        feat_root = out_session_dir
        (feat_root / "reg").mkdir(exist_ok=True)
        (feat_root / "mc").mkdir(exist_ok=True)
        feat_ica_dir = feat_root / "filtered_func_data.ica"
        feat_ica_dir.mkdir(exist_ok=True)

        shutil.copy2(fmri_file, feat_root / "filtered_func_data.nii.gz")
        if (melodic_ica_dir / "mask.nii.gz").exists():
            shutil.copy2(melodic_ica_dir / "mask.nii.gz", feat_root / "mask.nii.gz")
        if (melodic_ica_dir / "mean.nii.gz").exists():
            shutil.copy2(melodic_ica_dir / "mean.nii.gz", feat_root / "mean_func.nii.gz")

        for src_name, dst_name in (
            ("melodic_mix", "melodic_mix"),
            ("melodic_IC.nii.gz", "melodic_IC.nii.gz"),
            ("mean.nii.gz", "mean.nii.gz"),
            ("mask.nii.gz", "mask.nii.gz"),
        ):
            src = melodic_ica_dir / src_name
            dst = feat_ica_dir / dst_name
            if src.exists():
                shutil.copy2(src, dst)

        # Optional structural file from zip; fallback to mean_func.
        highres = feat_root / "reg" / "highres.nii.gz"
        if not extract_optional_member(
            zip_path,
            f"{args.subject}/MNINonLinear/T1w_restore.nii.gz",
            highres,
        ):
            if (feat_root / "mean_func.nii.gz").exists():
                shutil.copy2(feat_root / "mean_func.nii.gz", highres)
        standard = feat_root / "reg" / "standard.nii.gz"
        if highres.exists() and (not standard.exists()):
            shutil.copy2(highres, standard)

        ident_mat = "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n"
        (feat_root / "reg" / "highres2example_func.mat").write_text(ident_mat, encoding="utf-8")
        (feat_root / "reg" / "standard2example_func.mat").write_text(ident_mat, encoding="utf-8")

        # Motion parameters expected by pyfix (6 columns, MCFLIRT format).
        mc_par = feat_root / "mc" / "prefiltered_func_data_mcf.par"
        ntime = int(shape[3])
        if extracted["movement_regressors"] and Path(extracted["movement_regressors"]).exists():
            try:
                mp = _read_motion_txt(Path(extracted["movement_regressors"]))
                if not mp:
                    raise RuntimeError("empty motion regressors")
                fixed: List[List[float]] = []
                for row in mp:
                    if len(row) >= 6:
                        fixed.append(row[:6])
                    else:
                        fixed.append(row + [0.0] * (6 - len(row)))
                if len(fixed) > ntime:
                    fixed = fixed[:ntime]
                elif len(fixed) < ntime:
                    fixed.extend([[0.0] * 6 for _ in range(ntime - len(fixed))])
                _write_motion_par(mc_par, fixed)
                log(f"Prepared motion params: {mc_par} with rows={len(fixed)}", log_file)
            except Exception as exc:
                log(f"Failed parsing movement regressors ({exc}); writing zero motion params", log_file)
                _write_motion_par(mc_par, [[0.0] * 6 for _ in range(ntime)])
        else:
            _write_motion_par(mc_par, [[0.0] * 6 for _ in range(ntime)])
            log(f"Movement regressors missing; wrote zero motion params to {mc_par}", log_file)

        log(f"Prepared FEAT-like FIX input tree at {feat_root}", log_file)

        run_cmd(
            build_conda_cmd(args.conda_bin, args.conda_env, ["fix", "-f", str(feat_root)]),
            log_file,
            check=True,
            timeout_sec=args.fix_timeout_sec,
        )

        features = find_features_csv(feat_root)
        if features is None:
            raise RuntimeError("features.csv not found after fix -f")
        final_features = out_session_dir / "features.csv"
        shutil.copy2(features, final_features)
        if final_features.stat().st_size <= 0:
            raise RuntimeError("features.csv is empty")
        log(f"features.csv: {final_features} ({final_features.stat().st_size} bytes)", log_file)

        result["status"] = "success"
        result["features_csv"] = str(final_features)
        result["melodic_ica"] = str(melodic_ica_dir)
        result["runtime_sec"] = round(time.time() - t0, 2)
        log(f"=== DONE success runtime={result['runtime_sec']} sec ===", log_file)
    except Exception as exc:
        result["error"] = str(exc)
        result["runtime_sec"] = round(time.time() - t0, 2)
        log(f"ERROR: {exc}", log_file)
        log(f"=== DONE failed runtime={result['runtime_sec']} sec ===", log_file)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MELODIC + FIX feature extraction for one subject/session")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--conda-bin", default="local_data/miniconda3/bin/conda")
    parser.add_argument("--conda-env", default="fsl_melodic")
    parser.add_argument("--with-report", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--result-json", default=None)
    parser.add_argument(
        "--fslhd-timeout-sec",
        type=float,
        default=600.0,
        help="Hard timeout (seconds) for each fslhd invocation (default 600).",
    )
    parser.add_argument(
        "--melodic-timeout-sec",
        type=float,
        default=21600.0,
        help="Hard timeout (seconds) for MELODIC (default 21600 = 6h).",
    )
    parser.add_argument(
        "--fix-timeout-sec",
        type=float,
        default=10800.0,
        help="Hard timeout (seconds) for fix -f (default 10800 = 3h).",
    )
    args = parser.parse_args()

    res = run_pipeline(args)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.result_json:
        out = Path(args.result_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    if res.get("status") != "success" and res.get("status") != "skipped_completed":
        sys.exit(1)


if __name__ == "__main__":
    main()
