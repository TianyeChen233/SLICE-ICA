#!/usr/bin/env python3
"""Generate an artificial two-component annotation fixture, never participant data."""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    session = args.output / "synthetic_subject" / "synthetic_session" / "filtered_func_data.ica"
    target = session / "melodic_IC.nii.gz"
    if target.exists():
        raise SystemExit("Fixture exists; choose a new output directory")
    session.mkdir(parents=True, exist_ok=True)
    grid = np.indices((24, 24, 24), dtype=float)
    sphere = np.exp(-sum((grid[d] - 12) ** 2 for d in range(3)) / 32)
    rng = np.random.default_rng(42)
    stripe = np.sin(grid[0] * np.pi / 2) * sphere + rng.normal(0, 0.03, sphere.shape)
    maps = np.stack((sphere, stripe), axis=-1).astype(np.float32)
    nib.save(nib.Nifti1Image(maps, np.diag([2, 2, 2, 1])), str(target))
    print(f"Created SYNTHETIC fixture: {target}")


if __name__ == "__main__":
    main()
