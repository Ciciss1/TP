"""
Out-of-plane corrugation vs T: RMS of z (centred) over all atoms, over the defective atoms (grain
boundaries) and over the others, and the smoothed map of z.

Usage:
    python Corrugation.py <folder> [--sigma 25] [--grid 128]
"""
import argparse
from pathlib import Path

import numpy as np

import Properties as P


def analyze(path, maps, sigma, n_grid):
    xyz, bonds, Lx, Ly = P.load_crystal(path)
    sizes, ring_atoms = P.rings(xyz, bonds, Lx, Ly)
    defect = P.atoms_of(len(xyz), ring_atoms, sizes != 6)
    z = xyz[:, 2] - xyz[:, 2].mean()
    rms = lambda m: np.sqrt(np.mean(z[m] ** 2)) if m.any() else np.nan     # NaN when there is no defect
    res = dict(rms_global=np.sqrt(np.mean(z ** 2)), rms_bulk=rms(~defect), rms_defects=rms(defect))
    if maps:
        res.update(box=(Lx, Ly), z_map=P.smooth_map(xyz[:, :2], Lx, Ly, z, n_grid, sigma))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--sigma", type=float, default=25.0, help="width of the smoothing kernel (AA), maps only")
    ap.add_argument("--grid", type=int, default=128)
    args = ap.parse_args()
    folder = Path(args.folder)

    results = P.run_folder(folder, analyze, sigma=args.sigma, n_grid=args.grid)
    P.plot_vs_T(results, [
        ("Corrugation", r"RMS$(z)$ ($\mathrm{\AA}$)", [("rms_global", "")]),
        ("Grains vs grain boundaries", r"RMS$(z)$ ($\mathrm{\AA}$)", [("rms_bulk", "grains"), ("rms_defects", "defects")]),
    ], folder / "corrugation_vs_T")
    P.save_vs_T(results, ["rms_global", "rms_bulk", "rms_defects"], folder / "corrugation_data.npz")
    P.plot_maps(results, "z_map", folder / "corrugation_maps", cmap="RdBu_r", symmetric=True, label=r"$z$ ($\mathrm{\AA}$)")
