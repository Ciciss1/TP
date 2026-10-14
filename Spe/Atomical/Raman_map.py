"""
Optical analogues vs T. An atom is defective when it belongs to a non-hexagonal ring (what activates
the Raman D peak):
  - I_D/I_G = fraction of defective atoms, and its map smoothed over a "laser spot" of width sigma
  - types of defect clusters (connected 5/7 rings): isolated disclination / dipole / chain
  - local shear strain (anisotropy of the bond tensor, analogue of the G+/G- splitting), RMS over all
    atoms and over the defective ones, and its map

Usage:
    python Raman_map.py <folder> [--sigma 25] [--grid 128]
"""
import argparse
from pathlib import Path

import numpy as np

import Properties as P


def analyze(path, maps, sigma, n_grid):
    xyz, bonds, Lx, Ly = P.load_crystal(path)
    sizes, ring_atoms = P.rings(xyz, bonds, Lx, Ly)
    defect = P.atoms_of(len(xyz), ring_atoms, sizes != 6)
    _, _, component, _ = P.defect_graph(sizes, *P.ring_pairs(sizes, ring_atoms))
    cluster_sizes = np.unique(component[component >= 0], return_counts=True)[1]
    shear = P.shear_strain(xyz, bonds, Lx, Ly)

    mean = lambda x: x.mean() if x.size else np.nan              # NaN when there is no defect
    res = dict(
        id_ig=defect.mean(),
        f_isolated=mean(cluster_sizes == 1), f_dipole=mean(cluster_sizes == 2),
        f_chain=mean(cluster_sizes >= 3), mean_size=mean(cluster_sizes),
        shear_rms=np.sqrt(mean(shear ** 2)), shear_rms_defects=np.sqrt(mean(shear[defect] ** 2)),
    )
    if maps:
        xy = xyz[:, :2]
        res.update(box=(Lx, Ly),
                   id_ig_map=P.smooth_map(xy, Lx, Ly, defect, n_grid, sigma) / P.smooth_map(xy, Lx, Ly, None, n_grid, sigma),
                   shear_map=P.smooth_map(xy, Lx, Ly, shear, n_grid, sigma))
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
        (r"$I_D/I_G$", r"$I_D/I_G$", [("id_ig", "")]),
        ("Defect clusters", "fraction of clusters", [("f_isolated", "isolated"), ("f_dipole", "dipole"), ("f_chain", "chain")]),
        ("Local shear strain", r"$\varepsilon_{dev}$ (RMS)", [("shear_rms", "all atoms"), ("shear_rms_defects", "defects")]),
    ], folder / "raman_vs_T")
    P.save_vs_T(results, ["id_ig", "f_isolated", "f_dipole", "f_chain", "mean_size", "shear_rms", "shear_rms_defects"],
                folder / "raman_data.npz")
    P.plot_maps(results, "id_ig_map", folder / "raman_maps", label=r"$I_D/I_G$")
    P.plot_maps(results, "shear_map", folder / "shear_maps", cmap="magma", label=r"$\varepsilon_{dev}$")
