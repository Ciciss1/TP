"""
Geometry of the topological defects (5/7 rings) vs T, motivated by KTHNY:
  - separation of the isolated 5-7 dipoles (bound dislocations), expected to grow towards the
    solid -> hexatic unbinding; histogram pooled over k for each T
  - spatial correlation C(r) of the sizes of the defect clusters (are the long chains themselves
    grouped?), averaged over k with the number of pairs per bin as weight
  - enrichment of free disclinations near the triple junctions (branching points of the 5/7 chains)

Usage:
    python Dislocation_geometry.py <folder> [--r-bins 20] [--r-max R]
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import Properties as P


def analyze(path, maps, r_bins):
    xyz, bonds, Lx, Ly = P.load_crystal(path)
    sizes, ring_atoms = P.rings(xyz, bonds, Lx, Ly)
    is57, paired, component, degree = P.defect_graph(sizes, *P.ring_pairs(sizes, ring_atoms))
    centroids = P.ring_centroids(sizes, ring_atoms, xyz[:, :2], Lx, Ly)

    separations = P.dipole_separations(sizes, component, centroids, Lx, Ly)
    centers, cluster_sizes = P.group_mean(component[is57], centroids[is57], Lx, Ly)
    C, n_pairs = P.size_correlation(centers, cluster_sizes, Lx, Ly, r_bins)
    junctions, free = is57 & (degree >= 3), is57 & ~paired
    return dict(
        n_dipoles=len(separations), mean_sep=separations.mean() if len(separations) else np.nan,
        n_junctions=int(junctions.sum()), n_free=int(free.sum()),
        enrichment=P.junction_enrichment(centroids[junctions], centroids[free], Lx, Ly),
        separations=separations, C=C, n_pairs=n_pairs,
    )


def plot_per_T(results, r_bins, folder):
    '''
    Pooled histogram of the dipole separations and pair-weighted average of C(r), for each T
    '''
    folder.mkdir(exist_ok=True)
    r = 0.5 * (r_bins[:-1] + r_bins[1:])
    out = {"r": r}
    for T, runs in results.items():
        sep = np.concatenate([x["separations"] for x in runs])
        w = np.sum([x["n_pairs"] for x in runs], axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.nansum([x["C"] * x["n_pairs"] for x in runs], axis=0) / w
        out[f"C_T={T}"], out[f"separations_T={T}"] = C, sep

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        if len(sep):
            ax1.hist(sep, bins=30, color="tab:purple", alpha=0.8)
            ax1.axvline(sep.mean(), color="k", ls="--", label=f"mean = {sep.mean():.2f} " + r"$\mathrm{\AA}$")
            ax1.legend()
        ax1.set(xlabel=r"dipole separation ($\mathrm{\AA}$)", ylabel="count", title=f"T = {T} K ({len(sep)} dipoles)")
        ax2.plot(r, C, "o-", color="teal")
        ax2.axhline(0, color="k", lw=0.8, ls=":")
        ax2.set(xlabel=r"$r$ ($\mathrm{\AA}$)", ylabel=r"$C(r)$", title="Cluster-size correlation")
        for ax in (ax1, ax2):
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(folder / f"T={T}K.png", dpi=150)
        plt.close(fig)
    np.savez(folder.parent / "dislocation_geometry_per_T.npz", **out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--r-bins", type=int, default=20)
    ap.add_argument("--r-max", type=float, default=None, help="default: half the box")
    args = ap.parse_args()
    folder = Path(args.folder)

    _, _, first = P.crystal_files(folder)[0]
    r_max = args.r_max or min(P.load_crystal(first)[2:]) / 2
    r_bins = np.linspace(0, r_max, args.r_bins + 1)

    results = P.run_folder(folder, analyze, r_bins=r_bins)
    P.plot_vs_T(results, [
        ("Isolated 5-7 dipoles", r"$\langle r_{5-7} \rangle$ ($\mathrm{\AA}$)", [("mean_sep", "")]),
        ("Isolated 5-7 dipoles", "dipoles per crystal", [("n_dipoles", "")]),
        ("Free disclinations near triple junctions", "enrichment", [("enrichment", "")]),
    ], folder / "dislocation_geometry_vs_T")
    P.save_vs_T(results, ["mean_sep", "n_dipoles", "n_junctions", "n_free", "enrichment"], folder / "dislocation_geometry_data.npz")
    plot_per_T(results, r_bins, folder / "dislocation_geometry")
