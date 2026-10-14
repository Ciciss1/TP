"""
Ring statistics and first diffraction ring vs T: fractions of 5/6/7 rings, |m6| (azimuthal
broadening, orientational order) and radial width (translational order), plus the diffraction image
of the first ring for one crystal per temperature.

Usage:
    python Rings_Peaks.py <folder with the T=<T>K_k=<k>.npz>
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import Properties as P


def analyze(path, maps):
    xyz, bonds, Lx, Ly = P.load_crystal(path)
    sizes, _ = P.rings(xyz, bonds, Lx, Ly)
    peak = P.first_peak(xyz[:, :2], Lx, Ly)
    res = {f"ring_{s}": 100 * np.mean(sizes == s) for s in (5, 6, 7)}
    res.update(m6=peak["m6"], radial_width=peak["radial_width"])
    if maps:
        res["diffraction"] = P.diffraction_map(xyz[:, :2], Lx, Ly, 1.4 * peak["q0"])
    return res


def plot_diffraction(results, folder):
    '''
    First diffraction ring of the first k of each temperature (log scale)
    '''
    folder.mkdir(exist_ok=True)
    for T, r in results.items():
        S, qx, qy = r[0]["diffraction"]
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(np.log10(S.T + 1), origin="lower", extent=[qx[0], qx[-1], qy[0], qy[-1]], cmap="inferno")
        ax.set(xlabel=r"$q_x$ ($\mathrm{\AA}^{-1}$)", ylabel=r"$q_y$ ($\mathrm{\AA}^{-1}$)", title=f"T = {T} K (k = {r[0]['k']})")
        fig.colorbar(im, ax=ax, label=r"$\log_{10}(S(q) + 1)$")
        fig.tight_layout()
        fig.savefig(folder / f"T={T}K.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    results = P.run_folder(folder, analyze)
    P.plot_vs_T(results, [
        ("Ring sizes", "% of rings", [("ring_5", "5"), ("ring_6", "6"), ("ring_7", "7")]),
        ("Orientational order (azimuthal width)", r"$|m_6|$", [("m6", "")]),
        ("Translational order (radial width)", r"radial width ($\mathrm{\AA}^{-1}$)", [("radial_width", "")]),
    ], folder / "rings_peaks_vs_T")
    P.save_vs_T(results, ["ring_5", "ring_6", "ring_7", "m6", "radial_width"], folder / "rings_peaks_data.npz")
    plot_diffraction(results, folder / "diffraction")
