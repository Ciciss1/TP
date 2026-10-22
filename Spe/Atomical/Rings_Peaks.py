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
from scipy.ndimage import gaussian_filter

import Properties as P


def analyze(path, maps):
    xyz, bonds, Lx, Ly = P.load_crystal(path)
    sizes, _ = P.rings(xyz, bonds, Lx, Ly)
    peak = P.first_peak(xyz[:, :2], Lx, Ly)
    res = {f"ring_{s}": 100 * np.mean(sizes == s) for s in (5, 6, 7)}
    res.update(m6=peak["m6"], radial_width=peak["radial_width"])
    if maps:
        res["diffraction"] = (*P.diffraction_map(xyz[:, :2], Lx, Ly, 1.4 * peak["q0"]), peak["q0"])
    return res


def plot_diffraction(results, folder, q_cut = 0.5, blur = 0.02, sat = 99.0):
    '''
    First diffraction ring of the first k of each temperature, linear scale, TEM-pattern style:
    the central peak (|q| < q_cut q0) is removed, the map is blurred by a gaussian of width blur*q0
    (spots become visible blobs instead of single pixels) and the colour scale saturates at the
    sat-th percentile of the intensity on the first ring (|q - q0| < 0.15 q0), so the Bragg spots
    burn white while a diffuse ring stays visible
    '''
    folder.mkdir(exist_ok=True)
    for T, r in results.items():
        S, qx, qy, q0 = r[0]["diffraction"]
        S = gaussian_filter(S.T, blur * q0 / (qx[1] - qx[0]))
        q = np.hypot(*np.meshgrid(qx, qy))
        S[q < q_cut * q0] = 0
        vmax = np.percentile(S[np.abs(q - q0) < 0.15 * q0], sat)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(S, origin="lower", extent=[qx[0], qx[-1], qy[0], qy[-1]], cmap="afmhot",
                  vmin=0, vmax=vmax, interpolation="bilinear")
        ax.set_axis_off()
        ax.text(0.03, 0.97, f"T = {T} K", color="w", transform=ax.transAxes, va="top")
        fig.subplots_adjust(0, 0, 1, 1)
        fig.savefig(folder / f"T={T}K.png", dpi=150, facecolor="k")
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