"""
Tight-binding LDOS (KPM, GPU when available) near the Dirac point vs T, for three classes of atoms
probed in a single run: grains (atoms of hexagons only), bound dislocations (atoms of a 5 or 7 ring
next to one of the other kind) and free disclinations (atoms of an unpaired 5 or 7 ring).
Zero-energy weight (|E| < e_window) vs T, and spectra of a few temperatures.

Usage:
    python LDOS_map.py <folder> [--n-probes 128] [--n-moments 512] [--e-window 0.15]
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import Properties as P

CLASSES = {"grain": "grains", "paired": "dislocations (5-7)", "free": "free disclinations"}
E_GRID = np.linspace(-1.0, 1.0, 400)


def analyze(path, maps, n_probes, n_moments, e_window, seed = 0):
    xyz, bonds, Lx, Ly = P.load_crystal(path)
    sizes, ring_atoms = P.rings(xyz, bonds, Lx, Ly)
    is57, paired, _, _ = P.defect_graph(sizes, *P.ring_pairs(sizes, ring_atoms))
    N = len(xyz)
    masks = {"grain": ~P.atoms_of(N, ring_atoms, sizes != 6),
             "paired": P.atoms_of(N, ring_atoms, is57 & paired),
             "free": P.atoms_of(N, ring_atoms, is57 & ~paired)}

    rng = np.random.default_rng(seed)
    probes = {c: rng.choice(np.where(m)[0], min(n_probes, m.sum()), replace=False) for c, m in masks.items()}
    ldos, weight = P.kpm_ldos(xyz, bonds, Lx, Ly, np.concatenate(list(probes.values())), E_GRID, n_moments, e_window)

    res, start = {}, 0
    for c, p in probes.items():
        sl = slice(start, start + len(p))
        start += len(p)
        res[f"w_{c}"] = weight[sl].mean() if len(p) else np.nan
        res[f"n_{c}"] = int(masks[c].sum())
        if maps:
            res[f"spectrum_{c}"] = ldos[sl].mean(axis=0) if len(p) else np.full(len(E_GRID), np.nan)
    return res


def plot_spectra(results, path, n_show = 6):
    T = list(results)[::max(1, len(results) // n_show)]
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(6 * len(CLASSES), 4.5), sharey=True)
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(T)))
    for ax, (c, label) in zip(axes, CLASSES.items()):
        for t, col in zip(T, colors):
            ax.plot(E_GRID, results[t][0][f"spectrum_{c}"], color=col, label=f"T = {t} K")
        ax.set(xlabel="E (eV)", title=label)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("LDOS (1/eV)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{path}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--n-probes", type=int, default=128, help="probed atoms per class")
    ap.add_argument("--n-moments", type=int, default=512, help="Chebyshev moments (energy resolution)")
    ap.add_argument("--e-window", type=float, default=0.15, help="half-width of the zero-energy window (eV)")
    args = ap.parse_args()
    folder = Path(args.folder)

    results = P.run_folder(folder, analyze, n_probes=args.n_probes, n_moments=args.n_moments, e_window=args.e_window)
    P.plot_vs_T(results, [(f"LDOS weight, |E| < {args.e_window} eV", "weight", [(f"w_{c}", l) for c, l in CLASSES.items()])],
                folder / "ldos_vs_T")
    P.save_vs_T(results, [f"w_{c}" for c in CLASSES] + [f"n_{c}" for c in CLASSES], folder / "ldos_data.npz")
    plot_spectra(results, folder / "ldos_spectra")
