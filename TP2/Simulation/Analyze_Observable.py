"""
Orientational and translational order of the crystals of a folder (T=<T>K_k=<k>.npz) vs T:
  - G6(r) and GT(r) averaged over k, with the KTHNY bounds r^-1/4 and r^-1/3, against r/a with a the
    graphene lattice parameter (from the atom density of the relaxed crystals)
  - their power-law exponents eta6 and etaT, and the global order |<psi6>|
  - a map of the local lattice orientation arg(psi6)/6 for one crystal per T

Usage:
    python Analyze_Observable.py <folder> [--bins 60] [--fit-min 10]
"""
import re
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import Observables as obs

FNAME = re.compile(r"T=(\d+)K_k=(\d+)\.npz$")


def plot_correlations(T, r, C, bound, ylabel, a, path):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(T)))
    for t, c, col in zip(T, C, colors):
        m = np.isfinite(c)                                   # bins with no pair (small r, grid of 0.5 AA)
        ax.loglog(r[m] / a, c[m], color=col, label=f"T = {t} K")
    ax.loglog(r / a, (r / r[0]) ** (-bound), "k--", label=rf"$r^{{-{bound:.3g}}}$")
    ax.set(xlabel=rf"$r/a$   ($a$ = {a:.3f} $\mathrm{{\AA}}$)", ylabel=ylabel)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_orientation(xy, psi, Lx, Ly, title, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    s = 0.8 * (5.5 * 72) ** 2 / len(xy)                          # marker area so that the atoms tile the axes
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=s, marker="s", c=np.degrees(np.angle(psi)) / 6, cmap="hsv", vmin=-30, vmax=30, lw=0, rasterized=True)
    ax.set(xlim=(0, Lx), ylim=(0, Ly), aspect="equal", xlabel=r"$x$ ($\mathrm{\AA}$)", ylabel=r"$y$ ($\mathrm{\AA}$)", title=title)
    fig.colorbar(sc, ax=ax, label=r"$\arg(\psi_6)/6$ (deg)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--bins", type=int, default=60, help="logarithmic distance bins")
    ap.add_argument("--fit-min", type=float, default=10.0, help="smallest distance of the exponent fits (AA)")
    args = ap.parse_args()
    folder = Path(args.folder)

    files = sorted((int(m.group(1)), int(m.group(2)), p) for p in folder.glob("T=*K_k=*.npz") if (m := FNAME.search(p.name)))
    if not files:
        raise FileNotFoundError(f"No T=*K_k=*.npz crystal in {folder}")
    (folder / "orientation_maps").mkdir(exist_ok=True)

    res = defaultdict(lambda: defaultdict(list))
    r_bins = r = None
    for T, k, path in files:
        xyz, bonds, Lx, Ly = obs.load_crystal(path)
        if r_bins is None:
            r_bins = np.geomspace(2.0, min(Lx, Ly) / 2, args.bins + 1)
            r = np.sqrt(r_bins[:-1] * r_bins[1:])
            a = np.sqrt(4 * Lx * Ly / (np.sqrt(3) * len(xyz)))           # area per atom = sqrt(3) a^2 / 4
        G6, GT, psi, _ = obs.correlations(xyz, bonds, Lx, Ly, r_bins)
        for key, v in dict(G6=G6, GT=GT, psi6=abs(psi.mean()),
                           eta6=obs.power_law_exponent(r, G6, args.fit_min, min(Lx, Ly) / 4),
                           etaT=obs.power_law_exponent(r, GT, args.fit_min, min(Lx, Ly) / 4)).items():
            res[T][key].append(v)
        print(f"T={T} K k={k}  |psi6|={abs(psi.mean()):.3f}  eta6={res[T]['eta6'][-1]:.3f}  etaT={res[T]['etaT'][-1]:.3f}")
        if len(res[T]["G6"]) == 1:
            plot_orientation(xyz[:, :2], psi, Lx, Ly, f"T = {T} K (k = {k})", folder / "orientation_maps" / f"T={T}K.jpg")

    T = np.array(sorted(res))
    mean = {key: np.array([np.mean(res[t][key], axis=0) for t in T]) for key in ("G6", "GT", "psi6", "eta6", "etaT")}
    std = {key: np.array([np.std(res[t][key], axis=0) for t in T]) for key in ("psi6", "eta6", "etaT")}
    np.savez(folder / "observables_data.npz", T=T, r=r, a=a, **mean, **{f"{key}_std": v for key, v in std.items()})

    plot_correlations(T, r, mean["G6"], 1 / 4, r"$G_6(r)$", a, folder / "G6.pdf")
    plot_correlations(T, r, mean["GT"], 1 / 3, r"$G_T(r)$", a, folder / "GT.pdf")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, key, label, bound in ((axes[0], "psi6", r"$|\langle\psi_6\rangle|$", None),
                                  (axes[1], "eta6", r"$\eta_6$", 1 / 4), (axes[2], "etaT", r"$\eta_T$", 1 / 3)):
        ax.errorbar(T, mean[key], yerr=std[key], marker="o", ms=4, capsize=3)
        if bound:
            ax.axhline(bound, color="k", ls="--", lw=1)
        ax.set(xlabel=r"$T$ (K)", ylabel=label)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(folder / "order_vs_T.pdf")
    plt.close(fig)
