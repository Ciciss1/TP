"""
Spatial decomposition of the VDOS of thermalised crystals (outputs of sweep_thermalize.py) with the
topological defects of the relaxed crystal each trajectory comes from. Atoms are classified by the
connected clusters of 5/7 rings they belong to (as in Raman_map.py), and the atoms of hexagons by
their distance to the nearest 5/7 ring:
  - grain        : atoms of hexagons only (then split into distance shells)
  - disclination : atoms of an isolated 5 or 7 ring
  - dislocation  : atoms of an isolated 5-7 pair
  - boundary     : atoms of a chain of >= 3 5/7 rings (grain-boundary segment)
For each T, the spectra of the k runs are averaged with the number of atoms as weight:
  vdos_by_defect/T=<T>K.png : spectra per class and per shell (full range, low frequencies, G band)
  vdos_by_defect_vs_T.png   : weight of the G band per class, G-band half-width per shell (vs T and vs
                              distance), fraction of atoms per class
  vdos_by_defect_data.npz   : spectra, G-band fits and atom counts

Usage:
    python vdos_by_defect.py <vdos folder> <crystal folder> [--shells 0,3,6,10,20] [--fmax 2000]
"""
import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit

from thermalize import ThermalizationResult, vdos

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Atomical"))
import Properties as P

CLASSES = {"grain": "grain (hexagons)", "disclination": "isolated 5 or 7",
           "dislocation": "isolated 5-7", "boundary": "chains (boundaries)"}
COLORS = {"grain": "0.3", "disclination": "tab:red", "dislocation": "tab:orange", "boundary": "tab:blue"}
MIN_ATOMS = 20                                       # fewer atoms than this: spectrum too noisy, not fitted
FNAME = re.compile(r"(T=(\d+)K_k=(\d+))_thermalized_.*\.npz$")


def atom_classes(xyz, bonds, Lx, Ly, shells):
    '''
    Boolean masks of the atoms of each class and of each distance shell
    '''
    N = len(xyz)
    sizes, ring_atoms = P.rings(xyz, bonds, Lx, Ly)
    _, _, component, _ = P.defect_graph(sizes, *P.ring_pairs(sizes, ring_atoms))
    labels, counts = np.unique(component[component >= 0], return_counts=True)
    cluster = np.zeros(len(sizes), dtype=np.int64)                  # size of the 5/7 cluster of each ring
    cluster[component >= 0] = counts[np.searchsorted(labels, component[component >= 0])]
    masks = {"grain": ~P.atoms_of(N, ring_atoms, sizes != 6),
             "disclination": P.atoms_of(N, ring_atoms, cluster == 1),
             "dislocation": P.atoms_of(N, ring_atoms, cluster == 2),
             "boundary": P.atoms_of(N, ring_atoms, cluster >= 3)}

    defect = P.atoms_of(N, ring_atoms, component >= 0)
    xy = np.mod(xyz[:, :2], [Lx, Ly])
    xy[xy >= [Lx, Ly]] = 0.0
    dist = cKDTree(xy[defect], boxsize=[Lx, Ly]).query(xy)[0] if defect.any() else np.full(N, np.inf)
    edges = list(shells) + [np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        name = f"{lo:g}-{hi:g} A" if np.isfinite(hi) else f"> {lo:g} A"
        masks[name] = masks["grain"] & (dist >= lo) & ((dist < hi) | np.isinf(hi))
    return masks


def g_band(freq, spec, n):
    '''
    Centre and half-width (cm^-1) of a Lorentzian on a flat background fitted to the G band (NaN when
    there is no clear peak, as on the defect atoms), and the spectral weight between 1650 and 1850 cm^-1
    '''
    if n < MIN_ATOMS:
        return np.nan, np.nan, np.nan
    w = (freq >= 1650) & (freq <= 1850)
    weight = np.trapezoid(spec[w], freq[w])
    m = (freq > 1450) & (freq < 1950)
    f, s = freq[m], spec[m]
    lor = lambda x, a, x0, g, b: a * g ** 2 / ((x - x0) ** 2 + g ** 2) + b
    p0 = [s.max() - s.min(), np.clip(f[np.argmax(s)], 1651, 1849), 30, s.min()]
    try:
        p, _ = curve_fit(lor, f, s, p0=p0, bounds=([0, 1650, 2, 0], [np.inf, 1850, 300, np.inf]), maxfev=20000)
    except RuntimeError:
        return np.nan, np.nan, weight
    return (p[1], p[2], weight) if p[0] > p[3] else (np.nan, np.nan, weight)


def plot_one(T, freq, spectra, counts, shells_names, fmax, path):
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
    shell_colors = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(shells_names)))
    groups = [([(c, CLASSES[c], COLORS[c]) for c in CLASSES], "classes"),
              ([(s, f"grain, {s}", col) for s, col in zip(shells_names, shell_colors)], "distance to the nearest 5/7 ring")]
    for row, (items, title) in zip(axes, groups):
        for ax, (lo, hi), name in zip(row, ((0, fmax), (0, 600), (1450, 1950)), ("", "low frequencies", "G band")):
            m = (freq >= lo) & (freq <= hi)
            top = 0.0
            for key, label, col in items:
                if counts[key] < MIN_ATOMS:
                    continue
                ax.plot(freq[m], spectra[key][m], color=col, lw=1.0, label=f"{label} ({counts[key]} atoms)")
                top = max(top, spectra[key][m].max())
            ax.set(xlim=(lo, hi), ylim=(0, 1.1 * top if top else 1), xlabel=r"frequency (cm$^{-1}$)",
                   title=f"{title}{', ' + name if name else ''}")
        row[0].set_ylabel("VDOS (unit area)")
        row[0].legend(fontsize=7)
    fig.suptitle(f"$T_{{MC}}$ = {T} K", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("vdos_folder")
    ap.add_argument("crystal_folder")
    ap.add_argument("--shells", default="0,3,6,10,20", help="limits of the distance shells of the grain atoms (AA)")
    ap.add_argument("--fmax", type=float, default=2000.0, help="largest frequency shown (cm^-1)")
    args = ap.parse_args()
    shells = [float(x) for x in args.shells.split(",")]
    vdos_folder, crystal_folder = Path(args.vdos_folder), Path(args.crystal_folder)
    out = vdos_folder / "vdos_by_defect"
    out.mkdir(exist_ok=True)

    weighted = defaultdict(lambda: defaultdict(float))              # T -> key -> sum over k of n * spectrum
    counts = defaultdict(lambda: defaultdict(int))                  # T -> key -> number of atoms (all k)
    n_total = defaultdict(int)
    freq = None
    for path in sorted(vdos_folder.glob("T=*K_k=*_thermalized_*.npz")):
        m = FNAME.search(path.name)
        crystal = crystal_folder / f"{m.group(1)}.npz"
        if not crystal.exists():
            print(f"{path.name}: no {crystal.name} in {crystal_folder}, skipped")
            continue
        xyz, bonds, Lx, Ly = P.load_crystal(crystal)
        r = ThermalizationResult.load(path)
        if r.velocities.shape[1] != len(xyz):
            print(f"{path.name}: {r.velocities.shape[1]} atoms but {len(xyz)} in {crystal.name}, skipped")
            continue
        T = int(m.group(2))
        masks = atom_classes(xyz, bonds, Lx, Ly, shells)
        for key, mask in masks.items():
            n = int(mask.sum())
            if n == 0:
                continue
            f, spectra = vdos(r.velocities, r.times_ps[1] - r.times_ps[0], atoms=mask)
            if freq is None:
                freq = f
            if len(f) != len(freq) or not np.allclose(f, freq):
                raise ValueError(f"{path.name}: frequency grid differs from the other trajectories (other dt or length)")
            weighted[T][key] = weighted[T][key] + n * spectra["all"]
            counts[T][key] += n
        n_total[T] += len(xyz)
        print(f"{path.name}: " + "  ".join(f"{key}={int(mask.sum())}" for key, mask in masks.items()))

    if freq is None:
        raise SystemExit("No trajectory with its crystal")
    T_list = sorted(weighted)
    keys = list(CLASSES) + [k for k in masks if k not in CLASSES]
    shell_names = keys[len(CLASSES):]
    spectra = {T: {k: weighted[T][k] / counts[T][k] if counts[T][k] else np.full(len(freq), np.nan) for k in keys} for T in T_list}
    fits = {T: {k: g_band(freq, spectra[T][k], counts[T][k]) for k in keys} for T in T_list}

    print(f"\n{'T_MC (K)':>9s} " + " ".join(f"{k:>13s}" for k in keys))
    print(f"{'':>9s} " + " ".join(f"{'G weight':>13s}" for _ in CLASSES) + " ".join(f"{'HWHM':>14s}" for _ in shell_names))
    for T in T_list:
        print(f"{T:9d} " + " ".join(f"{fits[T][k][2]:13.3f}" for k in CLASSES) + " ".join(f"{fits[T][k][1]:14.1f}" for k in shell_names))
        plot_one(T, freq, spectra[T], counts[T], shell_names, args.fmax, out / f"T={T}K.png")

    T_arr = np.array(T_list)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    (ax_w, ax_h), (ax_d, ax_n) = axes
    shell_colors = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(shell_names)))
    for k in CLASSES:
        ax_w.plot(T_arr, [fits[T][k][2] for T in T_list], "o-", color=COLORS[k], label=CLASSES[k])
        ax_n.plot(T_arr, [counts[T][k] / n_total[T] if counts[T][k] else np.nan for T in T_list], "o-", color=COLORS[k], label=CLASSES[k])
    for k, col in zip(shell_names, shell_colors):
        ax_h.plot(T_arr, [fits[T][k][1] for T in T_list], "o-", color=col, label=f"grain, {k}")
    mid = np.array(shells + [shells[-1] * 1.5])[:len(shell_names)]
    mid = (mid + np.r_[mid[1:], mid[-1] * 1.5]) / 2                    # middle of each shell (last one: indicative)
    T_colors = plt.get_cmap("plasma")(np.linspace(0, 0.9, len(T_list)))
    for T, col in zip(T_list, T_colors):
        ax_d.plot(mid, [fits[T][k][1] for k in shell_names], "o-", color=col, label=f"{T} K")
    ax_w.set(xlabel=r"$T_{MC}$ (K)", ylabel=r"weight 1650-1850 cm$^{-1}$", title="G-band weight per class")
    ax_h.set(xlabel=r"$T_{MC}$ (K)", ylabel=r"G-band HWHM (cm$^{-1}$)", title="grain atoms, per distance to the nearest 5/7 ring")
    ax_d.set(xlabel="distance to the nearest 5/7 ring (AA, middle of the shell)", ylabel=r"G-band HWHM (cm$^{-1}$)",
             title="broadening vs distance to the defects")
    ax_n.set(xlabel=r"$T_{MC}$ (K)", ylabel="fraction of atoms", yscale="log", title="atoms per class")
    for ax in axes.flat:
        ax.legend(fontsize=7, ncol=2 if ax is ax_d else 1)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(vdos_folder / "vdos_by_defect_vs_T.png", dpi=150)
    plt.close(fig)

    np.savez(vdos_folder / "vdos_by_defect_data.npz", T=T_arr, freq=freq, keys=np.array(keys),
             spectra=np.array([[spectra[T][k] for k in keys] for T in T_list]),
             G=np.array([[fits[T][k][0] for k in keys] for T in T_list]),
             hwhm=np.array([[fits[T][k][1] for k in keys] for T in T_list]),
             g_weight=np.array([[fits[T][k][2] for k in keys] for T in T_list]),
             n_atoms=np.array([[counts[T][k] for k in keys] for T in T_list]), n_total=np.array([n_total[T] for T in T_list]))
