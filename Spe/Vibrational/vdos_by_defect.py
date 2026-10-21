"""
Spatial decomposition of the VDOS of thermalised crystals (outputs of sweep_thermalize.py) with the
topological defects of the relaxed crystal each trajectory comes from. Atoms are classified by the
connected clusters of 5/7 rings they belong to (defect_graph of Properties.py), and the atoms of hexagons by
their distance to the nearest 5/7 ring:
  - grain        : atoms of hexagons only (then split into distance shells)
  - disclination : atoms of an isolated 5 or 7 ring
  - dislocation  : atoms of an isolated 5-7 pair
  - boundary     : atoms of a chain of >= 3 5/7 rings (grain-boundary segment)
The power spectrum of every atom is computed once per trajectory (one FFT pass), then summed over the
atoms of each class and over the k runs of a temperature. Trajectories run in parallel.
  vdos_by_defect/T=<T>K.png : spectra per class and per shell (full range, low frequencies, G band)
  vdos_by_defect_vs_T.png   : weight of the G band per class, G-band half-width per shell (vs T and vs
                              distance), fraction of atoms per class
  vdos_by_defect_data.npz   : spectra, G-band fits and atom counts

Usage:
    python vdos_by_defect.py <vdos folder> <crystal folder> [--shells 0,3,6,10,20] [--workers 12] [--threads 6]
    (workers x threads = cores; each worker holds one trajectory in memory)
"""
import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")                  # parallelism comes from the workers and the FFT threads

import re
import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import scipy.fft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit

from thermalize import ThermalizationResult, THZ_TO_CM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Atomical"))
import Properties as P

CLASSES = {"grain": "grain (hexagons)", "disclination": "isolated 5 or 7",
           "dislocation": "isolated 5-7", "boundary": "chains (boundaries)"}
COLORS = {"grain": "0.3", "disclination": "tab:red", "dislocation": "tab:orange", "boundary": "tab:blue"}
MIN_ATOMS = 20                                       # fewer atoms than this: spectrum too noisy, not fitted
FNAME = re.compile(r"(T=(\d+)K_k=(\d+))_thermalized_.*\.npz$")


def shell_names(shells):
    edges = list(shells) + [np.inf]
    return [f"{lo:g}-{hi:g} A" if np.isfinite(hi) else f"> {lo:g} A" for lo, hi in zip(edges[:-1], edges[1:])]


def atom_classes(xyz, bonds, Lx, Ly, shells):
    '''
    Boolean masks (n_keys, N) of the atoms of each class, then of each distance shell
    '''
    N = len(xyz)
    sizes, ring_atoms = P.rings(xyz, bonds, Lx, Ly)
    _, _, component, _ = P.defect_graph(sizes, *P.ring_pairs(sizes, ring_atoms))
    labels, counts = np.unique(component[component >= 0], return_counts=True)
    cluster = np.zeros(len(sizes), dtype=np.int64)                  # size of the 5/7 cluster of each ring
    cluster[component >= 0] = counts[np.searchsorted(labels, component[component >= 0])]
    grain = ~P.atoms_of(N, ring_atoms, sizes != 6)
    masks = [grain, P.atoms_of(N, ring_atoms, cluster == 1), P.atoms_of(N, ring_atoms, cluster == 2),
             P.atoms_of(N, ring_atoms, cluster >= 3)]

    defect = P.atoms_of(N, ring_atoms, component >= 0)
    xy = np.mod(xyz[:, :2], [Lx, Ly])
    xy[xy >= [Lx, Ly]] = 0.0
    dist = cKDTree(xy[defect], boxsize=[Lx, Ly]).query(xy)[0] if defect.any() else np.full(N, np.inf)
    edges = list(shells) + [np.inf]
    masks += [grain & (dist >= lo) & ((dist < hi) | np.isinf(hi)) for lo, hi in zip(edges[:-1], edges[1:])]
    return np.array(masks)


def atom_power(velocities, threads, batch = 4000):
    '''
    Windowed power spectrum of the velocity of each atom (x + y + z), centre-of-mass drift removed:
    (N, n_frames // 2 + 1), float32
    '''
    n_frames, N, _ = velocities.shape
    v_com = velocities.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    window = np.hanning(n_frames).astype(np.float32)[:, None]
    power = np.empty((N, n_frames // 2 + 1), dtype=np.float32)
    for start in range(0, N, batch):
        v = ((velocities[:, start:start + batch] - v_com).reshape(n_frames, -1) * window).T   # (3 n, n_frames)
        F = scipy.fft.rfft(v, axis=1, workers=threads)
        power[start:start + batch] = (F.real ** 2 + F.imag ** 2).reshape(-1, 3, F.shape[1]).sum(axis=1)
    return power


def process(path, crystal, shells, threads):
    '''
    Summed power spectrum and number of atoms of each class / shell for one trajectory
    '''
    xyz, bonds, Lx, Ly = P.load_crystal(crystal)
    r = ThermalizationResult.load(path)
    if r.velocities.shape[1] != len(xyz):
        raise ValueError(f"{r.velocities.shape[1]} atoms but {len(xyz)} in {crystal.name}")
    masks = atom_classes(xyz, bonds, Lx, Ly, shells)
    power = masks.astype(np.float64) @ atom_power(r.velocities, threads)
    freq = np.fft.rfftfreq(r.velocities.shape[0], r.times_ps[1] - r.times_ps[0]) * THZ_TO_CM
    return freq, power, masks.sum(axis=1), len(xyz)


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


def plot_one(T, freq, spectra, counts, shells, fmax, path):
    n_cls = len(CLASSES)
    shell_colors = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(shells)))
    groups = [([(i, CLASSES[c], COLORS[c]) for i, c in enumerate(CLASSES)], "classes"),
              ([(n_cls + i, f"grain, {s}", col) for i, (s, col) in enumerate(zip(shells, shell_colors))],
               "distance to the nearest 5/7 ring")]
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
    for row, (items, title) in zip(axes, groups):
        for ax, (lo, hi), name in zip(row, ((0, fmax), (0, 600), (1450, 1950)), ("", ", low frequencies", ", G band")):
            m = (freq >= lo) & (freq <= hi)
            top = 0.0
            for i, label, col in items:
                if counts[i] >= MIN_ATOMS:
                    ax.plot(freq[m], spectra[i][m], color=col, lw=1.0, label=f"{label} ({counts[i]} atoms)")
                    top = max(top, spectra[i][m].max())
            ax.set(xlim=(lo, hi), ylim=(0, 1.1 * top or 1), xlabel=r"frequency (cm$^{-1}$)", title=title + name)
        row[0].set_ylabel("VDOS (unit area)")
        row[0].legend(fontsize=7)
    fig.suptitle(f"$T_{{MC}}$ = {T} K", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_vs_T(T, fits, frac, shells, shell_mid, path):
    n_cls = len(CLASSES)
    fig, ((ax_w, ax_h), (ax_d, ax_n)) = plt.subplots(2, 2, figsize=(13, 9))
    for i, c in enumerate(CLASSES):
        ax_w.plot(T, fits[:, i, 2], "o-", color=COLORS[c], label=CLASSES[c])
        ax_n.plot(T, frac[:, i], "o-", color=COLORS[c], label=CLASSES[c])
    for i, (s, col) in enumerate(zip(shells, plt.get_cmap("viridis")(np.linspace(0, 0.9, len(shells))))):
        ax_h.plot(T, fits[:, n_cls + i, 1], "o-", color=col, label=f"grain, {s}")
    for j, col in enumerate(plt.get_cmap("plasma")(np.linspace(0, 0.9, len(T)))):
        ax_d.plot(shell_mid, fits[j, n_cls:, 1], "o-", color=col, label=f"{T[j]} K")
    ax_w.set(xlabel=r"$T_{MC}$ (K)", ylabel=r"weight 1650-1850 cm$^{-1}$", title="G-band weight per class")
    ax_h.set(xlabel=r"$T_{MC}$ (K)", ylabel=r"G-band HWHM (cm$^{-1}$)", title="grain atoms, per distance to the nearest 5/7 ring")
    ax_d.set(xlabel="distance to the nearest 5/7 ring (AA, middle of the shell)", ylabel=r"G-band HWHM (cm$^{-1}$)",
             title="broadening vs distance to the defects")
    ax_n.set(xlabel=r"$T_{MC}$ (K)", ylabel="fraction of atoms", yscale="log", title="atoms per class")
    for ax in (ax_w, ax_h, ax_d, ax_n):
        ax.legend(fontsize=7, ncol=2 if ax is ax_d else 1)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("vdos_folder")
    ap.add_argument("crystal_folder")
    ap.add_argument("--shells", default="0,3,6,10,20", help="limits of the distance shells of the grain atoms (AA)")
    ap.add_argument("--fmax", type=float, default=2000.0, help="largest frequency shown (cm^-1)")
    ap.add_argument("--workers", type=int, default=12, help="trajectories processed at the same time")
    ap.add_argument("--threads", type=int, default=6, help="FFT threads per worker")
    args = ap.parse_args()
    shells = [float(x) for x in args.shells.split(",")]
    names = shell_names(shells)
    vdos_folder, crystal_folder = Path(args.vdos_folder), Path(args.crystal_folder)
    (vdos_folder / "vdos_by_defect").mkdir(exist_ok=True)

    jobs = []
    for path in sorted(vdos_folder.glob("T=*K_k=*_thermalized_*.npz")):
        m = FNAME.search(path.name)
        crystal = crystal_folder / f"{m.group(1)}.npz"
        if crystal.exists():
            jobs.append((int(m.group(2)), path, crystal))
        else:
            print(f"{path.name}: no {crystal.name} in {crystal_folder}, skipped")
    if not jobs:
        raise SystemExit("No trajectory with its crystal")

    power, counts, n_total, freq = {}, {}, {}, None
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        futures = {pool.submit(process, path, crystal, shells, args.threads): (T, path) for T, path, crystal in jobs}
        for fut in as_completed(futures):
            T, path = futures[fut]
            try:
                f, p, n, N = fut.result()
            except Exception as err:
                print(f"{path.name}: FAILED {err}")
                continue
            if freq is None:
                freq = f
            elif len(f) != len(freq) or not np.allclose(f, freq):
                print(f"{path.name}: other frequency grid (dt or length) than the first trajectory, skipped")
                continue
            power[T] = power.get(T, 0) + p
            counts[T] = counts.get(T, 0) + n
            n_total[T] = n_total.get(T, 0) + N
            print(f"{path.name}: " + "  ".join(f"{k}={c}" for k, c in zip(list(CLASSES) + names, n)), flush=True)

    T_list = sorted(power)
    keys = list(CLASSES) + names
    spectra = np.array([[p / np.trapezoid(p, freq) if n else np.full(len(freq), np.nan) for p, n in zip(power[T], counts[T])]
                        for T in T_list])
    cnt = np.array([counts[T] for T in T_list])
    fits = np.array([[g_band(freq, s, n) for s, n in zip(spectra[j], cnt[j])] for j in range(len(T_list))])
    frac = np.where(cnt > 0, cnt / np.array([n_total[T] for T in T_list])[:, None], np.nan)

    print(f"\n{'T_MC (K)':>9s} " + " ".join(f"{k:>13s}" for k in keys))
    print(f"{'':>9s} " + " ".join(f"{'G weight':>13s}" for _ in CLASSES) + " ".join(f"{'HWHM':>14s}" for _ in names))
    for j, T in enumerate(T_list):
        print(f"{T:9d} " + " ".join(f"{w:13.3f}" for w in fits[j, :len(CLASSES), 2])
              + " ".join(f"{h:14.1f}" for h in fits[j, len(CLASSES):, 1]))
    with ProcessPoolExecutor(max_workers=min(args.workers * args.threads, len(T_list))) as pool:   # one figure per T
        list(pool.map(plot_one, T_list, [freq] * len(T_list), spectra, cnt, [names] * len(T_list), [args.fmax] * len(T_list),
                      [vdos_folder / "vdos_by_defect" / f"T={T}K.png" for T in T_list]))

    edges = np.array(shells + [shells[-1] * 1.5])                      # last shell: indicative middle
    shell_mid = (edges[:-1] + edges[1:]) / 2
    plot_vs_T(np.array(T_list), fits, frac, names, shell_mid, vdos_folder / "vdos_by_defect_vs_T.png")
    np.savez(vdos_folder / "vdos_by_defect_data.npz", T=np.array(T_list), freq=freq, keys=np.array(keys), spectra=spectra,
             G=fits[:, :, 0], hwhm=fits[:, :, 1], g_weight=fits[:, :, 2], n_atoms=cnt,
             n_total=np.array([n_total[T] for T in T_list]))
