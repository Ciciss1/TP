"""
Sanity checks of the crystals of a folder (T=<T>K_k=<k>.npz written by Run_Simulation.py):
atom count, LAMMPS convergence, bond lengths, overlaps, corrugation, rings and Euler check.

Usage:
    python validate_crystals.py <folder> [--energy --lmp ~/opt/lammps/bin/lmp --airebo CH.airebo]

Output: table sorted by T, validation.csv and validation.png in the folder. --energy adds a single
AIREBO evaluation (E/atom and in-plane stress) of each crystal.
"""
import argparse
import csv
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

A_CC = 1.3967512290507305
Z_BOX = 10.0
FNAME = re.compile(r"T=(\d+)K_k=(\d+)\.npz$")


def min_image(d, Lx, Ly):
    return d - np.array([Lx, Ly]) * np.round(d / [Lx, Ly])


def ring_sizes(xy, bonds, Lx, Ly):
    '''
    Faces of the bond graph (neighbours sorted counter-clockwise, every atom has 3 bonds)
    '''
    N = len(xy)
    src, dst = np.r_[bonds[:, 0], bonds[:, 1]], np.r_[bonds[:, 1], bonds[:, 0]]
    d = min_image(xy[dst] - xy[src], Lx, Ly)
    nb = dst[np.lexsort((np.arctan2(d[:, 1], d[:, 0]), src))].reshape(N, 3).tolist()
    visited, sizes = set(), []
    for u0 in range(N):
        for v0 in nb[u0]:
            u, v, n = u0, v0, 0
            while (u, v) not in visited:
                visited.add((u, v))
                n += 1
                u, v = v, nb[v][(nb[v].index(u) - 1) % 3]
            if n:
                sizes.append(n)
    return np.array(sizes)


def single_point(xyz, Lx, Ly, lmp, airebo):
    '''
    E/atom (eV) and in-plane stresses (N/m) with full AIREBO, run 0
    '''
    N = len(xyz)
    with tempfile.TemporaryDirectory() as tmp:
        data, inp = os.path.join(tmp, "in.data"), os.path.join(tmp, "in.lmp")
        with open(data, "w") as f:
            f.write(f"check\n\n{N} atoms\n1 atom types\n\n0 {Lx:.6f} xlo xhi\n0 {Ly:.6f} ylo yhi\n-{Z_BOX} {Z_BOX} zlo zhi\n\n"
                    "Masses\n\n1 12.011\n\nAtoms\n\n")
            np.savetxt(f, np.column_stack([np.arange(1, N + 1), np.ones(N), xyz]), fmt="%d %d %.8f %.8f %.8f")
        with open(inp, "w") as f:
            f.write(f"units metal\natom_style atomic\nboundary p p p\nread_data {data}\n"
                    f"pair_style airebo 3.0 1 1\npair_coeff * * {os.path.abspath(airebo)} C\n"
                    "thermo_style custom pe pxx pyy pxy\nrun 0\nprint \"RESULT $(pe) $(pxx) $(pyy) $(pxy)\"\n")
        m = re.search(r"RESULT\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", subprocess.run([lmp, "-in", inp, "-log", "none"], capture_output=True, text=True).stdout)
    if not m:
        return dict(E_per_atom=np.nan, sxx=np.nan, syy=np.nan, sxy=np.nan)
    pe, pxx, pyy, pxy = map(float, m.groups())
    conv = -2 * Z_BOX * 6.24151e-7 * 16.0218                     # pressure (bar) -> 2D stress (N/m)
    return dict(E_per_atom=pe / N, sxx=pxx * conv, syy=pyy * conv, sxy=pxy * conv)


def analyse(path, args):
    d = np.load(path)
    xyz, bonds = d["atoms"], d["bonds"]
    Lx, Ly = (float(v) for v in d["lattice"])
    N = len(xyz)
    get = lambda key: d[key].item() if key in d.files else np.nan

    crit = d["lammps_criterion"] if "lammps_criterion" in d.files else []
    b = np.c_[min_image(xyz[bonds[:, 1], :2] - xyz[bonds[:, 0], :2], Lx, Ly), xyz[bonds[:, 1], 2] - xyz[bonds[:, 0], 2]]
    r = np.linalg.norm(b, axis=1)

    xy = np.mod(xyz[:, :2], [Lx, Ly])
    xy[xy >= [Lx, Ly]] = 0.0
    pairs = cKDTree(xy, boxsize=[Lx, Ly]).query_pairs(2.0, output_type="ndarray")
    dp = np.linalg.norm(np.c_[min_image(xy[pairs[:, 0]] - xy[pairs[:, 1]], Lx, Ly), xyz[pairs[:, 0], 2] - xyz[pairs[:, 1], 2]], axis=1)
    bonded = set((np.minimum(*bonds.T) * N + np.maximum(*bonds.T)).tolist())
    nonbonded = np.array([x not in bonded for x in (np.minimum(*pairs.T) * N + np.maximum(*pairs.T)).tolist()], dtype=bool)

    sizes = ring_sizes(xyz[:, :2], bonds, Lx, Ly)
    out = dict(file=path.name, T_K=get("meta_T_K"), N=N,
               N_excess_pct=100 * (N / (Lx * Ly / (3 * np.sqrt(3) / 4 * A_CC ** 2)) - 1),
               mc_converged=bool(get("meta_mc_converged") == 1), mc_energy=get("meta_mc_energy"),
               mc_misor=get("meta_mc_misorientation"),
               lmp_ok=bool(len(crit) and "force tolerance" in str(crit[-1])),
               bond_min=r.min(), bond_max=r.max(), bond_mean=r.mean(), bond_std=r.std(), n_bond_gt_1_7=int(np.sum(r > 1.7)),
               nonbonded_min=dp[nonbonded].min() if nonbonded.any() else np.nan,
               z_rms=xyz[:, 2].std(), z_ptp=np.ptp(xyz[:, 2]),
               ring5_pct=100 * np.mean(sizes == 5), ring7_pct=100 * np.mean(sizes == 7),
               ring_other_pct=100 * np.mean((sizes < 5) | (sizes > 7)), euler=int(np.sum(6 - sizes)))
    if args.energy:
        out.update(single_point(xyz, Lx, Ly, args.lmp, args.airebo))
    flags = [name for name, bad in (("LAMMPS", not out["lmp_ok"]), ("overlap", out["nonbonded_min"] < 1.8),
                                    ("bond", out["bond_max"] > 1.85), ("euler", out["euler"] != 0),
                                    ("N", abs(out["N_excess_pct"]) > 2)) if bad]
    out["flags"] = ",".join(flags) or "ok"
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--energy", action="store_true")
    ap.add_argument("--lmp", default="lmp")
    ap.add_argument("--airebo", default="CH.airebo")
    args = ap.parse_args()
    folder = Path(args.folder)

    files = sorted((p for p in folder.glob("T=*K_k=*.npz") if FNAME.search(p.name)),
                   key=lambda p: tuple(map(int, FNAME.search(p.name).groups())))
    rows = [analyse(p, args) for p in files]
    if not rows:
        raise SystemExit(f"No T=*K_k=*.npz crystal in {folder}")

    cols = ["T_K", "N", "N_excess_pct", "mc_misor", "lmp_ok", "bond_min", "bond_max", "n_bond_gt_1_7", "nonbonded_min",
            "z_rms", "ring5_pct", "ring7_pct", "euler"] + (["E_per_atom", "sxx", "syy"] if args.energy else []) + ["flags"]
    print("  ".join(f"{c:>13s}" for c in cols))
    for r in rows:
        print("  ".join(f"{r[c]:>13.5g}" if isinstance(r[c], (float, np.floating)) else f"{str(r[c]):>13s}" for c in cols))

    with open(folder / "validation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    panels = [("mc_misor", "MC misorientation"), ("N_excess_pct", "excess atoms (%)"), ("z_rms", r"z RMS ($\mathrm{\AA}$)"),
              ("ring5_pct", "5-rings (%)"), ("bond_max", r"longest bond ($\mathrm{\AA}$)"), ("nonbonded_min", r"closest non-bonded pair ($\mathrm{\AA}$)")]
    panels += [("E_per_atom", "E/atom (eV)"), ("sxx", r"$\sigma_{xx}$ (N/m)")] if args.energy else []
    fig, axes = plt.subplots((len(panels) + 2) // 3, 3, figsize=(12, 3.2 * ((len(panels) + 2) // 3)))
    T = [r["T_K"] for r in rows]
    for ax, (key, label) in zip(axes.ravel(), panels):
        ax.plot(T, [r[key] for r in rows], "o", ms=3)
        ax.set(xlabel="T (K)", ylabel=label)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(folder / "validation.png", dpi=150)
