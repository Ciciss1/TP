"""
Validation des cristaux de graphene polycristallin (.npz produits par Run_Simulation.py).

Usage :
    python validate_crystals.py /scratch/escarmel/crystals/eps=0/L=272.12/rho=0.00135
    python validate_crystals.py <dossier> --energy --lmp /home/escarmel/opt/lammps/bin/lmp --airebo CH.airebo

Sortie : tableau par temperature (trie par T), validation.csv, validation.png.
--energy relance un point unique AIREBO (run 0) pour obtenir E/atome et la contrainte dans le plan.
"""
import argparse
import glob
import os
import re
import subprocess
import tempfile

import numpy as np
from scipy.spatial import cKDTree

A_CC = 1.3967512290507305
Z_VACUUM = 10.0
EV_A2_TO_N_M = 16.0218
BAR_TO_EV_A3 = 6.24151e-7


def min_image(d, L):
    return d - L * np.round(d / L)


def scalar(data, key, default=np.nan):
    if key not in data.files:
        return default
    v = np.asarray(data[key])
    return v.item() if v.size == 1 else v


def build_neighbors(bonds, N):
    src = np.concatenate([bonds[:, 0], bonds[:, 1]])
    dst = np.concatenate([bonds[:, 1], bonds[:, 0]])
    deg = np.bincount(src, minlength=N)
    if not np.all(deg == 3):
        return None, deg
    order = np.argsort(src, kind="stable")
    return dst[order].reshape(N, 3), deg


def ring_statistics(atoms, bonds, L):
    '''
    Trace the faces of the (toroidal) bond graph. Returns ring-size histogram and Euler check
    (sum of (6 - n) over faces must be 0 on a torus for a trivalent graph).
    '''
    N = len(atoms)
    nb, deg = build_neighbors(bonds, N)
    if nb is None:
        return None
    d = min_image(atoms[nb, :2] - atoms[:, None, :2], L)
    ang = np.arctan2(d[..., 1], d[..., 0])
    order = np.argsort(ang, axis=1)
    ns = np.take_along_axis(nb, order, axis=1).tolist()

    visited = set()
    sizes = []
    for u0 in range(N):
        for v0 in ns[u0]:
            if (u0, v0) in visited:
                continue
            u, v, n = u0, v0, 0
            while (u, v) not in visited:
                visited.add((u, v))
                n += 1
                idx = ns[v].index(u)
                w = ns[v][(idx - 1) % 3]
                u, v = v, w
                if n > 10000:
                    break
            sizes.append(n)
    sizes = np.array(sizes)
    hist = {k: int(np.sum(sizes == k)) for k in (5, 6, 7)}
    hist["other"] = int(np.sum((sizes != 5) & (sizes != 6) & (sizes != 7)))
    hist["euler"] = int(np.sum(6 - sizes))
    hist["n_faces"] = int(len(sizes))
    return hist


def lammps_single_point(atoms, L, lmp, airebo):
    '''E/atom (eV) and in-plane stress (N/m) with full AIREBO, run 0.'''
    N = len(atoms)
    pos = atoms.copy()
    pos[:, 0] = np.mod(pos[:, 0], L)
    pos[:, 1] = np.mod(pos[:, 1], L)
    with tempfile.TemporaryDirectory(prefix="validate_") as tmp:
        data = os.path.join(tmp, "in.data")
        inp = os.path.join(tmp, "in.lmp")
        with open(data, "w") as f:
            f.write("check\n\n")
            f.write(f"{N} atoms\n1 atom types\n\n")
            f.write(f"0 {L:.6f} xlo xhi\n0 {L:.6f} ylo yhi\n-{Z_VACUUM} {Z_VACUUM} zlo zhi\n\n")
            f.write("Masses\n\n1 12.011\n\nAtoms\n\n")
            tab = np.column_stack([np.arange(1, N + 1), np.ones(N), pos])
            np.savetxt(f, tab, fmt="%d %d %.8f %.8f %.8f")
        with open(inp, "w") as f:
            f.write(f"""units metal
atom_style atomic
boundary p p p
read_data {data}
pair_style airebo 3.0 1 1
pair_coeff * * {os.path.abspath(airebo)} C
variable pe equal pe
variable pxx equal pxx
variable pyy equal pyy
variable pxy equal pxy
thermo_style custom step pe pxx pyy pxy
run 0
print "RESULT ${{pe}} ${{pxx}} ${{pyy}} ${{pxy}}"
""")
        r = subprocess.run([lmp, "-in", inp, "-log", "none"], capture_output=True, text=True)
        m = re.search(r"RESULT\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", r.stdout)
        if not m:
            return dict(E_per_atom=np.nan, sxx=np.nan, syy=np.nan, sxy=np.nan)
        pe, pxx, pyy, pxy = map(float, m.groups())
        conv = -2 * Z_VACUUM * BAR_TO_EV_A3 * EV_A2_TO_N_M       # pressure (bar) -> 2D stress (N/m)
        return dict(E_per_atom=pe / N, sxx=pxx * conv, syy=pyy * conv, sxy=pxy * conv)


def analyse(path, args):
    data = np.load(path)
    atoms = np.asarray(data["atoms"], dtype=np.float64)
    bonds = np.asarray(data["bonds"], dtype=np.int64)
    L = float(data["L"][0])
    N = len(atoms)
    xy = np.mod(atoms[:, :2], L)
    xy[xy >= L] = 0.0
    z = atoms[:, 2]

    out = {"file": os.path.basename(path)}
    out["T_K"] = scalar(data, "meta_T_K")
    out["N"] = N
    ideal = L * L / (3 * np.sqrt(3) / 4 * A_CC**2)
    out["N_excess_%"] = 100 * (N / ideal - 1)

    # MC
    out["mc_converged"] = bool(scalar(data, "meta_mc_converged", False))
    out["mc_sweeps"] = scalar(data, "meta_mc_sweeps")
    out["mc_energy"] = scalar(data, "meta_mc_energy")
    out["mc_misor"] = scalar(data, "meta_mc_misorientation")

    # LAMMPS convergence
    crit = data["lammps_criterion"] if "lammps_criterion" in data.files else np.array([])
    out["lmp_ok"] = bool(len(crit) == 2 and all("force tolerance" in str(c) for c in crit))
    fn = data["lammps_fnorm_final"] if "lammps_fnorm_final" in data.files else np.array([np.nan])
    out["fnorm_final"] = float(np.atleast_1d(fn)[-1])

    # Bond lengths (topological bonds, 3D, after relaxation)
    dxy = min_image(atoms[bonds[:, 0], :2] - atoms[bonds[:, 1], :2], L)
    dz = z[bonds[:, 0]] - z[bonds[:, 1]]
    r = np.sqrt((dxy**2).sum(1) + dz**2)
    out["bond_min"], out["bond_max"] = r.min(), r.max()
    out["bond_mean"], out["bond_std"] = r.mean(), r.std()
    out["frac_bond_gt_1.85"] = float(np.mean(r > 1.85))

    # All-pairs geometry
    tree = cKDTree(xy, boxsize=L)
    pairs = tree.query_pairs(2.0, output_type="ndarray")
    dxy = min_image(atoms[pairs[:, 0], :2] - atoms[pairs[:, 1], :2], L)
    dz = z[pairs[:, 0]] - z[pairs[:, 1]]
    dp = np.sqrt((dxy**2).sum(1) + dz**2)
    out["pair_min"] = dp.min() if len(dp) else np.nan
    out["n_pairs_lt_1.0"] = int(np.sum(dp < 1.0))
    geo = pairs[dp < 1.85]
    coord = np.bincount(geo.ravel(), minlength=N)
    out["frac_coord3_geo"] = float(np.mean(coord == 3))
    out["frac_coord_not3_geo"] = float(np.mean(coord != 3))
    bset = set((np.minimum(bonds[:, 0], bonds[:, 1]) * N + np.maximum(bonds[:, 0], bonds[:, 1])).tolist())
    gset = set((np.minimum(geo[:, 0], geo[:, 1]) * N + np.maximum(geo[:, 0], geo[:, 1])).tolist())
    out["bonds_topo_not_geo"] = len(bset - gset)
    out["bonds_geo_not_topo"] = len(gset - bset)

    # Roughness
    out["z_rms"] = float(np.std(z))
    out["z_ptp"] = float(np.ptp(z))

    # Rings
    rings = ring_statistics(atoms, bonds, L)
    if rings is not None:
        nf = rings["n_faces"]
        out["ring5_%"] = 100 * rings[5] / nf
        out["ring6_%"] = 100 * rings[6] / nf
        out["ring7_%"] = 100 * rings[7] / nf
        out["ring_other_%"] = 100 * rings["other"] / nf
        out["euler_check"] = rings["euler"]
    else:
        out["euler_check"] = np.nan

    if args.energy:
        out.update(lammps_single_point(atoms, L, args.lmp, args.airebo))
    return out


def flags(row):
    f = []
    if not row["lmp_ok"]:
        f.append("LAMMPS")
    if row["n_pairs_lt_1.0"] > 0 or row["pair_min"] < 1.0:
        f.append("overlap")
    if row["frac_coord_not3_geo"] > 0.02:
        f.append("coord")
    if row["z_rms"] > 1.0:
        f.append("buckling")
    if row["euler_check"] not in (0,) and not np.isnan(row["euler_check"]):
        f.append("euler")
    if abs(row["N_excess_%"]) > 2:
        f.append("N")
    return ",".join(f) if f else "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--energy", action="store_true")
    ap.add_argument("--lmp", default="lmp")
    ap.add_argument("--airebo", default="CH.airebo")
    ap.add_argument("--csv", default="validation.csv")
    ap.add_argument("--plot", default="validation.png")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.folder, "T=*K_k=*.npz")))
    if not files:
        raise SystemExit("Aucun .npz trouve")
    rows = []
    for p in files:
        try:
            rows.append(analyse(p, args))
        except Exception as e:
            print(f"ECHEC {p}: {type(e).__name__}: {e}")
    rows.sort(key=lambda r: (r["T_K"], r["file"]))
    for r in rows:
        r["flags"] = flags(r)

    cols = ["T_K", "N", "N_excess_%", "mc_sweeps", "mc_energy", "mc_misor", "lmp_ok", "bond_min", "bond_max",
            "pair_min", "frac_coord_not3_geo", "bonds_topo_not_geo", "z_rms", "ring5_%", "ring7_%", "euler_check"]
    if args.energy:
        cols += ["E_per_atom", "sxx", "syy", "sxy"]
    cols += ["flags"]

    print("  ".join(f"{c:>12s}" for c in cols))
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(f"{v:>12.4g}" if isinstance(v, (float, np.floating)) else f"{str(v):>12s}")
        print("  ".join(cells))

    import csv
    keys = sorted({k for r in rows for k in r})
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV -> {args.csv}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = np.array([r["T_K"] for r in rows])
    panels = [("mc_energy", "E MC moyenne (eV)"), ("mc_misor", "desorientation moyenne"),
              ("N_excess_%", "exces d'atomes (%)"), ("z_rms", "z rms (A)"),
              ("ring5_%", "anneaux 5 (%)"), ("ring7_%", "anneaux 7 (%)")]
    if args.energy:
        panels += [("E_per_atom", "E/atome (eV)"), ("sxx", "sigma_xx (N/m)")]
    n = len(panels)
    fig, axs = plt.subplots((n + 2) // 3, 3, figsize=(12, 3.2 * ((n + 2) // 3)))
    for ax, (k, lab) in zip(axs.ravel(), panels):
        ax.plot(T, [r.get(k, np.nan) for r in rows], "o-", ms=3)
        ax.set_xlabel("T (K)")
        ax.set_ylabel(lab)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.plot, dpi=150)
    print(f"Figure -> {args.plot}")


if __name__ == "__main__":
    main()
