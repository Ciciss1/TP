"""
Choice of KAPPA on the real crystals: the Monte Carlo states of a folder (T=<T>K_k=<k>.npz) are rebuilt
by the full pipeline (Lloyd + LAMMPS, as Run_Simulation.py) for several KAPPA, and the relaxed excess
energies are compared (reference: the same box with all grains aligned). The KAPPA giving the lowest
energy is the one the relaxed polycrystals prefer. The crystals are written in <out>_crystals/.

Usage:
    python kappa_scan.py <folder> scan.jsonl [--kappa 1,1.6,2] [--T 80,95,120] [--k 1]
                         [--workers 36] [--threads 2]
    python kappa_scan.py <folder> scan.jsonl --summary
    (T in 10^3 K as in parameters.txt, default all; k: runs of each T, default the first;
     fit = the KAPPA of Graphene.py, a number or a pair (k0, k1) for kappa(r) = k0 + k1 ln(r / b))
"""
import os
import sys
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "Spe", "Atomical"))
os.chdir(HERE)                                            # CH.airebo

import Graphene as G
import Properties as P

A_CC = 1.3967512290507305
FIT = G.KAPPA                                             # value of KAPPA in Graphene.py ("fit")


def label(kappa):
    return kappa if isinstance(kappa, str) else format(kappa, "g")


def build(path, kappa, threads, aligned = False):
    d = np.load(path)
    theta = np.zeros_like(d["theta"]) if aligned else d["theta"]
    G.KAPPA = FIT if kappa in (None, "fit") else kappa    # aligned reference: no boundary anyway
    vor = G.voronoi_from_points(float(np.ravel(d["L"])[0]), float(np.ravel(d["rho"])[0]), d["points"], theta)
    seed = int(np.ravel(d["lammps_seed"])[0]) if "lammps_seed" in d else 1
    return G.GrapheneCrystal(vor, a=A_CC, lloyd_kwargs={"method": "lbfgs"}, lammps_kwargs={"n_threads": threads, "seed": seed})


def job(task):
    T, k, path, kappa, threads, out_dir = task
    cr = build(path, kappa, threads, aligned=kappa is None)
    E = float(np.ravel(cr.lammps_info["energy_final"])[-1])
    row = dict(T=T, k=k, kappa=kappa, N=len(cr.atoms), E=E)
    if kappa is not None:
        sizes, _ = P.rings(np.asarray(cr.atoms, float), np.asarray(cr.bonds, np.int64), cr.Lx, cr.Ly)
        row.update(n57=int(((sizes == 5) | (sizes == 7)).sum()), other=int(((sizes < 5) | (sizes > 7)).sum()))
        cr.save_crystal(str(Path(out_dir) / f"T={T}K_k={k}_kappa={label(kappa)}.npz"))
    return row


def summary(out):
    rows = [json.loads(line) for line in open(out)]
    ref = [r for r in rows if r["kappa"] is None]
    if not ref:
        raise SystemExit("reference (aligned grains) not computed yet")
    e_bulk = ref[0]["E"] / ref[0]["N"]
    rows = [dict(r, excess=r["E"] - r["N"] * e_bulk) for r in rows if r["kappa"] is not None]
    kappas = sorted({r["kappa"] for r in rows}, key=lambda k: (isinstance(k, str), 0 if isinstance(k, str) else k))
    states = sorted({(r["T"], r["k"]) for r in rows})
    table = {(r["T"], r["k"], r["kappa"]): r for r in rows}
    print(f"e_bulk = {e_bulk:.5f} eV/atom; excess energy (eV) and number of 5/7 rings")
    print(f"{'T (K)':>8s} {'k':>2s} " + " ".join(f"{'kappa=' + label(k):>16s}" for k in kappas) + "   best")
    rel = {k: [] for k in kappas}
    for T, k in states:
        cells = [table.get((T, k, kap)) for kap in kappas]
        if any(c is None for c in cells):
            continue
        best = kappas[int(np.argmin([c["excess"] for c in cells]))]
        print(f"{T:8d} {k:2d} " + " ".join(f"{c['excess']:9.2f} ({c['n57']:4d})" for c in cells) + f"   {label(best)}")
        e_min = min(c["excess"] for c in cells)
        for kap, c in zip(kappas, cells):
            rel[kap].append(c["excess"] / e_min - 1)
    print("mean excess above the best KAPPA of each state: " +
          ", ".join(f"kappa={label(k)}: {100 * np.mean(v):.1f} %" for k, v in rel.items() if v))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("out")
    ap.add_argument("--kappa", default="1,1.6,2", help="values compared; fit = the KAPPA of Graphene.py")
    ap.add_argument("--T", help="temperatures (10^3 K), comma separated; default all")
    ap.add_argument("--k", default="1", help="runs of each T, comma separated")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    ap.add_argument("--threads", type=int, default=2, help="OpenMP threads per LAMMPS run")
    ap.add_argument("--summary", action="store_true", help="only print the table of a finished scan")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    if args.summary:
        summary(out)
        raise SystemExit

    out_dir = out.rsplit(".", 1)[0] + "_crystals"
    os.makedirs(out_dir, exist_ok=True)
    done = set()
    if os.path.isfile(out):
        done = {(r["T"], r["k"], r["kappa"]) for r in map(json.loads, open(out))}
    ks = {int(k) for k in args.k.split(",")}
    T_sel = {round(1000 * float(t)) for t in args.T.split(",")} if args.T else None
    files = [(T, k, str(p)) for T, k, p in P.crystal_files(args.folder) if k in ks and (T_sel is None or T in T_sel)]
    if not files:
        raise SystemExit("no crystal selected")
    tasks = [(files[0][0], files[0][1], files[0][2], None)]                         # aligned reference
    tasks += [(T, k, p, kap if kap == "fit" else float(kap)) for T, k, p in files for kap in args.kappa.split(",")]
    tasks = [t + (args.threads, out_dir) for t in tasks if (t[0], t[1], t[3]) not in done]
    print(f"{len(tasks)} crystals on {args.workers} workers x {args.threads} threads", flush=True)
    with ProcessPoolExecutor(args.workers) as pool, open(out, "a") as f:
        futures = {pool.submit(job, t): t for t in tasks}
        for fut in as_completed(futures):
            try:
                row = fut.result()
            except Exception as e:
                print(f"failed {futures[fut][:2]} kappa={futures[fut][3]}: {e}", flush=True)
                continue
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(json.dumps(row), flush=True)
    summary(out)
