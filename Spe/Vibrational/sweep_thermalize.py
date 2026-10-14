"""
Thermalise the crystals T=<T>K_k=<k>.npz of a folder (one k) at a physical temperature, in parallel,
and save the trajectories as <output>/T=<T>K_k=<k>_thermalized_<T_md>K.npz (existing files skipped).

Usage:
    python sweep_thermalize.py <crystal folder> <output folder> [--k 1] [--T 300] [--workers 9] [--threads 8] [--lmp lmp]
"""
import re
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from thermalize import Thermalizer

EQUIL_CHECK_PS = 2.0
EQUIL_MAX_PS = 20.0
PE_TOL = 0.005
PROD_PS = 5.0
AIREBO_FILE = "CH.airebo"


def process(path, out_path, T, threads, lmp):
    d = np.load(path)
    Lx, Ly = d["lattice"]
    result = Thermalizer(AIREBO_FILE, threads, lmp=lmp).run(
        d["atoms"], Lx, Ly, T=T, equil_check_ps=EQUIL_CHECK_PS, equil_max_ps=EQUIL_MAX_PS, pe_tol=PE_TOL, prod_ps=PROD_PS)
    result.save(out_path)
    return result.converged


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("output")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--T", type=float, default=300.0, help="MD temperature (K)")
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--threads", type=int, default=8, help="OpenMP threads per LAMMPS run")
    ap.add_argument("--lmp", default="lmp")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    todo = []
    for path in sorted(Path(args.folder).glob(f"T=*K_k={args.k}.npz"), key=lambda p: int(re.search(r"T=(\d+)", p.name).group(1))):
        out_path = out / f"{path.stem}_thermalized_{args.T:.0f}K.npz"
        if out_path.exists():
            print(f"{path.name}: already done")
        else:
            todo.append((path, out_path))

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, p, o, args.T, args.threads, args.lmp): p.name for p, o in todo}
        for fut in as_completed(futures):
            try:
                print(f"{futures[fut]}: done, converged = {fut.result()}")
            except Exception as err:
                print(f"{futures[fut]}: FAILED {err}")
