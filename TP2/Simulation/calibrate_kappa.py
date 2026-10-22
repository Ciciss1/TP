"""
Calibration of KAPPA (Graphene.py): a boundary is kept coherent (elastic strain, no dislocation) while
the misfit accumulated at its far end, dtheta r / b lattice vectors, stays below KAPPA.

One grain of size d (distance to its 6 neighbours) rotated by theta in an aligned matrix, built twice by
GrapheneCrystal: all boundaries coherent (KAPPA = inf) and all dislocated (KAPPA = 0). Both are relaxed
with AIREBO (planar stage, then 3D), the far matrix (r > 1.4 d) fixed and the core of the grain
(r < core d) moving rigidly without rotating, so that the misorientation cannot relax away. The crossing
theta* of the two excess energies gives KAPPA = theta* r / b, r mean distance from the centre of the
grain to the far end of its boundaries (side of the hexagon for a regular grain). seed 0 is the regular
hexagon; other seeds jitter the neighbours, rotate the pattern and shift the grain on the lattice.

Usage:
    python calibrate_kappa.py run kappa.jsonl [--n 19,26,32,45] [--theta 2,4,6,8,11,14,17,20,24]
                              [--seeds 4] [--jitter 0.15] [--core 0.25] [--workers 72] [--lmp lmp]
    python calibrate_kappa.py analyse kappa.jsonl [--core 0.25] [--fig kappa.png]
    n : box size in lattice parameters (L = n a, d = L / 3); run appends to the file and skips done tasks.
    Cost of one relaxation on one core: ~1 min for n = 19, ~3 min for n = 32, ~10 min for n = 45
    (2 per (n, seed, theta)).
"""
import os
import sys
import json
import argparse
import tempfile
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "Spe", "Atomical"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import Graphene as G
import CG_Relaxation as CG

A_CC = 1.3967512290507305
A = np.sqrt(3) * A_CC
AIREBO = os.path.join(HERE, "CH.airebo")
LMP = "lmp"


def no_lammps(self, atoms, *args, **kwargs):
    self.Lx = self.Ly = self.L
    self.lammps_info = {}
    return atoms


CG.CGRelaxation.relaxation_CG = no_lammps                     # constrained relaxation done below instead


def geometry(n, seed, jitter):
    '''
    Grain centres: grain 0 and its 6 neighbours at distance d = L / 3 (jittered for seed > 0)
    '''
    L = n * A
    d = L / 3
    rng = np.random.default_rng(seed)
    jitter = jitter if seed else 0.0
    c = np.array([L / 2, L / 2]) + rng.random(2) * A * (seed > 0)
    ang = np.radians(np.arange(6) * 60 + 30 + rng.random() * 60 * (seed > 0)) + rng.uniform(-jitter, jitter, 6)
    r = d * (1 + rng.uniform(-jitter, jitter, 6))
    return L, d, c, np.vstack([c, c + r[:, None] * np.c_[np.cos(ang), np.sin(ang)]])


def build(L, points, theta_deg, kappa):
    theta = np.zeros(len(points))
    theta[0] = np.radians(theta_deg)
    vor = G.voronoi_from_points(L, len(points) / L**2, points, theta)
    vor.areas[0] = 0.0                                    # mean orientation = matrix: same strain F in every run
    owner, _, segs, _ = G.boundary_views(vor.points, vor.theta, vor.all_points, vor.vor)
    segs = segs[owner == 0]
    l = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)
    r = np.linalg.norm(segs - points[0], axis=2).max(axis=1)       # far end of each boundary from the centre
    G.KAPPA = kappa
    cr = G.GrapheneCrystal(vor, a=A_CC, lloyd_kwargs={"method": "lbfgs"})
    return np.asarray(cr.atoms, float), l, r


def relax(atoms, L, types, seed = 1):
    '''
    AIREBO minimisation, type 2 fixed, type 3 rigid translation only; returns energy and positions
    '''
    with tempfile.TemporaryDirectory(prefix="kappa_") as tmp:
        data = os.path.join(tmp, "in.data")
        pos = atoms.copy()
        pos[:, :2] = np.mod(pos[:, :2], L)
        with open(data, "w") as f:
            f.write(f"grain\n\n{len(pos)} atoms\n3 atom types\n\n0 {L} xlo xhi\n0 {L} ylo yhi\n-30 30 zlo zhi\n\n")
            f.write("Masses\n\n1 12.011\n2 12.011\n3 12.011\n\nAtoms\n\n")
            np.savetxt(f, np.column_stack([np.arange(1, len(pos) + 1), types, pos]), fmt="%d %d %.8f %.8f %.8f")
        with open(os.path.join(tmp, "in.lmp"), "w") as f:
            f.write(f"""units metal
atom_style atomic
boundary p p p
read_data {data}
pair_style airebo 3.0 1 1
pair_coeff * * {AIREBO} C C C
group frozen type 2
group core type 3
group free type 1
fix hold frozen setforce 0.0 0.0 0.0
fix rigid core aveforce 0.0 0.0 0.0
fix planar all setforce NULL NULL 0.0
min_style cg
minimize 0.0 1.0e-3 20000 200000
unfix planar
displace_atoms free random 0.0 0.0 0.1 {seed} units box
minimize 0.0 1.0e-3 20000 200000
variable e equal pe
print "EFINAL ${{e}}"
write_dump all custom {tmp}/final.dump id x y z modify sort id
""")
        res = subprocess.run([LMP, "-in", "in.lmp", "-log", "none"], capture_output=True, text=True, cwd=tmp)
        line = [s for s in res.stdout.splitlines() if s.startswith("EFINAL")]
        if not line:
            raise RuntimeError(res.stdout[-2000:] + res.stderr[-2000:])
        return float(line[0].split()[1]), np.loadtxt(os.path.join(tmp, "final.dump"), skiprows=9)[:, 1:4]


def n_defects(pos, L, cut = 1.75):
    '''
    Number of 5 and 7 rings of the relaxed structure (bonds rebuilt from distances)
    '''
    import Properties as P
    xy = np.mod(pos[:, :2], L)
    xy[xy >= L] = 0
    pairs = cKDTree(np.c_[xy, pos[:, 2] + 5e5], boxsize=[L, L, 1e6]).query_pairs(cut, output_type="ndarray")
    try:
        sizes, _ = P.rings(pos, pairs.astype(np.int64), L, L)
    except ValueError:
        return -1
    return int(((sizes == 5) | (sizes == 7)).sum())


def job(task):
    n, seed, theta, mode, jitter, core = task
    L, d, c, points = geometry(n, seed, jitter)
    atoms, l, r_end = build(L, points, theta, {"ref": np.inf, "coh": np.inf, "disl": 0.0}[mode])
    r = np.linalg.norm(atoms[:, :2] - c, axis=1)
    types = np.where(r > 1.4 * d, 2, np.where(r < core * d, 3, 1))
    E, pos = relax(atoms, L, types)
    return dict(n=n, seed=seed, theta=theta, mode=mode, core=core, d=d, l_mean=float(l.mean()),
                r_mean=float(r_end.mean()), N=len(atoms), E=E,
                n57=n_defects(pos, L))


def run(args):
    global LMP
    LMP = args.lmp
    done = set()
    if os.path.isfile(args.out):
        for r in map(json.loads, open(args.out)):
            if r.get("core", 0.25) == args.core:
                done.add((r["n"], r["seed"], r["theta"], r["mode"]))
    ns = [int(x) for x in args.n.split(",")]
    thetas = [float(x) for x in args.theta.split(",")]
    tasks = [(n, 0, 0.0, "ref") for n in ns]
    tasks += [(n, s, t, m) for n in sorted(ns, reverse=True) for s in range(args.seeds) for t in thetas for m in ("coh", "disl")]
    tasks = [t + (args.jitter, args.core) for t in tasks if t not in done]
    print(f"{len(tasks)} relaxations on {args.workers} workers", flush=True)
    with ProcessPoolExecutor(args.workers) as pool, open(args.out, "a") as out:
        futures = {pool.submit(job, t): t for t in tasks}
        for f in as_completed(futures):
            try:
                r = f.result()
            except Exception as e:
                print(f"failed {futures[f]}: {e}", flush=True)
                continue
            out.write(json.dumps(r) + "\n")
            out.flush()
            print(f"n={r['n']} seed={r['seed']} theta={r['theta']:g} {r['mode']}: E = {r['E']:.3f} eV, {r['n57']} rings 5/7", flush=True)


def crossing(theta, dE, tol = 0.5):
    '''
    Angle where E_coh - E_disl becomes positive: first point above tol (eV, below it the two
    constructions are the same: no dislocation yet), linearly interpolated with the previous one.
    inf if the coherent grain is always the cheapest (theta* above the largest angle)
    '''
    k = np.flatnonzero(dE > tol)
    if not len(k):
        return np.inf
    k = k[0]
    if k == 0:
        return theta[0]
    return theta[k - 1] + (theta[k] - theta[k - 1]) * dE[k - 1] / (dE[k - 1] - dE[k])


def analyse(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in map(json.loads, open(args.out)) if r.get("core", 0.25) == args.core]
    e_bulk = {r["n"]: r["E"] / r["N"] for r in rows if r["mode"] == "ref"}
    data = {}
    for r in rows:
        if r["mode"] != "ref" and r["n"] in e_bulk:
            data.setdefault((r["n"], r["seed"]), {}).setdefault(r["theta"], {})[r["mode"]] = r

    fig, (ax_e, ax_k) = plt.subplots(1, 2, figsize=(12, 4.5))
    cmap = plt.get_cmap("viridis")
    ns = sorted({n for n, _ in data})
    kappas = {}
    print(f"{'n':>3s} {'seed':>4s} {'d (AA)':>7s} {'r (AA)':>7s} {'theta*':>7s} {'kappa':>6s}")
    for (n, seed), D in sorted(data.items()):
        th = np.array(sorted(t for t in D if {"coh", "disl"} <= set(D[t])))
        if len(th) < 2:
            continue
        exc = lambda m: np.array([D[t][m]["E"] - D[t][m]["N"] * e_bulk[n] for t in th])
        dE = exc("coh") - exc("disl")
        first = D[th[0]]["coh"]
        l, d = first.get("r_mean", first["l_mean"]), first["d"]   # regular hexagon: r = l
        ts = crossing(th, dE)
        kappa = np.radians(ts) * l / A
        kappas.setdefault(n, []).append(kappa)
        print(f"{n:3d} {seed:4d} {d:7.1f} {l:7.1f} {ts:7.2f} {kappa:6.2f}" if np.isfinite(ts) else
              f"{n:3d} {seed:4d} {d:7.1f} {l:7.1f}  > {th[-1]:g}  > {np.radians(th[-1]) * l / A:.2f}")
        col = cmap(ns.index(n) / max(1, len(ns) - 1))
        ax_e.plot(np.radians(th) * l / A, dE, "o-", color=col, lw=0.8, ms=3, label=f"d = {d:.0f} Å" if seed == 0 else None)
    ax_e.axhline(0, color="k", lw=0.8)
    ax_e.set_yscale("symlog", linthresh=5)
    ax_e.set(xlabel=r"$n = \theta\, r / b$", ylabel=r"$E_{coh} - E_{disl}$ (eV)", title="> 0 : dislocations moins chères")
    ax_e.legend(fontsize=8)

    if not kappas:
        raise SystemExit("no crystal with both constructions at two angles or more yet")
    all_k = np.concatenate([np.array(k) for k in kappas.values()])
    all_k = all_k[np.isfinite(all_k)]
    for n in ns:
        k_all = np.array(kappas.get(n, []))
        k = k_all[np.isfinite(k_all)]
        d = n * A / 3
        if len(k):
            ax_k.errorbar(d, k.mean(), k.std(), fmt="o", capsize=3, color="C0")
        print(f"d = {d:5.1f} AA : kappa = {k.mean() if len(k) else np.nan:.2f} +- {k.std() if len(k) else np.nan:.2f} "
              f"({len(k)} crystals, {len(k_all) - len(k)} more where the coherent grain always wins)")
    if len(all_k):
        ax_k.axhline(all_k.mean(), color="C1", ls="--", label=f"κ = {all_k.mean():.2f} ± {all_k.std():.2f}")
        print(f"all : KAPPA = {all_k.mean():.2f} +- {all_k.std():.2f} (standard error {all_k.std() / np.sqrt(len(all_k)):.2f})")
    ax_k.set(xlabel="taille du grain d (Å)", ylabel=r"$\kappa = \theta^* r / b$", title="seuil cohérent / disloqué")
    ax_k.legend(fontsize=8)
    for ax in (ax_e, ax_k):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.fig, dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "analyse"])
    ap.add_argument("out", help="jsonl file of the relaxations")
    ap.add_argument("--n", default="19,26,32,45", help="box sizes in lattice parameters (d = n a / 3)")
    ap.add_argument("--theta", default="2,4,6,8,11,14,17,20,24", help="misorientations (deg)")
    ap.add_argument("--seeds", type=int, default=4, help="grain shapes per size (0 = regular hexagon)")
    ap.add_argument("--jitter", type=float, default=0.15, help="relative jitter of the neighbours for seed > 0")
    ap.add_argument("--core", type=float, default=0.25, help="radius of the core held at theta, in units of d")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--lmp", default="lmp", help="LAMMPS executable")
    ap.add_argument("--fig", default="kappa.png")
    args = ap.parse_args()
    run(args) if args.action == "run" else analyse(args)
