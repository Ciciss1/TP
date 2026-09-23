import os
import sys
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from numba import njit
from tqdm import tqdm

from Voronoi import PeriodicVoronoi
from MonteCarlo import monte_carlo
from Graphene import GrapheneCrystal, voronoi_from_points

A_CC = 1.3967512290507305           # AIREBO equilibrium C-C distance (AA)
KB_EV = 8.61732814974056E-05        # Boltzmann constant (eV/K)


def w(message = ""):
    tqdm.write(message)


@njit
def _seed_numba(seed):
    np.random.seed(seed)


def seed_all(seed):
    '''
    Seed both the NumPy global RNG (substrate sites) and the Numba RNG (Metropolis moves)
    '''
    seed = int(seed) % (2**32 - 1)
    np.random.seed(seed)
    _seed_numba(seed)


def derived_seed(base_seed, *keys):
    return int(np.random.SeedSequence([base_seed, *keys]).generate_state(1)[0] % (2**31 - 2)) + 1


def sim_dir(p):
    '''
    output_dir/eps=.../L=.../rho=.../  (one folder per set of Hamiltonian/lattice parameters)
    '''
    return os.path.join(p["output_dir"], f"eps={p['epsilon']:.4g}", f"L={p['L']}", f"rho={p['rho']:g}")


def sim_path(p, T, run_idx):
    '''
    All temperatures and runs in the same folder:  T=<T in K>K_k=<run>.npz
    '''
    T_K = T / KB_EV
    return os.path.join(sim_dir(p), f"T={T_K:.0f}K_k={run_idx + 1}.npz")


# ─────────────────────────────── Phase 1 : Monte Carlo annealing ───────────────────────────────

def anneal_one_run(run_idx, p):
    '''
    Draw one Voronoi tessellation and anneal the grain orientations from the highest to the lowest
    temperature. Returns one task per temperature for the crystal construction.
    '''
    run_seed = derived_seed(p["seed"], run_idx)
    seed_all(run_seed)

    phi_s2 = p["phi_s"] ** 2
    phi_s4 = phi_s2 ** 2

    vor = PeriodicVoronoi(p["L"], p["rho"])
    theta = vor.theta.copy()

    tasks = []
    for t_idx, T in enumerate(sorted(p["T"], reverse=True)):
        t0 = time.perf_counter()
        theta, energy_history, misor_history = monte_carlo(
            theta, vor.adj_i, vor.adj_j, vor.adj_length, vor.areas, beta=1.0 / T,
            epsilon=p["epsilon"], gamma=p["gamma"], phi_s2=phi_s2, phi_s4=phi_s4,
            alpha=p["alpha"], beta_RS=p["beta_RS"], n_sweeps=p["n_monte_carlo"], use_tqdm=False
        )
        n_done = len(energy_history)
        converged = n_done < p["n_monte_carlo"]
        tail = slice(-min(1000, n_done), None)

        w(f"  [run {run_idx + 1} T={T:.4f} eV]  MC  {n_done} sweeps  "
          f"{'converged' if converged else 'NOT converged'}  E={energy_history[-1]:+.4f} eV  "
          f"({time.perf_counter() - t0:.1f} s)")

        save_path = sim_path(p, T, run_idx)
        tasks.append({
            "run_idx": run_idx,
            "T": T,
            "L": p["L"],
            "rho": p["rho"],
            "points": vor.points.copy(),
            "theta": theta.copy(),
            "save_path": save_path,
            "lammps_seed": derived_seed(p["seed"], run_idx, t_idx),
            "n_threads": p["n_threads"],
            "lloyd_method": p["lloyd_method"],
            "meta": {
                "T_eV": T,
                "T_K": T / KB_EV,
                "run": run_idx + 1,
                "base_seed": p["seed"],
                "run_seed": run_seed,
                "mc_converged": converged,
                "mc_sweeps": n_done,
                "mc_energy": float(np.mean(energy_history[tail])),
                "mc_misorientation": float(np.mean(misor_history[tail])),
                "epsilon": p["epsilon"], "gamma": p["gamma"], "phi_s": p["phi_s"],
                "alpha": p["alpha"], "beta_RS": p["beta_RS"],
            },
        })
    return tasks


# ─────────────────────────────── Phase 2 : crystal construction ───────────────────────────────

def build_crystal_task(task):
    '''
    Build, relax and save one crystal (runs in a worker process)
    '''
    t0 = time.perf_counter()
    try:
        vor = voronoi_from_points(task["L"], task["rho"], task["points"], task["theta"])
        crystal = GrapheneCrystal(
            vor, a = A_CC,
            lloyd_kwargs = {"method": task["lloyd_method"]},
            lammps_kwargs = {"n_threads": task["n_threads"], "seed": task["lammps_seed"]},
        )
        crystal.save_crystal(task["save_path"], **task["meta"])
        return {
            "ok": True,
            "task": task,
            "n_atoms": len(crystal.atoms),
            "timings": crystal.timings,
            "checks": crystal.checks,
            "lammps": crystal.lammps_info.get("criterion", []),
            "time": time.perf_counter() - t0,
        }
    except Exception as err:
        return {"ok": False, "task": task, "error": f"{type(err).__name__}: {err}", "time": time.perf_counter() - t0}


def report(result):
    task = result["task"]
    tag = f"[run {task['run_idx'] + 1} T={task['T']:.4f} eV]"
    if not result["ok"]:
        w(f"  {tag}  FAILED after {result['time']:.0f} s  {result['error']}")
        return
    tm = result["timings"]
    chk = result["checks"]
    topo = "ok" if (chk["atoms_equal_2_generators"] and chk["all_degree_3"]) else "TOPOLOGY PROBLEM"
    w(f"  {tag}  {result['n_atoms']} atoms  topo {topo}  bonds [{chk['bond_min']:.2f}, {chk['bond_max']:.2f}] AA  "
      f"Lloyd {tm['lloyd']:.0f} s  LAMMPS {tm['lammps']:.0f} s  stop: {', '.join(result['lammps'])}")
    w(f"  {tag}  saved → {task['save_path']}")


# ─────────────────────────────── Parameters ───────────────────────────────

def load_parameters(path):
    params = {}
    with open(path, "r") as f:
        exec(f.read(), {}, params)

    required_keys = ["output_dir", "epsilon", "gamma", "phi_s", "alpha", "beta_RS", "L", "rho", "n_monte_carlo", "n_runs", "T"]
    missing_keys = [key for key in required_keys if key not in params]
    if missing_keys:
        raise KeyError(f"Missing required parameters: {missing_keys}")

    T_raw = params["T"]
    T_list = [float(T_raw)] if isinstance(T_raw, (int, float)) else list(T_raw)
    params["T"] = [T * 1000 * KB_EV for T in T_list]           # 10^3 K -> eV

    # Optional parameters
    n_cpu = os.cpu_count() or 1
    params.setdefault("n_threads", 2)                            # OpenMP threads per LAMMPS process
    params.setdefault("n_workers", max(1, (n_cpu - 1) // params["n_threads"]))
    params.setdefault("seed", None)
    params.setdefault("skip_existing", False)
    params.setdefault("lloyd_method", "lbfgs")
    if params["seed"] is None:
        params["seed"] = int(np.random.SeedSequence().entropy % (2**32))

    return params


def main():
    param_file = sys.argv[1] if len(sys.argv) > 1 else "parameters.txt"
    p = load_parameters(param_file)
    out_dir = sim_dir(p)
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(param_file, os.path.join(out_dir, "parameters.txt"))     # trace of the parameters used

    w(f"  param file: {param_file}")
    w(f"  output dir: {out_dir}")
    w(f"  epsilon: {p['epsilon']}  gamma: {p['gamma']}  phi_s: {p['phi_s']}  alpha: {p['alpha']}  beta_RS: {p['beta_RS']}")
    w(f"  L: {p['L']}  rho: {p['rho']}  n_MC: {p['n_monte_carlo']}  n_runs: {p['n_runs']}  T (eV): {[round(T, 4) for T in p['T']]}")
    w(f"  seed: {p['seed']}  workers: {p['n_workers']} x {p['n_threads']} LAMMPS threads  Lloyd: {p['lloyd_method']}")

    n_total = p["n_runs"] * len(p["T"])
    t_start = time.perf_counter()

    # The Monte Carlo of run k (main process) overlaps with the crystal construction of run k-1 (workers)
    with ProcessPoolExecutor(max_workers=p["n_workers"]) as pool, \
         tqdm(total=n_total, desc="  Crystals", unit="crystal", position=0, leave=True, dynamic_ncols=True, colour="blue") as bar:

        futures = []
        for run_idx in range(p["n_runs"]):
            tasks = anneal_one_run(run_idx, p)
            for task in tasks:
                if p["skip_existing"] and os.path.isfile(task["save_path"]):
                    w(f"  [run {run_idx + 1} T={task['T']:.4f} eV]  exists, skipped")
                    bar.update(1)
                    continue
                futures.append(pool.submit(build_crystal_task, task))

            for fut in [f for f in futures if f.done()]:
                report(fut.result())
                bar.update(1)
                futures.remove(fut)

        for fut in as_completed(futures):
            report(fut.result())
            bar.update(1)

    w(f"  Total time: {(time.perf_counter() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()