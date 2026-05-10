import os
import sys
import time
import numpy as np
from tqdm import tqdm

from Voronoi import PeriodicVoronoi
from MonteCarlo import monte_carlo
from Graphene import GrapheneCrystal, load_crystal

def w(message = ""):
    tqdm.write(message)

def run_one_temp(args, outer_bar: tqdm):
    T, L, epsilon, rho, alpha, beta_RS, n_monte_carlo, output_dir = args

    sim_dir = os.path.join(output_dir, f"eps_{epsilon}/L_{L}/rho_{rho}/T_{T}/")
    os.makedirs(sim_dir, exist_ok=True)

    outer_bar.set_postfix_str(f"T={T}  1/3 Voronoi")
    t0 = time.perf_counter()
    vor = PeriodicVoronoi(L, rho)
    w(f"  [T={T:<5}]  Voronoi  {vor.N} grains  {time.perf_counter() - t0:5.2f} s")

    outer_bar.set_postfix_str(f"T={T}  2/3 Monte Carlo")
    t0 = time.perf_counter()

    thetas, energy_history = monte_carlo(
        vor.theta, vor.adj_i, vor.adj_j, vor.adj_length,
        beta=1.0 / T, epsilon=epsilon, rho=rho,
        alpha=alpha, beta_RS=beta_RS,
        n_sweeps=n_monte_carlo, use_tqdm=False,
    )

    vor.theta = thetas
    w(f"  [T={T:<5}]  Monte Carlo  E={energy_history[-1]:+.4f} eV   {time.perf_counter()-t0:5.2f}s")

    outer_bar.set_postfix_str(f"T={T}  3/3 Crystal")
    t0 = time.perf_counter()
    crystal = GrapheneCrystal(vor)
    w(f"  [T={T:<5}]  Crystal      {len(crystal.atoms)} atoms          {time.perf_counter()-t0:5.2f}s")
    
    save_path = os.path.join(sim_dir, "Crystal.npz")
    crystal.save_crystal(save_path)
    w(f"  [T={T:<5}]  Saved  →  {save_path}")

def load_parameters(path):
    params = {}
    with open(path, "r") as f:
        exec(f.read(), {}, params)

    required_keys = [
        "output_dir",
        "epsilon",
        "alpha",
        "beta_RS",
        "L",
        "rho",
        "n_monte_carlo",
        "T"
    ]

    missing_keys = [key for key in required_keys if key not in params]
    if missing_keys:
        raise KeyError(f"Missing required parameters: {missing_keys}")
    
    T_raw = params["T"]
    if isinstance(T_raw, (int, float)):
        params["T"] = [float(T_raw)]
    else:
        params["T"] = list(T_raw)

    return params

def main():
    param_file = sys.argv[1] if len(sys.argv) > 1 else "parameters.txt"

    params = load_parameters(param_file)

    output_dir = params["output_dir"]
    epsilon = params["epsilon"]
    alpha = params["alpha"]
    beta_RS = params["beta_RS"]
    L = params["L"]
    rho = params["rho"]
    n_monte_carlo = params["n_monte_carlo"]
    Ts = params["T"]

    os.makedirs(output_dir, exist_ok=True)

    w(f"  param file: {param_file}")
    w(f"  output dir: {output_dir}")
    w(f"  epsilon: {epsilon}  alpha: {alpha}  beta_RS: {beta_RS}  L: {L}  rho: {rho}  n_MC: {n_monte_carlo}  T: {Ts}")

    t_total = time.perf_counter()

    with tqdm(Ts, desc="  Progress", unit="T", position=0, leave=True, dynamic_ncols=True, colour = "blue") as outer_bar: 
        
        for T in outer_bar:
            t_sim = time.perf_counter()
            
            run_one_temp(
                (T, L, epsilon, rho, alpha, beta_RS, n_monte_carlo, output_dir),
                outer_bar
            )
            w(f"  [T={T:<5}]  ✓ done in {time.perf_counter()-t_sim:.1f}s\n")

    print(f"\n  All done — {time.perf_counter()-t_total:.1f}s total\n")

if __name__ == "__main__":
    main()