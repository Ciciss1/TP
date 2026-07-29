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

def run_one_run(run_idx, args, outer_bar: tqdm):
    L, epsilon, gamma, phi_s, rho, alpha, beta_RS, n_monte_carlo, T_list, output_dir = args

    phi_s2 = phi_s * phi_s
    phi_s4 = phi_s2 * phi_s2

    T_sorted = sorted(T_list, reverse=True)

    outer_bar.set_postfix_str(f"run {run_idx + 1} Voronoi")
    vor = PeriodicVoronoi(L, rho)
    theta = vor.theta.copy()

    for T in T_sorted:
        sim_dir = os.path.join(output_dir, f"eps_{epsilon}/L_{L}/rho_{rho}/T_{T}/")
        os.makedirs(sim_dir, exist_ok=True)

        outer_bar.set_postfix_str(f"run {run_idx + 1} : T={T} Monte Carlo")

        thetas, energy_history, misor_history = monte_carlo(
            theta, vor.adj_i, vor.adj_j, vor.adj_length, vor.areas, beta=1.0 / T, epsilon=epsilon, gamma=gamma, phi_s2=phi_s2, phi_s4=phi_s4, alpha=alpha, beta_RS=beta_RS, n_sweeps=n_monte_carlo, use_tqdm=False
        )
        w(f"  [run {run_idx + 1} T={T}]  Monte Carlo  E={energy_history[-1]:+.4f} eV")

        vor.theta = thetas.copy()

        outer_bar.set_postfix_str(f"run {run_idx + 1} : T={T} Crystal")

        crystal = GrapheneCrystal(vor)
        w(f"  [run {run_idx + 1} T={T}]  Crystal      {len(crystal.atoms)} atoms")

        save_path = os.path.join(sim_dir, f"Crystal_{run_idx + 1}.npz")
        crystal.save_crystal(save_path)
        w(f"  [run {run_idx + 1} T={T:<5}]  Saved  →  {save_path}")

def load_parameters(path):
    params = {}
    with open(path, "r") as f:
        exec(f.read(), {}, params)

    required_keys = [
        "output_dir",
        "epsilon",
        "gamma",
        "phi_s",
        "alpha",
        "beta_RS",
        "L",
        "rho",
        "n_monte_carlo",
        "n_runs",
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
    gamma = params["gamma"]
    phi_s = params["phi_s"]
    alpha = params["alpha"]
    beta_RS = params["beta_RS"]
    L = params["L"]
    rho = params["rho"]
    n_monte_carlo = params["n_monte_carlo"]
    n_runs = params["n_runs"]
    Ts = params["T"]

    os.makedirs(output_dir, exist_ok=True)

    w(f"  param file: {param_file}")
    w(f"  output dir: {output_dir}")
    w(f"  epsilon: {epsilon}  gamma: {gamma}  phi_s: {phi_s}  alpha: {alpha}  beta_RS: {beta_RS}  L: {L}  rho: {rho}  n_MC: {n_monte_carlo}  n_runs: {n_runs}  T: {Ts}")

    with tqdm(range(n_runs), desc="  Progress", unit="run", position=0, leave=True, dynamic_ncols=True, colour = "blue") as outer_bar: 
        
        for run_idx in outer_bar:            
            run_one_run(run_idx, (L, epsilon, gamma, phi_s, rho, alpha, beta_RS, n_monte_carlo, Ts, output_dir), outer_bar)

if __name__ == "__main__":
    main()