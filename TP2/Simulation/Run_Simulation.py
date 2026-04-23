import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from Voronoi import PeriodicVoronoi
from MonteCarlo import monte_carlo
from Graphene import GrapheneCrystal, load_crystal
from Run_anneal import run_anneal

def run_simulation(args):
    T, L, epsilon, rho, alpha, beta_RS, n_monte_carlo, output_dir = args
    output_dir = os.path.join(output_dir, f"L{L}/epsilon_{epsilon}/rho_{rho}/T_{T}/")
    os.makedirs(output_dir, exist_ok=True)

    vor = PeriodicVoronoi(L, rho)

    thetas, _ = monte_carlo(
                    vor.theta,
                    vor.adj_i,
                    vor.adj_j,
                    vor.adj_length,
                    1/T,
                    epsilon,
                    rho,
                    alpha,
                    beta_RS,
                    n_monte_carlo,
                    use_tqdm=False
    )

    vor.theta = thetas

    initial_crystal = GrapheneCrystal(vor)

    initial_crystal.save_crystal(os.path.join(output_dir, "initial_crystal.npz"))

    initial_crystal.save_crystal(os.path.join(output_dir, "final_crystal.npz"))

if __name__ == "__main__":
    params = {}
    with open("parameters.txt") as f:
        exec(f.read(), {}, params)

    core = params["core"]
    multi_threading = params["multi_threading"]

    epsilon = params["epsilon"]
    alpha = params["alpha"]
    beta_RS = params["beta_RS"]
    Ts = params["T"]

    L = params["L"]
    rho = params["rho"]
    n_monte_carlo = params["n_monte_carlo"]

    run_anneal_bool = params["run_anneal_bool"]
    unfreeze_dist = params["unfreeze_dist"]
    T_start = params["T_start"]
    T_end = params["T_end"]
    n_anneal = params["n_anneal"]
    n_min = params["n_min"]
    damping = params["damping"]
    dump_every = params["dump_every"]
    n_quench = params["n_quench"]
    n_iter = params["n_iter"]

    if multi_threading:
        n_workers = min(core, len(Ts))

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []

            for T in Ts:
                futures.append(executor.submit(
                    run_simulation,
                    (T, L, epsilon, rho, alpha, beta_RS, n_monte_carlo, params["output_dir"])
                ))

            for future in tqdm(as_completed(futures), total=len(futures), desc="Simulations"):
                future.result()  

    else:
        for T in Ts:

            output_dir = params["output_dir"]
            output_dir = os.path.join(output_dir, f"L{L}/epsilon_{epsilon}/rho_{rho}/T_{T}/")
            os.makedirs(output_dir, exist_ok=True)

            print("Creating Voronoi lattice...")

            vor = PeriodicVoronoi(L, rho)

            thetas, _ = monte_carlo(
                            vor.theta,
                            vor.adj_i,
                            vor.adj_j,
                            vor.adj_length,
                            1/T,
                            epsilon,
                            rho,
                            alpha,
                            beta_RS,
                            n_monte_carlo,
            )

            vor.theta = thetas

            print("Creating graphene crystal...")

            initial_crystal = GrapheneCrystal(vor)

            initial_crystal.save_crystal(os.path.join(output_dir, "initial_crystal.npz"))

            if run_anneal_bool:
                run_anneal(
                    unfreeze_dist,
                    T_start,
                    T_end,
                    n_anneal,
                    n_min,
                    damping,
                    dump_every,
                    n_quench,
                    n_iter,
                    out_dir = output_dir
                )
            else:
                print("Skipping annealing step")
                initial_crystal.save_crystal(os.path.join(output_dir, "final_crystal.npz"))