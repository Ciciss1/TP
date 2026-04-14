import os

from Voronoi import PeriodicVoronoi
from MonteCarlo import monte_carlo
from Graphene import GrapheneCrystal, load_crystal
from Run_anneal import run_anneal

if __name__ == "__main__":
    params = {}
    with open("parameters.txt") as f:
        exec(f.read(), {}, params)

    output_dir = params["output_dir"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    epsilon = params["epsilon"]
    alpha = params["alpha"]
    beta_RS = params["beta_RS"]
    T = params["T"]

    L = params["L"]
    rho = params["rho"]
    n_monte_carlo = params["n_monte_carlo"]

    unfreeze_dist = params["unfreeze_dist"]
    T_start = params["T_start"]
    T_end = params["T_end"]
    n_anneal = params["n_anneal"]
    n_min = params["n_min"]
    damping = params["damping"]
    dump_every = params["dump_every"]
    n_quench = params["n_quench"]
    n_iter = params["n_iter"]

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

    initial_crystal = GrapheneCrystal(vor)

    initial_crystal.save_crystal(os.path.join(output_dir, "initial_crystal.npz"))

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
        out_dir = output_dir,
    )