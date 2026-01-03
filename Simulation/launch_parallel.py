from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import MC_XY as mc_xy
import MC_ln as mc_ln


def run_one_sim_XY(geom, T, J, n_therm, n_meas, overrelax_interval, meas_interval):
    path = f"XY/J{J}/L{geom.L}/T{T:.3e}"
    sim = mc_xy.Simulation(geom, T, J, n_therm, n_meas, overrelax_interval, meas_interval, use_tqdm=False)
    sim.run(path)

def run_one_sim_ln(geom, T, epsilon, gamma, A, n_therm, n_meas, overrelax_interval, meas_interval):
    path = f"Model/epsilon{epsilon}_gamma{gamma}/L{geom.L}/T{T:.3e}"
    sim = mc_ln.Simulation(geom, T, epsilon, gamma, A, n_therm, n_meas, overrelax_interval, meas_interval, use_tqdm=False)
    sim.run(path)

if __name__ == "__main__":
    params = {}
    with open("parameters.txt") as f:
        exec(f.read(), {}, params)

    model = params["model"]
    L = params["L"]
    T = params["T"]
    J = params["J"]
    epsilon = params["epsilon"]
    gamma = params["gamma"]
    A = params["A"]
    rho = params["rho"]
    n_therm = params["n_therm"]
    n_meas = params["n_meas"]
    overrelax_interval = params["overrelax_interval"]
    meas_interval = params["meas_interval"]
    coeur = params["coeur"]

    if model == "XY":
        geom = mc_xy.Geometry(L, rho)
    elif model == "ln":
        geom = mc_ln.Geometry(L, rho)
    else:
        raise ValueError(f"Unknown model: {model}")

    n_workers = min(coeur, len(T))

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        
        for temp in T:
            if model == "XY":
                futures.append(executor.submit(
                    run_one_sim_XY,
                    geom,
                    temp,
                    J,
                    n_therm,
                    n_meas,
                    overrelax_interval,
                    meas_interval
                ))
            elif model == "ln":
                futures.append(executor.submit(
                    run_one_sim_ln,
                    geom,
                    temp,
                    epsilon,
                    gamma,
                    A,
                    n_therm,
                    n_meas,
                    overrelax_interval,
                    meas_interval
                ))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Simulations"):
            future.result()