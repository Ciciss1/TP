from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from tqdm import tqdm
import MC_XY as mc

def run_one_sim(L, T, J, rho, n_therm, n_meas, overrelax_interval, meas_interval):

    path = f"XY/J{J}/L{L}/T{T:.3e}"
    sim = mc.Simulation(L, T, J, rho, n_therm=n_therm, n_meas=n_meas, overrelax_interval=overrelax_interval, meas_interval=meas_interval)
    sim.run(path)

if __name__ == "__main__":
    L = 128
    T = [10**(-5), 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 2.0]
    J = 1.0
    rho = 1.0

    n_therm = 10**5
    n_meas = 10**4
    overrelax_interval = n_therm // 100
    meas_interval = n_meas // 10

    n_workers = min(4, len(T))

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for temp in T:
            futures.append(
                executor.submit(
                    run_one_sim,
                    L,
                    temp,
                    J,
                    rho,
                    n_therm,
                    n_meas,
                    overrelax_interval,
                    meas_interval,
                )
            )

        for future in tqdm(as_completed(futures), total=len(futures), desc="Simulations"):
            future.result()