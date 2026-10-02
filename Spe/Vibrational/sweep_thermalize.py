import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from thermalize import Thermalizer

RUSLAN_DIR = Path("/scratch/escarmel/crystals_2D/")
FOLDERS = ["eps=0/L=272.12/rho=0.00135"]
K_INDEX = 1

OUTPUT_DIR = Path("/scratch/escarmel/thermalize_results_2D/")

T_KELVIN = 300
EQUIL_CHECK_PS = 2.0
EQUIL_MAX_PS = 20.0
PE_TOL = 0.005
PROD_PS = 5.0
AIREBO_FILE = "CH.airebo"

THREADS_PER_WORKER = 8
N_WORKERS = 9


def find_crystals(folder: Path, k_index: int):
    pattern = re.compile(rf"T=(\d+)K?_k={k_index}\.npz$")
    found = []
    for f in folder.glob(f"T=*_k={k_index}.npz"):
        m = pattern.match(f.name)
        if m:
            found.append((int(m.group(1)), f))
    return sorted(found, key=lambda x: x[0])


def process_one(T_mc, path, out_path):
    thermalizer = Thermalizer(airebo_file=AIREBO_FILE, n_threads=THREADS_PER_WORKER)

    data = np.load(path)
    if "atoms" in data.files:
        atoms = data["atoms"]
        Lx = Ly = float(np.ravel(data["L"])[0])
    else:
        atoms, (Lx, Ly) = data["xyz"], data["lattice"]

    result = thermalizer.run(
        atoms, Lx, Ly, T=T_KELVIN,
        equil_check_ps=EQUIL_CHECK_PS, equil_max_ps=EQUIL_MAX_PS, pe_tol=PE_TOL,
        prod_ps=PROD_PS,
    )
    result.save(str(out_path))
    return T_mc, result.converged


if __name__ == "__main__":
    for folder_name in FOLDERS:
        src_folder = RUSLAN_DIR / folder_name
        dst_folder = OUTPUT_DIR / folder_name
        dst_folder.mkdir(parents=True, exist_ok=True)

        crystals = find_crystals(src_folder, K_INDEX)
        print(f"\n### {folder_name} : {len(crystals)} temperatures (k={K_INDEX}) ###")

        todo = []
        for T_mc, path in crystals:
            out_path = dst_folder / f"T={T_mc}_k={K_INDEX}_thermalized_{int(T_KELVIN)}K.npz"
            if out_path.exists():
                print(f"  T={T_mc} : deja fait, skip")
                continue
            todo.append((T_mc, path, out_path))

        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(process_one, *args): args[0] for args in todo}
            for future in as_completed(futures):
                T_mc = futures[future]
                try:
                    T_mc, converged = future.result()
                    print(f"  T={T_mc} : termine, converged={converged}")
                except Exception as e:
                    print(f"  T={T_mc} : ECHEC — {e}")
