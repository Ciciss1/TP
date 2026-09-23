import re
from pathlib import Path

import numpy as np
from thermalize import Thermalizer

RUSLAN_DIR = Path("../../TP2/Simulation/spe")
FOLDERS = ["eps=0/L=272.12/rho=0.00135"]
K_INDEX = 1

OUTPUT_DIR = Path("results")

T_KELVIN = 300       # temperature physique de thermalisation AIREBO
EQUIL_CHECK_PS = 2.0
EQUIL_MAX_PS = 20.0
PE_TOL = 0.005
PROD_PS = 5.0
AIREBO_FILE = "CH.airebo"
N_THREADS = 6


def find_crystals(folder: Path, k_index: int):
    """List (T_mc, path) for all files T=...[K]_k={k_index}.npz in the folder, sorted by T."""
    pattern = re.compile(rf"T=(\d+)K?_k={k_index}\.npz$")      # K optionnel : nouveaux fichiers et Ruslan
    found = []
    for f in folder.glob(f"T=*_k={k_index}.npz"):
        m = pattern.match(f.name)
        if m:
            found.append((int(m.group(1)), f))
    return sorted(found, key=lambda x: x[0])


if __name__ == "__main__":
    thermalizer = Thermalizer(airebo_file=AIREBO_FILE, n_threads=N_THREADS)

    for folder_name in FOLDERS:
        src_folder = RUSLAN_DIR / folder_name
        dst_folder = OUTPUT_DIR / folder_name
        dst_folder.mkdir(parents=True, exist_ok=True)

        crystals = find_crystals(src_folder, K_INDEX)
        print(f"\n### {folder_name} : {len(crystals)} temperatures (k={K_INDEX}) ###")

        for T_mc, path in crystals:
            out_path = dst_folder / f"T={T_mc}_k={K_INDEX}_thermalized_{int(T_KELVIN)}K.npz"
            if out_path.exists():
                print(f"  T={T_mc} : deja fait, skip")
                continue

            data = np.load(path)
            if "atoms" in data.files:                     # nouveaux fichiers (Run_Simulation)
                atoms = data["atoms"]
                Lx = Ly = float(np.ravel(data["L"])[0])
            else:                                         # fichiers de Ruslan
                atoms, (Lx, Ly) = data["xyz"], data["lattice"]
            print(f"  T={T_mc} : {len(atoms)} atomes, boite {Lx:.1f} x {Ly:.1f} A ...")

            result = thermalizer.run(
                atoms, Lx, Ly, T=T_KELVIN,
                equil_check_ps=EQUIL_CHECK_PS, equil_max_ps=EQUIL_MAX_PS, pe_tol=PE_TOL,
                prod_ps=PROD_PS,
            )
            result.save(str(out_path))
            print(f"    -> converged={result.converged}, sauve dans {out_path}")
