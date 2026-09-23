"""
benchmark_speed.py

Mesure la vitesse reelle de LAMMPS/AIREBO sur TON systeme (CPU, threads,
taille du cristal) via un court run de calibration, puis extrapole le
temps total attendu pour la pipeline complete (equilibration + production).

Necessaire car AIREBO est un potentiel couteux dont la vitesse varie
enormement selon le hardware -- pas de chiffre generique fiable.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from thermalize import Thermalizer, C_MASS, Z_VACUUM, _BACKENDS

INPUT_FILE = "../Ruslan/s_Ng=400/T=10000_k=1.npz"  # <- adapte
AIREBO_FILE = "CH.airebo"
N_THREADS = 6
BACKEND = "omp"

CALIBRATION_STEPS = 200   # pas de calibration (rapide, juste pour mesurer le debit)
DT_PS = 0.001             # 1 fs

# duree totale que tu comptes lancer pour de vrai (doit matcher run_thermalize.py)
EQUIL_MAX_PS = 20.0
PROD_PS = 5.0


def build_calibration_script(data_file, airebo_abs, backend_name, n_threads, steps) -> str:
    backend = _BACKENDS[backend_name]
    package_line = backend["package"].format(n_threads=n_threads)
    return f"""
    units metal
    atom_style atomic
    boundary p p p

    {package_line}
    read_data {data_file}

    pair_style airebo/{backend['pair_suffix']} 3.0 1 1
    pair_coeff * * {airebo_abs} C

    timestep {DT_PS}
    velocity all create 300.0 12345 mom yes rot no dist gaussian

    thermo 50
    fix nve_bench all nve
    run {steps}
    """


if __name__ == "__main__":
    data = np.load(INPUT_FILE)
    if "atoms" in data.files:
        atoms = data["atoms"]
        L = float(data["L"][0])
        Lx, Ly = L, L  # boite carree, nouveau format
    else:
        atoms = data["xyz"]
        Lx, Ly = data["lattice"]  # ancien format
    n_atoms = len(atoms)
    print(f"{n_atoms} atomes, boite {Lx:.2f} x {Ly:.2f} A")

    airebo_abs = str(Path(AIREBO_FILE).resolve())

    with tempfile.TemporaryDirectory(prefix="lammps_bench_") as tmp:
        tmp = Path(tmp)
        data_file = tmp / "input.data"
        in_file = tmp / "bench.in"

        Thermalizer._write_data(atoms, Lx, Ly, data_file)
        in_file.write_text(build_calibration_script(
            data_file, airebo_abs, BACKEND, N_THREADS, CALIBRATION_STEPS
        ))

        backend = _BACKENDS[BACKEND]
        cmd_extra = [a.format(n_threads=N_THREADS) for a in backend["cmd_extra"]]

        print(f"Calibration : {CALIBRATION_STEPS} pas NVE sur le vrai systeme...")
        result = subprocess.run(
            ["lmp", "-in", str(in_file), "-log", "none", *cmd_extra],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            sys.exit(f"LAMMPS a echoue :\n{result.stdout}\n{result.stderr}")

    # Parse la ligne "Loop time of X on N procs for M steps ..."
    match = re.search(r"Loop time of ([\d.]+) on \d+ procs for (\d+) steps", result.stdout)
    if not match:
        sys.exit("Impossible de trouver la ligne 'Loop time' dans la sortie LAMMPS.")

    loop_time_s, steps_done = float(match.group(1)), int(match.group(2))
    s_per_step = loop_time_s / steps_done
    steps_per_s = 1.0 / s_per_step

    total_steps = round((EQUIL_MAX_PS + PROD_PS) / DT_PS)
    est_total_s = total_steps * s_per_step

    print(f"\nDebit mesure : {steps_per_s:.2f} pas/s ({s_per_step*1000:.1f} ms/pas)")
    print(f"Pour {total_steps} pas au total (equil_max={EQUIL_MAX_PS}ps + prod={PROD_PS}ps) :")
    print(f"  Estimation MAJORANTE (si equil_max_ps atteint sans convergence) : {est_total_s/60:.1f} min")
    print("  (si la thermalisation converge avant equil_max_ps, ce sera plus rapide)")