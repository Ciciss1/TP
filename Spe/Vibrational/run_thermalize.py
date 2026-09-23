import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from thermalize import Thermalizer, compute_vdos

INPUT_FILE = r"graphene_pristine_small.npz"
OUTPUT_DIR = Path("results")

T_KELVIN = 10        # temperature for thermalization
EQUIL_CHECK_PS = 2.0    # NVT chunk length between convergence checks
EQUIL_MAX_PS = 20.0     # hard cap on NVT time if it never converges
PE_TOL = 0.005          # relative PE change threshold to stop equilibration early
PROD_PS = 5.0           # production NVE
AIREBO_FILE = "CH.airebo"
N_THREADS = 6
FREQ_MAX_THZ = 60.0

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(INPUT_FILE).stem  # ex: "T=10000_k=1", sans le chemin ../Ruslan/...

    data = np.load(INPUT_FILE)
    atoms = data["xyz"]
    Lx, Ly = data["lattice"]

    print(f"{len(atoms)} atomes, boite {Lx:.2f} x {Ly:.2f} A")

    thermalizer = Thermalizer(airebo_file=AIREBO_FILE, n_threads=N_THREADS)
    result = thermalizer.run(
        atoms, Lx, Ly,
        T=T_KELVIN,
        equil_check_ps=EQUIL_CHECK_PS, equil_max_ps=EQUIL_MAX_PS, pe_tol=PE_TOL,
        prod_ps=PROD_PS,
    )

    print("Converged :", result.converged, "| is_equilibrated() :", result.is_equilibrated())

    out_path = OUTPUT_DIR / f"{stem}_thermalized_{int(T_KELVIN)}K.npz"
    result.save(str(out_path))
    print(f"Sauve dans {out_path}")

    dt_frame_ps = result.times_ps[1] - result.times_ps[0]
    freq_thz, vdos = compute_vdos(result.velocities, dt_frame_ps=dt_frame_ps)

    mask = freq_thz <= FREQ_MAX_THZ
    freq_cm1 = freq_thz[mask] * 33.356  # converting THz -> cm^-1

    fig, (ax_temp, ax_vdos) = plt.subplots(
        1, 2, figsize=(18, 4.5), gridspec_kw={"width_ratios": [1, 3]}
    )

    ax_temp.plot(result.equil_steps, result.equil_temp)
    ax_temp.axhline(result.T, color="r", linestyle="--", label=f"target {result.T:.0f} K")
    ax_temp.set_xlabel("MD steps (equilibration)")
    ax_temp.set_ylabel("Temperature (K)")
    ax_temp.set_title(f"Convergence NVT - Temperature (converged: {result.converged})")
    ax_temp.legend()

    # --- Spectre vibrationnel ---
    ax_vdos.plot(freq_cm1, vdos[mask])
    ax_vdos.set_xlabel("Frequency (cm$^{-1}$)")
    ax_vdos.set_ylabel("VDOS (a.u.)")
    mc_t_match = re.search(r"T=(\d+)", stem)
    mc_t = mc_t_match.group(1) if mc_t_match else "?"
    ax_vdos.set_title(f"Spectrum - MC T={mc_t}")

    fig.tight_layout()
    plot_path = OUTPUT_DIR / f"{stem}_vdos_spectrum.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Plot sauve dans {plot_path}")