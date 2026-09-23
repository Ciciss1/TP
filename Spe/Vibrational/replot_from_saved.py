import sys
import re
from pathlib import Path
import matplotlib.pyplot as plt
from thermalize import ThermalizationResult, compute_vdos

OUTPUT_DIR = Path("results")
FREQ_MAX_THZ = 60.0


def plot_one(npz_path: Path):
    stem = npz_path.stem
    result = ThermalizationResult.load(str(npz_path))
    print(f"{npz_path} : converged={result.converged}, is_equilibrated()={result.is_equilibrated()}")

    dt_frame_ps = result.times_ps[1] - result.times_ps[0]
    freq_thz, vdos = compute_vdos(result.velocities, dt_frame_ps=dt_frame_ps)

    mask = freq_thz <= FREQ_MAX_THZ
    freq_cm1 = freq_thz[mask] * 33.356

    fig, (ax_temp, ax_vdos) = plt.subplots(
        1, 2, figsize=(18, 4.5), gridspec_kw={"width_ratios": [1, 3]}
    )

    ax_temp.plot(result.equil_steps, result.equil_temp)
    ax_temp.axhline(result.T, color="r", linestyle="--", label=f"target {result.T:.0f} K")
    ax_temp.set_xlabel("MD steps (equilibration)")
    ax_temp.set_ylabel("Temperature (K)")
    ax_temp.set_title(f"Convergence NVT - Temperature (converged: {result.converged})")
    ax_temp.legend()

    ax_vdos.plot(freq_cm1, vdos[mask])
    ax_vdos.set_xlabel("Frequency (cm$^{-1}$)")
    ax_vdos.set_ylabel("VDOS (a.u.)")
    mc_t_match = re.search(r"T=(\d+)", stem)
    mc_t = mc_t_match.group(1) if mc_t_match else "?"
    ax_vdos.set_title(f"Spectrum - MC T={mc_t}")

    fig.tight_layout()
    plot_path = npz_path.parent / f"{stem}_vdos_spectrum.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)  # evite d'accumuler les figures en memoire sur un gros dossier
    print(f"  -> {plot_path}")


if __name__ == "__main__":
    # Un seul fichier -> juste lui. Un dossier -> tous les .npz dedans
    # (recursif, donc marche aussi sur results/ avec les 4 sous-dossiers de sweep_thermalize.py).
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT_DIR

    if target.is_dir():
        files = sorted(target.rglob("*.npz"))
        print(f"{len(files)} fichier(s) trouve(s) dans {target}")
    else:
        files = [target]

    for npz_path in files:
        plot_one(npz_path)