import sys
from pathlib import Path
import matplotlib.pyplot as plt
from thermalize import ThermalizationResult, compute_vdos

FREQ_MAX_THZ = 60.0


def load_vdos(npz_path: Path):
    result = ThermalizationResult.load(str(npz_path))
    dt_frame_ps = result.times_ps[1] - result.times_ps[0]
    freq_thz, vdos = compute_vdos(result.velocities, dt_frame_ps=dt_frame_ps)
    mask = freq_thz <= FREQ_MAX_THZ
    freq_cm1 = freq_thz[mask] * 33.356
    return freq_cm1, vdos[mask], result


if __name__ == "__main__":
    # Usage : python compare_vdos.py fichier1.npz fichier2.npz fichier3.npz
    paths = ["/scratch/escarmel/thermalize_results/eps=0/L=272.12/rho=0.00135/T=10000_k=1_thermalized_300K.npz", "/scratch/escarmel/thermalize_results/eps=0/L=272.12/rho=0.00135/T=86000_k=1_thermalized_300K.npz", "/scratch/escarmel/thermalize_results/eps=0/L=272.12/rho=0.00135/T=120000_k=1_thermalized_300K.npz"]
    if not paths:
        raise SystemExit("Donne au moins un chemin .npz en argument")

    fig, ax = plt.subplots(figsize=(10, 6))

    for npz_path in paths:
        freq_cm1, vdos, result = load_vdos(npz_path)
        # extrait le "T=xxxx" (temperature MC) du nom de fichier pour la legende
        label = npz_path.split("_thermalized")[0]
        ax.plot(freq_cm1, vdos, label=label)
        print(f"{npz_path} : converged={result.converged}")

    ax.set_xlabel("Frequency (cm$^{-1}$)")
    ax.set_ylabel("VDOS (a.u.)")
    ax.set_title("Comparaison VDOS")
    ax.legend()
    fig.tight_layout()

    out_path = Path("vdos_comparison.png")
    fig.savefig(out_path, dpi=150)
    print(f"-> {out_path}")