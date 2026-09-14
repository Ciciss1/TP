"""
Boucle sur tous les fichiers T={temperature}_k={k}.npz d'un dossier,
calcule la distribution des anneaux, |m6| (largeur azimutale) et la
largeur radiale du premier pic (translationnel) pour chacun, moyenne
sur les k repetitions par temperature, et trace les resultats vs T.

Usage:
    python Batch_plot.py /chemin/vers/dossier
"""
import os
import sys
import re
import glob
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from Properties import (
    build_neighbor_array, find_rings_numba, ring_size_distribution,
    assign_sublattice, first_peak_qmag, first_peak_angle,
    structure_factor_ring, m6_from_profile,
    structure_factor_radial, radial_peak_width,
    plot_Sq_profile,
    structure_factor_fft_map, plot_fft_peak_map,
    structure_factor_fft_full, plot_fft_full_ring,
)


FNAME_RE = re.compile(r"T=(\d+)_k=(\d+)\.npz$")


def analyze_file(path, a_CC=1.42, n_theta=720, n_q=300, q_window=0.3):
    """
    Calcule la distribution des anneaux, |m6| (largeur azimutale) et la
    largeur radiale du premier pic (translationnel) pour un fichier.
    Retourne aussi les profils complets pour pouvoir tracer sans
    tout recalculer.
    """
    data = np.load(path)
    xy = data['xyz'][:, :2]
    Lx, Ly = data['lattice']

    nb_sorted, deg = build_neighbor_array(xy, Lx, Ly, bond_length=a_CC)
    ring_sizes, n_excluded = find_rings_numba(nb_sorted, deg)
    ring_dist = ring_size_distribution(ring_sizes)

    sub = assign_sublattice(nb_sorted, deg)
    atoms_A = xy[sub == 0]

    qmag = first_peak_qmag(a_CC)
    q_angle = first_peak_angle(a_CC)

    theta, Sq = structure_factor_ring(atoms_A, qmag, n_theta=n_theta)
    m6 = m6_from_profile(theta, Sq)

    q_values = np.linspace(qmag * (1 - q_window), qmag * (1 + q_window), n_q)
    Sq_radial = structure_factor_radial(atoms_A, q_angle, q_values)
    radial_width, q0 = radial_peak_width(q_values, Sq_radial)

    return {
        "ring_dist": ring_dist, "n_excluded": n_excluded,
        "m6": abs(m6), "theta": theta, "Sq": Sq,
        "radial_width": radial_width, "q0": q0,
        "q_values": q_values, "Sq_radial": Sq_radial,
        "atoms_A": atoms_A, "Lx": Lx, "Ly": Ly,
    }


def collect_results(folder, a_CC=1.42, example_k=None):
    """
    Parcourt tous les T=*_k=*.npz du dossier, calcule les observables,
    et regroupe les resultats par temperature.
    Pour chaque T, garde aussi le resultat complet d'UN fichier
    representatif (le premier k rencontre, ou `example_k` si precise)
    pour pouvoir tracer le profil de diffraction sans tout recalculer.
    Outputs:
        results[T] = {'ring_5': [...], 'ring_6': [...], 'ring_7': [...],
                       'm6': [...], 'radial_width': [...]}  (une valeur par k)
        example_profiles[T] = dict (sortie de analyze_file + 'k')
    """
    files = sorted(glob.glob(f"{folder}/T=*_k=*.npz"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier T=*_k=*.npz trouve dans {folder}")

    results = defaultdict(lambda: defaultdict(list))
    example_profiles = {}

    for path in files:
        m = FNAME_RE.search(path)
        if not m:
            print(f"  (ignore, nom non reconnu) {path}")
            continue
        T, k = int(m.group(1)), int(m.group(2))

        print(f"T={T} k={k} ...", end=" ", flush=True)
        try:
            res = analyze_file(path, a_CC=a_CC)
        except Exception as e:
            print(f"ECHEC ({e})")
            continue

        for size in [5, 6, 7]:
            results[T][f"ring_{size}"].append(res["ring_dist"].get(size, 0.0))
        results[T]["m6"].append(res["m6"])
        results[T]["radial_width"].append(res["radial_width"])
        print(f"OK  (6-rings={res['ring_dist'].get(6,0):.1f}%  "
              f"|m6|={res['m6']:.3f}  radial_width={res['radial_width']:.4f}  "
              f"excl={res['n_excluded']})")

        keep = (T not in example_profiles) if example_k is None else (k == example_k)
        if keep:
            example_profiles[T] = {**res, "k": k}

    return dict(results), example_profiles


def plot_results(results, out_dir=".", out_prefix="phase_transition"):
    """ Trace % of rings, |m6| and radial width vs T, with error bars (std over k). """
    T_values = sorted(results.keys())

    def mean_std(T, key):
        vals = np.array(results[T][key])
        return vals.mean(), vals.std()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # --- panel 1: ring size distribution ---
    colors = {"ring_5": "tab:orange", "ring_6": "tab:green", "ring_7": "tab:red"}
    labels = {"ring_5": "Pentagons (5)", "ring_6": "Hexagons (6)", "ring_7": "Heptagons (7)"}
    for key in ["ring_5", "ring_6", "ring_7"]:
        means = [mean_std(T, key)[0] for T in T_values]
        stds = [mean_std(T, key)[1] for T in T_values]
        ax1.errorbar(T_values, means, yerr=stds, marker='o', capsize=3,
                     color=colors[key], label=labels[key])
    ax1.set_xlabel(rf"$T$(K)")
    ax1.set_ylabel("% of rings")
    ax1.set_title("Distribution of ring sizes vs T")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # --- panel 2: |m6| (orientational order, azimuthal broadening) ---
    means = [mean_std(T, "m6")[0] for T in T_values]
    stds = [mean_std(T, "m6")[1] for T in T_values]
    ax2.errorbar(T_values, means, yerr=stds, marker='o', capsize=3, color="tab:blue")
    ax2.set_xlabel(rf"$T$(K)")
    ax2.set_ylabel(r"$|m_6|$")
    ax2.set_title("Orientational order (azimuthal broadening of S(q))")
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3)

    # --- panel 3: radial peak width (translational order) ---
    means = [mean_std(T, "radial_width")[0] for T in T_values]
    stds = [mean_std(T, "radial_width")[1] for T in T_values]
    ax3.errorbar(T_values, means, yerr=stds, marker='o', capsize=3, color="tab:purple")
    ax3.set_xlabel(rf"$T$(K)")
    ax3.set_ylabel(r"radial width ($\mathrm{\AA}^{-1}$)")
    ax3.set_title("Translational order (radial broadening of S(q))")
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(out_dir, f"{out_prefix}.png")
    pdf_path = os.path.join(out_dir, f"{out_prefix}.pdf")
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=150)
    print(f"Figures sauvegardees : {pdf_path} / {png_path}")


def save_results_npz(results, out_dir=".", out_name="rings_peaks_data.npz"):
    """
    Sauvegarde les arrays moyennes+std (sur k) par temperature, utilises
    pour le plot phase_transition -- pour retracer ou retraiter sans
    tout relancer.
    Contenu du npz:
        T : (n_T,) temperatures triees
        ring_5_mean, ring_5_std, ring_6_mean, ring_6_std,
        ring_7_mean, ring_7_std, m6_mean, m6_std,
        radial_width_mean, radial_width_std : (n_T,)
    """
    T_values = np.array(sorted(results.keys()))
    out = {"T": T_values}
    for key in ["ring_5", "ring_6", "ring_7", "m6", "radial_width"]:
        means = np.array([np.mean(results[T][key]) for T in T_values])
        stds = np.array([np.std(results[T][key]) for T in T_values])
        out[f"{key}_mean"] = means
        out[f"{key}_std"] = stds
    out_path = os.path.join(out_dir, out_name)
    np.savez(out_path, **out)
    print(f"Donnees sauvegardees : {out_path}")


def save_diffraction_profiles(example_profiles, out_dir=".", subdir="diffraction_profiles"):
    """
    Sauvegarde un profil de diffraction (figure + npz) par temperature :
    profil angulaire (theta, Sq) ET profil radial (q_values, Sq_radial),
    pour voir visuellement les deux types d'elargissement et reexploiter
    les donnees brutes plus tard.
    """
    full_dir = os.path.join(out_dir, subdir)
    os.makedirs(full_dir, exist_ok=True)
    for T in sorted(example_profiles):
        res = example_profiles[T]

        png_path = os.path.join(full_dir, f"T={T}.png")
        plot_Sq_profile(
            res["theta"], res["Sq"],
            title=f"T={T} (k={res['k']})  |m6|={res['m6']:.3f}  radial_width={res['radial_width']:.4f}",
            save_path=png_path,
            q_values=res["q_values"], Sq_radial=res["Sq_radial"],
        )
        plt.close('all')

        npz_path = os.path.join(full_dir, f"T={T}.npz")
        np.savez(npz_path,
                 theta=res["theta"], Sq=res["Sq"],
                 q_values=res["q_values"], Sq_radial=res["Sq_radial"],
                 m6=res["m6"], radial_width=res["radial_width"],
                 T=T, k=res["k"])

    print(f"\nProfils de diffraction (figures + npz) sauvegardes dans {full_dir}/")


def save_fft_maps(example_profiles, a_CC=1.42, out_dir=".", subdir="fft_peak_maps",
                   window=0.6, n_grid=2048):
    """
    Sauvegarde une vraie carte FFT 2D (image du spot de diffraction) par
    temperature -- complementaire des profils angulaire/radial calcules
    par sommation directe (plus precis pour les nombres, moins visuel).
    """
    full_dir = os.path.join(out_dir, subdir)
    os.makedirs(full_dir, exist_ok=True)

    qmag = first_peak_qmag(a_CC)
    q_angle = first_peak_angle(a_CC)
    q_center = (qmag * np.cos(q_angle), qmag * np.sin(q_angle))

    for T in sorted(example_profiles):
        res = example_profiles[T]
        sub_Sq, sub_qx, sub_qy = structure_factor_fft_map(
            res["atoms_A"], res["Lx"], res["Ly"], q_center,
            window=window, n_grid=n_grid,
        )

        png_path = os.path.join(full_dir, f"T={T}.png")
        plot_fft_peak_map(sub_Sq, sub_qx, sub_qy,
                           title=f"T={T} (k={res['k']})", save_path=png_path)
        plt.close('all')

        npz_path = os.path.join(full_dir, f"T={T}.npz")
        np.savez(npz_path, Sq_map=sub_Sq, qx=sub_qx, qy=sub_qy, T=T, k=res["k"])

    print(f"\nCartes FFT 2D sauvegardees dans {full_dir}/")


def save_fft_full_rings(example_profiles, a_CC=1.42, out_dir=".", subdir="fft_full_ring",
                         q_max_factor=1.4, n_grid=2048, log_scale=False):
    """
    Sauvegarde la carte FFT complete du 1er anneau par temperature --
    pour voir les 6 pics d'un coup (solide/hexatique) ou l'anneau
    isotrope complet (liquide), au lieu d'une fenetre sur un seul pic.
    """
    full_dir = os.path.join(out_dir, subdir)
    os.makedirs(full_dir, exist_ok=True)

    qmag = first_peak_qmag(a_CC)
    q_max = qmag * q_max_factor

    for T in sorted(example_profiles):
        res = example_profiles[T]
        sub_Sq, sub_qx, sub_qy = structure_factor_fft_full(
            res["atoms_A"], res["Lx"], res["Ly"], q_max, n_grid=n_grid,
        )

        png_path = os.path.join(full_dir, f"T={T}.png")
        plot_fft_full_ring(sub_Sq, sub_qx, sub_qy,
                            title=f"T={T} (k={res['k']})",
                            save_path=png_path, log_scale=log_scale)
        plt.close('all')

        npz_path = os.path.join(full_dir, f"T={T}.npz")
        np.savez(npz_path, Sq_map=sub_Sq, qx=sub_qx, qy=sub_qy, T=T, k=res["k"])

    print(f"\nCartes FFT (anneau complet) sauvegardees dans {full_dir}/")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"{folder} n'est pas un dossier valide")
    results, example_profiles = collect_results(folder)
    plot_results(results, out_dir=folder)
    save_results_npz(results, out_dir=folder)
    save_diffraction_profiles(example_profiles, out_dir=folder)
    save_fft_maps(example_profiles, out_dir=folder)
    save_fft_full_rings(example_profiles, out_dir=folder)