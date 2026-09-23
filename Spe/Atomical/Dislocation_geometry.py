"""
Run separe (independant de Batch_plot.py / Raman_map.py) : deux
analyses purement geometriques/topologiques sur les anneaux 5/6/7
deja detectes, motivees directement par KTHNY (pas par une technique
experimentale particuliere) :

  - n1) Distance de separation des dipoles 5-7 (dislocations liees
    isolees, composantes connexes de taille 2 dans le sous-graphe
    5/7). KTHNY predit que cette separation croit a l'approche de la
    transition solide->hexatique puis diverge au depiegeage -- on
    trace sa moyenne (et sa distribution poolee) vs T.

  - n3) Correlation spatiale des tailles de cluster de defauts (5/7),
    C(r), pour voir si les grands clusters (segments de joint de
    grain) sont eux-memes regroupes spatialement, ou si les tailles
    sont spatialement independantes. Statistique faible par fichier :
    on moyenne C(r) sur les k repetitions d'une meme T, pondere par le
    nombre de paires par bin.

Usage:
    python Dislocation_geometry.py /chemin/vers/dossier [--r-bins 20] [--r-max R]
"""
import os
import sys
import re
import glob
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from Properties import (
    build_neighbor_array, find_rings_with_atoms_numba,
    dipole_57_separations, plot_dipole_separation_hist,
    cluster_centroids_and_sizes, cluster_size_spatial_correlation,
    plot_cluster_size_correlation,
)


FNAME_RE = re.compile(r"T=(\d+)_k=(\d+)\.npz$")


def analyze_file_dislocation(path, a_CC=1.42, r_bins=None):
    """
    Calcule, pour un fichier :
        - les distances de separation des dipoles 5-7 isoles
        - les centroides/tailles des clusters de defauts 5/7
        - C(r) (correlation spatiale des tailles de cluster) sur les
          bins de distance r_bins (fournis par l'appelant, communs a
          tout le batch pour pouvoir moyenner sur les k repetitions)
    """
    data = np.load(path)
    xy = data['xyz'][:, :2]
    Lx, Ly = data['lattice']

    if r_bins is None:
        r_bins = np.linspace(0, Lx / 2, 21)

    nb_sorted, deg = build_neighbor_array(xy, Lx, Ly, bond_length=a_CC)
    ring_sizes, ring_atoms, n_excluded = find_rings_with_atoms_numba(nb_sorted, deg)

    separations = dipole_57_separations(ring_sizes, ring_atoms, xy, Lx, Ly)

    centroids, sizes = cluster_centroids_and_sizes(ring_sizes, ring_atoms, xy, Lx, Ly)
    r_centers, C_r, n_pairs = cluster_size_spatial_correlation(centroids, sizes, Lx, Ly, r_bins)

    return {
        "separations": separations,
        "n_dipoles": len(separations),
        "mean_sep": float(separations.mean()) if len(separations) else np.nan,
        "centroids": centroids, "sizes": sizes,
        "r_centers": r_centers, "C_r": C_r, "n_pairs": n_pairs,
        "Lx": Lx, "Ly": Ly, "n_excluded": n_excluded,
    }


def collect_results_dislocation(folder, a_CC=1.42, n_r_bins=20, r_max=None):
    """
    Parcourt tous les T=*_k=*.npz du dossier. Pour chaque T :
        - poole les separations de dipole sur tous les k (histogramme
          + moyenne/std de la moyenne par fichier)
        - moyenne C(r) sur les k, pondere par n_pairs par bin (evite
          qu'un fichier avec peu de clusters pese autant qu'un fichier
          bien plus statistique)
    """
    files = sorted(glob.glob(f"{folder}/T=*_k=*.npz"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier T=*_k=*.npz trouve dans {folder}")

    if r_max is None:
        first_data = np.load(files[0])
        r_max = first_data['lattice'][0] / 2
    r_bins = np.linspace(0, r_max, n_r_bins + 1)

    per_T_mean_sep = defaultdict(list)
    per_T_n_dipoles = defaultdict(list)
    pooled_separations = defaultdict(list)
    sum_C_weighted = defaultdict(lambda: np.zeros(n_r_bins))
    sum_weights = defaultdict(lambda: np.zeros(n_r_bins))

    for path in files:
        m = FNAME_RE.search(path)
        if not m:
            print(f"  (ignore, nom non reconnu) {path}")
            continue
        T, k = int(m.group(1)), int(m.group(2))

        print(f"T={T} k={k} ...", end=" ", flush=True)
        try:
            res = analyze_file_dislocation(path, a_CC=a_CC, r_bins=r_bins)
        except Exception as e:
            print(f"ECHEC ({e})")
            continue

        per_T_mean_sep[T].append(res["mean_sep"])
        per_T_n_dipoles[T].append(res["n_dipoles"])
        pooled_separations[T].extend(res["separations"].tolist())

        valid = ~np.isnan(res["C_r"])
        sum_C_weighted[T][valid] += res["C_r"][valid] * res["n_pairs"][valid]
        sum_weights[T][valid] += res["n_pairs"][valid]

        print(f"OK  n_dipoles={res['n_dipoles']}  "
              f"mean_sep={res['mean_sep']:.3f}  n_clusters={len(res['sizes'])}  "
              f"excl={res['n_excluded']}")

    T_values = sorted(per_T_mean_sep.keys())
    C_r_by_T = {}
    for T in T_values:
        with np.errstate(invalid='ignore', divide='ignore'):
            C_r_by_T[T] = np.where(sum_weights[T] > 0,
                                    sum_C_weighted[T] / sum_weights[T], np.nan)

    return {
        "T_values": T_values,
        "mean_sep": per_T_mean_sep,
        "n_dipoles": per_T_n_dipoles,
        "pooled_separations": {T: np.array(v) for T, v in pooled_separations.items()},
        "r_centers": 0.5 * (r_bins[:-1] + r_bins[1:]),
        "C_r_by_T": C_r_by_T,
        "n_pairs_total": {T: sum_weights[T] for T in T_values},
    }


def plot_results_vs_T(results, out_dir=".", out_prefix="dislocation_geometry_vs_T"):
    """ Trace la separation moyenne des dipoles et le nb de dipoles/fichier vs T. """
    T_values = results["T_values"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    means = [np.nanmean(results["mean_sep"][T]) for T in T_values]
    stds = [np.nanstd(results["mean_sep"][T]) for T in T_values]
    ax1.errorbar(T_values, means, yerr=stds, marker='o', capsize=3, color="tab:purple")
    ax1.set_xlabel(r"$T$ (K)")
    ax1.set_ylabel(r"$\langle r_{5-7} \rangle$ ($\mathrm{\AA}$)")
    ax1.set_title("Dipole 5-7 separation vs T")
    ax1.grid(alpha=0.3)

    means_n = [np.mean(results["n_dipoles"][T]) for T in T_values]
    stds_n = [np.std(results["n_dipoles"][T]) for T in T_values]
    ax2.errorbar(T_values, means_n, yerr=stds_n, marker='o', capsize=3, color="tab:orange")
    ax2.set_xlabel(r"$T$ (K)")
    ax2.set_ylabel("dipoles per file")
    ax2.set_title("Number of isolated 5-7 dipoles vs T")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(out_dir, f"{out_prefix}.png")
    pdf_path = os.path.join(out_dir, f"{out_prefix}.pdf")
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=150)
    print(f"Figures sauvegardees : {pdf_path} / {png_path}")


def save_results_npz(results, out_dir=".", out_name="dislocation_geometry_data.npz"):
    """ Sauvegarde les arrays moyenne+std (sur k) par T, et C(r) par T. """
    T_values = np.array(results["T_values"])
    mean_sep_mean = np.array([np.nanmean(results["mean_sep"][T]) for T in T_values])
    mean_sep_std = np.array([np.nanstd(results["mean_sep"][T]) for T in T_values])
    n_dipoles_mean = np.array([np.mean(results["n_dipoles"][T]) for T in T_values])
    n_dipoles_std = np.array([np.std(results["n_dipoles"][T]) for T in T_values])
    C_r_matrix = np.array([results["C_r_by_T"][T] for T in T_values])

    out_path = os.path.join(out_dir, out_name)
    np.savez(out_path, T=T_values,
             mean_sep_mean=mean_sep_mean, mean_sep_std=mean_sep_std,
             n_dipoles_mean=n_dipoles_mean, n_dipoles_std=n_dipoles_std,
             r_centers=results["r_centers"], C_r_by_T=C_r_matrix)
    print(f"Donnees sauvegardees : {out_path}")


def save_example_plots(results, out_dir=".", subdir="dislocation_geometry_examples"):
    """
    Sauvegarde, par T, l'histogramme poole des separations de dipole
    et la courbe C(r) moyennee sur les k -- pour une inspection directe
    de l'evolution avec T (au-dela des scalaires moyens de
    plot_results_vs_T).
    """
    full_dir = os.path.join(out_dir, subdir)
    os.makedirs(full_dir, exist_ok=True)

    for T in results["T_values"]:
        sep_path = os.path.join(full_dir, f"separations_T={T}.png")
        plot_dipole_separation_hist(
            results["pooled_separations"][T],
            title=f"T={T}  (n={len(results['pooled_separations'][T])} dipoles, "
                  f"pooled over k)",
            save_path=sep_path,
        )
        plt.close('all')

        corr_path = os.path.join(full_dir, f"cluster_corr_T={T}.png")
        plot_cluster_size_correlation(
            results["r_centers"], results["C_r_by_T"][T],
            results["n_pairs_total"][T],
            title=f"T={T}  (cluster-size correlation, k-averaged)",
            save_path=corr_path,
        )
        plt.close('all')

    print(f"\nHistogrammes de separation + courbes C(r) sauvegardes dans {full_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", default=".",
                         help="Dossier contenant les T=*_k=*.npz")
    parser.add_argument("--r-bins", type=int, default=20,
                         help="Nombre de bins de distance pour C(r)")
    parser.add_argument("--r-max", type=float, default=None,
                         help="Distance max (Angstrom) pour C(r). "
                              "Par defaut, Lx/2 du premier fichier.")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        raise NotADirectoryError(f"{args.folder} n'est pas un dossier valide")

    results = collect_results_dislocation(
        args.folder, n_r_bins=args.r_bins, r_max=args.r_max)
    plot_results_vs_T(results, out_dir=args.folder)
    save_results_npz(results, out_dir=args.folder)
    save_example_plots(results, out_dir=args.folder)
