"""
Run separe : teste si les disclinaisons libres (non appariees, cf.
classify_dislocations) s'accumulent spatialement pres des jonctions
triples du reseau de defauts 5/7 (points de branchement, degre >= 3
dans le sous-graphe 5/7 -- purement topologique, aucune donnee de
grain/Voronoi necessaire, cf. Properties.defect_chain_degree).

Motive par la litterature TEM/mecanique des dislocations (Beausir &
Fressengeas) : une jonction triple porte generiquement une charge de
disclinaison residuelle (nulle seulement pour une jonction symetrique
a 120 deg), donc on s'attend a un enrichissement des disclinaisons
libres a proximite -- teste ici contre une baseline CSR (points
uniformement distribues) via la distance au plus proche voisin.

Usage:
    python Triple_junctions.py /chemin/vers/dossier [--radius-mindeg 3] [--n-random 500]
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
    triple_junction_disclination_enrichment,
    plot_triple_junction_map, plot_junction_distance_distribution,
)


FNAME_RE = re.compile(r"T=(\d+)_k=(\d+)\.npz$")


def analyze_file_junctions(path, a_CC=1.42, min_degree=3, n_random=None, seed=0):
    """
    Calcule, pour un fichier, les jonctions triples, les disclinaisons
    libres, et l'enrichissement spatial des secondes pres des
    premieres (cf. triple_junction_disclination_enrichment).
    """
    data = np.load(path)
    xy = data['xyz'][:, :2]
    Lx, Ly = data['lattice']

    nb_sorted, deg = build_neighbor_array(xy, Lx, Ly, bond_length=a_CC)
    ring_sizes, ring_atoms, n_excluded = find_rings_with_atoms_numba(nb_sorted, deg)

    res = triple_junction_disclination_enrichment(
        ring_sizes, ring_atoms, xy, Lx, Ly,
        n_random=n_random, min_degree=min_degree, seed=seed)
    res["Lx"], res["Ly"] = Lx, Ly
    res["n_excluded"] = n_excluded
    return res


def collect_results_junctions(folder, a_CC=1.42, min_degree=3, n_random=None, example_k=None):
    """
    Parcourt tous les T=*_k=*.npz du dossier. Regroupe par T :
        - n_junctions, n_free_total (comptes bruts)
        - median_free, median_random, enrichment (une valeur par k)
    Garde aussi UN fichier representatif par T (positions + distances)
    pour les figures detaillees (carte spatiale, ECDF).
    """
    files = sorted(glob.glob(f"{folder}/T=*_k=*.npz"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier T=*_k=*.npz trouve dans {folder}")

    results = defaultdict(lambda: defaultdict(list))
    example_results = {}

    for path in files:
        m = FNAME_RE.search(path)
        if not m:
            print(f"  (ignore, nom non reconnu) {path}")
            continue
        T, k = int(m.group(1)), int(m.group(2))

        print(f"T={T} k={k} ...", end=" ", flush=True)
        try:
            res = analyze_file_junctions(
                path, a_CC=a_CC, min_degree=min_degree, n_random=n_random, seed=k)
        except Exception as e:
            print(f"ECHEC ({e})")
            continue

        results[T]["n_junctions"].append(res["n_junctions"])
        results[T]["n_free_total"].append(res["n_free_total"])
        results[T]["median_free"].append(res["median_free"])
        results[T]["median_random"].append(res["median_random"])
        results[T]["enrichment"].append(res["enrichment"])
        print(f"OK  n_junctions={res['n_junctions']}  n_free={res['n_free_total']}  "
              f"enrichment={res['enrichment']:.3f}  excl={res['n_excluded']}")

        keep = (T not in example_results) if example_k is None else (k == example_k)
        if keep:
            example_results[T] = {**res, "k": k}

    return dict(results), example_results


def plot_results_vs_T(results, out_dir=".", out_prefix="triple_junctions_vs_T"):
    """ Trace l'enrichissement et le nb de jonctions/disclinaisons libres vs T. """
    T_values = sorted(results.keys())

    def mean_std(T, key):
        vals = np.array(results[T][key])
        return np.nanmean(vals), np.nanstd(vals)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    means = [mean_std(T, "enrichment")[0] for T in T_values]
    stds = [mean_std(T, "enrichment")[1] for T in T_values]
    ax1.errorbar(T_values, means, yerr=stds, marker='o', capsize=3, color="tab:red")
    ax1.axhline(1.0, color="black", ls=":", lw=1, label="CSR (pas d'enrichissement)")
    ax1.set_xlabel(r"$T$ (K)")
    ax1.set_ylabel(r"enrichment ($d_{random}^{med} / d_{free}^{med}$)")
    ax1.set_title("Free disclination enrichment near triple junctions")
    ax1.legend()
    ax1.grid(alpha=0.3)

    colors = {"n_junctions": "tab:red", "n_free_total": "tab:orange"}
    labels = {"n_junctions": "Triple junctions", "n_free_total": "Free disclinations"}
    for key in ["n_junctions", "n_free_total"]:
        means = [mean_std(T, key)[0] for T in T_values]
        stds = [mean_std(T, key)[1] for T in T_values]
        ax2.errorbar(T_values, means, yerr=stds, marker='o', capsize=3,
                     color=colors[key], label=labels[key])
    ax2.set_xlabel(r"$T$ (K)")
    ax2.set_ylabel("count per file")
    ax2.set_title("Triple junctions & free disclinations vs T")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(out_dir, f"{out_prefix}.png")
    pdf_path = os.path.join(out_dir, f"{out_prefix}.pdf")
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=150)
    print(f"Figures sauvegardees : {pdf_path} / {png_path}")


def save_results_npz(results, out_dir=".", out_name="triple_junctions_data.npz"):
    """ Sauvegarde les arrays moyenne+std (sur k) par T. """
    T_values = np.array(sorted(results.keys()))
    out = {"T": T_values}
    for key in ["n_junctions", "n_free_total", "median_free", "median_random", "enrichment"]:
        vals = [np.array(results[T][key], dtype=float) for T in T_values]
        out[f"{key}_mean"] = np.array([np.nanmean(v) for v in vals])
        out[f"{key}_std"] = np.array([np.nanstd(v) for v in vals])
    out_path = os.path.join(out_dir, out_name)
    np.savez(out_path, **out)
    print(f"Donnees sauvegardees : {out_path}")


def save_example_plots(example_results, out_dir=".", subdir="triple_junction_examples"):
    """
    Sauvegarde, par T, la carte spatiale (jonctions + disclinaisons
    libres) et l'ECDF des distances (libres vs CSR) d'un fichier
    representatif -- pour une inspection visuelle directe.
    """
    full_dir = os.path.join(out_dir, subdir)
    os.makedirs(full_dir, exist_ok=True)

    for T in sorted(example_results):
        res = example_results[T]

        map_path = os.path.join(full_dir, f"map_T={T}.png")
        plot_triple_junction_map(
            res["junction_xy"], res["free_xy"], res["Lx"], res["Ly"],
            radius=res["median_free"],
            title=f"T={T} (k={res['k']})  enrichment={res['enrichment']:.2f}",
            save_path=map_path,
        )
        plt.close('all')

        ecdf_path = os.path.join(full_dir, f"ecdf_T={T}.png")
        plot_junction_distance_distribution(
            res["d_free"], res["d_random"],
            title=f"T={T} (k={res['k']})",
            save_path=ecdf_path,
        )
        plt.close('all')

        npz_path = os.path.join(full_dir, f"T={T}.npz")
        np.savez(npz_path, junction_xy=res["junction_xy"], free_xy=res["free_xy"],
                 d_free=res["d_free"], d_random=res["d_random"],
                 median_free=res["median_free"], median_random=res["median_random"],
                 enrichment=res["enrichment"], T=T, k=res["k"])

    print(f"\nCartes + ECDF sauvegardees dans {full_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", default=".",
                         help="Dossier contenant les T=*_k=*.npz")
    parser.add_argument("--min-degree", type=int, default=3,
                         help="Degre minimal (dans le sous-graphe 5/7) pour "
                              "qu'un anneau soit considere comme une jonction triple")
    parser.add_argument("--n-random", type=int, default=None,
                         help="Nb de points CSR tires pour la baseline "
                              "(defaut: max(n_free, 200))")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        raise NotADirectoryError(f"{args.folder} n'est pas un dossier valide")

    results, example_results = collect_results_junctions(
        args.folder, min_degree=args.min_degree, n_random=args.n_random)
    plot_results_vs_T(results, out_dir=args.folder)
    save_results_npz(results, out_dir=args.folder)
    save_example_plots(example_results, out_dir=args.folder)
