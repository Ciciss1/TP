"""
Run separe (independant de Batch_plot.py) : teste l'analogue optique
du mapping Raman I_D/I_G comme sonde de desordre spatialement resolue.

Un atome est "defectueux" s'il appartient a au moins un anneau != 6
(exactement ce qui active le pic D en Raman : un defaut ou un bord de
cristal). On calcule :
  - I_D/I_G global (fraction d'atomes defectueux) par fichier, moyenne
    sur les k repetitions par T, trace vs T -- pour voir si cette
    observable montre une transition (comparable au panel ring_5/7 de
    Batch_plot.py, mais definie comme une vraie quantite "a la Raman").
  - une carte spatiale I_D(r)/I_G(r) par T (un k representatif),
    lissee par un noyau gaussien de largeur sigma_A (taille de "spot"),
    pour voir si les joints de grains apparaissent visuellement, comme
    dans le mapping Raman experimental sur graphene CVD.

Usage:
    python Raman_map.py /chemin/vers/dossier [--sigma 5.0] [--grid 128]
"""
import os
import sys
import re
import glob
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from Properties import (build_neighbor_array,
    find_rings_with_atoms_numba, defect_atom_mask,
    raman_id_ig_map, plot_raman_map,
    classify_defect_clusters, cluster_size_fractions,
    local_strain_field, scalar_field_map, plot_strain_map, masked_rms,
)


FNAME_RE = re.compile(r"T=(\d+)_k=(\d+)\.npz$")


def analyze_file_raman(path, a_CC=1.42, n_grid=128, sigma_A=5.0):
    """
    Calcule le masque de defaut, la carte I_D/I_G, le scalaire global,
    la decomposition en composantes connexes de defauts (isolee /
    dipole / chaine etendue), et le champ de deformation locale
    (analogue G+/G-) pour un fichier. Garde xy/Lx/Ly pour pouvoir
    retracer les cartes sans tout recalculer.
    """
    data = np.load(path)
    xy = data['xyz'][:, :2]
    Lx, Ly = data['lattice']

    nb_sorted, deg = build_neighbor_array(xy, Lx, Ly, bond_length=a_CC)
    ring_sizes, ring_atoms, n_excluded = find_rings_with_atoms_numba(nb_sorted, deg)
    is_defect = defect_atom_mask(len(xy), ring_sizes, ring_atoms)

    ratio_map, id_map, ig_map, global_id_ig = raman_id_ig_map(
        xy, is_defect, Lx, Ly, n_grid=n_grid, sigma_A=sigma_A)

    _, component_sizes = classify_defect_clusters(ring_sizes, ring_atoms)
    f_isolated, f_dipole, f_chain, mean_size = cluster_size_fractions(component_sizes)

    eps_hydro, eps_dev = local_strain_field(xy, nb_sorted, deg, Lx, Ly, a_CC=a_CC)
    dev_map, dev_mean, dev_rms = scalar_field_map(
        xy, eps_dev, Lx, Ly, n_grid=n_grid, sigma_A=sigma_A)
    dev_rms_defects = masked_rms(eps_dev, is_defect)

    return {
        "global_id_ig": global_id_ig, "ratio_map": ratio_map,
        "xy": xy, "Lx": Lx, "Ly": Ly, "n_excluded": n_excluded,
        "n_defect": int(is_defect.sum()), "N": len(xy),
        "f_isolated": f_isolated, "f_dipole": f_dipole,
        "f_chain": f_chain, "mean_size": mean_size,
        "dev_map": dev_map, "dev_rms": dev_rms,
        "dev_rms_defects": dev_rms_defects,
    }


def collect_results_raman(folder, a_CC=1.42, n_grid=128, sigma_A=5.0, example_k=None):
    """
    Parcourt tous les T=*_k=*.npz du dossier. Regroupe global_id_ig par
    T (une valeur par k), et garde la carte d'UN fichier representatif
    par T pour l'inspection visuelle.
    """
    files = sorted(glob.glob(f"{folder}/T=*_k=*.npz"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier T=*_k=*.npz trouve dans {folder}")

    results = defaultdict(lambda: defaultdict(list))
    example_maps = {}

    for path in files:
        m = FNAME_RE.search(path)
        if not m:
            print(f"  (ignore, nom non reconnu) {path}")
            continue
        T, k = int(m.group(1)), int(m.group(2))

        print(f"T={T} k={k} ...", end=" ", flush=True)
        try:
            res = analyze_file_raman(path, a_CC=a_CC, n_grid=n_grid, sigma_A=sigma_A)
        except Exception as e:
            print(f"ECHEC ({e})")
            continue

        results[T]["id_ig"].append(res["global_id_ig"])
        results[T]["f_isolated"].append(res["f_isolated"])
        results[T]["f_dipole"].append(res["f_dipole"])
        results[T]["f_chain"].append(res["f_chain"])
        results[T]["mean_size"].append(res["mean_size"])
        results[T]["dev_rms"].append(res["dev_rms"])
        results[T]["dev_rms_defects"].append(res["dev_rms_defects"])
        print(f"OK  I_D/I_G={res['global_id_ig']:.4f}  "
              f"isolated={res['f_isolated']:.3f} dipole={res['f_dipole']:.3f} "
              f"chain={res['f_chain']:.3f} mean_size={res['mean_size']:.2f}  "
              f"dev_rms={res['dev_rms']:.4f} (defects={res['dev_rms_defects']:.4f})  "
              f"excl={res['n_excluded']}")

        keep = (T not in example_maps) if example_k is None else (k == example_k)
        if keep:
            example_maps[T] = {**res, "k": k}

    return dict(results), example_maps


def plot_results_vs_T(results, out_dir=".", out_prefix="raman_vs_T"):
    """ Trace I_D/I_G, les fractions de cluster, et la deformation deviatorique vs T. """
    T_values = sorted(results.keys())

    def mean_std(T, key):
        vals = np.array(results[T][key])
        return vals.mean(), vals.std()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))

    means = [mean_std(T, "id_ig")[0] for T in T_values]
    stds = [mean_std(T, "id_ig")[1] for T in T_values]
    ax1.errorbar(T_values, means, yerr=stds, marker='o', capsize=3, color="tab:red")
    ax1.set_xlabel(r"$T$ (K)")
    ax1.set_ylabel(r"$I_D/I_G$")
    ax1.set_title(r"$I_D/I_G$ vs T")
    ax1.grid(alpha=0.3)

    colors = {"f_isolated": "tab:blue", "f_dipole": "tab:green", "f_chain": "tab:orange"}
    labels = {"f_isolated": "Isolated", "f_dipole": "Dipole", "f_chain": "Chain"}
    for key in ["f_isolated", "f_dipole", "f_chain"]:
        means = [mean_std(T, key)[0] for T in T_values]
        stds = [mean_std(T, key)[1] for T in T_values]
        ax2.errorbar(T_values, means, yerr=stds, marker='o', capsize=3,
                     color=colors[key], label=labels[key])
    ax2.set_xlabel(r"$T$ (K)")
    ax2.set_ylabel("Fraction of clusters")
    ax2.set_title("Defect cluster type vs T")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.grid(alpha=0.3)

    means = [mean_std(T, "dev_rms")[0] for T in T_values]
    stds = [mean_std(T, "dev_rms")[1] for T in T_values]
    ax3.errorbar(T_values, means, yerr=stds, marker='o', capsize=3,
                 color="tab:brown", label="All atoms")

    means = [mean_std(T, "dev_rms_defects")[0] for T in T_values]
    stds = [mean_std(T, "dev_rms_defects")[1] for T in T_values]
    ax3.errorbar(T_values, means, yerr=stds, marker='o', capsize=3,
                 color="tab:pink", label="At defects")

    ax3.set_xlabel(r"$T$ (K)")
    ax3.set_ylabel(r"$\varepsilon_{dev}$ (RMS)")
    ax3.set_title("Local shear strain vs T")
    ax3.legend()
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(out_dir, f"{out_prefix}.png")
    pdf_path = os.path.join(out_dir, f"{out_prefix}.pdf")
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=150)
    print(f"Figures sauvegardees : {pdf_path} / {png_path}")


def save_results_data(results, out_dir=".", out_name="raman_data.npz"):
    """ Sauvegarde moyenne+std (sur k) par T pour toutes les observables. """
    T_values = np.array(sorted(results.keys()))
    out = {"T": T_values}
    for key in ["id_ig", "f_isolated", "f_dipole", "f_chain", "mean_size",
                "dev_rms", "dev_rms_defects"]:
        means = np.array([np.mean(results[T][key]) for T in T_values])
        stds = np.array([np.std(results[T][key]) for T in T_values])
        out[f"{key}_mean"] = means
        out[f"{key}_std"] = stds
    out_path = os.path.join(out_dir, out_name)
    np.savez(out_path, **out)
    print(f"Donnees sauvegardees : {out_path}")


def save_example_maps(example_maps, out_dir=".", subdir="raman_maps", sigma_A=5.0):
    """
    Sauvegarde une carte spatiale I_D/I_G (figure + npz) par T -- pour
    voir visuellement si les joints de grains apparaissent (comme un
    mapping Raman experimental), et comment ca evolue avec T.
    Utilise un vmax commun (percentile 99 sur toutes les cartes) pour
    que les couleurs soient comparables d'une T a l'autre.
    """
    full_dir = os.path.join(out_dir, subdir)
    os.makedirs(full_dir, exist_ok=True)

    all_vals = np.concatenate([res["ratio_map"].ravel() for res in example_maps.values()])
    vmax = np.percentile(all_vals, 99)

    for T in sorted(example_maps):
        res = example_maps[T]
        png_path = os.path.join(full_dir, f"T={T}.png")
        plot_raman_map(
            res["ratio_map"], res["Lx"], res["Ly"],
            title=f"T={T}, I_D/I_G={res['global_id_ig']:.3f}",
            save_path=png_path, vmax=vmax,
        )
        plt.close('all')

        npz_path = os.path.join(full_dir, f"T={T}.npz")
        np.savez(npz_path, ratio_map=res["ratio_map"], Lx=res["Lx"], Ly=res["Ly"],
                 global_id_ig=res["global_id_ig"], T=T, k=res["k"])

    print(f"\nCartes I_D/I_G sauvegardees dans {full_dir}/ (vmax commun = {vmax:.4f})")


def save_strain_maps(example_maps, out_dir=".", subdir="strain_maps"):
    """
    Sauvegarde une carte spatiale de deformation deviatorique eps_dev
    (figure + npz) par T -- analogue de l'amplitude locale du splitting
    G+/G-. Meme principe de vmax commun que save_example_maps.
    """
    full_dir = os.path.join(out_dir, subdir)
    os.makedirs(full_dir, exist_ok=True)

    all_vals = np.concatenate([res["dev_map"].ravel() for res in example_maps.values()])
    vmax = np.percentile(all_vals, 99)

    for T in sorted(example_maps):
        res = example_maps[T]
        png_path = os.path.join(full_dir, f"T={T}.png")
        plot_strain_map(
            res["dev_map"], res["Lx"], res["Ly"],
            title=f"T={T}, eps_dev(RMS)={res['dev_rms']:.4f}",
            save_path=png_path, vmax=vmax, cmap='magma',
        )
        plt.close('all')

        npz_path = os.path.join(full_dir, f"T={T}.npz")
        np.savez(npz_path, dev_map=res["dev_map"], Lx=res["Lx"], Ly=res["Ly"],
                 dev_rms=res["dev_rms"], T=T, k=res["k"])

    print(f"\nCartes de deformation sauvegardees dans {full_dir}/ (vmax commun = {vmax:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", default=".",
                         help="Dossier contenant les T=*_k=*.npz")
    parser.add_argument("--sigma", type=float, default=25.0,
                         help="Largeur du noyau gaussien en Angstrom "
                              "(taille effective du 'spot laser'). Affecte "
                              "uniquement l'aspect des cartes spatiales, pas "
                              "le scalaire I_D/I_G global. ~25-30 A donnait "
                              "le meilleur contraste grain/joint de grain "
                              "sur les tests, contre du bruit filamentaire a 5 A.")
    parser.add_argument("--grid", type=int, default=128,
                         help="Resolution de la grille de binning")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        raise NotADirectoryError(f"{args.folder} n'est pas un dossier valide")

    results, example_maps = collect_results_raman(
        args.folder, n_grid=args.grid, sigma_A=args.sigma)
    plot_results_vs_T(results, out_dir=args.folder)
    save_results_data(results, out_dir=args.folder)
    save_example_maps(example_maps, out_dir=args.folder, sigma_A=args.sigma)
    save_strain_maps(example_maps, out_dir=args.folder)