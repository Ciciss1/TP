"""
Run separe (independant de Batch_plot.py / Raman_map.py) : etude de la
corrugation hors-plan (z) comme observable de la transition solide/
hexatique/liquide.

z est deja present dans les fichiers relaxes LAMMPS/AIREBO (colonne 3
de 'xyz'). On calcule, comme pour I_D/I_G dans Raman_map.py :
  - RMS(z_centre) global par fichier, moyenne sur les k repetitions
    par T, trace vs T -- pour voir si la corrugation croit avec T
    (deploiement thermique) et/ou montre une signature a la transition.
  - RMS(z_centre) restreint aux atomes defectueux (GB) vs aux atomes
    "bulk" (dans les grains), via le meme masque is_defect que
    Raman_map.py -- pour verifier la prediction Yazyev & Chen / Carraro
    & Nelson que les GB a petit angle (forte densite de dislocations)
    sont preferentiellement corrugues, plutot qu'une simple derive
    thermique uniforme de tout le feuillet.
  - une carte spatiale z_centre(r) par T (un k representatif), lissee
    par le meme noyau gaussien que les cartes I_D/I_G et eps_dev, pour
    voir si les joints de grains apparaissent comme des lignes de
    flambage (hillocks / ripples, cf. Fig. 2 de Yazyev & Chen).

z est centre par soustraction de sa moyenne globale (pas de PBC en z,
donc pas de retrait de tilt de plan -- juste le mode zero trivial de
derive verticale du feuillet libre).

Usage:
    python Corrugation.py /chemin/vers/dossier [--sigma 25.0] [--grid 128]
"""
import os
import re
import glob
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from Properties import (build_neighbor_array,
    find_rings_with_atoms_numba, defect_atom_mask,
    scalar_field_map, plot_strain_map, masked_rms,
)


FNAME_RE = re.compile(r"T=(\d+)_k=(\d+)\.npz$")


def analyze_file_corrugation(path, a_CC=1.42, n_grid=128, sigma_A=25.0):
    """
    Calcule z centre, le masque de defaut (meme detection d'anneaux que
    Raman_map.py), le RMS global / bulk / defauts, et la carte spatiale
    lissee. Garde xy/Lx/Ly pour pouvoir retracer sans tout recalculer.
    """
    data = np.load(path)
    xyz = data['xyz']
    xy = xyz[:, :2]
    z = xyz[:, 2]
    Lx, Ly = data['lattice']

    z_c = z - z.mean()

    nb_sorted, deg = build_neighbor_array(xy, Lx, Ly, bond_length=a_CC)
    ring_sizes, ring_atoms, n_excluded = find_rings_with_atoms_numba(nb_sorted, deg)
    is_defect = defect_atom_mask(len(xy), ring_sizes, ring_atoms)

    rms_global = float(np.sqrt(np.mean(z_c ** 2)))
    rms_defects = masked_rms(z_c, is_defect)
    rms_bulk = masked_rms(z_c, ~is_defect)

    z_map, z_map_mean, z_map_rms = scalar_field_map(
        xy, z_c, Lx, Ly, n_grid=n_grid, sigma_A=sigma_A)

    return {
        "rms_global": rms_global, "rms_defects": rms_defects, "rms_bulk": rms_bulk,
        "z_map": z_map, "xy": xy, "Lx": Lx, "Ly": Ly,
        "n_excluded": n_excluded, "n_defect": int(is_defect.sum()), "N": len(xy),
    }


def collect_results_corrugation(folder, a_CC=1.42, n_grid=128, sigma_A=25.0, example_k=None):
    """
    Parcourt tous les T=*_k=*.npz du dossier. Regroupe les RMS par T
    (une valeur par k), et garde la carte d'UN fichier representatif
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
            res = analyze_file_corrugation(path, a_CC=a_CC, n_grid=n_grid, sigma_A=sigma_A)
        except Exception as e:
            print(f"ECHEC ({e})")
            continue

        results[T]["rms_global"].append(res["rms_global"])
        results[T]["rms_defects"].append(res["rms_defects"])
        results[T]["rms_bulk"].append(res["rms_bulk"])
        print(f"OK  RMS(z)={res['rms_global']:.4f}  "
              f"bulk={res['rms_bulk']:.4f}  defects={res['rms_defects']:.4f}  "
              f"excl={res['n_excluded']}")

        keep = (T not in example_maps) if example_k is None else (k == example_k)
        if keep:
            example_maps[T] = {**res, "k": k}

    return dict(results), example_maps


def plot_results_vs_T(results, out_dir=".", out_prefix="corrugation_vs_T"):
    """ Trace RMS(z) global et bulk vs defauts, en fonction de T. """
    T_values = sorted(results.keys())

    def mean_std(T, key):
        vals = np.array(results[T][key])
        return vals.mean(), vals.std()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    means = [mean_std(T, "rms_global")[0] for T in T_values]
    stds = [mean_std(T, "rms_global")[1] for T in T_values]
    ax1.errorbar(T_values, means, yerr=stds, marker='o', capsize=3, color="tab:purple")
    ax1.set_xlabel(r"$T$ (K)")
    ax1.set_ylabel(r"RMS$(z)$ ($\mathrm{\AA}$)")
    ax1.set_title("Corrugation globale vs T")
    ax1.grid(alpha=0.3)

    means_bulk = [mean_std(T, "rms_bulk")[0] for T in T_values]
    stds_bulk = [mean_std(T, "rms_bulk")[1] for T in T_values]
    ax2.errorbar(T_values, means_bulk, yerr=stds_bulk, marker='o', capsize=3,
                 color="tab:brown", label="Bulk (intra-grain)")

    means_def = [mean_std(T, "rms_defects")[0] for T in T_values]
    stds_def = [mean_std(T, "rms_defects")[1] for T in T_values]
    ax2.errorbar(T_values, means_def, yerr=stds_def, marker='o', capsize=3,
                 color="tab:red", label="Defauts (GB)")

    ax2.set_xlabel(r"$T$ (K)")
    ax2.set_ylabel(r"RMS$(z)$ ($\mathrm{\AA}$)")
    ax2.set_title("Corrugation : bulk vs joints de grains")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(out_dir, f"{out_prefix}.png")
    pdf_path = os.path.join(out_dir, f"{out_prefix}.pdf")
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=150)
    print(f"Figures sauvegardees : {pdf_path} / {png_path}")


def save_results_data(results, out_dir=".", out_name="corrugation_data.npz"):
    """ Sauvegarde moyenne+std (sur k) par T pour toutes les observables. """
    T_values = np.array(sorted(results.keys()))
    out = {"T": T_values}
    for key in ["rms_global", "rms_bulk", "rms_defects"]:
        means = np.array([np.mean(results[T][key]) for T in T_values])
        stds = np.array([np.std(results[T][key]) for T in T_values])
        out[f"{key}_mean"] = means
        out[f"{key}_std"] = stds
    out_path = os.path.join(out_dir, out_name)
    np.savez(out_path, **out)
    print(f"Donnees sauvegardees : {out_path}")


def save_example_maps(example_maps, out_dir=".", subdir="corrugation_maps", sigma_A=25.0):
    """
    Sauvegarde une carte spatiale z_centre (figure + npz) par T -- pour
    voir si les joints de grains apparaissent comme des lignes de
    flambage, et comment l'amplitude evolue avec T. vmax symetrique
    (percentile 99 en valeur absolue) commun a toutes les T, et cmap
    divergent (le signe de z compte, contrairement a eps_dev).
    """
    full_dir = os.path.join(out_dir, subdir)
    os.makedirs(full_dir, exist_ok=True)

    all_vals = np.concatenate([res["z_map"].ravel() for res in example_maps.values()])
    vmax = np.percentile(np.abs(all_vals), 99)

    for T in sorted(example_maps):
        res = example_maps[T]
        png_path = os.path.join(full_dir, f"T={T}.png")
        plot_strain_map(
            res["z_map"], res["Lx"], res["Ly"],
            title=f"T={T}, RMS(z)={res['rms_global']:.4f} A",
            save_path=png_path, vmax=vmax, cmap='RdBu_r',
        )
        plt.close('all')

        npz_path = os.path.join(full_dir, f"T={T}.npz")
        np.savez(npz_path, z_map=res["z_map"], Lx=res["Lx"], Ly=res["Ly"],
                 rms_global=res["rms_global"], T=T, k=res["k"])

    print(f"\nCartes de corrugation sauvegardees dans {full_dir}/ (vmax commun = {vmax:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", default=".",
                         help="Dossier contenant les T=*_k=*.npz")
    parser.add_argument("--sigma", type=float, default=25.0,
                         help="Largeur du noyau gaussien en Angstrom, "
                              "memes valeurs que pour les cartes de "
                              "deformation dans Raman_map.py.")
    parser.add_argument("--grid", type=int, default=128,
                         help="Resolution de la grille de binning")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        raise NotADirectoryError(f"{args.folder} n'est pas un dossier valide")

    results, example_maps = collect_results_corrugation(
        args.folder, n_grid=args.grid, sigma_A=args.sigma)
    plot_results_vs_T(results, out_dir=args.folder)
    save_results_data(results, out_dir=args.folder)
    save_example_maps(example_maps, out_dir=args.folder, sigma_A=args.sigma)
