"""
Run separe, meme squelette que LDOS_map.py / Raman_map.py : teste si la
LDOS tight-binding (KPM) distingue les dislocations liees (paire 5-7,
attendu hexatique/ordonne) des disclinaisons libres (attendu liquide/
desordonne) -- plutot que la simple presence d'un defaut (GB vs bulk,
teste par LDOS_map.py, qui melange les deux regimes a tout T).

Reutilise classify_dislocations / defect_type_atom_masks (deja dans
Properties.py, section anneaux -- meme graphe nb_sorted/deg que
Raman_map.py et LDOS_map.py, pas de nouveau calcul geometrique) et
Properties.ldos_paired_vs_free pour la partie KPM.

A T tres bas, il peut y avoir tres peu (voire aucune) disclinaison
libre -- les fichiers ou une des deux classes est vide sont ignores
(voir "skip" dans les logs), pas une erreur.

Usage:
    python LDOS_paired_free_map.py /chemin/vers/dossier [--n-probes 128]
                                    [--n-moments 512] [--backend torch]
"""
import os
import re
import glob
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib.pyplot as plt

from Properties import (
    build_neighbor_array, find_rings_with_atoms_numba, ldos_paired_vs_free,
)


FNAME_RE = re.compile(r"T=(\d+)_k=(\d+)\.npz$")


def _prepare_file(path, a_CC=1.42):
    """ Partie CPU (chargement + graphe de voisinage + anneaux) -- voir LDOS_map.py. """
    data = np.load(path)
    xyz = data['xyz']
    xy = xyz[:, :2]
    Lx, Ly = data['lattice']
    nb_sorted, deg = build_neighbor_array(xy, Lx, Ly, bond_length=a_CC)
    ring_sizes, ring_atoms, n_excluded = find_rings_with_atoms_numba(nb_sorted, deg)
    return dict(xyz=xyz, Lx=Lx, Ly=Ly, nb_sorted=nb_sorted, deg=deg,
                ring_sizes=ring_sizes, ring_atoms=ring_atoms,
                n_excluded=n_excluded, N=len(xyz))


def analyze_prepared_ldos(prep, n_probes_per_class=128, n_moments=512,
                           e_window=0.15, e_grid=None, backend="torch", seed=0):
    out = ldos_paired_vs_free(
        prep["xyz"], prep["nb_sorted"], prep["deg"], prep["Lx"], prep["Ly"],
        prep["ring_sizes"], prep["ring_atoms"],
        n_probes_per_class=n_probes_per_class, n_moments=n_moments,
        e_grid=e_grid, e_window=e_window, seed=seed, backend=backend,
    )
    if out is None:
        return None
    out["N"] = prep["N"]
    out["n_excluded"] = prep["n_excluded"]
    return out


def collect_results_ldos_pf(folder, a_CC=1.42, n_probes_per_class=128, n_moments=512,
                             e_window=0.15, backend="torch", example_k=None, prefetch=True):
    """ Meme logique que collect_results_ldos (LDOS_map.py), classes paired/free. """
    files = sorted(glob.glob(f"{folder}/T=*_k=*.npz"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier T=*_k=*.npz trouve dans {folder}")

    results = defaultdict(lambda: defaultdict(list))
    example_spectra = {}
    e_grid = None

    executor = ThreadPoolExecutor(max_workers=1) if prefetch else None
    next_future = executor.submit(_prepare_file, files[0], a_CC) if executor else None

    for i, path in enumerate(files):
        m = FNAME_RE.search(path)
        if not m:
            print(f"  (ignore, nom non reconnu) {path}")
            if executor and i + 1 < len(files):
                next_future = executor.submit(_prepare_file, files[i + 1], a_CC)
            continue
        T, k = int(m.group(1)), int(m.group(2))

        print(f"T={T} k={k} ...", end=" ", flush=True)
        try:
            if executor:
                prep = next_future.result()
                if i + 1 < len(files):
                    next_future = executor.submit(_prepare_file, files[i + 1], a_CC)
            else:
                prep = _prepare_file(path, a_CC=a_CC)

            res = analyze_prepared_ldos(
                prep, n_probes_per_class=n_probes_per_class, n_moments=n_moments,
                e_window=e_window, e_grid=e_grid, backend=backend,
            )
        except Exception as e:
            print(f"ECHEC ({e})")
            continue

        if res is None:
            print("skip (classe liee ou libre vide sur ce fichier)")
            continue

        if e_grid is None:
            e_grid = res["e_grid"]

        results[T]["paired_zero_E_weight"].append(res["paired_zero_E_weight"])
        results[T]["free_zero_E_weight"].append(res["free_zero_E_weight"])
        print(f"OK  paired_w={res['paired_zero_E_weight']:.5f}  free_w={res['free_zero_E_weight']:.5f}  "
              f"n_paired_atoms={res['n_paired_atoms']}  n_free_atoms={res['n_free_atoms']}  "
              f"excl={res['n_excluded']}")

        keep = (T not in example_spectra) if example_k is None else (k == example_k)
        if keep:
            example_spectra[T] = {**res, "k": k}

    if executor:
        executor.shutdown(wait=True)

    return dict(results), example_spectra, e_grid


def plot_results_vs_T(results, e_window, out_dir=".", out_prefix="ldos_paired_free_vs_T"):
    """ Trace le poids LDOS a energie nulle (liee vs libre) vs T. """
    T_values = sorted(results.keys())

    def mean_std(T, key):
        vals = np.array(results[T][key])
        return vals.mean(), vals.std()

    fig, ax = plt.subplots(figsize=(6.5, 5))

    for key, color, label in [("paired_zero_E_weight", "tab:purple", "dislocations liees (5-7)"),
                               ("free_zero_E_weight", "tab:orange", "disclinaisons libres")]:
        means = [mean_std(T, key)[0] for T in T_values]
        stds = [mean_std(T, key)[1] for T in T_values]
        ax.errorbar(T_values, means, yerr=stds, marker='o', capsize=3, color=color, label=label)

    ax.set_xlabel(r"$T$ (K)")
    ax.set_ylabel(f"LDOS weight, |E|<{e_window} eV")
    ax.set_title("Zero-energy LDOS weight: liees vs libres, vs T")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(out_dir, f"{out_prefix}.png")
    pdf_path = os.path.join(out_dir, f"{out_prefix}.pdf")
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=150)
    print(f"Figures sauvegardees : {pdf_path} / {png_path}")


def save_results_data(results, out_dir=".", out_name="ldos_paired_free_data.npz"):
    T_values = np.array(sorted(results.keys()))
    out = {"T": T_values}
    for key in ["paired_zero_E_weight", "free_zero_E_weight"]:
        means = np.array([np.mean(results[T][key]) for T in T_values])
        stds = np.array([np.std(results[T][key]) for T in T_values])
        out[f"{key}_mean"] = means
        out[f"{key}_std"] = stds
    out_path = os.path.join(out_dir, out_name)
    np.savez(out_path, **out)
    print(f"Donnees sauvegardees : {out_path}")


def save_spectra_plot(example_spectra, e_grid, out_dir=".", out_name="ldos_paired_free_spectra_vs_T.png",
                       n_show=6):
    """ Spectres liee (trait plein) vs libre (pointille) pour quelques T representatifs. """
    T_sorted = sorted(example_spectra.keys())
    show_T = T_sorted[::max(1, len(T_sorted) // n_show)]

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap('viridis')
    for i, T in enumerate(show_T):
        color = cmap(i / max(1, len(show_T) - 1))
        res = example_spectra[T]
        ax.plot(e_grid, res["paired_spectrum"], color=color, ls='-', label=f"T={T} (liee)")
        ax.plot(e_grid, res["free_spectrum"], color=color, ls='--', alpha=0.6)

    ax.set_xlabel("E (eV)")
    ax.set_ylabel("LDOS (a.u.)")
    ax.set_title("Dislocation liee (plein) vs disclinaison libre (pointille) LDOS spectra")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, out_name)
    plt.savefig(out_path, dpi=150)
    print(f"Spectres sauvegardes : {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", default=".",
                         help="Dossier contenant les T=*_k=*.npz")
    parser.add_argument("--n-probes", type=int, default=128,
                         help="Nb d'atomes sondes par classe (liee / libre).")
    parser.add_argument("--n-moments", type=int, default=512,
                         help="Ordre du developpement de Chebyshev (KPM).")
    parser.add_argument("--e-window", type=float, default=0.15,
                         help="Demi-largeur (eV) de la fenetre autour de E=0.")
    parser.add_argument("--backend", choices=["numpy", "torch"], default="torch")
    parser.add_argument("--no-prefetch", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        raise NotADirectoryError(f"{args.folder} n'est pas un dossier valide")

    results, example_spectra, e_grid = collect_results_ldos_pf(
        args.folder, n_probes_per_class=args.n_probes, n_moments=args.n_moments,
        e_window=args.e_window, backend=args.backend, prefetch=not args.no_prefetch)
    plot_results_vs_T(results, args.e_window, out_dir=args.folder)
    save_results_data(results, out_dir=args.folder)
    save_spectra_plot(example_spectra, e_grid, out_dir=args.folder)