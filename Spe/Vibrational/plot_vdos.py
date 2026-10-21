"""
VDOS of thermalised crystals (outputs of sweep_thermalize.py, T=<T_MC>K_k=<k>_thermalized_<T_MD>K.npz).
For each file: NVT temperature history and spectrum (total, in-plane, out-of-plane), G-band
position/width and weight below 30 cm^-1, saved as <file>_vdos.png; then all the total spectra
(area 1, coloured by T_MC) in vdos_all_T.png, and the G-peak width (HWHM) and height versus T_MC
in vdos_G_vs_T.png, in the folder of the first file.

Usage:
    python plot_vdos.py <files or folders> [--fmax 2000]
"""
import re
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from thermalize import ThermalizationResult, vdos, fit_G, low_freq_weight


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--fmax", type=float, default=2000.0, help="largest frequency shown (cm^-1)")
    args = ap.parse_args()
    files = [f for p in map(Path, args.paths) for f in (p.rglob("*_thermalized_*.npz") if p.is_dir() else [p])]
    files = sorted((int(re.search(r"T=(\d+)K", f.name).group(1)), f) for f in files)
    if not files:
        raise SystemExit("No *_thermalized_*.npz file")

    cmap = plt.get_cmap("viridis")
    norm = Normalize(files[0][0], files[-1][0])
    fig_all, ax_all = plt.subplots(figsize=(10, 5))
    T_list, hwhm_list, height_list = [], [], []
    print(f"{'file':>45s} {'conv':>5s} {'G (cm-1)':>9s} {'HWHM':>6s} {'height':>9s} {'w<30':>7s}")
    for T_mc, path in files:
        r = ThermalizationResult.load(path)
        dt = r.times_ps[1] - r.times_ps[0]
        freq, spectra = vdos(r.velocities, dt)
        spec = spectra["all"]
        G, hwhm = fit_G(freq, spec)

        m = freq <= args.fmax
        spec_n = spec[m] / (spec[m].sum() * (freq[1] - freq[0]))  # area 1
        win = (freq[m] >= G - hwhm) & (freq[m] <= G + hwhm)
        height = spec_n[win].max()
        T_list.append(T_mc)
        hwhm_list.append(hwhm)
        height_list.append(height)
        print(f"{path.name:>45s} {str(r.converged):>5s} {G:9.1f} {hwhm:6.1f} {height:9.2e} {100 * low_freq_weight(freq, spec):6.2f}%")

        fig, (ax_t, ax_v) = plt.subplots(1, 2, figsize=(18, 4.5), gridspec_kw={"width_ratios": [1, 3]})
        ax_t.plot(r.equil_steps, r.equil_temp)
        ax_t.axhline(r.T, color="r", ls="--", label=f"target {r.T:.0f} K")
        ax_t.set(xlabel="NVT step", ylabel="T (K)", title=f"Equilibration (converged: {r.converged})")
        ax_t.legend()
        for comp, s in spectra.items():
            ax_v.plot(freq[m], s[m], label=comp, lw=1.5 if comp == "all" else 0.8)
        ax_v.axvline(G, color="k", ls=":", lw=0.8)
        ax_v.set(xlabel=r"frequency (cm$^{-1}$)", ylabel="VDOS", title=f"{path.stem}   G = {G:.0f} cm$^{{-1}}$")
        ax_v.legend()
        fig.tight_layout()
        fig.savefig(path.with_name(f"{path.stem}_vdos.png"), dpi=150)
        plt.close(fig)

        ax_all.plot(freq[m], spec_n, color=cmap(norm(T_mc)), lw=0.8)

    fig_all.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax_all, label=r"$T_{MC}$ (K)")
    ax_all.set(xlabel=r"frequency (cm$^{-1}$)", ylabel="VDOS", xlim=(0, args.fmax), title=files[0][1].parent.name)
    fig_all.tight_layout()
    fig_all.savefig(files[0][1].parent / "vdos_all_T.png", dpi=150)

    fig_g, (ax_w, ax_h) = plt.subplots(1, 2, figsize=(11, 4))
    ax_w.plot(T_list, hwhm_list, "o-")
    ax_w.set(xlabel=r"$T_{MC}$ (K)", ylabel=r"G width, HWHM (cm$^{-1}$)")
    ax_h.plot(T_list, height_list, "o-", color="C1")
    ax_h.set(xlabel=r"$T_{MC}$ (K)", ylabel="G height (VDOS, area 1)")
    fig_g.suptitle(files[0][1].parent.name)
    fig_g.tight_layout()
    fig_g.savefig(files[0][1].parent / "vdos_G_vs_T.png", dpi=150)