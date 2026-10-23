"""
Sensitivity of the observables to KAPPA: the crystals rebuilt by kappa_scan.py
(<out>_crystals/T=<T>K_k=<k>_kappa=<kappa>.npz) are analysed for each KAPPA and compared to a reference
KAPPA, as a function of T_MC:
  - fraction of 5 and 7 rings (%), |<psi6>|, G6 at r = L/4, eta6 (G6 ~ r^-eta6)
  - |m6| and radial width of the first diffraction ring
Writes in <folder>/kappa_observables/: observables_vs_T.png (top: values, bottom: difference to the
reference), G6.png (G6(r) at a few T), observables.csv, and prints for each observable the mean
difference to the reference against the scatter of the reference curve from one T to the next.

Usage:
    python kappa_analysis.py /scratch/$USER/kappa/scan_crystals [--ref 1.6] [--kappa 1,1.6]
                                [--T 30,40,60] [--workers 36]
"""
import os
import re
import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "Spe", "Atomical"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import Observables as obs
import Properties as P

FNAME = re.compile(r"T=(\d+)K_k=(\d+)_kappa=([^/]+)\.npz$")
KEYS = [("ring_57", "anneaux 5 + 7 (%)"), ("psi6", r"$|\langle\psi_6\rangle|$"), ("G6_L4", r"$G_6(L/4)$"),
        ("eta6", r"$\eta_6$"), ("m6", r"$|m_6|$ (diffraction)"), ("radial_width", r"largeur radiale (Å$^{-1}$)")]


def analyse(path, n_bins = 40):
    xyz, bonds, Lx, Ly = obs.load_crystal(path)
    sizes, _ = P.rings(xyz, bonds, Lx, Ly)
    r_bins = np.geomspace(2.0, min(Lx, Ly) / 2, n_bins + 1)
    r = np.sqrt(r_bins[:-1] * r_bins[1:])
    G6, _, psi, _ = obs.correlations(xyz, bonds, Lx, Ly, r_bins)
    peak = P.first_peak(xyz[:, :2], Lx, Ly)
    return dict(ring_57=100 * np.mean((sizes == 5) | (sizes == 7)), psi6=abs(psi.mean()),
                G6_L4=float(np.interp(min(Lx, Ly) / 4, r, G6)),
                eta6=obs.power_law_exponent(r, G6, 10.0, min(Lx, Ly) / 4),
                m6=peak["m6"], radial_width=peak["radial_width"], r=r, G6=G6)


def label_of(text):
    return text if text == "fit" else format(float(text), "g")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="the *_crystals folder written by kappa_scan.py")
    ap.add_argument("--ref", default="1.6", help="reference KAPPA")
    ap.add_argument("--kappa", help="KAPPA compared, comma separated (default: all found)")
    ap.add_argument("--T", help="temperatures (10^3 K), comma separated (default: all)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    folder = Path(args.folder)
    out = folder / "kappa_observables"
    out.mkdir(exist_ok=True)

    files = [(int(m.group(1)), int(m.group(2)), label_of(m.group(3)), p)
             for p in folder.glob("T=*K_k=*_kappa=*.npz") if (m := FNAME.search(p.name))]
    ref = label_of(args.ref)
    keep = {label_of(k) for k in args.kappa.split(",")} | {ref} if args.kappa else None
    T_sel = {round(1000 * float(t)) for t in args.T.split(",")} if args.T else None
    files = [f for f in files if (keep is None or f[2] in keep) and (T_sel is None or f[0] in T_sel)]
    if not any(f[2] == ref for f in files):
        raise SystemExit(f"no crystal with the reference kappa={ref} in {folder}")

    with ProcessPoolExecutor(args.workers) as pool:
        results = dict(zip([f[:3] for f in files], pool.map(analyse, [f[3] for f in files])))

    kappas = sorted({f[2] for f in files}, key=lambda k: (k == "fit", float(k) if k != "fit" else 0))
    temps = sorted({f[0] for f in files})

    def curve(kap, key):
        '''mean over k of each T (nan where missing)'''
        return np.array([np.nanmean([v[key] for (T, k, kk), v in results.items() if T == t and kk == kap] or [np.nan])
                         for t in temps])

    with open(out / "observables.csv", "w") as f:
        f.write("T_K,k,kappa," + ",".join(k for k, _ in KEYS) + "\n")
        for (T, k, kap), v in sorted(results.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
            f.write(f"{T},{k},{kap}," + ",".join(f"{v[key]:.6g}" for key, _ in KEYS) + "\n")

    T = np.array(temps) / 1000
    cmap = plt.get_cmap("viridis")
    colors = {kap: ("k" if kap == ref else cmap(i / max(1, len(kappas) - 1))) for i, kap in enumerate(kappas)}
    fig, axes = plt.subplots(2, len(KEYS), figsize=(3.6 * len(KEYS), 6.5), sharex=True)
    print(f"difference to kappa={ref}, mean over T (+- standard error) vs scatter of the reference from one T to the next")
    for j, (key, ylabel) in enumerate(KEYS):
        y_ref = curve(ref, key)
        # scatter of the reference: deviation of each point from the mean of its two neighbours
        noise = np.nanstd(y_ref[1:-1] - 0.5 * (y_ref[:-2] + y_ref[2:])) / np.sqrt(1.5) if len(T) > 2 else np.nan
        line = f"  {key:>13s} (scatter {noise:.3g}):"
        for kap in kappas:
            y = curve(kap, key)
            axes[0, j].plot(T, y, "o-", ms=3, lw=1.5 if kap == ref else 0.9, color=colors[kap], label=f"κ = {kap}")
            if kap != ref:
                d = y - y_ref
                axes[1, j].plot(T, d, "o-", ms=3, lw=0.9, color=colors[kap])
                n = np.isfinite(d).sum()
                line += f"   kappa={kap}: {np.nanmean(d):+.3g} +- {np.nanstd(d) / np.sqrt(max(n, 1)):.2g}"
        print(line)
        axes[1, j].axhline(0, color="k", lw=0.8)
        axes[1, j].axhspan(-noise, noise, color="0.85", zorder=0)
        axes[0, j].set(ylabel=ylabel)
        axes[1, j].set(xlabel=r"$T_{MC}$ ($10^3$ K)", ylabel=f"écart à κ = {ref}")
        for ax in axes[:, j]:
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{folder}   (bande grise : dispersion de la référence d'une T à l'autre)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "observables_vs_T.png", dpi=150)
    plt.close(fig)

    show = [temps[i] for i in np.unique(np.linspace(0, len(temps) - 1, min(4, len(temps))).round().astype(int))]
    fig, axes = plt.subplots(1, len(show), figsize=(4 * len(show), 3.6), squeeze=False)
    for ax, t in zip(axes[0], show):
        for kap in kappas:
            v = [v for (TT, k, kk), v in results.items() if TT == t and kk == kap]
            if v:
                ax.plot(v[0]["r"], np.clip(v[0]["G6"], 1e-3, None), lw=1.5 if kap == ref else 0.9, color=colors[kap], label=f"κ = {kap}")
        ax.set_xscale("log")
        lo, hi = ax.get_ylim()
        if hi / max(lo, 1e-3) > 3:                                   # log scale only when G6 really decays
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
        elif hi - lo < 0.05:                                         # nearly constant: no zoom on the noise
            ax.set_ylim(max(0, (lo + hi) / 2 - 0.05), (lo + hi) / 2 + 0.05)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set(title=f"$T_{{MC}}$ = {t / 1000:g}·10³ K", xlabel="r (Å)", ylabel=r"$G_6(r)$")
        ax.grid(alpha=0.3, which="both")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "G6.png", dpi=150)
    print(f"written in {out}")
