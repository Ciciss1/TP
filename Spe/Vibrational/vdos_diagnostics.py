"""
Diagnostics VDOS pour le projet TP2 (graphène polycristallin) : décalage du pic G
et excès de poids spectral à omega -> 0.

Adapté aux deux formats de fichiers réels du pipeline :

1) Fichier de cristal (sortie Voronoi -> MC -> Lloyd -> LAMMPS), ex. "T10000K_k1.npz" :
   keys utiles : atoms (N,3) [x,y,z en A], bonds (Nb,2) [indices 0-based],
   L (1,) [cote de boite carree, A], check_n_atoms, lammps_fnorm_final,
   lammps_criterion, meta_T_K, meta_epsilon, meta_gamma, meta_phi_s, ...

2) Fichier de VDOS (trajectoire NVT de production), colonnes :
   positions (N,3), velocities (T,N,3), times_ps (T,), equil_steps, equil_temp,
   equil_pe, T, Lx, Ly, dt_ps, seed, n_atoms, converged
"""
import numpy as np
from scipy.optimize import curve_fit

A_CC_PERFECT = 1.3967512290507305  # A, longueur de liaison C-C non contrainte, mesurée sur TON graphène parfait AIREBO relaxé (pas la valeur manuel 1.42 A)
A_PERFECT = np.sqrt(3) / 4 * (2.46) ** 2   # 2.62 A^2/atome
GAMMA_G = 1.8                 # parametre de Gruneisen du mode G (Yoon et al. 2011 ~1.8-2.0)


# =====================================================================
# 1. Fichier de cristal : compression via longueur de liaison réelle
# =====================================================================
def load_crystal(path):
    d = np.load(path)
    return dict(
        atoms=d["atoms"], bonds=d["bonds"], L=float(d["L"][0]),
        n_atoms=int(d["check_n_atoms"]),
        fnorm=d["lammps_fnorm_final"], criterion=d["lammps_criterion"],
        T_K=float(d["meta_T_K"]), epsilon=float(d["meta_epsilon"]),
    )


def bond_length_report(path, label=""):
    c = load_crystal(path)
    atoms, bonds, L = c["atoms"], c["bonds"], c["L"]
    i, j = bonds[:, 0], bonds[:, 1]
    dxy = atoms[i, :2] - atoms[j, :2]
    dxy -= L * np.round(dxy / L)                 # minimum image en xy (periodique)
    dz = atoms[i, 2] - atoms[j, 2]                # z : pas de PBC (feuille finie en z)
    d = np.sqrt((dxy ** 2).sum(1) + dz ** 2)

    eps = 1 - d.mean() / A_CC_PERFECT             # compression linéaire moyenne (>0 = comprimé)
    z = atoms[:, 2]

    print(f"[{label}] N={len(atoms)}  liaisons={len(d)}")
    print(f"[{label}] <d_CC>={d.mean():.4f} A  std={d.std():.4f}  "
          f"(référence non contrainte {A_CC_PERFECT} A)")
    print(f"[{label}] compression moyenne eps = {eps*100:+.2f} %")
    print(f"[{label}] décalage G attendu (Grüneisen) ≈ {2*GAMMA_G*eps*100:+.1f} %")
    print(f"[{label}] std(z)={z.std():.3f} A   ptp(z)={np.ptp(z):.3f} A  (corrugation)")
    print(f"[{label}] convergence LAMMPS : fnorm_final={c['fnorm']}  critère={c['criterion']}")
    if np.any(c["fnorm"] > 1.0):
        print(f"[{label}] ATTENTION : fnorm résiduel élevé -> relaxation probablement "
              f"incomplète (arrêtée par max_iter ou tol énergie, pas par la force). "
              f"Les longueurs de liaison ci-dessus peuvent inclure de la contrainte "
              f"résiduelle non physique.")
    return d, z


# =====================================================================
# 2. Fichier VDOS : VACF par composante (xy vs z), position du pic G,
#    poids spectral basse fréquence
# =====================================================================
def load_vdos_run(path):
    d = np.load(path)
    return dict(
        pos=d["positions"], vel=d["velocities"], times_ps=d["times_ps"],
        dt_ps=float(d["dt_ps"]), Lx=float(d["Lx"]), Ly=float(d["Ly"]),
        T=float(d["T"]), n_atoms=int(d["n_atoms"]),
        converged=bool(d["converged"]),
    )


def vdos(vel, dt_ps, comp="all", window=True, remove_com=True):
    """VDOS normalisée à intégrale=1 sur [0, Nyquist]. comp in {'all','xy','z'}."""
    v = np.asarray(vel, dtype=np.float64)
    if remove_com:
        v = v - v.mean(axis=1, keepdims=True)     # retire la dérive/translation du centre de masse
    idx = {"all": [0, 1, 2], "xy": [0, 1], "z": [2]}[comp]
    v = v[:, :, idx]
    T = v.shape[0]
    n = 2 * T
    F = np.fft.rfft(v, n=n, axis=0)
    acf = np.fft.irfft(F * np.conj(F), axis=0)[:T]
    acf /= (T - np.arange(T))[:, None, None]       # estimateur non biaisé de la VACF
    acf = acf.mean(axis=(1, 2))
    if window:
        acf = acf * np.hanning(2 * T)[T:]
    spec = np.abs(np.fft.rfft(np.r_[acf, acf[-1:0:-1]]))
    dt_fs = dt_ps * 1000.0
    c_cm_per_fs = 2.99792458e-5
    freq = np.fft.rfftfreq(2 * T - 1, d=dt_fs) / c_cm_per_fs   # cm^-1
    spec = spec / np.trapz(spec, freq)
    return freq, spec


def fit_G(freq, spec, lo=1450, hi=1950):
    m = (freq > lo) & (freq < hi)
    f, s = freq[m], spec[m]
    lor = lambda x, a, x0, g, b: a * g ** 2 / ((x - x0) ** 2 + g ** 2) + b
    p0 = [s.max(), f[np.argmax(s)], 30, s.min()]
    p, _ = curve_fit(lor, f, s, p0=p0, maxfev=20000)
    print(f"  G : argmax brut={f[np.argmax(s)]:.0f} cm^-1   "
          f"centre ajusté={p[1]:.1f} cm^-1   HWHM={abs(p[2]):.0f} cm^-1")
    return p[1], abs(p[2])


def low_freq_weight(freq, spec, fmax=30):
    m = freq < fmax
    w = np.trapz(spec[m], freq[m])
    print(f"  poids spectral < {fmax} cm^-1 : {w*100:.2f} % du total")
    return w


def compare_runs(paths_by_label, comps=("all", "xy", "z")):
    """paths_by_label = {'cristal parfait': 'pristine.npz', 'poly T=10000': 'T10000K_k1_thermalized.npz', ...}"""
    results = {}
    for label, path in paths_by_label.items():
        run = load_vdos_run(path)
        print(f"\n=== {label}  (T_cible={run['T']} K, converged={run['converged']}) ===")
        results[label] = {}
        for comp in comps:
            freq, spec = vdos(run["vel"], run["dt_ps"], comp)
            print(f" -- composante {comp} --")
            w = low_freq_weight(freq, spec)
            gpos = gwidth = None
            if comp != "z":
                gpos, gwidth = fit_G(freq, spec)
            results[label][comp] = dict(freq=freq, spec=spec, low_freq_weight=w,
                                          G_pos=gpos, G_width=gwidth)
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        bond_length_report(sys.argv[1], label=sys.argv[1])
