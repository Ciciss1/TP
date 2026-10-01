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


def atom_slope_score(atoms, bonds, L, bond_cutoff=2.0):
    """
    Pente locale hors-plan par atome : moyenne de (dz/d_liaison)^2 sur ses liaisons.
    C'est la quantite qui controle la deformation dans le plan induite par un
    deplacement hors-plan en theorie des membranes (terme non-lineaire de von
    Karman, epsilon ~ (grad z)^2/2) -- differente de la courbure (Laplacien de z)
    utilisee par atom_curvature_score : un flanc raide d'ondulation peut etre
    localement "plat" au sens du Laplacien tout en ayant une pente forte.
    """
    N = len(atoms)
    i, j = bonds[:, 0], bonds[:, 1]
    dxy = atoms[i, :2] - atoms[j, :2]
    dxy -= L * np.round(dxy / L)
    dz = atoms[i, 2] - atoms[j, 2]
    dist2 = (dxy ** 2).sum(1) + dz ** 2
    keep = dist2 < bond_cutoff ** 2
    i, j, dz, dist2 = i[keep], j[keep], dz[keep], dist2[keep]
    slope2 = dz ** 2 / dist2   # sin^2(angle de la liaison hors du plan)

    score = np.zeros(N)
    count = np.zeros(N)
    for a, b, s in zip(i, j, slope2):
        score[a] += s; count[a] += 1
        score[b] += s; count[b] += 1
    count[count == 0] = 1
    score /= count

    print(f"  [slope_score] moyenne={score.mean():.5f}  "
          f"p50={np.percentile(score,50):.5f}  p80={np.percentile(score,80):.5f}  "
          f"max={score.max():.5f}")
    return score


def atom_curvature_score(atoms, bonds, L, bond_cutoff=2.0):
    """
    Amplitude de courbure locale hors-plan par atome : |z_atome - moyenne(z_voisins)|
    (Laplacien discret de z). Contrairement a atom_defect_mask (qui mesure la
    distorsion angulaire, sensible a la topologie 5/7-membres), ceci mesure la
    corrugation/flambage, independamment de la deformation dans le plan.
    """
    N = len(atoms)
    i, j = bonds[:, 0], bonds[:, 1]
    dxy = atoms[i, :2] - atoms[j, :2]
    dxy -= L * np.round(dxy / L)
    dz = atoms[i, 2] - atoms[j, 2]
    dist = np.sqrt((dxy ** 2).sum(1) + dz ** 2)
    keep = dist < bond_cutoff
    i, j = i[keep], j[keep]

    neighbors = [[] for _ in range(N)]
    for a, b in zip(i, j):
        neighbors[a].append(b)
        neighbors[b].append(a)

    score = np.zeros(N)
    for a in range(N):
        nb = neighbors[a]
        if len(nb) == 0:
            continue
        score[a] = abs(atoms[nb, 2].mean() - atoms[a, 2])

    print(f"  [curvature_score] moyenne={score.mean():.4f} A  "
          f"p50={np.percentile(score,50):.4f}  p80={np.percentile(score,80):.4f}  "
          f"max={score.max():.4f}")
    return score


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
    vel = d["velocities"]
    times_ps = d["times_ps"]
    n_time = len(times_ps)

    print(f"  [diagnostic chargement] velocities.shape={vel.shape}  "
          f"times_ps: {n_time} points, dt_ps (champ)={float(d['dt_ps']):.6g}")

    # Devine quel axe de `velocities` est le temps, en le comparant a len(times_ps),
    # plutot que de supposer (T, N, 3) a l'aveugle.
    if vel.ndim != 3:
        raise ValueError(f"velocities a {vel.ndim} dimensions, 3 attendues (temps, atomes, xyz)")
    if vel.shape[0] == n_time:
        pass  # deja (T, N, 3)
    elif vel.shape[1] == n_time:
        vel = np.transpose(vel, (1, 0, 2))   # (N, T, 3) -> (T, N, 3)
        print("  [diagnostic chargement] velocities etait (N, T, 3), transpose en (T, N, 3)")
    else:
        print(f"  [diagnostic chargement] ATTENTION : aucun axe de velocities.shape={vel.shape} "
              f"ne correspond a len(times_ps)={n_time} -> le decoupage temps/atomes est incertain, "
              f"resultats a prendre avec precaution")

    # dt reel = espacement median de times_ps, plus fiable que le champ 'dt_ps'
    # (qui peut etre le pas d'integration MD, pas le pas d'echantillonnage des vitesses)
    dt_ps_reel = float(np.median(np.diff(times_ps)))
    print(f"  [diagnostic chargement] dt_ps reel (deduit de times_ps)={dt_ps_reel:.6g}  "
          f"{'(= champ dt_ps)' if abs(dt_ps_reel - float(d['dt_ps'])) < 1e-9 else '(!= champ dt_ps, on utilise celui-ci)'}")

    return dict(
        pos=d["positions"], vel=vel, times_ps=times_ps,
        dt_ps=dt_ps_reel, Lx=float(d["Lx"]), Ly=float(d["Ly"]),
        T=float(d["T"]), n_atoms=int(d["n_atoms"]),
        converged=bool(d["converged"]),
    )


def atom_defect_mask(atoms, bonds, L, threshold_deg=15.0, bond_cutoff=2.0):
    """
    Classe chaque atome comme 'defaut/joint' (True) ou 'coeur' (False), a partir de la
    deviation angulaire de ses liaisons par rapport a 120 deg (hexagonal parfait) --
    plus simple qu'une vraie detection de cycles, mais correle directement avec
    l'appartenance a un anneau non-hexagonal (5/7-membres) pres des joints.

    threshold_deg : deviation max acceptee pour etre considere "coeur"
    bond_cutoff : ignore les pseudo-liaisons de topologie devenues trop longues
                  apres relaxation (cf. les quelques liaisons a >2 A trouvees)
    """
    N = len(atoms)
    i, j = bonds[:, 0], bonds[:, 1]
    dxy = atoms[i, :2] - atoms[j, :2]
    dxy -= L * np.round(dxy / L)
    dz = atoms[i, 2] - atoms[j, 2]
    dist = np.sqrt((dxy ** 2).sum(1) + dz ** 2)
    keep = dist < bond_cutoff
    i, j, dist = i[keep], j[keep], dist[keep]

    vec = np.zeros((N, 3))  # non utilise directement, on construit une liste de voisins
    neighbors = [[] for _ in range(N)]
    for a, b in zip(i, j):
        neighbors[a].append(b)
        neighbors[b].append(a)

    defect_score = np.full(N, np.nan)
    for a in range(N):
        nb = neighbors[a]
        if len(nb) < 2:
            defect_score[a] = 90.0  # atome mal coordonne -> traite comme defaut
            continue
        vecs = []
        for b in nb:
            d = atoms[b, :2] - atoms[a, :2]
            d -= L * np.round(d / L)
            d3 = np.array([d[0], d[1], atoms[b, 2] - atoms[a, 2]])
            vecs.append(d3 / np.linalg.norm(d3))
        max_dev = 0.0
        for p in range(len(vecs)):
            for q in range(p + 1, len(vecs)):
                ang = np.degrees(np.arccos(np.clip(np.dot(vecs[p], vecs[q]), -1, 1)))
                max_dev = max(max_dev, abs(ang - 120.0))
        defect_score[a] = max_dev

    mask = defect_score > threshold_deg
    print(f"  [defect_mask] {mask.sum()}/{N} atomes classes 'joint/defaut' "
          f"({100*mask.mean():.1f} %), seuil={threshold_deg} deg")
    return mask


def vdos(vel, dt_ps, comp="all", window=True, remove_com=True, atom_idx=None):
    """VDOS normalisée à intégrale=1 sur [0, Nyquist]. comp in {'all','xy','z'}.
    atom_idx : indices (ou masque booleen) des atomes a inclure ; None = tous."""
    v = np.asarray(vel, dtype=np.float64)
    if remove_com:
        v = v - v.mean(axis=1, keepdims=True)     # retire la dérive/translation du centre de masse
    if atom_idx is not None:
        v = v[:, atom_idx, :]
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
    spec = spec / np.trapezoid(spec, freq)
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
    w = np.trapezoid(spec[m], freq[m])
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
    if len(sys.argv) < 2:
        print("Usage : python vdos_diagnostics.py fichier1.npz [fichier2.npz ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        keys = np.load(path).files
        print(f"\n########## {path} ##########")
        if "velocities" in keys:
            run = load_vdos_run(path)
            for comp in ("all", "xy", "z"):
                print(f"-- composante {comp} --")
                freq, spec = vdos(run["vel"], run["dt_ps"], comp)
                low_freq_weight(freq, spec)
                if comp != "z":
                    fit_G(freq, spec)
        elif "bonds" in keys:
            bond_length_report(path, label=path)
        else:
            print(f"Format non reconnu, clés trouvées : {keys}")