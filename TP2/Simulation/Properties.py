import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from collections import defaultdict

def build_neighbor_array(xy, Lx, Ly, bond_length=1.42, tol=0.15, max_deg=8):
    """
    Construit un tableau (N, max_deg) de voisins, tries par angle CCW,
    pad avec -1. Format compatible numba (pas de listes python).
    """
    cutoff = bond_length * (1 + tol)
    tree = cKDTree(xy, boxsize=[Lx, Ly])
    pairs = np.array(list(tree.query_pairs(cutoff)))

    N = len(xy)
    counts = np.zeros(N, dtype=np.int64)
    nb = -np.ones((N, max_deg), dtype=np.int64)

    for i, j in pairs:
        if counts[i] < max_deg:
            nb[i, counts[i]] = j
            counts[i] += 1
        if counts[j] < max_deg:
            nb[j, counts[j]] = i
            counts[j] += 1

    nb_sorted, deg = _sort_neighbors_ccw(xy, nb, counts, Lx, Ly)
    return nb_sorted, deg


@njit(parallel=True)
def _sort_neighbors_ccw(xy, nb, counts, Lx, Ly):
    N = xy.shape[0]
    max_deg = nb.shape[1]
    nb_sorted = -np.ones((N, max_deg), dtype=np.int64)
    for i in prange(N):
        d = counts[i]
        if d == 0:
            continue
        angles = np.empty(d, dtype=np.float64)
        for k in range(d):
            j = nb[i, k]
            dx = xy[j, 0] - xy[i, 0]
            dy = xy[j, 1] - xy[i, 1]
            dx -= Lx * round(dx / Lx)
            dy -= Ly * round(dy / Ly)
            angles[k] = np.arctan2(dy, dx)
        order = np.argsort(angles)
        for k in range(d):
            nb_sorted[i, k] = nb[i, order[k]]
    return nb_sorted, counts


# Anneaux (face-tracing en numba)

@njit
def find_rings_numba(nb_sorted, deg, max_ring_size=12):
    N = nb_sorted.shape[0]
    max_deg = nb_sorted.shape[1]
    visited = np.zeros((N, max_deg), dtype=np.bool_)

    ring_sizes = np.empty(N * max_deg, dtype=np.int64)
    n_rings = 0
    n_excluded = 0

    for i in range(N):
        for si in range(deg[i]):
            if visited[i, si]:
                continue
            j = nb_sorted[i, si]
            start_i, start_j = i, j
            cur_a, cur_b, cur_slot = i, j, si
            length = 0
            broken = False

            while True:
                visited[cur_a, cur_slot] = True
                length += 1

                idx = -1
                for k in range(deg[cur_b]):
                    if nb_sorted[cur_b, k] == cur_a:
                        idx = k
                        break
                if idx == -1:
                    broken = True
                    break

                next_slot = (idx - 1) % deg[cur_b]
                c = nb_sorted[cur_b, next_slot]

                if cur_b == start_i and c == start_j:
                    break

                cur_a, cur_b, cur_slot = cur_b, c, next_slot

                if length > max_ring_size:
                    broken = True
                    break

            if broken:
                n_excluded += 1
            else:
                ring_sizes[n_rings] = length
                n_rings += 1

    return ring_sizes[:n_rings], n_excluded


def ring_size_distribution(ring_sizes):
    sizes, counts = np.unique(ring_sizes, return_counts=True)
    total = counts.sum()
    return {int(s): 100.0 * c / total for s, c in zip(sizes, counts)}


# Sous-reseau A/B (BFS, rapide meme en python pour ce cout)

def assign_sublattice(nb_sorted, deg):
    N = nb_sorted.shape[0]
    label = np.full(N, -1, dtype=np.int64)
    for start in range(N):
        if label[start] != -1 or deg[start] == 0:
            continue
        label[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            i = queue[head]; head += 1
            for k in range(deg[i]):
                j = nb_sorted[i, k]
                if label[j] == -1:
                    label[j] = 1 - label[i]
                    queue.append(j)
    return label


# S(q) : elargissement du 1er pic (numba, sommation directe)

def first_peak_qmag(a_CC=1.42):
    return 4 * np.pi / (3 * a_CC)

@njit(parallel=True)
def structure_factor_ring(xy, qmag, n_theta=720):
    N = xy.shape[0]
    theta = np.arange(n_theta) * (2 * np.pi / n_theta)
    Sq = np.zeros(n_theta)
    for t in prange(n_theta):
        qx = qmag * np.cos(theta[t])
        qy = qmag * np.sin(theta[t])
        re = 0.0
        im = 0.0
        for i in range(N):
            phase = qx * xy[i, 0] + qy * xy[i, 1]
            re += np.cos(phase)
            im += np.sin(phase)
        Sq[t] = (re * re + im * im) / N
    return theta, Sq

def m6_from_profile(theta, Sq):
    m6 = np.sum(Sq * np.exp(1j * 6 * theta)) / np.sum(Sq)
    return m6


# Visualisation du profil de diffraction

def plot_Sq_profile(theta, Sq, title="", save_path=None, q_values=None, Sq_radial=None):
    """
    Trace le profil de S(q) le long du cercle : vue polaire (comme une
    image de diffraction) + vue lineaire (angle vs intensite, pour
    quantifier la largeur des pics).
    """
    import matplotlib.pyplot as plt

    n_panels = 3 if q_values is not None else 2

    fig = plt.figure(figsize=(5.5 * n_panels, 5))

    # --- vue polaire (comme une figure de diffraction) ---
    ax1 = fig.add_subplot(1, n_panels, 1, projection='polar')
    ax1.plot(theta, Sq, color='black', lw=1)
    ax1.fill(theta, Sq, alpha=0.3)
    ax1.set_title(f"S(q) along the first peak\n{title}")
    ax1.set_yticklabels([])

    # --- vue lineaire (plus facile a lire quantitativement) ---
    ax2 = fig.add_subplot(1, n_panels, 2)
    ax2.plot(np.degrees(theta), Sq, color='tab:blue')
    ax2.set_xlabel("angle (deg)")
    ax2.set_ylabel("S(q)")
    ax2.set_title("Angular profile of S(q)")
    ax2.grid(alpha=0.3)

    # --- coupe radiale ---
    if q_values is not None:
        ax3 = fig.add_subplot(1, n_panels, 3)
        ax3.plot(q_values, Sq_radial, color='tab:red')
        ax3.set_xlabel(r"$|q|$ ($\AA^{-1}$)")
        ax3.set_ylabel("S(q)")
        ax3.set_title("Radial profile of S(q)")
        ax3.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure sauvegardee: {save_path}")
    return fig

# 6) Elargissement RADIAL du premier pic (ordre translationnel)

@njit(parallel=True)
def structure_factor_radial(xy, q_angle, q_values):
    """
    S(q) le long d'une ligne radiale a angle fixe (celui du pic),
    en balayant |q|. Mesure l'elargissement RADIAL (ordre translationnel),
    complementaire de structure_factor_ring (elargissement azimutal).
    Inputs:
        xy : (N,2) positions (sous-reseau A recommande)
        q_angle : angle (rad) de la direction du pic (ex: angle de G_base)
        q_values : array de normes |q| a scanner
    Outputs:
        Sq : intensite a chaque q_values
    """
    N = xy.shape[0]
    n_q = len(q_values)
    Sq = np.zeros(n_q)
    cos_a = np.cos(q_angle)
    sin_a = np.sin(q_angle)
    for t in prange(n_q):
        qx = q_values[t] * cos_a
        qy = q_values[t] * sin_a
        re = 0.0
        im = 0.0
        for i in range(N):
            phase = qx * xy[i, 0] + qy * xy[i, 1]
            re += np.cos(phase)
            im += np.sin(phase)
        Sq[t] = (re * re + im * im) / N
    return Sq
 
 
def radial_peak_width(q_values, Sq):
    """
    Largeur du pic radial (ecart-type pondere par l'intensite autour
    du maximum), en A^-1. Plus la valeur est grande, plus l'ordre
    translationnel est perdu (peak Bragg -> pic diffus).
    """
    q0 = q_values[np.argmax(Sq)]
    weights = Sq / Sq.sum()
    variance = np.sum(weights * (q_values - q0) ** 2)
    return np.sqrt(variance), q0
 
 
def first_peak_angle(a_CC=1.42):
    """ Angle de G_base (meme convention que Observables.py) """
    G_base = np.array([1.0, -1.0 / np.sqrt(3)])
    return np.arctan2(G_base[1], G_base[0])


# ---------------------------------------------------------------------
# 7) Vraie FFT 2D : image du spot de diffraction (ce que Ruslan decrit)
# ---------------------------------------------------------------------

def structure_factor_fft_map(xy, Lx, Ly, q_center, window=0.6, n_grid=2048):
    """
    Calcule S(q) par vraie FFT 2D (binning des positions + np.fft.fft2),
    puis retourne une fenetre zoomee autour du pic q_center. Contrairement
    a structure_factor_ring/radial (sommation directe, plus precise mais
    ciblee sur un cercle/une ligne), ceci donne une image 2D complete du
    spot de diffraction -- utile pour une inspection visuelle directe de
    l'elargissement (radial ET azimutal en meme temps).
    Inputs:
        xy : (N,2) positions (sous-reseau A recommande)
        Lx, Ly : taille de la boite
        q_center : (qx0, qy0) centre de la fenetre (ex: G_base)
        window : demi-largeur de la fenetre en q (A^-1)
        n_grid : resolution du binning (plus grand = plus fin pres du pic)
    Outputs:
        sub_Sq : image 2D (n,n) de S(q) autour du pic
        sub_qx, sub_qy : coordonnees q correspondantes
    """
    hist, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=n_grid,
                                 range=[[0, Lx], [0, Ly]])
    F = np.fft.fftshift(np.fft.fft2(hist))
    Sq = np.abs(F) ** 2 / len(xy)

    qx_freqs = np.fft.fftshift(np.fft.fftfreq(n_grid, d=Lx / n_grid)) * 2 * np.pi
    qy_freqs = np.fft.fftshift(np.fft.fftfreq(n_grid, d=Ly / n_grid)) * 2 * np.pi

    qx0, qy0 = q_center
    ix = np.argmin(np.abs(qx_freqs - qx0))
    iy = np.argmin(np.abs(qy_freqs - qy0))
    nwin = max(1, int(window / (qx_freqs[1] - qx_freqs[0])))

    sub_Sq = Sq[ix - nwin:ix + nwin, iy - nwin:iy + nwin]
    sub_qx = qx_freqs[ix - nwin:ix + nwin]
    sub_qy = qy_freqs[iy - nwin:iy + nwin]
    return sub_Sq, sub_qx, sub_qy


def plot_fft_peak_map(sub_Sq, sub_qx, sub_qy, title="", save_path=None):
    """ Trace la carte 2D du spot de diffraction (imshow). """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sub_Sq.T, origin='lower',
                    extent=[sub_qx[0], sub_qx[-1], sub_qy[0], sub_qy[-1]],
                    cmap='inferno', aspect='auto')
    ax.set_xlabel(r"$q_x$ ($\mathrm{\AA}^{-1}$)")
    ax.set_ylabel(r"$q_y$ ($\mathrm{\AA}^{-1}$)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="S(q)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure sauvegardee: {save_path}")
    return fig


def structure_factor_fft_full(xy, Lx, Ly, q_max, n_grid=2048):
    """
    Comme structure_factor_fft_map, mais centre sur q=0 et couvre tout
    le premier anneau (au lieu d'une fenetre autour d'un seul pic) --
    pour voir les 6 pics d'un coup (solide/hexatique) ou l'anneau
    isotrope complet (liquide).
    Inputs:
        xy : (N,2) positions (sous-reseau A recommande)
        Lx, Ly : taille de la boite
        q_max : rayon max couvert (prendre ~1.3-1.5x le rayon du 1er pic)
        n_grid : resolution du binning
    Outputs:
        sub_Sq, sub_qx, sub_qy : carte 2D centree sur q=0
    """
    hist, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=n_grid,
                                 range=[[0, Lx], [0, Ly]])
    F = np.fft.fftshift(np.fft.fft2(hist))
    Sq = np.abs(F) ** 2 / len(xy)

    qx_freqs = np.fft.fftshift(np.fft.fftfreq(n_grid, d=Lx / n_grid)) * 2 * np.pi
    qy_freqs = np.fft.fftshift(np.fft.fftfreq(n_grid, d=Ly / n_grid)) * 2 * np.pi

    ix = np.where(np.abs(qx_freqs) <= q_max)[0]
    iy = np.where(np.abs(qy_freqs) <= q_max)[0]
    sub_Sq = Sq[ix[0]:ix[-1] + 1, iy[0]:iy[-1] + 1]
    sub_qx = qx_freqs[ix[0]:ix[-1] + 1]
    sub_qy = qy_freqs[iy[0]:iy[-1] + 1]
    return sub_Sq, sub_qx, sub_qy


def plot_fft_full_ring(sub_Sq, sub_qx, sub_qy, title="", save_path=None,
                        log_scale=False, mask_radius=None):
    """
    Trace la carte complete du 1er anneau. Par defaut (log_scale=False),
    masque le pic central (q=0, bien plus intense que les pics d'ordre 1)
    pour que l'echelle lineaire reste lisible sur les 6 pics -- sinon
    utiliser log_scale=True a la place.
    Inputs:
        mask_radius : rayon (A^-1) autour de q=0 a masquer. Si None,
                      calcule automatiquement (30% du bord de la carte).
    """
    import matplotlib.pyplot as plt
    QX, QY = np.meshgrid(sub_qx, sub_qy, indexing='ij')
    data = sub_Sq.copy()

    if log_scale:
        data = np.log10(data + 1)
        label = "log10(S(q)+1)"
    else:
        if mask_radius is None:
            mask_radius = 0.3 * min(abs(sub_qx[0]), abs(sub_qx[-1]))
        mask = (QX**2 + QY**2) < mask_radius**2
        data = data.astype(float)
        data[mask] = np.nan
        label = "S(q)"

    cmap = plt.get_cmap('inferno').copy()
    cmap.set_bad('black')

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(data.T, origin='lower',
                    extent=[sub_qx[0], sub_qx[-1], sub_qy[0], sub_qy[-1]],
                    cmap=cmap, aspect='auto')
    ax.set_xlabel(r"$q_x$ ($\mathrm{\AA}^{-1}$)")
    ax.set_ylabel(r"$q_y$ ($\mathrm{\AA}^{-1}$)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=label)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure sauvegardee: {save_path}")
    return fig

#  ---------------------------------------------------------------------
# 8) Anneaux AVEC la liste des atomes (extension de find_rings_numba)
# ---------------------------------------------------------------------
 
@njit
def find_rings_with_atoms_numba(nb_sorted, deg, max_ring_size=12):
    """
    Meme face-tracing que find_rings_numba, mais garde en plus la liste
    des atomes de chaque anneau (necessaire pour localiser spatialement
    les defauts, pas juste compter leur proportion globale).
    Outputs:
        ring_sizes : (n_rings,)
        ring_atoms : (n_rings, max_ring_size), pad -1
        n_excluded : int
    """
    N = nb_sorted.shape[0]
    max_deg = nb_sorted.shape[1]
    visited = np.zeros((N, max_deg), dtype=np.bool_)
 
    ring_sizes = np.empty(N * max_deg, dtype=np.int64)
    ring_atoms = -np.ones((N * max_deg, max_ring_size), dtype=np.int64)
    n_rings = 0
    n_excluded = 0
 
    for i in range(N):
        for si in range(deg[i]):
            if visited[i, si]:
                continue
            j = nb_sorted[i, si]
            start_i, start_j = i, j
            cur_a, cur_b, cur_slot = i, j, si
            length = 0
            broken = False
            buf = np.empty(max_ring_size, dtype=np.int64)
 
            while True:
                visited[cur_a, cur_slot] = True
                if length < max_ring_size:
                    buf[length] = cur_a
                length += 1
 
                idx = -1
                for k in range(deg[cur_b]):
                    if nb_sorted[cur_b, k] == cur_a:
                        idx = k
                        break
                if idx == -1:
                    broken = True
                    break
 
                next_slot = (idx - 1) % deg[cur_b]
                c = nb_sorted[cur_b, next_slot]
 
                if cur_b == start_i and c == start_j:
                    break
 
                cur_a, cur_b, cur_slot = cur_b, c, next_slot
 
                if length > max_ring_size:
                    broken = True
                    break
 
            if broken:
                n_excluded += 1
            else:
                ring_sizes[n_rings] = length
                for k in range(length):
                    ring_atoms[n_rings, k] = buf[k]
                n_rings += 1
 
    return ring_sizes[:n_rings], ring_atoms[:n_rings], n_excluded
 
 
def defect_atom_mask(N, ring_sizes, ring_atoms):
    """
    Marque comme "defectueux" tout atome appartenant a au moins un
    anneau de taille != 6 -- l'analogue topologique exact de ce qui
    active le pic D en Raman (un defaut ou un bord de cristal).
    Inputs:
        N : nombre total d'atomes
        ring_sizes, ring_atoms : sortie de find_rings_with_atoms_numba
    Output:
        is_defect : (N,) bool
    """
    is_defect = np.zeros(N, dtype=bool)
    non_hex = ring_atoms[ring_sizes != 6]
    flat = non_hex[non_hex >= 0]
    is_defect[flat] = True
    return is_defect
 
 
# ---------------------------------------------------------------------
# 9) Carte I_D(r)/I_G(r) -- analogue optique du mapping Raman
# ---------------------------------------------------------------------
 
def raman_id_ig_map(xy, is_defect, Lx, Ly, n_grid=128, sigma_A=5.0):
    """
    Coarse-graine la densite d'atomes defectueux (~I_D) et la densite
    totale (~I_G) par un noyau gaussien periodique de largeur sigma_A
    (en Angstrom) -- l'equivalent de la taille du spot laser en Raman.
    Retourne la carte 2D I_D/I_G ainsi qu'un scalaire global (integre
    sur tout l'echantillon), directement comparable a un ratio I_D/I_G
    mesure experimentalement sur l'ensemble d'un grain/echantillon.
 
    Inputs:
        xy : (N,2) positions de TOUS les atomes (A et B, pas seulement A --
             contrairement a S(q), ici on veut la densite reelle d'atomes)
        is_defect : (N,) bool, sortie de defect_atom_mask
        Lx, Ly : taille de boite (periodique)
        n_grid : resolution de la grille de binning
        sigma_A : largeur (Angstrom) du noyau gaussien. A ajuster selon
                  la taille de boite : trop petit -> bruit atome-par-atome
                  (pas de "spot"), trop grand -> tout se moyenne et les
                  joints de grains disparaissent. Un bon point de depart
                  est une fraction (~5-10%) de la taille de grain typique.
 
    Outputs:
        ratio_map : (n_grid, n_grid), I_D/I_G local
        id_map, ig_map : cartes brutes lissees (densites)
        global_id_ig : scalaire, I_D/I_G integre sur tout l'echantillon
                       (= fraction d'atomes defectueux, independante du
                       choix de sigma_A -- c'est la quantite a tracer vs T)
    """
    defect_xy = xy[is_defect]
 
    ig_hist, _, _ = np.histogram2d(
        xy[:, 0], xy[:, 1], bins=n_grid, range=[[0, Lx], [0, Ly]])
    if defect_xy.shape[0] > 0:
        id_hist, _, _ = np.histogram2d(
            defect_xy[:, 0], defect_xy[:, 1], bins=n_grid, range=[[0, Lx], [0, Ly]])
    else:
        id_hist = np.zeros_like(ig_hist)
 
    sigma_px = sigma_A / (Lx / n_grid)
    ig_map = gaussian_filter(ig_hist, sigma_px, mode='wrap')
    id_map = gaussian_filter(id_hist, sigma_px, mode='wrap')
 
    eps = 1e-12 * max(ig_map.max(), 1.0)
    ratio_map = id_map / (ig_map + eps)
 
    global_id_ig = id_hist.sum() / ig_hist.sum()
 
    return ratio_map, id_map, ig_map, global_id_ig
 
 
def plot_raman_map(ratio_map, Lx, Ly, title="", save_path=None, vmax=None):
    """ Trace la carte spatiale I_D/I_G (imshow), analogue d'un mapping Raman. """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(ratio_map.T, origin='lower', extent=[0, Lx, 0, Ly],
                    cmap='inferno', aspect='equal', vmax=vmax)
    ax.set_xlabel(r"$x$ ($\mathrm{\AA}$)")
    ax.set_ylabel(r"$y$ ($\mathrm{\AA}$)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=r"$I_D/I_G$")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure sauvegardee: {save_path}")
    return fig
 
 
# ---------------------------------------------------------------------
# 10) Analogue I_D/I_D' -- dislocations liees vs disclinaisons libres
# ---------------------------------------------------------------------
#
# KTHNY distingue les deux transitions par le CARACTERE du desordre
# topologique, pas juste sa quantite :
#   - solide -> hexatique : depiegeage de PAIRES de dislocations liees
#     (un anneau 5 et un anneau 7 adjacents, partageant une arete)
#   - hexatique -> liquide : depiegeage de DISCLINAISONS LIBRES
#     (un 5 ou un 7 sans partenaire complementaire adjacent)
# C'est l'exact analogue topologique du rapport experimental I_D/I_D',
# qui differencie le TYPE de defaut (sp3 ~13, lacune ~7, GB ~3.5) plutot
# que juste I_D qui n'en mesure que la quantite.
 
def _ring_ring_edges(ring_sizes, ring_atoms):
    """
    Toutes les paires d'anneaux adjacents (partageant une arete),
    calculees de facon vectorisee -- meme technique que
    classify_defect_clusters (cle d'arete unique + tri), mais SANS
    restriction aux anneaux 5/7. Remplace l'ancienne construction par
    defaultdict Python (ring_adjacency), qui domine le temps de calcul
    sur des systemes reels (~95% du temps total mesure sur un fichier
    a 60k anneaux) et est appelee deux fois pour rien (une fois dans
    classify_dislocations, une fois dans defect_chain_degree).
    Output: r1, r2 : (n_edges,) indices d'anneaux, une paire par arete
            partagee entre exactement 2 anneaux
    """
    n_r, max_size = ring_atoms.shape
    valid = ring_atoms >= 0
    lengths = valid.sum(axis=1)
    shifted = np.full_like(ring_atoms, -1)
    for L in np.unique(lengths):
        if L == 0:
            continue
        rows = np.where(lengths == L)[0]
        shifted[rows, :L] = np.roll(ring_atoms[rows, :L], -1, axis=1)

    ring_idx = np.repeat(np.arange(n_r), max_size)
    a = ring_atoms.ravel()
    b = shifted.ravel()
    keep = (a >= 0) & (b >= 0)
    ring_idx, a, b = ring_idx[keep], a[keep], b[keep]

    key = np.minimum(a, b).astype(np.int64) * (int(a.max()) + 1) + np.maximum(a, b)
    order = np.argsort(key, kind='stable')
    key_sorted = key[order]
    ring_sorted = ring_idx[order]

    same_as_next = key_sorted[:-1] == key_sorted[1:]
    r1 = ring_sorted[:-1][same_as_next]
    r2 = ring_sorted[1:][same_as_next]
    return r1, r2


def ring_adjacency(ring_sizes, ring_atoms):
    """
    Determine quels anneaux partagent une arete (bond) -- chaque arete
    du reseau planaire borde exactement 2 anneaux, ce qui donne
    directement les voisins topologiques de chaque anneau. Conserve
    pour compatibilite/exploration (ex. notebook) ; en interne,
    classify_dislocations et defect_chain_degree n'appellent plus
    cette fonction et utilisent directement _ring_ring_edges (evite la
    conversion couteuse en liste de sets Python quand on n'en a pas
    besoin).
    Output: liste de sets, neighbor_rings[i] = indices des anneaux
            adjacents a l'anneau i.
    """
    r1, r2 = _ring_ring_edges(ring_sizes, ring_atoms)
    neighbor_rings = [set() for _ in range(len(ring_sizes))]
    for i, j in zip(r1.tolist(), r2.tolist()):
        neighbor_rings[i].add(j)
        neighbor_rings[j].add(i)
    return neighbor_rings


def classify_dislocations(ring_sizes, ring_atoms):
    """
    Classe chaque anneau 5 ou 7 comme "apparie" (fait partie d'une
    dislocation liee : un 5 et un 7 voisins, partageant une arete) ou
    "libre" (disclinaison non appariee -- pas de partenaire
    complementaire adjacent). Les anneaux 6 (et autres tailles) ne sont
    pas concernes, is_paired vaut False pour eux par defaut.
    Output:
        is_paired : (n_rings,) bool
    """
    r1, r2 = _ring_ring_edges(ring_sizes, ring_atoms)
    complementary = ((ring_sizes[r1] == 5) & (ring_sizes[r2] == 7)) | \
                    ((ring_sizes[r1] == 7) & (ring_sizes[r2] == 5))

    is_paired = np.zeros(len(ring_sizes), dtype=bool)
    is_paired[r1[complementary]] = True
    is_paired[r2[complementary]] = True
    return is_paired
 
 
def dislocation_disclination_fractions(ring_sizes, is_paired):
    """
    Reduit la classification par anneau a des scalaires globaux.
    Outputs:
        n_paired : nb d'anneaux 5/7 dans une paire de dislocation
        n_free : nb d'anneaux 5/7 libres (disclinaisons non appariees)
        free_fraction : n_free / (n_free + n_paired), borne [0,1] --
                        0 = tous les defauts sont des dislocations liees
                        (attendu solide/hexatique), 1 = tous sont des
                        disclinaisons libres (attendu liquide). C'est
                        la quantite a tracer vs T (bien definie partout).
        id_idprime : n_free / n_paired, analogue direct du rapport
                     experimental I_D/I_D' (diverge si n_paired -> 0,
                     donc plus bruite a basse T -- garder free_fraction
                     comme observable principale).
    """
    non_hex = (ring_sizes == 5) | (ring_sizes == 7)
    n_paired = int(np.sum(is_paired[non_hex]))
    n_free = int(np.sum(non_hex) - n_paired)
    denom = n_free + n_paired
    free_fraction = n_free / denom if denom > 0 else np.nan
    id_idprime = n_free / n_paired if n_paired > 0 else np.inf
    return n_paired, n_free, free_fraction, id_idprime
 
 
def defect_type_atom_masks(N, ring_sizes, ring_atoms, is_paired):
    """
    Comme defect_atom_mask, mais separe les atomes appartenant a une
    dislocation liee de ceux appartenant a une disclinaison libre --
    pour une carte spatiale distinguant les deux types de defaut
    (utile pour verifier que les disclinaisons libres apparaissent bien
    plutot au coeur des joints de grains desordonnes en regime liquide).
    """
    is_paired_atom = np.zeros(N, dtype=bool)
    is_free_atom = np.zeros(N, dtype=bool)
 
    non_hex_idx = np.where((ring_sizes == 5) | (ring_sizes == 7))[0]
    for ridx in non_hex_idx:
        atoms = ring_atoms[ridx]
        atoms = atoms[atoms >= 0]
        if is_paired[ridx]:
            is_paired_atom[atoms] = True
        else:
            is_free_atom[atoms] = True
 
    return is_paired_atom, is_free_atom
 
 
# ---------------------------------------------------------------------
# 11) Composantes connexes de defauts -- separer defaut thermique
#     (isole) de segment de joint de grain structurel (chaine longue)
# ---------------------------------------------------------------------
#
# classify_dislocations/is_paired ci-dessus comptait toute une CHAINE de
# joint de grain comme "appariee" des qu'un anneau a un seul voisin
# complementaire -- ce qui noie le signal thermique dans le fond
# structurel des joints de grains (fixes des T~0 par la construction
# Voronoi). Ici on regroupe plutot les anneaux 5/7 en composantes
# connexes du sous-graphe restreint aux anneaux 5/7 entre eux, et on
# classe chaque COMPOSANTE (pas chaque anneau) par sa taille :
#   - taille 1 : disclinaison isolee (vraiment libre, aucun voisin 5/7)
#   - taille 2 (un 5 + un 7) : dislocation isolee -- le vrai dipole
#     lie de KTHNY, distinct d'un simple maillon de chaine
#   - taille >= 3 : segment etendu -- typiquement un morceau de joint
#     de grain structurel (chaine 5-7-5-7...), presente meme a basse T
 
def classify_defect_clusters(ring_sizes, ring_atoms):
    """
    Version vectorisee (numpy + scipy.sparse.csgraph, code C compile)
    de la meme idee que ci-dessus -- BEAUCOUP plus rapide que des
    boucles Python pour les gros systemes (N ~ 1e5 atomes).
 
    Output:
        component_id : (n_rings,) int, -1 pour les anneaux 6/autres,
                        sinon indice de la composante connexe
        component_sizes : (n_components,) taille (en nb d'anneaux) de
                           chaque composante
    """
    n_rings = len(ring_sizes)
    non_hex = (ring_sizes == 5) | (ring_sizes == 7)
 
    # 1) Aplatir tous les anneaux en aretes (ring_idx, atom_a, atom_b),
    #    par decalage cyclique vectorise (groupe par longueur d'anneau
    #    pour eviter une boucle Python par anneau).
    n_r, max_size = ring_atoms.shape
    valid = ring_atoms >= 0
    lengths = valid.sum(axis=1)
    shifted = np.full_like(ring_atoms, -1)
    for L in np.unique(lengths):
        if L == 0:
            continue
        rows = np.where(lengths == L)[0]
        shifted[rows, :L] = np.roll(ring_atoms[rows, :L], -1, axis=1)
 
    ring_idx = np.repeat(np.arange(n_r), max_size)
    a = ring_atoms.ravel()
    b = shifted.ravel()
    keep = (a >= 0) & (b >= 0)
    ring_idx, a, b = ring_idx[keep], a[keep], b[keep]
 
    # 2) Chaque arete non-orientee est partagee par exactement 2
    #    anneaux -- les retrouver en triant une cle unique par arete et
    #    en reperant les paires consecutives egales (sort au lieu d'un
    #    dict Python).
    key = np.minimum(a, b).astype(np.int64) * (int(a.max()) + 1) + np.maximum(a, b)
    order = np.argsort(key, kind='stable')
    key_sorted = key[order]
    ring_sorted = ring_idx[order]
 
    same_as_next = key_sorted[:-1] == key_sorted[1:]
    r1 = ring_sorted[:-1][same_as_next]
    r2 = ring_sorted[1:][same_as_next]
 
    # 3) Ne garder que les aretes entre deux anneaux non-hexagonaux
    #    (5 ou 7), puis composantes connexes via scipy (C compile).
    both_non_hex = non_hex[r1] & non_hex[r2]
    r1, r2 = r1[both_non_hex], r2[both_non_hex]
 
    graph = coo_matrix((np.ones(len(r1)), (r1, r2)), shape=(n_rings, n_rings))
    _, labels = connected_components(graph, directed=False)
 
    component_id = np.where(non_hex, labels, -1)
    unique_labels, component_sizes = np.unique(labels[non_hex], return_counts=True)
 
    return component_id, component_sizes
 
 
def cluster_size_fractions(component_sizes):
    """
    Reduit les tailles de composantes a des fractions globales, en
    nombre de COMPOSANTES (pas d'anneaux) -- une longue chaine ne
    compte que pour 1, ce qui l'empeche de dominer numeriquement le
    compte a cause de sa taille.
    Outputs:
        f_isolated : fraction de composantes de taille 1
        f_dipole : fraction de composantes de taille 2
        f_chain : fraction de composantes de taille >= 3
        mean_size : taille moyenne des composantes (indicateur continu
                    complementaire, sensible a la croissance des chaines)
    """
    n = len(component_sizes)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    f_isolated = float(np.mean(component_sizes == 1))
    f_dipole = float(np.mean(component_sizes == 2))
    f_chain = float(np.mean(component_sizes >= 3))
    mean_size = float(np.mean(component_sizes))
    return f_isolated, f_dipole, f_chain, mean_size
 
 
# ---------------------------------------------------------------------
# 12) Champ de deformation locale -- analogue du splitting Raman G+/G-
# ---------------------------------------------------------------------
#
# Le splitting G+/G- vient de la levee de degenerescence du mode E2g
# sous contrainte anisotrope (effet Gruneisen) : une deformation
# ISOTROPE decale G globalement (sans le splitter), une deformation
# DEVIATORIQUE (cisaillement/uniaxiale) le splitte en G+/G-.
# Analogue topologique : pour chaque atome 3-coordonne, le tenseur de
# second moment Q = <b_k ⊗ b_k> sur ses 3 vecteurs de liaison est
# ISOTROPE (independamment de l'orientation locale du reseau) si les 3
# liaisons sont a 120° et de meme longueur -- toute anisotropie de Q
# trahit une distorsion locale (contrainte de cisaillement), et toute
# variation de sa trace trahit une dilatation/compression isotrope.
 
@njit(parallel=True)
def local_strain_field(xy, nb_sorted, deg, Lx, Ly, a_CC=1.42):
    """
    Outputs (arrays (N,), NaN pour les atomes non 3-coordonnes -- bords
    d'anneaux exclus/casses) :
        eps_hydro : (L_moyenne - a_CC)/a_CC -- deformation isotrope,
                    analogue du decalage global du pic G (parametre de
                    Gruneisen isotrope)
        eps_dev : sqrt((Qxx-Qyy)^2 + 4*Qxy^2) / (Qxx+Qyy) -- anisotropie
                  normalisee (sans dimension, ~0 a ~1) du tenseur de
                  liaison, analogue de l'amplitude du splitting G+/G-
    """
    N = xy.shape[0]
    eps_hydro = np.full(N, np.nan)
    eps_dev = np.full(N, np.nan)
 
    for i in prange(N):
        if deg[i] != 3:
            continue
        bx = np.empty(3)
        by = np.empty(3)
        for k in range(3):
            j = nb_sorted[i, k]
            dx = xy[j, 0] - xy[i, 0]
            dy = xy[j, 1] - xy[i, 1]
            dx -= Lx * round(dx / Lx)
            dy -= Ly * round(dy / Ly)
            bx[k] = dx
            by[k] = dy
 
        Lsum = 0.0
        Qxx = 0.0
        Qyy = 0.0
        Qxy = 0.0
        for k in range(3):
            Lk = np.sqrt(bx[k] * bx[k] + by[k] * by[k])
            Lsum += Lk
            Qxx += bx[k] * bx[k]
            Qyy += by[k] * by[k]
            Qxy += bx[k] * by[k]
        Qxx /= 3.0
        Qyy /= 3.0
        Qxy /= 3.0
 
        eps_hydro[i] = (Lsum / 3.0 - a_CC) / a_CC
        trace = Qxx + Qyy
        if trace > 0.0:
            eps_dev[i] = np.sqrt((Qxx - Qyy) ** 2 + 4.0 * Qxy ** 2) / trace
 
    return eps_hydro, eps_dev
 
 
def scalar_field_map(xy, values, Lx, Ly, n_grid=128, sigma_A=25.0):
    """
    Carte 2D lissee d'un champ scalaire par atome (ex: eps_dev), meme
    lissage gaussien "spot laser" que raman_id_ig_map. Les NaN (atomes
    non 3-coordonnes) sont exclus du poids comme du compte.
    Outputs:
        field_map : (n_grid, n_grid), moyenne locale ponderee
        global_mean, global_rms : scalaires sur tous les atomes valides
    """
    valid = ~np.isnan(values)
    xy_v = xy[valid]
    val_v = values[valid]
 
    weighted_hist, _, _ = np.histogram2d(
        xy_v[:, 0], xy_v[:, 1], bins=n_grid, range=[[0, Lx], [0, Ly]], weights=val_v)
    count_hist, _, _ = np.histogram2d(
        xy_v[:, 0], xy_v[:, 1], bins=n_grid, range=[[0, Lx], [0, Ly]])
 
    sigma_px = sigma_A / (Lx / n_grid)
    weighted_smooth = gaussian_filter(weighted_hist, sigma_px, mode='wrap')
    count_smooth = gaussian_filter(count_hist, sigma_px, mode='wrap')
 
    eps = 1e-12
    field_map = weighted_smooth / (count_smooth + eps)
    global_mean = float(val_v.mean())
    global_rms = float(np.sqrt(np.mean(val_v ** 2)))
    return field_map, global_mean, global_rms
 
 
def plot_strain_map(field_map, Lx, Ly, title="", save_path=None, vmax=None, cmap='viridis'):
    """ Trace la carte spatiale d'un champ de deformation (imshow). """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(field_map.T, origin='lower', extent=[0, Lx, 0, Ly],
                    cmap=cmap, aspect='equal', vmax=vmax)
    ax.set_xlabel(r"$x$ ($\mathrm{\AA}$)")
    ax.set_ylabel(r"$y$ ($\mathrm{\AA}$)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure sauvegardee: {save_path}")
    return fig
 
 
def masked_rms(values, mask):
    """
    RMS de 'values' restreint aux atomes ou mask est True (et values
    non NaN). Reutilise des champs deja calcules (eps_dev, is_defect)
    sans repasser dessus -- cout quasi nul, juste de l'indexation
    booleenne numpy. Sert a isoler l'intensite de la contrainte AU
    NIVEAU DES DEFAUTS plutot que de la diluer sur tout l'echantillon
    (la plupart des atomes, a l'interieur des grains, ont eps_dev~0
    a toute temperature puisque seule l'orientation des grains est
    thermalisee -- les moyenner avec noie le signal).
    """
    valid = mask & ~np.isnan(values)
    if not np.any(valid):
        return np.nan
    return float(np.sqrt(np.mean(values[valid] ** 2)))


# ---------------------------------------------------------------------
# 13) Centroide d'anneau (unwrapped, periodique) -- brique commune aux
#     deux analyses geometriques ci-dessous
# ---------------------------------------------------------------------

def ring_centroid(ring_atoms, xy, Lx, Ly):
    """
    Position centroide de chaque anneau, moyenne des positions de ses
    atomes en tenant compte de la periodicite (chaque atome est ramene
    a l'image minimale par rapport au premier atome de l'anneau avant
    la moyenne) -- sinon un anneau a cheval sur le bord de boite donne
    un centroide faux (moyenne de deux images opposees).
    Le resultat peut sortir de [0, L) (unwrapped) : c'est voulu, il
    reste correct pour des distances MIC ulterieures entre centroides.
    Inputs:
        ring_atoms : (n_rings, max_size) indices d'atomes, pad -1
        xy : (N,2) positions
        Lx, Ly : taille de boite
    Output:
        centroids : (n_rings, 2)
    """
    n_rings, max_size = ring_atoms.shape
    valid = ring_atoms >= 0
    ref_idx = ring_atoms[:, 0]
    ref_xy = xy[ref_idx]

    acc_x = np.zeros(n_rings)
    acc_y = np.zeros(n_rings)
    for k in range(max_size):
        mask_k = valid[:, k]
        atoms_k = np.where(mask_k, ring_atoms[:, k], ref_idx)
        pos_k = xy[atoms_k]
        dx = pos_k[:, 0] - ref_xy[:, 0]
        dy = pos_k[:, 1] - ref_xy[:, 1]
        dx -= Lx * np.round(dx / Lx)
        dy -= Ly * np.round(dy / Ly)
        acc_x += np.where(mask_k, dx, 0.0)
        acc_y += np.where(mask_k, dy, 0.0)

    counts = valid.sum(axis=1)
    centroids = np.empty((n_rings, 2))
    centroids[:, 0] = ref_xy[:, 0] + acc_x / counts
    centroids[:, 1] = ref_xy[:, 1] + acc_y / counts
    return centroids


def _group_defect_clusters(ring_sizes, ring_atoms):
    """
    Regroupe les anneaux 5/7 par cluster (composante connexe de
    classify_defect_clusters) via UN SEUL tri global des labels, au
    lieu de chercher les membres de chaque cluster un par un
    (np.where(component_id==lbl) repete) -- O(n_rings log n_rings)
    au lieu de O(n_clusters * n_rings), ce qui compte pour les gros
    systemes (N ~ 1e5) ou pres de la transition (beaucoup de clusters).
    Les membres de chaque cluster sont contigus dans idx_sorted, donc
    directement exploitables avec np.add.reduceat pour des sommes
    par groupe sans boucle Python.
    Output:
        component_id : (n_rings,), -1 pour les anneaux 6/autres
        unique_labels : (n_clusters,) valeurs de label des clusters
        start_idx : (n_clusters,) indice de debut de chaque groupe
                    dans idx_sorted
        counts : (n_clusters,) taille (nb d'anneaux) de chaque cluster,
                 alignee avec unique_labels
        idx_sorted : (n_non_hex,) indices d'anneaux tries par cluster
                     -- idx_sorted[start_idx[i]:start_idx[i]+counts[i]]
                     donne les anneaux du cluster i
    """
    component_id, _ = classify_defect_clusters(ring_sizes, ring_atoms)
    idx_non_hex = np.where(component_id >= 0)[0]
    labels_non_hex = component_id[idx_non_hex]

    order = np.argsort(labels_non_hex, kind='stable')
    idx_sorted = idx_non_hex[order]
    labels_sorted = labels_non_hex[order]

    unique_labels, start_idx, counts = np.unique(
        labels_sorted, return_index=True, return_counts=True)

    return component_id, unique_labels, start_idx, counts, idx_sorted


# ---------------------------------------------------------------------
# 14) Distance de separation des dipoles 5-7 (dislocations liees isolees)
# ---------------------------------------------------------------------
#
# KTHNY: la transition solide -> hexatique est le depiegeage de PAIRES
# de disclinaisons liees (dislocations, un 5 et un 7 adjacents). Le
# mecanisme de Kosterlitz-Thouless predit que la separation typique r
# de ces paires liees CROIT a l'approche de Tc puis diverge au
# depiegeage -- c'est une prediction geometrique directe, indepen-
# dante de toute technique experimentale, testable directement sur les
# anneaux deja detectes (classify_defect_clusters isole precisement
# les paires 5-7 isolees, composantes de taille 2).

def dipole_57_separations(ring_sizes, ring_atoms, xy, Lx, Ly):
    """
    Distance (Angstrom) entre le 5 et le 7 de chaque dipole isole
    (dislocation liee non voisine d'un autre defaut 5/7 -- composante
    connexe de taille exactement 2 dans le sous-graphe des anneaux
    5/7, cf. classify_defect_clusters). Les rares composantes de
    taille 2 formees de deux anneaux de meme signe (5-5 ou 7-7
    adjacents, pas un vrai dipole KTHNY) sont exclues.
    Output:
        separations : (n_dipoles,) une valeur par dipole isole trouve
    """
    _, unique_labels, start_idx, counts, idx_sorted = _group_defect_clusters(ring_sizes, ring_atoms)
    centroids = ring_centroid(ring_atoms, xy, Lx, Ly)

    dipole_mask = counts == 2
    m1 = idx_sorted[start_idx[dipole_mask]]
    m2 = idx_sorted[start_idx[dipole_mask] + 1]

    is_57 = ((ring_sizes[m1] == 5) & (ring_sizes[m2] == 7)) | \
            ((ring_sizes[m1] == 7) & (ring_sizes[m2] == 5))
    m1, m2 = m1[is_57], m2[is_57]

    dx = centroids[m1, 0] - centroids[m2, 0]
    dy = centroids[m1, 1] - centroids[m2, 1]
    dx -= Lx * np.round(dx / Lx)
    dy -= Ly * np.round(dy / Ly)

    return np.hypot(dx, dy)


def plot_dipole_separation_hist(separations, title="", save_path=None, bins=30, r_max=None):
    """ Histogramme des distances de separation des dipoles 5-7 d'un fichier. """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))
    if len(separations) > 0:
        rng = (0, r_max) if r_max is not None else (0, separations.max() * 1.05)
        ax.hist(separations, bins=bins, range=rng, color="tab:purple", alpha=0.8)
        ax.axvline(separations.mean(), color="black", ls="--", lw=1,
                   label=f"mean = {separations.mean():.2f} \u00c5")
        ax.legend()
    ax.set_xlabel(r"dipole separation $r$ ($\mathrm{\AA}$)")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure sauvegardee: {save_path}")
    return fig


# ---------------------------------------------------------------------
# 15) Correlation spatiale des tailles de cluster de defauts
# ---------------------------------------------------------------------
#
# Complementaire de la distribution des tailles seule (deja captee par
# cluster_size_fractions) : teste si les GRANDS clusters (segments
# etendus de joint de grain) sont eux-memes regroupes spatialement
# (attendu si les defauts s'organisent le long de structures de joint
# de grain plutot que d'etre distribues au hasard), ou si au contraire
# les tailles de cluster sont spatialement independantes (attendu pour
# des defauts purement thermiques, non correles).

def cluster_centroids_and_sizes(ring_sizes, ring_atoms, xy, Lx, Ly):
    """
    Position centroide (moyenne periodique des centroides de ses
    anneaux membres) et taille (nb d'anneaux) de chaque cluster de
    defauts 5/7 (composante connexe, cf. classify_defect_clusters).
    Outputs:
        centroids : (n_clusters, 2)
        sizes : (n_clusters,) nb d'anneaux par cluster
    """
    _, unique_labels, start_idx, counts, idx_sorted = _group_defect_clusters(ring_sizes, ring_atoms)
    ring_centroids = ring_centroid(ring_atoms, xy, Lx, Ly)

    n_clusters = len(unique_labels)
    ref_idx_per_group = idx_sorted[start_idx]          # 1er anneau de chaque cluster
    group_id = np.repeat(np.arange(n_clusters), counts)  # aligne avec idx_sorted
    ref_xy_per_member = ring_centroids[ref_idx_per_group[group_id]]
    member_xy = ring_centroids[idx_sorted]

    dx = member_xy[:, 0] - ref_xy_per_member[:, 0]
    dy = member_xy[:, 1] - ref_xy_per_member[:, 1]
    dx -= Lx * np.round(dx / Lx)
    dy -= Ly * np.round(dy / Ly)

    sum_dx = np.add.reduceat(dx, start_idx)
    sum_dy = np.add.reduceat(dy, start_idx)

    centroids = np.empty((n_clusters, 2))
    centroids[:, 0] = ring_centroids[ref_idx_per_group, 0] + sum_dx / counts
    centroids[:, 1] = ring_centroids[ref_idx_per_group, 1] + sum_dy / counts

    return centroids, counts


def cluster_size_spatial_correlation(centroids, sizes, Lx, Ly, r_bins):
    """
    Fonction de correlation spatiale des tailles de cluster :
        C(r) = < (s_i - <s>)(s_j - <s>) >_{|r_ij| in bin} / var(s)
    moyennee sur toutes les paires de clusters separees d'une distance
    dans chaque bin de r_bins (distance MIC, periodique). Normalisee
    par var(s) pour que C ~ 1 a tres courte distance (r plus petit que
    la taille typique d'un cluster, peu de paires) et C -> 0 si les
    tailles sont spatialement independantes a grande distance.
    Inputs:
        centroids, sizes : sortie de cluster_centroids_and_sizes
        Lx, Ly : taille de boite
        r_bins : bords des bins de distance (Angstrom), array croissant
    Outputs:
        r_centers : (n_bins,) centre de chaque bin
        C_r : (n_bins,) NaN si aucune paire dans le bin
        n_pairs : (n_bins,) nb de paires par bin (statistique typique-
                  ment faible sur un seul fichier -- cumuler sur les k
                  repetitions d'une meme T avant d'interpreter la forme
                  de C(r), pas seulement sa moyenne/std globale)
    """
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    n_bins = len(r_centers)
    n = len(sizes)
    if n < 2:
        return r_centers, np.full(n_bins, np.nan), np.zeros(n_bins, dtype=int)

    mean_s = sizes.mean()
    var_s = sizes.var()
    ds = sizes - mean_s

    ii, jj = np.triu_indices(n, k=1)
    dx = centroids[ii, 0] - centroids[jj, 0]
    dy = centroids[ii, 1] - centroids[jj, 1]
    dx -= Lx * np.round(dx / Lx)
    dy -= Ly * np.round(dy / Ly)
    r = np.hypot(dx, dy)
    prod = ds[ii] * ds[jj]

    bin_idx = np.digitize(r, r_bins) - 1
    C_r = np.full(n_bins, np.nan)
    n_pairs = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        sel = bin_idx == b
        n_pairs[b] = int(sel.sum())
        if n_pairs[b] > 0 and var_s > 0:
            C_r[b] = float(prod[sel].mean() / var_s)

    return r_centers, C_r, n_pairs


def plot_cluster_size_correlation(r_centers, C_r, n_pairs, title="", save_path=None):
    """
    Trace C(r) (avec les points sans paires simplement absents) et
    annote chaque point avec son nombre de paires -- utile pour juger
    d'un coup d'oeil la fiabilite statistique de chaque bin.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    valid = ~np.isnan(C_r)
    ax.plot(r_centers[valid], C_r[valid], marker='o', color="teal")
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel(r"$r$ ($\mathrm{\AA}$)")
    ax.set_ylabel(r"$C(r)$ (taille de cluster, normalisee)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure sauvegardee: {save_path}")
    return fig


# ---------------------------------------------------------------------
# 16) Densite de disclinaisons libres aux jonctions triples
# ---------------------------------------------------------------------
#
# Une jonction triple (Beausir & Fressengeas, litterature TEM/mecanique
# des dislocations) est le point ou 3 segments de joint de grain se
# rencontrent -- purement topologique ici : dans le sous-graphe des
# anneaux 5/7 (deja construit par ring_adjacency pour
# classify_dislocations), un maillon INTERIEUR de chaine a 2 voisins
# 5/7, une EXTREMITE ou un DIPOLE isole en a 1, et un point de
# BRANCHEMENT (jonction triple ou plus) en a >= 3. Aucune donnee de
# germes/Voronoi n'est necessaire.
#
# Une jonction triple generale (angles de desorientation quelconques)
# porte intrinsequement une charge de disclinaison residuelle (nulle
# seulement dans le cas symetrique a 120 deg) -- on teste ici si les
# disclinaisons LIBRES (non appariees, cf. classify_dislocations) sont
# spatialement enrichies pres de ces jonctions plutot qu'uniformement
# reparties le long des joints de grain.

def defect_chain_degree(ring_sizes, ring_atoms):
    """
    Degre de chaque anneau 5/7 DANS LE SOUS-GRAPHE DES ANNEAUX 5/7
    (nb de voisins -- anneaux partageant une arete -- qui sont
    eux-memes 5 ou 7). 0 pour les anneaux hexagonaux/autres.
        degre 1 : extremite de chaine ou dipole isole
        degre 2 : maillon interieur d'une chaine (joint de grain lineaire)
        degre >= 3 : point de branchement -- jonction triple (ou plus)
    """
    r1, r2 = _ring_ring_edges(ring_sizes, ring_atoms)
    non_hex = (ring_sizes == 5) | (ring_sizes == 7)
    both_non_hex = non_hex[r1] & non_hex[r2]
    r1f, r2f = r1[both_non_hex], r2[both_non_hex]

    degree = np.bincount(
        np.concatenate([r1f, r2f]), minlength=len(ring_sizes))
    return degree


def triple_junction_disclination_enrichment(ring_sizes, ring_atoms, xy, Lx, Ly,
                                             n_random=None, min_degree=3, seed=None):
    """
    Teste l'enrichissement spatial des disclinaisons libres pres des
    jonctions triples via la distance au plus proche voisin (fonction
    G de statistique spatiale), plutot qu'un rayon fixe -- un rayon
    fixe sature des que les disques de voisinage autour de chaque
    jonction se recouvrent, ce qui arrive vite quand il y a beaucoup
    de jonctions (observe sur des systemes reels ~540x545 Angstrom
    avec plusieurs centaines de jonctions).
    Methode (test contre l'hypothese nulle de repartition spatiale
    completement aleatoire, CSR) :
      - d_free : distance (MIC) de chaque disclinaison libre a la
        jonction la plus proche
      - d_random : meme chose pour n_random points uniformement
        distribues dans la boite (baseline CSR)
      Si les disclinaisons libres s'accumulent pres des jonctions,
      d_free doit etre systematiquement plus petite que d_random.
    Inputs:
        n_random : nb de points CSR a tirer (defaut: max(n_free, 200))
        min_degree : degre minimal pour qu'un anneau 5/7 soit une
                     jonction triple (3 par defaut)
        seed : graine du tirage aleatoire (reproductibilite)
    Output: dict avec
        n_junctions, n_free_total, junction_xy, free_xy,
        d_free, d_random (arrays de distances, Angstrom),
        median_free, median_random,
        enrichment (= median_random / median_free ; > 1 = les
        disclinaisons libres sont, en mediane, plus proches des
        jonctions qu'attendu au hasard)
    """
    r1, r2 = _ring_ring_edges(ring_sizes, ring_atoms)
    non_hex = (ring_sizes == 5) | (ring_sizes == 7)

    complementary = ((ring_sizes[r1] == 5) & (ring_sizes[r2] == 7)) | \
                    ((ring_sizes[r1] == 7) & (ring_sizes[r2] == 5))
    is_paired = np.zeros(len(ring_sizes), dtype=bool)
    is_paired[r1[complementary]] = True
    is_paired[r2[complementary]] = True

    both_non_hex = non_hex[r1] & non_hex[r2]
    degree = np.bincount(
        np.concatenate([r1[both_non_hex], r2[both_non_hex]]),
        minlength=len(ring_sizes))

    centroids = ring_centroid(ring_atoms, xy, Lx, Ly)

    junction_mask = non_hex & (degree >= min_degree)
    free_mask = non_hex & (~is_paired)

    n_junctions = int(junction_mask.sum())
    n_free_total = int(free_mask.sum())
    junction_xy = centroids[junction_mask] % [Lx, Ly]
    free_xy = centroids[free_mask] % [Lx, Ly]

    result = {
        "n_junctions": n_junctions, "n_free_total": n_free_total,
        "junction_xy": junction_xy, "free_xy": free_xy,
    }

    if n_junctions == 0 or n_free_total == 0:
        result.update(d_free=np.array([]), d_random=np.array([]),
                       median_free=np.nan, median_random=np.nan, enrichment=np.nan)
        return result

    tree = cKDTree(junction_xy, boxsize=[Lx, Ly])
    d_free, _ = tree.query(free_xy, k=1)

    rng = np.random.default_rng(seed)
    if n_random is None:
        n_random = max(n_free_total, 200)
    random_xy = np.column_stack([
        rng.uniform(0, Lx, n_random), rng.uniform(0, Ly, n_random)])
    d_random, _ = tree.query(random_xy, k=1)

    median_free = float(np.median(d_free))
    median_random = float(np.median(d_random))
    enrichment = median_random / median_free if median_free > 0 else np.inf

    result.update(d_free=d_free, d_random=d_random,
                  median_free=median_free, median_random=median_random,
                  enrichment=enrichment)
    return result


def plot_junction_distance_distribution(d_free, d_random, title="", save_path=None):
    """
    Compare les distributions cumulees (ECDF) de d_free (disclinaisons
    libres -> jonction la plus proche) et d_random (baseline CSR) --
    d_free au-dessus de d_random = enrichissement pres des jonctions.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for d, color, label in [(d_free, "tab:orange", "free disclinations"),
                             (d_random, "tab:gray", "CSR baseline")]:
        if len(d) == 0:
            continue
        d_sorted = np.sort(d)
        ecdf = np.arange(1, len(d_sorted) + 1) / len(d_sorted)
        ax.step(d_sorted, ecdf, where='post', color=color, label=label)

    ax.set_xlabel(r"distance to nearest triple junction ($\mathrm{\AA}$)")
    ax.set_ylabel("cumulative fraction")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure sauvegardee: {save_path}")
    return fig


def plot_triple_junction_map(junction_xy, free_xy, Lx, Ly, radius=None,
                              title="", save_path=None):
    """
    Carte spatiale : jonctions triples (etoiles rouges), disclinaisons
    libres (points oranges), et optionnellement le rayon de voisinage
    utilise (cercles pointilles) -- pour verifier visuellement si les
    points oranges s'accumulent bien autour des etoiles.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 5.5))

    if len(free_xy) > 0:
        ax.scatter(free_xy[:, 0], free_xy[:, 1], s=14, color="tab:orange",
                   label="free disclinations", zorder=2)
    if len(junction_xy) > 0:
        ax.scatter(junction_xy[:, 0], junction_xy[:, 1], s=90, marker='*',
                   color="tab:red", edgecolor='black', linewidth=0.5,
                   label="triple junctions", zorder=3)
        if radius is not None:
            for jx, jy in junction_xy:
                circle = plt.Circle((jx, jy), radius, fill=False,
                                     ls=':', color='tab:red', alpha=0.4)
                ax.add_patch(circle)

    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_xlabel(r"$x$ ($\mathrm{\AA}$)")
    ax.set_ylabel(r"$y$ ($\mathrm{\AA}$)")
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure sauvegardee: {save_path}")
    return fig