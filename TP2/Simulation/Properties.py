import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree

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