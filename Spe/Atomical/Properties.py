"""
Structural observables of the relaxed polycrystalline graphene crystals written by
TP2/Simulation/Run_Simulation.py (files T=<T>K_k=<k>.npz: atoms, bonds, lattice = [Lx, Ly]).

The topology is read from the saved bonds instead of being rebuilt from distances (atoms with a number
of bonds other than 3, left by the construction, only produce unusual rings).
"""
import re
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

FNAME = re.compile(r"T=(\d+)K_k=(\d+)\.npz$")


# ─────────────────────────────── Reading ───────────────────────────────

def crystal_files(folder):
    '''
    (T in K, k, path) of the crystals of a folder, sorted by T then k
    '''
    files = sorted((int(m.group(1)), int(m.group(2)), p)
                   for p in Path(folder).glob("T=*K_k=*.npz") if (m := FNAME.search(p.name)))
    if not files:
        raise FileNotFoundError(f"No T=*K_k=*.npz crystal in {folder}")
    return files


def load_crystal(path):
    '''
    Relaxed positions (N, 3), bonds (Nb, 2) and box size Lx, Ly of a crystal
    '''
    d = np.load(path)
    Lx, Ly = (float(v) for v in d["lattice"])
    return np.asarray(d["atoms"], dtype=np.float64), np.asarray(d["bonds"], dtype=np.int64), Lx, Ly


def min_image(d, Lx, Ly):
    d = np.array(d, dtype=np.float64)
    d[..., 0] -= Lx * np.round(d[..., 0] / Lx)
    d[..., 1] -= Ly * np.round(d[..., 1] / Ly)
    return d


def wrap(xy, Lx, Ly):
    xy = np.mod(xy, [Lx, Ly])
    xy[xy[:, 0] >= Lx, 0] = 0.0
    xy[xy[:, 1] >= Ly, 1] = 0.0
    return xy


def bond_vectors(xyz, bonds, Lx, Ly):
    '''
    Vectors i -> j of the bonds (Nb, 3), minimum image in x and y
    '''
    return min_image(xyz[bonds[:, 1]] - xyz[bonds[:, 0]], Lx, Ly)


# ─────────────────────────────── Rings and topological defects ───────────────────────────────

@njit(cache=True)
def _trace_faces(src, dst, start, max_size):
    '''
    Faces of the planar graph: dst[start[i]:start[i + 1]] are the neighbours of atom i sorted
    counter-clockwise (src = i). Each directed bond belongs to exactly one face, which is followed by
    always turning to the next neighbour clockwise. Works for any number of bonds per atom.
    '''
    M = len(dst)
    visited = np.zeros(M, dtype=np.bool_)
    sizes = np.empty(M, dtype=np.int64)
    atoms = -np.ones((M, max_size), dtype=np.int64)
    n = 0
    for e0 in range(M):
        if visited[e0]:
            continue
        e, length = e0, 0
        while not visited[e]:
            visited[e] = True
            if length < max_size:
                atoms[n, length] = src[e]
            length += 1
            a, b = src[e], dst[e]
            k = start[b]
            while dst[k] != a:
                k += 1
            e = start[b] + (k - start[b] - 1) % (start[b + 1] - start[b])
        sizes[n] = length
        n += 1
    return sizes[:n], atoms[:n]


def rings(xyz, bonds, Lx, Ly, max_size = 20):
    '''
    Rings of the crystal (faces of the bond graph; atoms with a number of bonds other than 3 are
    allowed and only produce unusual rings)
    Outputs:
        sizes : number of atoms of each ring (n_rings,)
        ring_atoms : atoms of each ring, padded with -1 (n_rings, max_size)
    '''
    N = len(xyz)
    d = bond_vectors(xyz, bonds, Lx, Ly)
    src = np.r_[bonds[:, 0], bonds[:, 1]]
    dst = np.r_[bonds[:, 1], bonds[:, 0]]
    ang = np.arctan2(np.r_[d[:, 1], -d[:, 1]], np.r_[d[:, 0], -d[:, 0]])
    order = np.lexsort((ang, src))
    start = np.r_[0, np.cumsum(np.bincount(src, minlength=N))]
    sizes, ring_atoms = _trace_faces(src[order], dst[order], start, max_size)
    if sizes.max() > max_size:
        raise ValueError(f"ring larger than max_size = {max_size}")
    return sizes, ring_atoms


def ring_pairs(sizes, ring_atoms):
    '''
    Pairs of rings sharing a bond (r1, r2)
    '''
    n, m = ring_atoms.shape
    cols = np.arange(m)
    valid = cols < sizes[:, None]
    nxt = ring_atoms[np.arange(n)[:, None], (cols + 1) % sizes[:, None]]
    a, b = ring_atoms[valid], nxt[valid]
    r = np.repeat(np.arange(n), sizes)
    key = np.minimum(a, b) * (a.max() + 1) + np.maximum(a, b)
    order = np.argsort(key, kind="stable")
    key, r = key[order], r[order]
    same = key[:-1] == key[1:]
    return r[:-1][same], r[1:][same]


def atoms_of(N, ring_atoms, selected):
    '''
    Atoms belonging to at least one of the selected rings (e.g. sizes != 6: defective atoms)
    '''
    mask = np.zeros(N, dtype=bool)
    r = ring_atoms[selected]
    mask[r[r >= 0]] = True
    return mask


def defect_graph(sizes, r1, r2):
    '''
    Topological defects. A 5 or 7 ring is paired when it shares a bond with a ring of the other kind
    (bound dislocation) and free otherwise (disclination). The connected components of the subgraph
    of 5/7 rings are isolated disclinations (size 1), isolated dislocations (5-7 dipoles, size 2) or
    chains (grain-boundary segments, size >= 3); a 5/7 ring with >= 3 such neighbours is a branching
    point (triple junction).
    Outputs:
        is57, paired : (n_rings,) masks
        component : connected component of each 5/7 ring, -1 for the other rings
        degree : number of 5/7 neighbours of each ring
    '''
    n = len(sizes)
    is57 = (sizes == 5) | (sizes == 7)
    both = is57[r1] & is57[r2]
    r1, r2 = r1[both], r2[both]
    paired = np.zeros(n, dtype=bool)
    opposite = sizes[r1] != sizes[r2]
    paired[r1[opposite]] = paired[r2[opposite]] = True
    _, labels = connected_components(coo_matrix((np.ones(len(r1)), (r1, r2)), shape=(n, n)), directed=False)
    component = np.where(is57, labels, -1)
    degree = np.bincount(np.r_[r1, r2], minlength=n)
    return is57, paired, component, degree


def ring_centroids(sizes, ring_atoms, xy, Lx, Ly):
    '''
    Centre of each ring (periodic average of its atoms, may lie outside the box)
    '''
    valid = ring_atoms >= 0
    ref = xy[ring_atoms[:, 0]]
    d = min_image(xy[np.where(valid, ring_atoms, ring_atoms[:, :1])] - ref[:, None], Lx, Ly)
    return ref + (d * valid[..., None]).sum(axis=1) / sizes[:, None]


def group_mean(labels, xy, Lx, Ly):
    '''
    Periodic mean position and size of each group of points with the same label
    '''
    _, first, inv, counts = np.unique(labels, return_index=True, return_inverse=True, return_counts=True)
    ref = xy[first]
    d = min_image(xy - ref[inv], Lx, Ly)
    return ref + np.stack([np.bincount(inv, d[:, 0]), np.bincount(inv, d[:, 1])], axis=1) / counts[:, None], counts


# ─────────────────────────────── Dislocation geometry ───────────────────────────────

def dipole_separations(sizes, component, centroids, Lx, Ly):
    '''
    Distance between the 5 and the 7 of each isolated dislocation (component made of one 5 and one 7)
    '''
    idx = np.where(component >= 0)[0]
    _, inv, counts = np.unique(component[idx], return_inverse=True, return_counts=True)
    idx = idx[counts[inv] == 2]
    idx = idx[np.argsort(component[idx], kind="stable")]
    m1, m2 = idx[0::2], idx[1::2]
    keep = sizes[m1] != sizes[m2]
    return np.linalg.norm(min_image(centroids[m1[keep]] - centroids[m2[keep]], Lx, Ly), axis=1)


def size_correlation(centers, cluster_sizes, Lx, Ly, r_bins):
    '''
    C(r) = <(s_i - <s>)(s_j - <s>)> / var(s) over the pairs of defect clusters at distance r
    Outputs:
        C : (n_bins,), NaN for empty bins
        n_pairs : (n_bins,)
    '''
    if len(centers) < 2:
        return np.full(len(r_bins) - 1, np.nan), np.zeros(len(r_bins) - 1, dtype=int)
    xy = wrap(centers, Lx, Ly)
    pairs = cKDTree(xy, boxsize=[Lx, Ly]).query_pairs(r_bins[-1], output_type="ndarray")
    r = np.linalg.norm(min_image(xy[pairs[:, 0]] - xy[pairs[:, 1]], Lx, Ly), axis=1)
    ds = cluster_sizes - cluster_sizes.mean()
    n_pairs = np.histogram(r, r_bins)[0]
    s = np.histogram(r, r_bins, weights=ds[pairs[:, 0]] * ds[pairs[:, 1]])[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        C = s / n_pairs / cluster_sizes.var()
    return C, n_pairs


def junction_enrichment(junctions, free, Lx, Ly, seed = 0):
    '''
    Distance of the free disclinations to the nearest triple junction, compared with uniformly random
    points: median(d_random) / median(d_free) > 1 means free disclinations gather near the junctions
    '''
    if len(junctions) == 0 or len(free) == 0:
        return np.nan
    tree = cKDTree(wrap(junctions, Lx, Ly), boxsize=[Lx, Ly])
    rand = np.random.default_rng(seed).random((max(len(free), 200), 2)) * [Lx, Ly]
    return float(np.median(tree.query(rand)[0]) / np.median(tree.query(wrap(free, Lx, Ly))[0]))


# ─────────────────────────────── Smoothed maps ───────────────────────────────

def smooth_map(xy, Lx, Ly, values = None, n_grid = 128, sigma = 25.0):
    '''
    Local average of a per-atom quantity (local atom density if values is None) on an n_grid x n_grid
    grid, with a periodic Gaussian kernel of width sigma (AA)
    '''
    rng = [[0, Lx], [0, Ly]]
    count = np.histogram2d(xy[:, 0], xy[:, 1], bins=n_grid, range=rng)[0]
    s = (sigma * n_grid / Lx, sigma * n_grid / Ly)
    count = gaussian_filter(count, s, mode="wrap")
    if values is None:
        return count
    total = gaussian_filter(np.histogram2d(xy[:, 0], xy[:, 1], bins=n_grid, range=rng, weights=values)[0], s, mode="wrap")
    return total / np.maximum(count, 1e-12)


# ─────────────────────────────── Diffraction ───────────────────────────────

@njit(parallel=True, cache=True)
def structure_factor(xy, q):
    '''
    S(q) = |sum_i exp(i q.r_i)|^2 / N for each wave vector q (n_q, 2)
    '''
    out = np.empty(len(q))
    for t in prange(len(q)):
        re = 0.0
        im = 0.0
        for i in range(len(xy)):
            ph = q[t, 0] * xy[i, 0] + q[t, 1] * xy[i, 1]
            re += np.cos(ph)
            im += np.sin(ph)
        out[t] = (re * re + im * im) / len(xy)
    return out


def first_peak(xy, Lx, Ly, n_theta = 720, n_q = 300, q_window = 0.3):
    '''
    First diffraction ring. Its radius is first estimated from the atom density, the angular profile
    S(theta) on that circle gives the direction of the strongest peak, the radial profile along it
    gives the exact radius q0 and the radial width (intensity-weighted std around the maximum,
    translational order), and the angular profile at q0 gives |m6| = |sum S exp(6 i theta)| / sum S
    (orientational order)
    '''
    a_cc = np.sqrt(Lx * Ly / len(xy) / (3 * np.sqrt(3) / 4))
    q_mag = 4 * np.pi / (3 * a_cc)
    theta = np.arange(n_theta) * 2 * np.pi / n_theta
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    direction = circle[np.argmax(structure_factor(xy, q_mag * circle))]

    q = np.linspace(q_mag * (1 - q_window), q_mag * (1 + q_window), n_q)
    S_radial = structure_factor(xy, q[:, None] * direction)
    q0 = q[np.argmax(S_radial)]
    width = np.sqrt(np.sum(S_radial * (q - q0) ** 2) / S_radial.sum())

    S_ring = structure_factor(xy, q0 * circle)
    m6 = abs(np.sum(S_ring * np.exp(6j * theta)) / S_ring.sum())
    return dict(q0=q0, m6=m6, radial_width=width, theta=theta, S_ring=S_ring, q=q, S_radial=S_radial)


def diffraction_map(xy, Lx, Ly, q_max, n_grid = 2048):
    '''
    |FFT|^2 of the atom density binned on n_grid x n_grid, cropped to |qx|, |qy| <= q_max
    Outputs:
        S (n_qx, n_qy), qx, qy
    '''
    hist = np.histogram2d(xy[:, 0], xy[:, 1], bins=n_grid, range=[[0, Lx], [0, Ly]])[0]
    S = np.abs(np.fft.fftshift(np.fft.fft2(hist))) ** 2 / len(xy)
    qx = np.fft.fftshift(np.fft.fftfreq(n_grid, Lx / n_grid)) * 2 * np.pi
    qy = np.fft.fftshift(np.fft.fftfreq(n_grid, Ly / n_grid)) * 2 * np.pi
    ix, iy = np.abs(qx) <= q_max, np.abs(qy) <= q_max
    return S[np.ix_(ix, iy)], qx[ix], qy[iy]


# ─────────────────────────────── Batch over a folder ───────────────────────────────

def run_folder(folder, analyze, **kwargs):
    '''
    analyze(path, maps, **kwargs) -> dict for every crystal of the folder; maps is True for the first
    k of each temperature (spatial maps are only computed for one crystal per T)
    Output:
        {T: [result of each k]}, sorted by T
    '''
    results = defaultdict(list)
    for T, k, path in crystal_files(folder):
        res = analyze(path, maps=not results[T], **kwargs)
        res["k"] = k
        results[T].append(res)
        print(f"T={T} K k={k}  " + "  ".join(f"{key}={v:.4g}" for key, v in res.items() if np.isscalar(v) and key != "k"))
    return dict(results)


def mean_std(results, key):
    '''
    Temperatures, mean and std over k of a scalar result
    '''
    T = np.array(list(results))
    vals = [np.array([r[key] for r in results[t]], dtype=float) for t in T]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN temperatures
        return T, np.array([np.nanmean(v) for v in vals]), np.array([np.nanstd(v) for v in vals])


def plot_vs_T(results, panels, path):
    '''
    One panel per (title, ylabel, [(key, label), ...]), mean +/- std over k; saved as path.png/.pdf
    '''
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.5), squeeze=False)
    for ax, (title, ylabel, curves) in zip(axes[0], panels):
        for key, label in curves:
            T, m, s = mean_std(results, key)
            ax.errorbar(T, m, yerr=s, marker="o", ms=4, capsize=3, label=label)
        ax.set(xlabel=r"$T$ (K)", ylabel=ylabel, title=title)
        if len(curves) > 1:
            ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{path}.pdf")
    fig.savefig(f"{path}.png", dpi=150)
    plt.close(fig)


def save_vs_T(results, keys, path):
    '''
    T, <key>_mean and <key>_std for each scalar key
    '''
    out = {}
    for key in keys:
        out["T"], out[f"{key}_mean"], out[f"{key}_std"] = mean_std(results, key)
    np.savez(path, **out)


def plot_maps(results, key, folder, cmap = "inferno", symmetric = False, label = ""):
    '''
    The map `key` of the first k of each temperature (box size in the result as "box"), with a common
    color scale, as folder/T=<T>K.png
    '''
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    maps = {T: r[0] for T, r in results.items() if key in r[0]}
    vals = np.concatenate([m[key].ravel() for m in maps.values()])
    vmax = np.percentile(np.abs(vals) if symmetric else vals, 99)
    for T, m in maps.items():
        Lx, Ly = m["box"]
        fig, ax = plt.subplots(figsize=(6, 5.2))
        im = ax.imshow(m[key].T, origin="lower", extent=[0, Lx, 0, Ly], cmap=cmap,
                       vmin=-vmax if symmetric else None, vmax=vmax)
        ax.set(xlabel=r"$x$ ($\mathrm{\AA}$)", ylabel=r"$y$ ($\mathrm{\AA}$)", title=f"T = {T} K (k = {m['k']})")
        fig.colorbar(im, ax=ax, label=label)
        fig.tight_layout()
        fig.savefig(folder / f"T={T}K.png", dpi=150)
        plt.close(fig)
