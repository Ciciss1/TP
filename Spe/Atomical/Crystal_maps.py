"""
Pictures of the relaxed crystals of a folder (T=<T>K_k=<k>.npz, first k of each T):
  - crystal_maps/overview.png : one panel per T, hexagons coloured by their orientation (mod 60 deg),
    pentagons red, heptagons blue, other rings black, Voronoi boundaries of the MC dashed
  - crystal_maps/zoom_T=<T>K.png : the least and the most misoriented boundaries of one crystal

Usage:
    python Crystal_maps.py <folder> [--T 10,70,85,120] [--zoom 85] [--half-width 22]
    (temperatures in 10^3 K as in parameters.txt; default: 6 temperatures spread over the folder)
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.colors import hsv_to_rgb
from scipy.spatial import Voronoi

import Properties as P


def ring_polygons(xyz, bonds, Lx, Ly):
    '''
    Rings as polygons (unwrapped, centroid in the box), their size and orientation (deg, mod 60)
    '''
    sizes, ring_atoms = P.rings(xyz, bonds, Lx, Ly)
    polys, orient = [], np.empty(len(sizes))
    for n, (s, ra) in enumerate(zip(sizes, ring_atoms)):
        p = xyz[ra[:s], :2]
        p = p[0] + P.min_image(p - p[0], Lx, Ly)
        c = p.mean(axis=0)
        orient[n] = np.degrees(np.angle(np.exp(6j * np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])).sum())) / 6
        polys.append(p - np.floor(c / [Lx, Ly]) * [Lx, Ly])
    return sizes, polys, orient


def ring_colors(sizes, orient):
    h = (orient + 30) / 60
    rgb = hsv_to_rgb(np.stack([h, np.full_like(h, 0.45), np.full_like(h, 0.97)], axis=1))
    rgb[sizes == 5] = (0.80, 0.10, 0.10)
    rgb[sizes == 7] = (0.10, 0.20, 0.75)
    rgb[(sizes < 5) | (sizes > 7)] = (0.1, 0.1, 0.1)
    return rgb


def periodic_copies(polys, rgb, Lx, Ly):
    '''
    Polygons crossing the box edges are also drawn on the other side
    '''
    out, col = [], []
    for p, c in zip(polys, rgb):
        for shift in np.array([[dx, dy] for dx in (-Lx, 0, Lx) for dy in (-Ly, 0, Ly)]):
            q = p + shift
            if q[:, 0].max() > 0 and q[:, 0].min() < Lx and q[:, 1].max() > 0 and q[:, 1].min() < Ly:
                out.append(q)
                col.append(c)
    return out, col


def grain_boundaries(points, theta, L, Lx, Ly):
    '''
    Voronoi boundaries of the MC grains (rescaled to the relaxed box) and their misorientation (deg)
    '''
    N = len(points)
    pts = np.mod(points, L) * [Lx / L, Ly / L]
    vor = Voronoi(np.vstack([pts + [dx, dy] for dx in (-Lx, 0, Lx) for dy in (-Ly, 0, Ly)]))
    rp, rv = vor.ridge_points, np.asarray(vor.ridge_vertices)
    keep = (rv >= 0).all(axis=1) & ((rp // N) == 4).any(axis=1) & (rp[:, 0] % N != rp[:, 1] % N)
    dtheta = np.abs((np.degrees(theta[rp[keep, 0] % N] - theta[rp[keep, 1] % N]) + 30) % 60 - 30)
    return vor.vertices[rv[keep]], dtheta


def draw(ax, path, window = None, edges = False):
    '''
    Draw one crystal (whole box, or window = (xmin, xmax, ymin, ymax)); returns the ring sizes
    '''
    d = np.load(path)
    xyz, bonds = d["atoms"], d["bonds"]
    Lx, Ly = d["lattice"]
    sizes, polys, orient = ring_polygons(xyz, bonds, Lx, Ly)
    polys, col = periodic_copies(polys, ring_colors(sizes, orient), Lx, Ly)
    ax.add_collection(PolyCollection(polys, facecolors=col, edgecolors="0.35" if edges else "none", linewidths=0.3 * edges))
    segs, _ = grain_boundaries(d["points"], d["theta"], float(np.ravel(d["L"])[0]), Lx, Ly)
    ax.add_collection(LineCollection(segs, colors="k", linewidths=1.2 if edges else 0.6, linestyles=(0, (3, 3)), alpha=0.6))
    xmin, xmax, ymin, ymax = window or (0, Lx, 0, Ly)
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal", xticks=[], yticks=[])
    return sizes


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--T", help="temperatures of the overview (10^3 K), comma separated")
    ap.add_argument("--zoom", type=float, help="temperature of the zoom figure (10^3 K)")
    ap.add_argument("--half-width", type=float, default=22.0, help="half width of the zooms (AA)")
    args = ap.parse_args()
    folder = Path(args.folder)
    out = folder / "crystal_maps"
    out.mkdir(exist_ok=True)

    files = {}
    for T, k, path in P.crystal_files(folder):
        files.setdefault(T, path)                                  # first k of each T
    T_all = np.array(sorted(files))
    T_show = ([T_all[np.argmin(abs(T_all - 1000 * float(t)))] for t in args.T.split(",")] if args.T
              else T_all[np.unique(np.linspace(0, len(T_all) - 1, min(6, len(T_all))).round().astype(int))])

    n_col = min(3, len(T_show))
    n_row = -(-len(T_show) // n_col)
    fig, axes = plt.subplots(n_row, n_col, figsize=(4.2 * n_col, 4.4 * n_row), squeeze=False)
    for ax, T in zip(axes.flat, T_show):
        sizes = draw(ax, files[T])
        ax.set_title(f"$T_{{MC}}$ = {T / 1000:g}·10³ K  —  {int(((sizes == 5) | (sizes == 7)).sum())} anneaux 5/7", fontsize=10)
    for ax in axes.flat[len(T_show):]:
        ax.axis("off")
    fig.suptitle("hexagones colorés par orientation, 5 = rouge, 7 = bleu, autres = noir, pointillés = joints de Voronoi du MC", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=170)
    plt.close(fig)

    T_zoom = T_all[np.argmin(abs(T_all - 1000 * args.zoom))] if args.zoom else T_show[len(T_show) // 2]
    d = np.load(files[T_zoom])
    Lx, Ly = d["lattice"]
    segs, dtheta = grain_boundaries(d["points"], d["theta"], float(np.ravel(d["L"])[0]), Lx, Ly)
    mid, length = segs.mean(axis=1), np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)
    hw = args.half_width
    ok = (length > 10) & np.all((mid > hw) & (mid < [Lx - hw, Ly - hw]), axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8))
    for ax, k, name in ((axes[0], np.flatnonzero(ok)[np.argmin(dtheta[ok])] if ok.any() else None, "joint le moins désorienté"),
                        (axes[1], np.flatnonzero(ok)[np.argmax(dtheta[ok])] if ok.any() else None, "joint le plus désorienté")):
        if k is None:
            ax.axis("off")
            continue
        c = mid[k]
        draw(ax, files[T_zoom], window=(c[0] - hw, c[0] + hw, c[1] - hw, c[1] + hw), edges=True)
        ax.set_title(f"{name} : Δθ = {dtheta[k]:.1f}°", fontsize=10)
    fig.suptitle(f"$T_{{MC}}$ = {T_zoom / 1000:g}·10³ K", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"zoom_T={T_zoom}K.png", dpi=170)
    plt.close(fig)
