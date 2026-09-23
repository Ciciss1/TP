import os
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from numba import njit
from shapely.geometry import Polygon
from shapely import contains_xy
from scipy.spatial import Voronoi, cKDTree

from Voronoi import PeriodicVoronoi
from Lloyd import Lloyd, _periodic_images
from CG_Relaxation import CGRelaxation


@njit(cache=True)
def generate_triangular_lattice(L, a_CC = 1.42):
    '''
    Generate a triangular lattice with lattice constant a
    Inputs:
        L : size of the box
        a_CC : carbon-carbon bond length
    Outputs:
        atoms : coordinates of the atoms in the triangular lattice
    '''
    a = a_CC * np.sqrt(3)
    a1 = np.array([a, 0])
    a2 = np.array([a * 0.5, a * np.sqrt(3) * 0.5])

    nmax = int(L / a) + 5
    buf = np.empty((16 * nmax**2, 2), dtype=np.float64)
    idx = 0

    for i in range(-nmax, nmax):
        for j in range(-nmax, nmax):
            r = i * a1 + j * a2
            buf[idx] = r
            idx += 1
    atoms = buf[:idx]
    return atoms

@njit(cache=True)
def rotate_and_move_atoms(atoms, theta, center):
    '''
    Rotate atoms in each grain by the corresponding angle in theta and move them to the center of the grain
    Inputs:
        atoms : coordinates of the atoms in the graphene lattice
        theta : orientation of the grain
        center : center of the grain
    Outputs:
        rotated_atoms : coordinates of the rotated and moved atoms
    '''
    c, s = np.cos(theta), np.sin(theta)
    rotated = np.empty_like(atoms)

    for i in range(len(atoms)):
        x = atoms[i, 0]
        y = atoms[i, 1]
        rotated[i, 0] = c * x - s * y + center[0]
        rotated[i, 1] = s * x + c * y + center[1]
    return rotated

@njit(cache=True)
def compute_neighbors(atoms, bonds):
    '''
    Compute the 3 nearest neighbors for each atom based on the bonds
    Inputs:
        atoms : coordinates of the atoms
        bonds : list of bonds between atoms
    Outputs:
        neighbors : list of nearest neighbors for each atom
    '''
    N = len(atoms)
    neighbors = -np.ones((N, 3), dtype=np.int64)
    for k in range(len(bonds)):
        i, j = bonds[k, 0], bonds[k, 1]
        for slot in range(3):
            if neighbors[i, slot] == -1:
                neighbors[i, slot] = j
                break
        for slot in range(3):
            if neighbors[j, slot] == -1:
                neighbors[j, slot] = i
                break
    return neighbors


def voronoi_from_points(L, rho, points, theta):
    '''
    Rebuild a PeriodicVoronoi from saved grain centres and orientations (no random draw)
    '''
    vor = PeriodicVoronoi.__new__(PeriodicVoronoi)
    vor.L = L
    vor.rho = rho
    vor.points = np.asarray(points, dtype=np.float64)
    vor.theta = np.asarray(theta, dtype=np.float64).copy()
    vor.N = len(vor.points)
    vor.build_periodic_voronoi()
    vor.get_adjacency()
    return vor


def periodic_bond_lengths(atoms, bonds, L):
    d = atoms[bonds[:, 0], :2] - atoms[bonds[:, 1], :2]
    d -= L * np.round(d / L)
    return np.linalg.norm(d, axis=1)


def load_crystal(path):
    data = np.load(path)

    L = float(data['L'][0])
    rho = float(data['rho'][0])

    vor = voronoi_from_points(L, rho, data['points'], data['theta'])

    crystal = GrapheneCrystal.__new__(GrapheneCrystal)
    crystal.lattice = vor
    crystal.vor = vor.vor
    crystal.all_points = vor.all_points
    crystal.L = L
    crystal.N = vor.N
    crystal.points = vor.points
    crystal.theta = vor.theta

    crystal.relaxed_generators = data['relaxed_generators']
    crystal.boundary_mask = crystal.get_boundary_mask(crystal.relaxed_generators)
    crystal.atoms = data['atoms']
    if 'bonds' in data.files:
        crystal.bonds = data['bonds']
    else:
        # Old files: rebuild the bonds with the original atom ordering
        _, crystal.bonds = crystal.vertices_from_generators(crystal.relaxed_generators, legacy=True)
    crystal.neighbors = compute_neighbors(crystal.atoms, crystal.bonds)

    return crystal


class GrapheneCrystal(Lloyd, CGRelaxation):
    '''
    Create a polycrystalline graphene structure based on the Voronoi diagram
    Attributes:
        lattice : Voronoi lattice
        L : size of the box
        N : number of grains
        points : coordinates of the grain centers
        theta : orientation of the grains
        atoms : coordinates of the atoms
        bonds : list of bonds between atoms
        neighbors : list of nearest neighbors for each atom
        timings, lloyd_info, lammps_info, checks : diagnostics of the construction
    '''
    def __init__(self, voronoi: PeriodicVoronoi, a = 1.42, margin = 5, lloyd_kwargs = None, lammps_kwargs = None):
        self.lattice = voronoi
        self.vor = voronoi.vor
        self.L = voronoi.L
        self.N = voronoi.N
        self.points = voronoi.points
        self.all_points = voronoi.all_points
        self.theta = voronoi.theta.copy()
        self.build_polycrystal(a, margin = margin, lloyd_kwargs = lloyd_kwargs or {}, lammps_kwargs = lammps_kwargs or {})

    def remove_close_generators(self, generators, min_dist = 0.5):
        '''
        Remove generators closer than min_dist to another one, taking the periodicity into account
        (generators must be in [0, L))
        '''
        tree = cKDTree(generators, boxsize=self.L)
        close_pairs = tree.query_pairs(min_dist, output_type='ndarray')
        to_remove = set()
        for i, j in close_pairs:
            if i not in to_remove and j not in to_remove:
                to_remove.add(i)

        mask = np.ones(len(generators), dtype=bool)
        mask[list(to_remove)] = False

        return generators[mask]

    def get_boundary_mask(self, generators, margin = 10, k = 16):
        '''
        Identify generators that are close to the grains boundaries.
        The distance of a point to the boundary of its (convex) Voronoi cell is the minimum over the
        neighbouring grain centres c_k of the distance to the bisector with c_k:
            (d_k^2 - d_1^2) / (2 |c_k - c_1|)
        Inputs:
            generators : coordinates of the generators
            margin : distance from the boundary
            k : number of nearest grain centres considered
        Outputs:
            boundary_mask : boolean mask
        '''
        centres = self.all_points
        k = min(k, len(centres))
        tree = cKDTree(centres)
        d, idx = tree.query(np.mod(generators, self.L), k=k)
        c = centres[idx]
        sep = np.linalg.norm(c[:, 1:] - c[:, :1], axis=2)
        dist = (d[:, 1:]**2 - d[:, :1]**2) / (2 * np.maximum(sep, 1e-12))
        self.boundary_mask = dist.min(axis=1) < margin
        return self.boundary_mask

    def vertices_from_generators(self, generators, legacy = False, tol = 1e-4, pad = 10.0):
        '''
        Compute the vertices of the Voronoi diagram from the generators
        Inputs:
            generators : coordinates of the generators
            legacy : reproduce the atom ordering of the old implementation (for old files without bonds)
            tol : merging tolerance of periodic copies of a vertex
            pad : width of the periodic images (non-legacy mode)
        Outputs:
            atoms : coordinates of the atoms in the graphene lattice
            bonds : list of bonds between atoms
        '''
        L = self.L
        M = len(generators)

        if legacy:
            images = [generators + np.array([dx, dy]) for dx in [-L, 0, L] for dy in [-L, 0, L]]
            all_gen = np.vstack(images)
            c0, c1 = 4 * M, 5 * M
        else:
            all_gen = _periodic_images(np.mod(generators, L), L, pad)
            c0, c1 = 0, M
        vor = Voronoi(all_gen)

        rp = vor.ridge_points
        rv = np.asarray(vor.ridge_vertices)
        central = ((rp >= c0) & (rp < c1)).any(axis=1)
        finite = (rv >= 0).all(axis=1)
        seq = rv[central & finite].ravel()          # vertices in order of first encounter

        wpos = np.mod(vor.vertices[seq], L)
        nL = int(round(L / tol))
        keys = np.round(wpos / tol).astype(np.int64)
        if not legacy:
            keys %= nL                              # vertices at x = 0 and x = L are the same atom
        kid = keys[:, 0] * (nL + 2) + keys[:, 1]

        _, first, inv = np.unique(kid, return_index=True, return_inverse=True)
        order = np.argsort(first)
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        atom_of = rank[inv.ravel()]
        atoms = wpos[first[order]]

        pairs = atom_of.reshape(-1, 2)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        pairs.sort(axis=1)
        bonds = np.unique(pairs, axis=0).astype(np.int64)

        return atoms, bonds

    def check_topology(self, n_generators):
        '''
        Sanity checks: a generic Voronoi diagram on a torus has exactly 2 vertices per generator
        and every vertex has exactly 3 neighbours
        '''
        deg = np.bincount(self.bonds.ravel(), minlength=len(self.atoms))
        r = periodic_bond_lengths(self.atoms, self.bonds, self.L)
        self.checks = {
            "n_generators": int(n_generators),
            "n_atoms": int(len(self.atoms)),
            "atoms_equal_2_generators": bool(len(self.atoms) == 2 * n_generators),
            "all_degree_3": bool(np.all(deg == 3)),
            "bond_min": float(r.min()),
            "bond_max": float(r.max()),
        }
        if not (self.checks["atoms_equal_2_generators"] and self.checks["all_degree_3"]):
            warnings.warn(f"Topology check failed: {self.checks}")
        return self.checks

    def build_polycrystal(self, a_CC = 1.42, margin = 10, lloyd_kwargs = None, lammps_kwargs = None):
        '''
        Build the polycrystalline graphene structure
        Inputs:
            a_CC : carbon-carbon bond length
            margin : distance from the grain boundaries
            lloyd_kwargs : options of the Lloyd relaxation
            lammps_kwargs : options of the LAMMPS relaxation
        '''
        self.timings = {}
        t0 = time.perf_counter()

        # Generate base Lattice
        mean_grain_radius = 1 / np.sqrt(np.pi * self.lattice.rho)
        patch_L = min(self.L, 6 * mean_grain_radius)
        base_lattice = generate_triangular_lattice(patch_L, a_CC)
        all_generators = []

        # Construct the generators for each grain by rotating and moving the base lattice
        for grain in range(len(self.all_points)):
            region_idx = self.vor.point_region[grain]
            vertices = self.vor.regions[region_idx]

            if -1 in vertices or len(vertices) == 0:
                continue

            polygon = Polygon(self.vor.vertices[vertices]).buffer(0.5)

            min_x, min_y, max_x, max_y = polygon.bounds
            if (max_x < 0 or min_x > self.L or max_y < 0 or min_y > self.L):
                continue

            diag = np.hypot(max_x - min_x, max_y - min_y)
            if diag > patch_L:
                src = generate_triangular_lattice(diag + 4 * a_CC, a_CC)
            else:
                src = base_lattice

            theta = self.theta[grain % self.N]
            center = self.all_points[grain]

            rot_atoms = rotate_and_move_atoms(src, theta, center)

            mask = (rot_atoms[:, 0] >= min_x) & (rot_atoms[:, 0] <= max_x) & (rot_atoms[:, 1] >= min_y) & (rot_atoms[:, 1] <= max_y)

            rot_atoms = rot_atoms[mask]

            inside = contains_xy(polygon, rot_atoms[:, 0], rot_atoms[:, 1])
            all_generators.append(rot_atoms[inside])

        generators = np.vstack(all_generators)

        # Keep only generators that are within the box, wrapped into [0, L)
        mask = (generators[:, 0] >= 0) & (generators[:, 0] <= self.L) & (generators[:, 1] >= 0) & (generators[:, 1] <= self.L)
        generators = np.mod(generators[mask], self.L)

        # Periodic removal: also removes the duplicates across the box edges
        generators = self.remove_close_generators(generators)

        self.boundary_mask = self.get_boundary_mask(generators, margin)
        self.timings["generators"] = time.perf_counter() - t0

        # Relax the generators using Lloyd's algorithm
        t0 = time.perf_counter()
        self.relaxed_generators = self.relaxation(generators, self.boundary_mask, **(lloyd_kwargs or {}))
        self.timings["lloyd"] = time.perf_counter() - t0

        # Construct the atoms and bonds from the relaxed generators
        t0 = time.perf_counter()
        self.atoms, self.bonds = self.vertices_from_generators(self.relaxed_generators)
        self.check_topology(len(self.relaxed_generators))
        self.atoms = np.hstack([self.atoms, np.zeros((len(self.atoms), 1))])
        self.timings["vertices"] = time.perf_counter() - t0

        # Relax the atoms using LAMMPS
        t0 = time.perf_counter()
        self.atoms = self.relaxation_CG(
            atoms=self.atoms,
            generators=self.relaxed_generators,
            generator_boundary_mask=self.boundary_mask,
            **(lammps_kwargs or {}),
        )
        self.timings["lammps"] = time.perf_counter() - t0

        # Compute the neighbors for each atom
        self.neighbors = compute_neighbors(self.atoms, self.bonds)

        del self.vor
        del self.all_points

    def plot_atoms(self, fig_size = 6, dot_size = 1):

        plt.figure(figsize=(fig_size, fig_size))

        plt.scatter(self.atoms[:, 0], self.atoms[:, 1], s=dot_size, color='black')
        plt.xlim(0, self.L)
        plt.ylim(0, self.L)
        plt.gca().set_aspect('equal')
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.tight_layout()

    def plot_bonds(self, fig_size = 6, dot_size = 1, lw = 0.5):
        plt.figure(figsize=(fig_size, fig_size))

        lines = [(self.atoms[i, :2], self.atoms[j, :2]) for i, j in self.bonds if np.linalg.norm(self.atoms[i, :2] - self.atoms[j, :2]) < 4]
        lc = LineCollection(lines, colors='black', linewidths=lw)
        plt.gca().add_collection(lc)
        plt.scatter(self.relaxed_generators[self.boundary_mask][:, 0], self.relaxed_generators[self.boundary_mask][:, 1], s=dot_size, color='green')
        plt.scatter(self.relaxed_generators[~self.boundary_mask][:, 0], self.relaxed_generators[~self.boundary_mask][:, 1], s=dot_size, color='red')
        plt.xlim(0, self.L)
        plt.ylim(0, self.L)
        plt.gca().set_aspect('equal')
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.tight_layout()

    def plot_all(self, fig_size = 6, dot_size = 1, lw = 0.5, atom_phase = None):
        fig, ax = plt.subplots(figsize=(fig_size*1.25, fig_size))

        if atom_phase is not None:
            phase_norm = (atom_phase + np.pi) / (2 * np.pi)

            mp = ax.scatter(self.atoms[:, 0], self.atoms[:, 1], s=dot_size, c=phase_norm, cmap='hsv', vmin=0, vmax=1, zorder=2)

            cbar = fig.colorbar(mp, ax=ax, ticks=[0, 0.25, 0.5, 0.75, 1])
            cbar.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'], fontsize=10)
            cbar.ax.set_ylabel(r'$\arg(\psi_6) \,\mathrm{mod}\, \pi/3$', fontsize=12)
        else:
            ax.scatter(self.atoms[:, 0], self.atoms[:, 1], s=dot_size, color='black')

        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.L)
        ax.set_aspect('equal')
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        plt.tight_layout()

    def plot_lattice(self):
        self.lattice.plot()

    def save_crystal(self, path, **extra):
        '''
        Save the crystal. Diagnostics (timings, Lloyd, LAMMPS, topology checks) are stored with
        the prefixes timing_, lloyd_, lammps_, check_; extra keyword arguments are stored as meta_<key>.
        '''
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        diag = {}
        for prefix, d in (("timing_", getattr(self, "timings", {})),
                          ("lloyd_", getattr(self, "lloyd_info", {})),
                          ("lammps_", getattr(self, "lammps_info", {})),
                          ("check_", getattr(self, "checks", {})),
                          ("meta_", extra)):
            for key, value in d.items():
                diag[prefix + key] = np.asarray(value)

        np.savez_compressed(
            path,
            points = self.points,
            theta = self.theta,
            L = np.array([self.L]),
            rho = np.array([self.lattice.rho]),
            relaxed_generators = self.relaxed_generators,
            atoms = self.atoms,
            bonds = self.bonds,
            **diag,
        )


# Test

if __name__ == "__main__":
    import Observables as obs

    _ = generate_triangular_lattice(10.0)

    configs = [
        (200,  0.0007,  "10 grains / 200Å  — test de base"),
    ]

    for L, rho, desc in configs:
        print(f"\n{'─'*55}")
        print(f"  {desc}")
        print(f"{'─'*55}")

        t0 = time.time()

        vor = PeriodicVoronoi(L, rho)
        crystal = GrapheneCrystal(vor)

        angles = np.array([np.angle(obs.compute_psi6(i, crystal.atoms, crystal.neighbors, crystal.L))
                           for i in range(len(crystal.atoms))])

        crystal.plot_all(atom_phase=angles)
        plt.savefig(f"results/test_{L:.0f}_{rho:.0e}_all.png", dpi=300)
        plt.close()

        crystal.plot_bonds()
        plt.savefig(f"results/test_{L:.0f}_{rho:.0e}_bonds.png", dpi=300)
        plt.close()

        t1 = time.time()
        print(f"  Grains   : {vor.N}")
        print(f"  Atomes   : {len(crystal.atoms):,} Shape: {crystal.atoms.shape}")
        print(f"  Temps    : {t1 - t0:.2f} s   {crystal.timings}")
        print(f"  Checks   : {crystal.checks}")
        print(f"  LAMMPS   : {crystal.lammps_info}")

        crystal.save_crystal(f"results/test_{L:.0f}_{rho:.0e}.npz")