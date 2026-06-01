import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from numba import njit
from shapely.geometry import Polygon
from shapely import contains_xy
from scipy.spatial import Voronoi, cKDTree
from scipy.interpolate import griddata

import Observables as obs
from Voronoi import PeriodicVoronoi
from Lloyd import Lloyd
from CG_Relaxation import CGRelaxation

@njit
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

@njit
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

@njit
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

def load_crystal(path):
    data = np.load(path)
    
    L = float(data['L'][0])
    rho = float(data['rho'][0])

    vor = PeriodicVoronoi.__new__(PeriodicVoronoi)
    vor.L = L
    vor.rho = rho
    vor.points = data['points']
    vor.theta = data['theta']
    vor.N = len(vor.points)
    vor.build_periodic_voronoi()
    vor.get_adjacency()

    crystal = GrapheneCrystal.__new__(GrapheneCrystal)
    crystal.lattice = vor
    crystal.vor = vor.vor
    crystal.L = L
    crystal.N = vor.N
    crystal.points = vor.points
    crystal.theta = vor.theta

    crystal.relaxed_generators = data['relaxed_generators']
    crystal.boundary_mask = crystal.get_boundary_mask(crystal.relaxed_generators)
    _ , crystal.bonds = crystal.vertices_from_generators(crystal.relaxed_generators)
    crystal.atoms = data['atoms']
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
    '''
    def __init__(self, voronoi: PeriodicVoronoi, a = 1.42):
        self.lattice = voronoi
        self.vor = voronoi.vor
        self.L = voronoi.L
        self.N = voronoi.N
        self.points = voronoi.points
        self.all_points = voronoi.all_points
        self.theta = voronoi.theta
        self.build_polycrystal(a)

    def remove_close_generators(self, generators, min_dist = 0.5):
        tree = cKDTree(generators)
        close_pairs = tree.query_pairs(min_dist)
        to_remove = set()
        for i, j in close_pairs:
            if i not in to_remove and j not in to_remove:
                to_remove.add(i)

        mask = np.ones(len(generators), dtype=bool)
        mask[list(to_remove)] = False

        return generators[mask]

    def get_boundary_mask(self, generators, margin = 10):
        '''
        Identify generators that are close to the grains boundaries
        Inputs:
            generators : coordinates of the generators
            margin : distance from the boundary
        Outputs:
            boundary_mask : boolean mask
        '''
        L = self.L

        vertices = np.array(self.vor.ridge_vertices)
        valid = (vertices[:, 0] != -1) & (vertices[:, 1] != -1)
        vertices = vertices[valid]

        v1 = self.vor.vertices[vertices[:, 0]]
        v2 = self.vor.vertices[vertices[:, 1]]

        self.boundary_mask = np.zeros(len(generators), dtype=bool)

        for i in range(len(v1)):
            edge_vec = v2[i] - v1[i]
            edge_length = np.linalg.norm(edge_vec)
            if edge_length < 1e-8:
                continue
            edge_dir = edge_vec / edge_length

            to_v1 = generators - v1[i]
            proj_length = np.dot(to_v1, edge_dir)
            proj_length = np.clip(proj_length, 0, edge_length)
            closest_point = v1[i] + np.outer(proj_length, edge_dir)
            dist_to_edge = np.linalg.norm(generators - closest_point, axis=1)

            self.boundary_mask |= (dist_to_edge < margin)

        return self.boundary_mask

    def vertices_from_generators(self, generators):
        '''
        Compute the vertices of the Voronoi diagram from the generators
        Inputs:
            generators : coordinates of the generators
        Outputs:
            atoms : coordinates of the atoms in the graphene lattice
            bonds : list of bonds between atoms
        '''
        L = self.L
        images = [generators + np.array([dx, dy]) 
                    for dx in [-L, 0, L]
                    for dy in [-L, 0, L]]
        all_gen = np.vstack(images)
        M = len(generators)
        vor = Voronoi(all_gen)

        central_cells = set(range(4 * M, 5 * M))
        tol = 1e-4

        v2atom = {}
        atom_list = []
        pos_to_atom = {}

        def get_atom(raw_idx):
            if raw_idx in v2atom:
                return v2atom[raw_idx]
            wpos = vor.vertices[raw_idx] % L
            key = (int(round(wpos[0] / tol)), int(round(wpos[1] / tol)))
            if key not in pos_to_atom:
                pos_to_atom[key] = len(atom_list)
                atom_list.append(wpos)
            v2atom[raw_idx] = pos_to_atom[key]
            return v2atom[raw_idx]

        bonds = set()
        for k in range(len(vor.ridge_points)):
            pi, pj = vor.ridge_points[k]
            vi, vj = vor.ridge_vertices[k]
            if vi == -1 or vj == -1:
                continue
            if pi not in central_cells and pj not in central_cells:
                continue
            
            i, j = get_atom(vi), get_atom(vj)
            if i != j:
                bonds.add((min(i, j), max(i, j)))

        atoms = np.array(atom_list)
        bonds = np.array(sorted(bonds), dtype=np.int64)

        return atoms, bonds

    def build_polycrystal(self, a_CC = 1.42, margin = 10):
        '''
        Build the polycrystalline graphene structure
        Inputs:
            a_CC : carbon-carbon bond length
            margin : distance from the grain boundaries
            n_iter : maximum number of iterations for relaxation
            tol : tolerance for convergence of relaxation
        '''
        # Generate base Lattice
        base_lattice = generate_triangular_lattice(self.L, a_CC)
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

            theta = self.theta[grain % self.N]
            center = self.all_points[grain]
            
            rot_atoms = rotate_and_move_atoms(base_lattice, theta, center)

            mask = (rot_atoms[:, 0] >= min_x) & (rot_atoms[:, 0] <= max_x) & (rot_atoms[:, 1] >= min_y) & (rot_atoms[:, 1] <= max_y)

            rot_atoms = rot_atoms[mask]

            inside = contains_xy(polygon, rot_atoms[:, 0], rot_atoms[:, 1])
            all_generators.append(rot_atoms[inside])

        generators = np.vstack(all_generators)

        # Keep only generators that are within the box
        mask = (generators[:, 0] >= 0) & (generators[:, 0] <= self.L) & (generators[:, 1] >= 0) & (generators[:, 1] <= self.L)

        generators = generators[mask]

        generators = self.remove_close_generators(generators)

        self.boundary_mask = self.get_boundary_mask(generators, margin)

        # Relax the generators using Lloyd's algorithm
        self.relaxed_generators = self.relaxation(generators, self.boundary_mask)

        # Construct the atoms and bonds from the relaxed generators
        self.atoms, self.bonds = self.vertices_from_generators(self.relaxed_generators)

        self.atoms = np.hstack([self.atoms, np.zeros((len(self.atoms), 1))])

        # Relax the atoms using LAMMPS
        self.atoms = self.relaxation_CG(
            atoms=self.atoms,
            generators=self.relaxed_generators,
            generator_boundary_mask=self.boundary_mask,
        )

        # Compute the neighbors for each atom
        self.neighbors = compute_neighbors(self.atoms, self.bonds)

        del self.vor
        del self.all_points

    def compute_observables(self, a = 1.42, n_samples = 5_000_000):
        '''
        Compute the observables for the graphene crystal
        '''
        r_max = self.L / 2
        dr = a * np.sqrt(3) / 2
        num_bins = int(r_max / dr)
        bin_bounds = np.linspace(0, r_max, num_bins + 1)
        bin_centers = 0.5 * (bin_bounds[:-1] + bin_bounds[1:])

        grain_centers = self.points
        tree = cKDTree(grain_centers)
        grain_of_atoms = tree.query(self.atoms[:, :2], k=1, workers=-1)[1]
        grain_of_atoms = grain_of_atoms.astype(np.int64)

        G6 = obs.compute_orientational_correlation(self.atoms, self.neighbors, bin_bounds, n_samples)
        GT = obs.compute_translational_correlation(self.atoms, self.neighbors, grain_of_atoms, self.points, self.theta, bin_bounds, self.L)

        return bin_centers, G6, GT

    def plot_atoms(self, fig_size = 6, dot_size = 1):

        plt.figure(figsize=(fig_size, fig_size))

        plt.scatter(self.atoms[:, 0], self.atoms[:, 1], s=dot_size, color='black')
        plt.xlim(0, self.L)
        plt.ylim(0, self.L)
        plt.gca().set_aspect('equal')
        plt.title('Graphene Crystal')
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.tight_layout()

    def plot_bonds(self, fig_size = 6, dot_size = 1, lw = 0.5):
        plt.figure(figsize=(fig_size, fig_size))

        lines = [(self.atoms[i, :2], self.atoms[j, :2]) for i, j in self.bonds]
        lc = LineCollection(lines, colors='black', linewidths=lw)
        plt.gca().add_collection(lc)
        plt.scatter(self.relaxed_generators[self.boundary_mask][:, 0], self.relaxed_generators[self.boundary_mask][:, 1], s=dot_size, color='green')
        plt.scatter(self.relaxed_generators[~self.boundary_mask][:, 0], self.relaxed_generators[~self.boundary_mask][:, 1], s=dot_size, color='red')
        plt.xlim(0, self.L)
        plt.ylim(0, self.L)
        plt.gca().set_aspect('equal')
        plt.title('Graphene Bonds')
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.tight_layout()

    def plot_all(self, fig_size = 6, dot_size = 1, lw = 0.5, atom_phase = None):
        fig, ax = plt.subplots(figsize=(fig_size * 1.2, fig_size))

        lines = [(self.atoms[i, :2], self.atoms[j, :2]) for i, j in self.bonds if np.linalg.norm(self.atoms[i, :2] - self.atoms[j, :2]) < 4]
        lc = LineCollection(lines, colors='black', linewidths=lw)
        ax.add_collection(lc)        
        if atom_phase is not None:
            phase_norm = atom_phase / (np.pi/3)

            mp = ax.scatter(self.atoms[:, 0], self.atoms[:, 1], s=dot_size, c=phase_norm, cmap='hsv', vmin=0, vmax=1, zorder=2)

            cbar = fig.colorbar(mp, ax=ax, ticks=[0, 0.25, 0.5, 0.75, 1])
            cbar.ax.set_yticklabels(['0', r'$\pi/12$', r'$\pi/6$', r'$\pi/4$', r'$\pi/3$'], fontsize=10)
            cbar.ax.set_ylabel(r'$\arg(\psi_6) mod \pi/3$', fontsize=12)
        else:
            ax.scatter(self.atoms[:, 0], self.atoms[:, 1], s=dot_size, color='black')

        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.L)
        ax.set_aspect('equal')
        ax.set_title('Graphene Crystal')
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        plt.tight_layout()

    def plot_lattice(self):
        self.lattice.plot()

    def save_crystal(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        np.savez_compressed(
            path,
            points = self.points,
            theta = self.theta,
            L = np.array([self.L]),
            rho = np.array([self.lattice.rho]),
            relaxed_generators = self.relaxed_generators,
            atoms = self.atoms,
        )


# Test

if __name__ == "__main__":
    import time

    _ = generate_triangular_lattice(10.0)

    configs = [
        (120,  0.0005,  "10 grains / 120Å  — test de base"),
        # (200,  0.0003,  "12 grains / 200Å  — test de base"),
        # (500,  0.0003,  "75 grains / 500Å  — polycristal moyen"),
        # (500,  0.001,   "250 grains / 500Å — grains plus petits"),
        # (1000, 0.0003,  "300 grains / 1000Å — grande boîte"),
        # (1000, 0.001,   "1000 grains / 1000Å — haute densité"),
    ]

    for L, rho, desc in configs:
        print(f"\n{'─'*55}")
        print(f"  {desc}")
        print(f"{'─'*55}")

        t0 = time.time()

        vor = PeriodicVoronoi(L, rho)

        crystal = GrapheneCrystal(vor)

        angles = []
        for i, atom in enumerate(crystal.atoms):
            psi_6_i = obs.compute_psi6(i, crystal.atoms, crystal.neighbors, crystal.L)
            angle = np.angle(psi_6_i) % (np.pi / 3)
            angles.append(angle)

        angles = np.array(angles)

        crystal.plot_all(atom_phase=angles)
        plt.savefig(f"results/test_{L:.0f}_{rho:.0e}_all.png", dpi=300)
        plt.close()


        t1 = time.time()
        print(f"  Grains   : {vor.N}")
        print(f"  Atomes   : {len(crystal.atoms):,}")
        print(f"  Temps    : {t1 - t0:.2f} s")