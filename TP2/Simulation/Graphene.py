import sys
sys.path.insert(0, "TP2/Simulation")
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from numba import njit
from shapely.geometry import Polygon
from shapely import contains_xy
from scipy.spatial import Voronoi, cKDTree
import torch

import Observables as obs
from Voronoi import PeriodicVoronoi
from Lloyd_GPU import LloydHybrid
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

    vor = PeriodicVoronoi(L, rho)
    vor.points = data['points']
    vor.theta = data['theta']
    vor.N = len(vor.points)
    vor.build_periodic_voronoi()
    vor.get_adjacency()

    crystal = GrapheneCrystal.__new__(GrapheneCrystal)
    crystal.lattice = vor
    crystal.L = L
    crystal.N = vor.N
    crystal.points = vor.points
    crystal.theta = vor.theta

    crystal.relaxed_generators = data['relaxed_generators']
    crystal.atoms, crystal.bonds = crystal.vertices_from_generators(crystal.relaxed_generators)
    crystal.neighbors = compute_neighbors(crystal.atoms, crystal.bonds)

    return crystal

class GrapheneCrystal(LloydHybrid, CGRelaxation):
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

    def get_boundary_mask(self, generators, margin = 5):
        '''
        Identify generators that are close to the grains boundaries
        Inputs:
            generators : coordinates of the generators
            margin : distance from the boundary
        Outputs:
            boundary_mask : boolean mask
        '''
        tree = cKDTree(generators)
        boundary_mask = np.zeros(len(generators), dtype=bool)

        for v1, v2 in zip(self.lattice.ridge_v1, self.lattice.ridge_v2):
            edge_len = np.linalg.norm(v2 - v1)
            n_samples = max(2, int(edge_len / (1.42 * 0.5)))
            for t in np.linspace(0, 1, n_samples):
                pt = v1 * (1 - t) + v2 * t
                idxs = tree.query_ball_point(pt, margin)
                for idx in idxs:
                    boundary_mask[idx] = True

        return boundary_mask

    def vertices_from_generators(self, generators):
        '''
        Compute the vertices of the Voronoi diagram from the generators
        Inputs:
            generators : coordinates of the generators
        Outputs:
            vertices : coordinates of the atoms in the graphene lattice
        '''

        images = [generators + np.array([dx, dy]) 
                    for dx in [-self.L, 0, self.L]
                    for dy in [-self.L, 0, self.L]]
        all_gen = np.vstack(images)
        M = len(generators)
        vor = Voronoi(all_gen)

        vertex_indices = set()
        for all_idx in range(4*M, 5*M):
            region = vor.regions[vor.point_region[all_idx]]
            if -1 in region or len(region) == 0:
                continue
            vertex_indices.update(region)

        vertex_list = sorted(vertex_indices)
        old_to_new = {old: new for new, old in enumerate(vertex_list)}
        atoms = vor.vertices[vertex_list]

        bonds = []
        for (vi, vj) in vor.ridge_vertices:
            if vi == -1 or vj == -1:
                continue
            
            if vi not in old_to_new or vj not in old_to_new:
                continue

            bonds.append((old_to_new[vi], old_to_new[vj]))

        bonds = np.array(bonds, dtype=np.int64)

        mask = (atoms[:, 0] >= 0) & (atoms[:, 0] <= self.L) & (atoms[:, 1] >= 0) & (atoms[:, 1] <= self.L)

        bond_mask = mask[bonds[:, 0]] & mask[bonds[:, 1]]
        bonds = bonds[bond_mask]
        new_indices = np.full(len(atoms), -1, dtype=np.int64)
        new_indices[mask] = np.arange(np.sum(mask))
        bonds = new_indices[bonds]
        bonds = bonds[(bonds[:, 0] >= 0) & (bonds[:, 1] >= 0)]

        atoms = atoms[mask]

        return atoms, bonds
    
    def build_polycrystal(self, a_CC = 1.42, margin = 5, n_iter = 50, tol = 0.1):
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

            polygon = Polygon(self.vor.vertices[vertices]).buffer(0.1)

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

        boundary_mask = self.get_boundary_mask(generators, margin)

        # Relax the generators using Lloyd's algorithm
        if torch.cuda.is_available():
            self.relaxed_generators = self.relaxation_GPU(generators, boundary_mask, n_iter=n_iter, tol=tol)
        else:
            print("Using CPU for relaxation.")
            self.relaxed_generators = self.relaxation_CPU(generators, boundary_mask, n_iter=n_iter, tol=tol)

        # Construct the atoms and bonds from the relaxed generators
        self.atoms, self.bonds = self.vertices_from_generators(self.relaxed_generators)

        # Relax the atoms using LAMMPS
        self.atoms = self.relaxation_CG(
            atoms=self.atoms,
            generators=self.relaxed_generators,
            generator_boundary_mask=boundary_mask,
            ftol=0.1,
            max_steps=100,
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

        G6 = obs.compute_orientational_correlation(self.atoms, self.neighbors, bin_bounds, n_samples)

        return bin_centers, G6

    def plot_atoms(self):
        fig_size = max(4, min(20, self.L / 30))
        plt.figure(figsize=(fig_size, fig_size))

        dot_size = max(0.5, 500 / self.L**2)

        plt.scatter(self.atoms[:, 0], self.atoms[:, 1], s=dot_size, color='black')
        plt.xlim(0, self.L)
        plt.ylim(0, self.L)
        plt.gca().set_aspect('equal')
        plt.title('Graphene Crystal')
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.tight_layout()

    def plot_bonds(self):
        fig_size = max(4, min(20, self.L / 30))
        plt.figure(figsize=(fig_size, fig_size))

        lw = max(0.2, 5 / self.L)
        dot_size = max(0.5, 500 / self.L**2)

        lines = [(self.atoms[i], self.atoms[j]) for i, j in self.bonds]
        lc = LineCollection(lines, colors='black', linewidths=lw)
        plt.gca().add_collection(lc)
        plt.scatter(self.relaxed_generators[:, 0], self.relaxed_generators[:, 1], s=dot_size, color='red') 
        plt.xlim(0, self.L)
        plt.ylim(0, self.L)
        plt.gca().set_aspect('equal')
        plt.title('Graphene Bonds')
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.tight_layout()

    def save_crystal(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        np.savez_compressed(
            path,
            relaxed_generators = self.relaxed_generators,
            points = self.points,
            theta = self.theta,
            L = np.array([self.L]),
            rho = np.array([self.lattice.rho]),
        )


# Test

if __name__ == "__main__":
    import time

    _ = generate_triangular_lattice(10.0)

    configs = [
        (200,  0.0003,  "12 grains / 200Å  — test de base"),
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

        crystal.plot_atoms()
        plt.savefig(f"results/test_{L:.0f}_{rho:.0e}_atoms.png", dpi=300)
        plt.close()
        crystal.plot_bonds()
        plt.savefig(f"results/test_{L:.0f}_{rho:.0e}_bonds.png", dpi=300)
        plt.close()


        t1 = time.time()
        print(f"  Grains   : {vor.N}")
        print(f"  Atomes   : {len(crystal.atoms):,}")
        print(f"  Temps    : {t1 - t0:.2f} s")