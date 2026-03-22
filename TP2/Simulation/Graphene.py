import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from numba import njit
from shapely.geometry import Point, Polygon
from scipy.spatial import cKDTree
from collections import defaultdict

import Observables as obs

@njit
def generate_graphene_lattice(L, a_CC = 1.42):
    '''
    Generate a graphene lattice with lattice constant a
    '''
    a = a_CC * np.sqrt(3)

    a1 = np.array([a, 0])
    a2 = np.array([a/2, a*np.sqrt(3)/2])

    bA = np.array([0, 0])
    bB = np.array([a/2, a*np.sqrt(3)/6])

    nmax = int(L / a) + 3

    atoms = np.empty((8 * nmax**2, 2), dtype=np.float64)

    idx = 0

    for i in range(-nmax, nmax):
        for j in range(-nmax, nmax):
            r = i * a1 + j * a2

            atoms[idx] = r + bA
            idx += 1
            atoms[idx] = r + bB
            idx += 1

    atoms = atoms[:idx]
    return atoms

@njit
def rotate_atoms(atoms, theta):
    '''
    Rotate atoms in each grain by the corresponding angle in theta
    '''
    rotated_atoms = np.empty_like(atoms)
    c, s = np.cos(theta), np.sin(theta)
    for i in range(len(atoms)):
        dx = atoms[i, 0]
        dy = atoms[i, 1]

        x_rot = c * dx - s * dy
        y_rot = s * dx + c * dy
        rotated_atoms[i, 0] = x_rot
        rotated_atoms[i, 1] = y_rot
    return rotated_atoms

@njit
def compute_neighbors(atoms, bonds):
    N = len(atoms)
    neighbors = -np.ones((N, 3), dtype=np.int64)
    for i, j in bonds:
        if neighbors[i, 0] == -1:
            neighbors[i, 0] = j
        elif neighbors[i, 1] == -1:
            neighbors[i, 1] = j
        elif neighbors[i, 2] == -1:
            neighbors[i, 2] = j

        if neighbors[j, 0] == -1:
            neighbors[j, 0] = i
        elif neighbors[j, 1] == -1:
            neighbors[j, 1] = i
        elif neighbors[j, 2] == -1:
            neighbors[j, 2] = i
    return neighbors

class GrapheneCrystal:
    def __init__(self, voronoi, a = 1.42):
        self.lattice = voronoi
        self.vor = voronoi.vor
        self.L = voronoi.L
        self.N = voronoi.N
        self.points = voronoi.points
        self.all_points = voronoi.all_points
        self.theta = voronoi.theta
        self.build_polycrystal(a)

    def remove_close_atoms(self, min_dist):
        tree = cKDTree(self.atoms)
        pairs = tree.query_pairs(min_dist)

        to_remove = set()
        for i, j in pairs:
            if j in to_remove or i in to_remove:
                continue
            to_remove.add(j)

        mask = np.ones(len(self.atoms), dtype=bool)
        mask[list(to_remove)] = False
        self.atoms = self.atoms[mask]

    def build_graphene_bonds(self, a = 1.42, max_bonds = 3):
        tree = cKDTree(self.atoms)
        pairs = tree.query_pairs(a * 1.2)

        self.bonds = []
        for i, j in pairs:
            d = np.linalg.norm(self.atoms[i] - self.atoms[j])
            if a * 0.8 <= d <= a * 1.2:
                self.bonds.append((i, j))
        self.bonds = np.array(self.bonds)

        bond_count = np.zeros(len(self.atoms), dtype=np.int64)
        for i, j in self.bonds:
            bond_count[i] += 1
            bond_count[j] += 1
        overcoordinated = np.where(bond_count > max_bonds)[0]
        
        while len(overcoordinated) > 0:
            neighbors_dict = defaultdict(list)
            for i, j in self.bonds:
                neighbors_dict[i].append(j)
                neighbors_dict[j].append(i)

            to_remove = set()

            for idx in overcoordinated:
                if idx in to_remove:
                    continue

                if bond_count[idx] <= max_bonds:
                    continue
                
                candidates = [idx] + neighbors_dict[idx]
                worst = max(candidates, key=lambda x: bond_count[x])
                to_remove.add(worst)

                bond_count[worst] = 0
                for neighbor in neighbors_dict[worst]:
                    bond_count[neighbor] -= 1
                
            mask = np.ones(len(self.atoms), dtype=bool)
            mask[list(to_remove)] = False
            self.atoms = self.atoms[mask]

            new_index = np.full(len(mask), -1, dtype=np.int64)
            new_index[mask] = np.arange(np.sum(mask))

            new_bonds = []
            for i, j in self.bonds:
                if mask[i] and mask[j]:
                    new_bonds.append((new_index[i], new_index[j]))

            self.bonds = np.array(new_bonds)

            bond_count = np.zeros(len(self.atoms), dtype=np.int64)
            for i, j in self.bonds:
                bond_count[i] += 1
                bond_count[j] += 1
            overcoordinated = np.where(bond_count > max_bonds)[0]
        
        bond_count = np.zeros(len(self.atoms), dtype=np.int64)
        for i, j in self.bonds:
            bond_count[i] += 1
            bond_count[j] += 1
        undercoordinated = np.where(bond_count < max_bonds)[0]
        max_bond_length = a * 1.7

        tree = cKDTree(self.atoms)
        bonds_set = set(tuple(sorted((i, j))) for i, j in self.bonds)
        new_bonds = list(self.bonds)

        while len(undercoordinated) > 0:
            made_any = False
            for idx in undercoordinated:
                if bond_count[idx] >= max_bonds:
                    continue

                neighbors = tree.query_ball_point(self.atoms[idx], max_bond_length)
                neighbors = [n for n in neighbors if n != idx and bond_count[n] < max_bonds and tuple(sorted((idx, n))) not in bonds_set]
                if not neighbors:
                    continue

                nearest = min(neighbors, key=lambda n: np.linalg.norm(self.atoms[idx] - self.atoms[n]))

                bond_key = tuple(sorted((idx, nearest)))
                bonds_set.add(bond_key)
                new_bonds.append(bond_key)
                bond_count[idx] += 1
                bond_count[nearest] += 1
                made_any = True

            if not made_any:
                break

            undercoordinated = np.where(bond_count < max_bonds)[0]

        self.bonds = np.array(new_bonds)
    
    def build_polycrystal(self, a = 1.42):
        
        base_lattice = generate_graphene_lattice(3 * self.L / 2, a)
        base_lattice += np.array([self.L, self.L])

        all_atoms = []

        for grain in range(len(self.all_points)):
            region_idx = self.vor.point_region[grain]
            vertices = self.vor.regions[region_idx]

            if -1 in vertices or len(vertices) == 0:
                continue

            polygon = Polygon(self.vor.vertices[vertices]).buffer(a * 0.8)

            theta = self.theta[grain % self.N]
            rot_atoms = rotate_atoms(base_lattice, theta)

            min_x, min_y, max_x, max_y = polygon.bounds
            mask = (rot_atoms[:, 0] >= min_x) & (rot_atoms[:, 0] <= max_x) & (rot_atoms[:, 1] >= min_y) & (rot_atoms[:, 1] <= max_y)

            rot_atoms = rot_atoms[mask]

            for atom in rot_atoms:
                if polygon.covers(Point(atom)):
                    all_atoms.append(atom)

        self.atoms = np.vstack(all_atoms)

        mask = (self.atoms[:, 0] >= 0) & (self.atoms[:, 0] <= self.L) & (self.atoms[:, 1] >= 0) & (self.atoms[:, 1] <= self.L)
        self.atoms = self.atoms[mask]
        
        self.remove_close_atoms(a * 0.8)
        self.build_graphene_bonds()
        self.neighbors = compute_neighbors(self.atoms, self.bonds)

    def compute_observables(self):
        self.psi6 = obs.compute_psi6(self.atoms, self.neighbors)

        bin_bounds = np.linspace(0, self.L / 2, 51)
        self.G6 = obs.compute_orientational_correlation(self.psi6, self.atoms, bin_bounds)

    def plot_atoms(self):
        plt.figure(figsize=(6,6))
        plt.scatter(self.atoms[:, 0], self.atoms[:, 1], s=1, color='black')
        plt.xlim(0, self.L)
        plt.ylim(0, self.L)
        plt.gca().set_aspect('equal')
        plt.title('Graphene Crystal')
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.tight_layout()

    def plot_bonds(self):
        plt.figure(figsize=(6,6))
        lines = [(self.atoms[i], self.atoms[j]) for i, j in self.bonds]
        lc = LineCollection(lines, colors='black', linewidths=0.5)
        plt.gca().add_collection(lc)
        plt.xlim(0, self.L)
        plt.ylim(0, self.L)
        plt.gca().set_aspect('equal')
        plt.title('Graphene Bonds')
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.tight_layout()