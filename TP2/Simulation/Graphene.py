import sys
sys.path.insert(0, "TP2/Simulation")
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from numba import njit
from shapely.geometry import Polygon
from shapely.vectorized import contains
from scipy.spatial import cKDTree
from collections import defaultdict

import Observables as obs
from Voronoi import PeriodicVoronoi

@njit
def generate_graphene_lattice(L, a_CC = 1.42):
    '''
    Generate a graphene lattice with lattice constant a
    Inputs:
        L : size of the box
        a_CC : carbon-carbon bond length
    Outputs:
        atoms : coordinates of the atoms in the graphene lattice
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

    crystal.atoms = data['atoms']
    crystal.bonds = data['bonds']
    crystal.neighbors = compute_neighbors(crystal.atoms, crystal.bonds)

    return crystal

class GrapheneCrystal:
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

    def remove_close_atoms(self, min_dist):
        '''
        Remove atoms that are closer than min_dist to each other
        Inputs:
            min_dist : minimum distance between atoms
        '''
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
        '''
        Build the bonds between atoms based on their distances and ensure that each atom has at most 3 bonds
        Inputs:
            a : carbon-carbon bond length
            max_bonds : maximum number of bonds per atom
        '''
        tree = cKDTree(self.atoms)
        raw_pairs = np.array(list(tree.query_pairs(a * 1.2)))

        if len(raw_pairs) == 0:
            self.bonds = np.empty((0, 2), dtype=np.int64)
            return
        
        diff = self.atoms[raw_pairs[:, 0]] - self.atoms[raw_pairs[:, 1]]
        dists = np.hypot(diff[:, 0], diff[:, 1])
        mask = (dists >= a * 0.8) & (dists <= a * 1.2)
        self.bonds = raw_pairs[mask]

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
        '''
        Build the polycrystalline graphene structure
        Inputs:
            a : carbon-carbon bond length
        '''
        
        base_lattice = generate_graphene_lattice(self.L, a)

        all_atoms = []

        for grain in range(len(self.all_points)):
            region_idx = self.vor.point_region[grain]
            vertices = self.vor.regions[region_idx]

            if -1 in vertices or len(vertices) == 0:
                continue

            polygon = Polygon(self.vor.vertices[vertices]).buffer(a * 0.5)

            min_x, min_y, max_x, max_y = polygon.bounds
            if (max_x < 0 or min_x > self.L or max_y < 0 or min_y > self.L):
                continue

            theta = self.theta[grain % self.N]
            center = self.all_points[grain]
            
            rot_atoms = rotate_and_move_atoms(base_lattice, theta, center)

            mask = (rot_atoms[:, 0] >= min_x) & (rot_atoms[:, 0] <= max_x) & (rot_atoms[:, 1] >= min_y) & (rot_atoms[:, 1] <= max_y)

            rot_atoms = rot_atoms[mask]

            inside = contains(polygon, rot_atoms[:, 0], rot_atoms[:, 1])
            all_atoms.append(rot_atoms[inside])

        self.atoms = np.vstack(all_atoms)

        mask = (self.atoms[:, 0] >= 0) & (self.atoms[:, 0] <= self.L) & (self.atoms[:, 1] >= 0) & (self.atoms[:, 1] <= self.L)
        self.atoms = self.atoms[mask]
        
        self.remove_close_atoms(a * 0.8)
        self.build_graphene_bonds()
        self.neighbors = compute_neighbors(self.atoms, self.bonds)

        del self.vor
        del self.all_points

    def compute_grain_mask(self):
        '''
        Compute a mask indicating which grain each atom belongs to
        '''
        tree = cKDTree(self.points)
        _, grain_mask = tree.query(self.atoms)
        return grain_mask.astype(np.int32)

    def compute_observables(self):
        '''
        Compute the observables for the graphene crystal
        '''
        bin_bounds = np.linspace(0, self.L / 2, 101)
        bin_centers = 0.5 * (bin_bounds[:-1] + bin_bounds[1:])

        grain_mask = self.compute_grain_mask()

        G6 = obs.compute_orientational_correlation(self.atoms, self.neighbors, bin_bounds)

        GT = obs.compute_translational_correlation(self.atoms, grain_mask, self.points, self.theta, bin_bounds, self.L)

        return bin_centers, G6, GT

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

    def save_crystal(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        np.savez(
            path,
            bonds = self.bonds,
            atoms = self.atoms,
            points = self.points,
            theta = self.theta,
            L = np.array([self.L]),
            rho = np.array([self.lattice.rho]),
        )