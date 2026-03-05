import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from shapely.geometry import Point, Polygon
from scipy.spatial import cKDTree

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
            to_remove.add(j)

        mask = np.ones(len(self.atoms), dtype=bool)
        mask[list(to_remove)] = False
        self.atoms = self.atoms[mask]

    def build_graphene_bonds(self, r_min = 1.2, r_max = 1.8, max_neighbors = 3):
        tree = cKDTree(self.atoms)
        pairs = tree.query_pairs(r_max)

        neighbors = [[] for _ in range(len(self.atoms))]
        for i, j in pairs:
            d = np.linalg.norm(self.atoms[i] - self.atoms[j])

            if r_min < d < r_max:
                neighbors[i].append((j, d))
                neighbors[j].append((i, d))

        self.bonds = []
        for i in range(len(self.atoms)):
            if len(neighbors[i]) == 0:
                continue

            neigh = sorted(neighbors[i], key=lambda x: x[1])[:max_neighbors]

            for j, d in neigh:
                if i < j:
                    self.bonds.append((i, j))
        self.bonds = np.array(self.bonds, dtype=np.int32)

    def build_polycrystal(self, a = 1.42):
        
        base_lattice = generate_graphene_lattice(2*self.L, a_CC=a)
        base_lattice += np.array([3*self.L/2, 3*self.L/2])

        all_atoms = []

        start = 4 * self.N
        end = 5 * self.N

        for grain in range(len(self.all_points)):
            region_idx = self.vor.point_region[grain]
            vertices = self.vor.regions[region_idx]

            if -1 in vertices or len(vertices) == 0:
                continue

            polygon = Polygon(self.vor.vertices[vertices]).buffer(1e-10)

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

    def plot(self):
        plt.figure(figsize=(6,6))
        plt.scatter(self.atoms[:, 0], self.atoms[:, 1], s=1, color='black')
        plt.xlim(0, self.L)
        plt.ylim(0, self.L)
        plt.gca().set_aspect('equal')
        plt.title('Graphene Crystal')
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.tight_layout()
