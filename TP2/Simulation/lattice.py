import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from shapely.geometry import Polygon

class PerodicVoronoi:
    def __init__(self, L, rho):
        '''
        Create a periodic Voronoi lattice.
        Attributes:
            L : size of the box
            rho : density of points
            N : number of points
            points : coordinates of the points
            adj_i, adj_j : indices of adjacent points
            adj_length : length of the edge between adjacent points
        '''
        self.L = L
        self.rho = rho
        self.N = int(L**2 * rho)

        self.points = np.random.rand(self.N, 2) * L
               
        self.build_periodic_voronoi()
        self.get_adjacency()

        
    def build_periodic_voronoi(self):
        images = []
        shifts = [-self.L, 0, self.L]

        for dx in shifts:
            for dy in shifts:
                images.append(self.points + np.array([dx, dy]))

        self.all_points = np.vstack(images)
        self.vor = Voronoi(self.all_points)
    
    def get_adjacency(self):
        adj_i = []
        adj_j = []
        adj_length = []

        start = 4 * self.N
        end = 5 * self.N

        for (i, j), verts in zip(self.vor.ridge_points, self.vor.ridge_vertices):
            
            if not (start <= i < end and start <= j < end):
                continue

            i -= start
            j -= start

            if -1 in verts or len(verts) != 2:
                raise RuntimeError(f"Ridge between points {i} and {j} has an infinite vertex, which should not happen in a periodic Voronoi diagram.")
            
            v1, v2 = self.vor.vertices[verts]
            length = np.linalg.norm(v1 - v2)

            if length > 1e-6:
                adj_i.append(i)
                adj_j.append(j)
                adj_length.append(length)
                
        self.adj_i = np.array(adj_i, dtype=np.int64)
        self.adj_j = np.array(adj_j, dtype=np.int64)
        self.adj_length = np.array(adj_length, dtype=np.float64)

    # def get_angles(self):

    def plot(self):

        plt.figure(figsize=(8, 8))

        total_points = len(self.all_points)

        for idx in range(total_points):
            region_idx = self.vor.point_region[idx]
            vertices = self.vor.regions[region_idx]

            if -1 in vertices or len(vertices) == 0:
                continue

            polygon = Polygon(self.vor.vertices[vertices])

            if polygon.is_empty:
                continue

            x, y = polygon.exterior.xy
            plt.fill(x, y, alpha=0.5)

        plt.scatter(self.points[:, 0], self.points[:, 1], color='red', s=10)
        plt.xlim(0, self.L)
        plt.ylim(0, self.L)
        plt.title(f"Periodic Voronoi Lattice (L={self.L}, rho={self.rho})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid()
        plt.tight_layout()
        plt.show()