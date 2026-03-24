import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.spatial import Voronoi
from shapely.geometry import Polygon

class PeriodicVoronoi:
    def __init__(self, L, rho):
        '''
        Create a periodic Voronoi lattice
        Attributes:
            L : size of the box
            rho : density of points
            N : number of points
            points : coordinates of the points
            adj_i, adj_j : indices of adjacent points
            adj_length : length of the edge between adjacent points
            theta : random orientation of the grains
        '''
        self.L = L
        self.rho = rho
        self.N = int(L**2 * rho)

        self.points = np.random.rand(self.N, 2) * L
               
        self.build_periodic_voronoi()
        self.get_adjacency()

        self.theta = np.random.uniform(-np.pi/6, np.pi/6, self.N).astype(np.float64)

        
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

        v1_list = []
        v2_list = []

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
                v1_list.append(v1)
                v2_list.append(v2)

        self.adj_i = np.array(adj_i, dtype=np.int32)
        self.adj_j = np.array(adj_j, dtype=np.int32)
        self.adj_length = np.array(adj_length, dtype=np.float64)
        self.ridge_v1 = np.array(v1_list, dtype=np.float64)
        self.ridge_v2 = np.array(v2_list, dtype=np.float64)

    def plot(self):

        fig, ax = plt.subplots(figsize=(8, 6))

        total_points = len(self.all_points)

        norm = mcolors.Normalize(vmin=-np.pi/6, vmax=np.pi/6)
        cmap = cm.hsv

        for idx in range(total_points):
            region_idx = self.vor.point_region[idx]
            vertices = self.vor.regions[region_idx]

            if -1 in vertices or len(vertices) == 0:
                continue

            polygon = Polygon(self.vor.vertices[vertices])

            if polygon.is_empty:
                continue

            x, y = polygon.exterior.xy
            
            color = cmap(norm(self.theta[idx % self.N]))
            ax.fill(x, y, alpha=0.5, color=color)

        ax.scatter(self.points[:, 0], self.points[:, 1], color='black', s=10)
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.L)
        ax.set_aspect('equal')
        ax.set_title(rf"Periodic Voronoi Lattice ($L=${self.L}, $\rho=${self.rho})")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.grid()
        
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, ticks=[-np.pi/6, 0, np.pi/6])
        cbar.set_label(r"$\theta$ (rad)")
        cbar.set_ticklabels([r"$-\pi/6$", r"$0$", r"$\pi/6$"])

        plt.tight_layout()