import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.spatial import Voronoi, cKDTree
from shapely.geometry import Polygon

@njit
def generate_ir_111_substrate(L, a_sub = 2.715):
    '''
    Generate a Ir(111) lattice with lattice constant a_sub
    Inputs:
        L : size of the box
        a_sub : lattice constant of the substrate
    Outputs:
        coords : coordinates of the substrate points
    '''
    a1 = np.array([a_sub, 0])
    a2 = np.array([a_sub * 0.5, a_sub * np.sqrt(3) * 0.5])

    nmax = int(L / a_sub) + 5
    coords = []

    for i in range(-nmax, nmax):
        for j in range(-nmax, nmax):
            r = i * a1 + j * a2
            coords.append(r)

    return coords

def pick_substrate_points(substrate_coords, N, L):
    '''
    Randomly pick N substrate points from the generated substrate lattice
    Inputs:
        substrate_coords : coordinates of the substrate points
        N : number of points to pick
        L : size of the box
    Outputs:
        chosen_coords : coordinates of the chosen substrate points
    '''
    mask = (substrate_coords[:, 0] >= 0) & (substrate_coords[:, 0] < L) & (substrate_coords[:, 1] >= 0) & (substrate_coords[:, 1] < L)
    substrate_coords = substrate_coords[mask]

    M = len(substrate_coords)
    if N > M:
        raise ValueError(f"Requested {N} points but only {M} are available.")
    
    indices = np.random.choice(M, N, replace=False)
    chosen_coords = substrate_coords[indices]
    return chosen_coords

class PeriodicVoronoi:
    def __init__(self, L, rho):
        '''
        Create a periodic Voronoi lattice
        Attributes:
            L : size of the box
            rho : density of points
            N : number of points
            points : coordinates of the points
            all_points : coordinates of all points (including images)
            vor : Voronoi diagram
            adj_i, adj_j : indices of adjacent points
            adj_length : length of the edge between adjacent points
            theta : random orientation of the grains
        '''
        self.L = L
        self.rho = rho
        self.N = max(1, int(rho * L * L))

        self.points = pick_substrate_points(np.array(generate_ir_111_substrate(2 * L)), self.N, L)
               
        self.build_periodic_voronoi()
        self.get_adjacency()

        self.theta = np.random.uniform(-np.pi/6, np.pi/6, self.N).astype(np.float64)

        
    def build_periodic_voronoi(self):
        '''
        Build the periodic Voronoi diagram by creating 9 images of the points
        '''
        images = []
        shifts = [-self.L, 0, self.L]

        for dx in shifts:
            for dy in shifts:
                images.append(self.points + np.array([dx, dy]))

        self.all_points = np.vstack(images)
        self.vor = Voronoi(self.all_points)
    
    def get_adjacency(self):
        '''
        Get the adjacency list, edge lengths from the Voronoi diagram
        '''
        ridge_points = np.array(self.vor.ridge_points)
        ridge_vertices = np.array(self.vor.ridge_vertices)
        
        start = 4 * self.N
        end = 5 * self.N

        i = ridge_points[:, 0]
        j = ridge_points[:, 1]

        in_i = (i >= start) & (i < end)
        in_j = (j >= start) & (j < end)

        finite = (ridge_vertices[:, 0] != -1) & (ridge_vertices[:, 1] != -1)

        gi = i % self.N
        gj = j % self.N

        interior = in_i & in_j & finite

        boundary = (in_i ^ in_j) & finite & ((in_i & (gi < gj)) | (in_j & (gj < gi)))

        mask = interior | boundary
        rv = ridge_vertices[mask]
        gi = gi[mask]
        gj = gj[mask]

        v1 = self.vor.vertices[rv[:, 0]]
        v2 = self.vor.vertices[rv[:, 1]]
        lengths = np.linalg.norm(v1 - v2, axis=1)

        keep = (lengths > 1e-10) & (gi != gj)
        self.adj_i = gi[keep].astype(np.int32)
        self.adj_j = gj[keep].astype(np.int32)
        self.adj_length = lengths[keep]

        self.areas = np.zeros(self.N)
        for idx in range(start, end):
            region_idx = self.vor.point_region[idx]
            vertices = self.vor.regions[region_idx]

            if -1 in vertices or len(vertices) == 0:
                continue

            polygon = Polygon(self.vor.vertices[vertices])
            self.areas[idx - start] = polygon.area

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