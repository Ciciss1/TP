import time
import numpy as np
from numba import njit
from scipy.spatial import Voronoi

@njit
def polygon_centroid(vertices):
    '''
    Compute the centroid of a polygon given its vertices
    Inputs:
        vertices : coordinates of the vertices
    Outputs:
        centroid : coordinates of the centroid
    '''
    x, y = vertices[:, 0], vertices[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    A = 0.5 * np.sum(cross)
    cx = np.sum((x + np.roll(x, -1)) * cross) / (6 * A)
    cy = np.sum((y + np.roll(y, -1)) * cross) / (6 * A)
    return np.array([cx, cy])

def relaxation_CPU(L, generators, boundary_mask, n_iter = 200, tol = 1e-3):
    '''
    Minimize the distance between the generators and the centroids of their Voronoi cells using Lloyd's algorithm
    Inputs:
        L : physical size of the box
        generators : coordinates of the generators
        boundary_mask : boolean mask indicating which generators are close to the boundaries
        n_iter : maximum number of iterations
        tol : tolerance for convergence
    Outputs:
        relaxed_generators : coordinates of the relaxed generators
    '''
    generators_relax = generators.copy()
    free_idx = np.where(boundary_mask)[0]
    N = len(generators_relax)

    for it in range(n_iter):
        t0 = time.perf_counter()
        images = [generators_relax + np.array([dx, dy]) 
                    for dx in [-L, 0, L]
                    for dy in [-L, 0, L]]
        all_gen = np.vstack(images)
        vor = Voronoi(all_gen)

        new_positions = generators_relax.copy()
        point_region = np.array(vor.point_region)

        for idx in free_idx:
            region = vor.regions[point_region[idx + 4*N]]

            if -1 in region or len(region) == 0:
                continue

            verts = vor.vertices[region]

            ref = generators_relax[idx]
            verts = verts - L * np.round((verts - ref) / L)

            centroid = polygon_centroid(verts)
            new_positions[idx] = centroid % L

        delta = np.max(np.linalg.norm(new_positions[free_idx] - generators_relax[free_idx], axis=1))
        generators_relax = new_positions

        # print(f"Iteration {it+1}, max displacement: {delta:.6f}, time: {time.perf_counter() - t0:.2f} s")

        if delta < tol:
            break

    return generators_relax

class Lloyd:

    def relaxation(self, generators, boundary_mask, n_iter = 50, tol = 1e-3):
        
        return relaxation_CPU(self.L, generators, boundary_mask, n_iter, tol)
