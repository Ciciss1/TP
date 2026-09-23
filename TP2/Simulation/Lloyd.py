import numpy as np
from numba import njit
from scipy.spatial import Delaunay, cKDTree
from scipy.optimize import minimize


@njit(cache=True)
def _cell_moments(pts, simplices, n_cells):
    '''
    Area, first moment and second moment of the Voronoi cells of pts[:n_cells], computed from the
    Delaunay triangulation (no Voronoi region lists needed). For every counter-clockwise triangle
    (i, j, k) with circumcentre o, the cell of i receives the two signed triangles (i, m_ij, o) and
    (i, o, m_ik), m being the edge midpoints. The signed sum is exactly the Voronoi cell (obtuse
    triangles included).
    Inputs:
        pts : all points of the triangulation (cells first, then context and periodic images)
        simplices : Delaunay triangles
        n_cells : number of cells to integrate
    Outputs:
        A : cell areas
        Sx, Sy : first moments relative to the generator, so that centroid = g + S / A
        I : second moments  int |x - g|^2 dA  (CVT energy of the cell)
    '''
    A = np.zeros(n_cells)
    Sx = np.zeros(n_cells)
    Sy = np.zeros(n_cells)
    I = np.zeros(n_cells)

    for t in range(simplices.shape[0]):
        a = simplices[t, 0]
        b = simplices[t, 1]
        c = simplices[t, 2]
        if a >= n_cells and b >= n_cells and c >= n_cells:
            continue

        bx = pts[b, 0] - pts[a, 0]
        by = pts[b, 1] - pts[a, 1]
        cx = pts[c, 0] - pts[a, 0]
        cy = pts[c, 1] - pts[a, 1]
        d = 2.0 * (bx * cy - by * cx)
        if d == 0.0:
            continue
        if d < 0.0:                       # counter-clockwise orientation
            b, c = c, b
            bx, by, cx, cy = cx, cy, bx, by
            d = -d
        b2 = bx * bx + by * by
        c2 = cx * cx + cy * cy
        ox = pts[a, 0] + (cy * b2 - by * c2) / d
        oy = pts[a, 1] + (bx * c2 - cx * b2) / d

        for r in range(3):
            if r == 0:
                i, j, k = a, b, c
            elif r == 1:
                i, j, k = b, c, a
            else:
                i, j, k = c, a, b
            if i >= n_cells:
                continue
            xi = pts[i, 0]
            yi = pts[i, 1]
            mjx = 0.5 * (pts[j, 0] - xi)
            mjy = 0.5 * (pts[j, 1] - yi)
            mkx = 0.5 * (pts[k, 0] - xi)
            mky = 0.5 * (pts[k, 1] - yi)
            px = ox - xi
            py = oy - yi
            a1 = 0.5 * (mjx * py - mjy * px)
            a2 = 0.5 * (px * mky - py * mkx)
            A[i] += a1 + a2
            Sx[i] += (a1 * (mjx + px) + a2 * (px + mkx)) / 3.0
            Sy[i] += (a1 * (mjy + py) + a2 * (py + mky)) / 3.0
            I[i] += a1 / 6.0 * (mjx * mjx + mjy * mjy + px * px + py * py + mjx * px + mjy * py) \
                  + a2 / 6.0 * (px * px + py * py + mkx * mkx + mky * mky + px * mkx + py * mky)
    return A, Sx, Sy, I


def _wrap(x, L):
    x = np.mod(x, L)
    x[x >= L] = 0.0
    return x


def _periodic_images(points, L, pad):
    '''
    Return points followed by the periodic images of the points within pad of the box edges
    '''
    x, y = points[:, 0], points[:, 1]
    images = [points]
    for dx in (-L, 0.0, L):
        for dy in (-L, 0.0, L):
            if dx == 0.0 and dy == 0.0:
                continue
            sel = np.ones(len(points), dtype=bool)
            if dx > 0: sel &= x < pad
            if dx < 0: sel &= x > L - pad
            if dy > 0: sel &= y < pad
            if dy < 0: sel &= y > L - pad
            if np.any(sel):
                images.append(points[sel] + np.array([dx, dy]))
    return np.vstack(images)


class _LocalCVT:
    '''
    Free generators + the fixed generators within a shell around them. Fixed generators further away
    cannot change the free cells nor the cells adjacent to them, so they are left out of the
    triangulation. The free generators come first.
    '''
    def __init__(self, L, generators, boundary_mask, shell_factor=4.0):
        self.L = L
        gen = _wrap(np.array(generators, dtype=np.float64), L)
        self.free = np.where(boundary_mask)[0]
        fixed = np.where(~np.asarray(boundary_mask, dtype=bool))[0]

        d_nn, _ = cKDTree(gen, boxsize=L).query(gen[:min(len(gen), 2000)], k=2)
        self.spacing = float(np.median(d_nn[:, 1]))
        shell = shell_factor * self.spacing
        self.pad = shell + 2.0 * self.spacing

        if len(fixed) > 0 and len(self.free) > 0:
            d, _ = cKDTree(gen[self.free], boxsize=L).query(gen[fixed], k=1, distance_upper_bound=shell)
            self.fixed_pos = gen[fixed[np.isfinite(d)]]
        else:
            self.fixed_pos = np.empty((0, 2))
        self.gen = gen
        self.n_free = len(self.free)
        self.n_local = self.n_free + len(self.fixed_pos)
        self.n_eval = 0

    def moments(self, free_pos):
        self.n_eval += 1
        loc = np.vstack([_wrap(free_pos.reshape(-1, 2).copy(), self.L), self.fixed_pos])
        pts = _periodic_images(loc, self.L, self.pad)
        tri = Delaunay(pts)
        return _cell_moments(pts, tri.simplices.astype(np.int64), self.n_local)

    def energy_and_grad(self, x):
        # CVT energy F = sum_i int_{V_i} |x - g_i|^2 ; dF/dg_i = 2 m_i (g_i - c_i) = -2 S_i
        A, Sx, Sy, I = self.moments(x)
        grad = -2.0 * np.column_stack([Sx[:self.n_free], Sy[:self.n_free]])
        return float(I.sum()), grad.ravel()

    def residual(self, x):
        # distance of each free generator to its centroid (the Lloyd displacement)
        A, Sx, Sy, I = self.moments(x)
        A = np.maximum(A[:self.n_free], 1e-12)
        return np.hypot(Sx[:self.n_free] / A, Sy[:self.n_free] / A), A


def relaxation_CPU(L, generators, boundary_mask, n_iter=1000, tol=1e-2, step_size=0.99, method="lbfgs"):
    '''
    Relax the free generators towards a centroidal Voronoi tessellation
    Inputs:
        L : physical size of the box
        generators : coordinates of the generators
        boundary_mask : True for the generators that are allowed to move
        n_iter : maximum number of iterations
        tol : convergence threshold on max |g_i - c_i| (AA), the Lloyd displacement
        step_size : Lloyd step (method="lloyd" only)
        method : "lbfgs" (L-BFGS on the CVT energy, Liu et al. 2009) or "lloyd"
    Outputs:
        relaxed_generators : coordinates of the relaxed generators
        info : number of energy evaluations and final max |g - c|
    '''
    cvt = _LocalCVT(L, generators, boundary_mask)
    gen = cvt.gen
    if cvt.n_free == 0:
        return gen, {"iterations": 0, "evaluations": 0, "max_displacement": 0.0}

    x = gen[cvt.free].ravel().copy()

    if method == "lbfgs":
        res0, A0 = cvt.residual(x)
        # |dF/dg_i| = 2 m_i |g_i - c_i|  ->  gradient tolerance equivalent to tol
        gtol = 2.0 * float(np.median(A0)) * tol
        res = minimize(cvt.energy_and_grad, x, jac=True, method="L-BFGS-B",
                       options=dict(maxiter=n_iter, gtol=gtol, ftol=0.0, maxcor=20))
        x = res.x
        n_it = int(res.nit)
    elif method == "lloyd":
        n_it = 0
        for n_it in range(1, n_iter + 1):
            A, Sx, Sy, _ = cvt.moments(x)
            A = np.maximum(A[:cvt.n_free], 1e-12)
            disp = step_size * np.column_stack([Sx[:cvt.n_free] / A, Sy[:cvt.n_free] / A])
            x = (x.reshape(-1, 2) + disp).ravel()
            if np.sqrt(np.max(np.sum(disp * disp, axis=1))) < tol:
                break
    else:
        raise ValueError(f"Unknown method {method}")

    final_res, _ = cvt.residual(x)
    gen[cvt.free] = _wrap(x.reshape(-1, 2).copy(), L)
    return gen, {"iterations": n_it, "evaluations": cvt.n_eval, "max_displacement": float(final_res.max())}


class Lloyd:

    def relaxation(self, generators, boundary_mask, n_iter=1000, tol=1e-2, step_size=0.99, method="lbfgs"):
        relaxed, info = relaxation_CPU(self.L, generators, boundary_mask, n_iter, tol, step_size, method)
        self.lloyd_info = info
        return relaxed