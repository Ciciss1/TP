import numpy as np
from numba import njit

@njit
def lower_bound(arr, x):
    '''
    Find the index of the first element in arr that is greater than or equal to x
    Inputs:
        arr : sorted array
        x : value to find
    Outputs:
        index of the first element in arr that is greater than or equal to x
    '''
    left, right = 0, arr.size
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < x:
            left = mid + 1
        else:
            right = mid
    return left

@njit
def bin_index(r, bin_bounds):
    '''
    Find the index of the bin that r belongs to
    Inputs:
        r : distance
        bin_bounds : array of bin boundaries
    Outputs:
        index of the bin that r belongs to
    '''
    if r < bin_bounds[0] or r >= bin_bounds[bin_bounds.size - 1]:
        return -1
    k = lower_bound(bin_bounds, r)
    return k - 1

@njit
def compute_psi6(atoms, neighbors):
    '''
    Compute the local orientational order parameter
    Inputs:
        atoms : coordinates of the atoms
        neighbors : list of nearest neighbors for each atom
    Outputs:
        psi6 : local orientational order parameter for each atom
    '''
    N = len(atoms)
    psi6 = np.zeros(N, dtype=np.complex128)

    for i in range(N):
        nb = neighbors[i]
        sum_psi = 0.0 + 0.0j
        cnt = 0
        for j in range(len(nb)):
            if nb[j] < 0:
                continue
            dx = atoms[nb[j], 0] - atoms[i, 0]
            dy = atoms[nb[j], 1] - atoms[i, 1]
            theta = np.arctan2(dy, dx)
            sum_psi += np.exp(6j * theta)
            cnt += 1
        psi6[i] = sum_psi / cnt if cnt > 0 else 0.0 + 0.0j
    return psi6            

@njit
def compute_orientational_correlation(psi6, coords, bin_bounds):
    '''
    Compute the orientational correlation function G6(r) 
    Inputs:
        psi6 : local orientational order parameter for each atom
        coords : coordinates of the atoms
        bin_bounds : array of bin boundaries
    Outputs:
        G6 : orientational correlation function for each bin
    '''
    N = len(psi6)
    num_bins = len(bin_bounds) - 1
    G6 = np.zeros(num_bins, dtype=np.float64)
    count = np.zeros(num_bins, dtype=np.int64)

    for i in range(N):
        xi = coords[i, 0]
        yi = coords[i, 1]

        for j in range(i, N):
            dx = xi - coords[j, 0]
            dy = yi - coords[j, 1]
            r = np.sqrt(dx * dx + dy * dy)
            b = bin_index(r, bin_bounds)
            if b < 0:
                continue
            G6[b] += psi6[i].real * psi6[j].real + psi6[i].imag * psi6[j].imag
            count[b] += 1

    for b in range(num_bins):
        if count[b] > 0:
            G6[b] /= count[b]
    return G6

@njit
def compute_Sq_grid(x, y, qx_vals, qy_vals, q_min = 0.5):
    '''
    Compute S(q) on a grid
    Inputs:
        x : x coordinates of the atoms
        y : y coordinates of the atoms
        qx_vals : array of qx values for the grid
        qy_vals : array of qy values for the grid
        q_min : minimum q value to consider
    Outputs:
        best_qx : qx value that maximizes S(q)
        best_qy : qy value that maximizes S(q)
    '''
    N = len(x)
    n_qx = len(qx_vals)
    n_qy = len(qy_vals)

    best_Sq = -1.0
    best_qx = 0.0
    best_qy = 0.0

    for i in range(n_qx):
        qx = qx_vals[i]
        for j in range(n_qy):
            qy = qy_vals[j]
            
            if qx * qx + qy * qy < q_min * q_min:
                continue

            re = 0.0
            im = 0.0

            for k in range(N):
                phi = qx * x[k] + qy * y[k]
                re += np.cos(phi)
                im += np.sin(phi)

            Sq = (re * re + im * im) / N

            if Sq > best_Sq:
                best_Sq = Sq
                best_qx = qx
                best_qy = qy

    return best_qx, best_qy

@njit
def compute_translationnal_correlation_grain(coords, bin_bounds, G):
    '''
    Compute the translational correlation function CG(r) for a single grain
    Inputs:
        coords : coordinates of the atoms
        bin_bounds : array of bin boundaries
        G : lattice vectors of the system
    Outputs:
        CG : translational correlation function for each bin
    '''
    N = len(coords)
    num_bins = len(bin_bounds) - 1
    CG = np.zeros(num_bins, dtype=np.float64)
    count = np.zeros(num_bins, dtype=np.int64)

    for i in range(N):
        xi = coords[i, 0]
        yi = coords[i, 1]

        phase_i_re = np.cos(G[0] * xi + G[1] * yi)
        phase_i_im = np.sin(G[0] * xi + G[1] * yi)

        for j in range(i, N):
            dx = xi - coords[j, 0]
            dy = yi - coords[j, 1]
            r = np.sqrt(dx * dx + dy * dy)
            b = bin_index(r, bin_bounds)
            if b < 0:
                continue

            xj = coords[j, 0]
            yj = coords[j, 1]

            phase_j_re = np.cos(G[0] * xj + G[1] * yj)
            phase_j_im = np.sin(G[0] * xj + G[1] * yj)

            CG[b] += phase_i_re * phase_j_re + phase_i_im * phase_j_im
            count[b] += 1

    for b in range(num_bins):
        if count[b] > 0:
            CG[b] /= count[b]
    return CG

def compute_translationnal_correlation_total(atoms, grain_mask, bin_bounds, n_q = 200, q_max = 6.0):
    '''
    Compute the translational correlation function CG(r) for the whole system
    Inputs:
        atoms : coordinates of the atoms
        grain_mask : array of grain labels for each atom
        bin_bounds : array of bin boundaries
        n_q, q_max : parameters for the grid
    Outputs:
        CG_total : translational correlation function for the whole system
    '''
    qx_vals = np.linspace(-q_max, q_max, n_q)
    qy_vals = np.linspace(-q_max, q_max, n_q)
    n_bins = len(bin_bounds) - 1

    grain_ids = np.unique(grain_mask)
    grain_ids = grain_ids[grain_ids >= 0]

    curves = []
    for grain_id in grain_ids:
        idx = np.where(grain_mask == grain_id)[0]
        if len(idx) < 50:
            continue

        grain_atoms = atoms[idx]
        x = np.ascontiguousarray(grain_atoms[:, 0])
        y = np.ascontiguousarray(grain_atoms[:, 1])

        best_qx, best_qy = compute_Sq_grid(x, y, qx_vals, qy_vals)
        G = np.array([best_qx, best_qy])

        CG = compute_translationnal_correlation_grain(grain_atoms, bin_bounds, G)
        curves.append(CG)

    curves = np.array(curves)
    CG_total = np.mean(curves, axis=0)
    return CG_total