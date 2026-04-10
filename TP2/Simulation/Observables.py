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
def compute_psi6(i, atoms, neighbors):
    '''
    Compute the local orientational order parameter
    Inputs:
        atoms : coordinates of the atoms
        neighbors : list of nearest neighbors for each atom
    Outputs:
        psi6 : local orientational order parameter for atom i
    '''
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
    psi6 = sum_psi / cnt if cnt > 0 else 0.0 + 0.0j
    return psi6            

@njit
def compute_orientational_correlation(coords, neighbors, bin_bounds, n_samples = 50_000):
    '''
    Compute the orientational correlation function G6(r) 
    Inputs:
        coords : coordinates of the atoms
        neighbors : list of nearest neighbors for each atom
        bin_bounds : array of bin boundaries
        n_samples : number of samples to use for the correlation function
    Outputs:
        G6 : orientational correlation function for each bin
    '''
    N = len(coords)
    num_bins = len(bin_bounds) - 1
    r_max = bin_bounds[num_bins]

    G6 = np.zeros(num_bins, dtype=np.float64)
    count = np.zeros(num_bins, dtype=np.int64)

    for _ in range(n_samples):
        i = np.random.randint(0, N)
        j = np.random.randint(i, N)
        if i == j:
            continue
        
        dx = coords[i, 0] - coords[j, 0]
        dy = coords[i, 1] - coords[j, 1]
        r = np.sqrt(dx * dx + dy * dy)
        if r >= r_max:
            continue

        b = bin_index(r, bin_bounds)
        if b < 0:
            continue

        psi_6_i = compute_psi6(i, coords, neighbors)
        psi_6_j = compute_psi6(j, coords, neighbors)

        G6[b] += psi_6_i.real * psi_6_j.real + psi_6_i.imag * psi_6_j.imag
        count[b] += 1

    for b in range(num_bins):
        if count[b] > 0:
            G6[b] /= count[b]
    return G6

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
    phase_x = np.exp(1j * np.outer(qx_vals, x))
    phase_y = np.exp(1j * np.outer(qy_vals, y))

    fx = np.sum(phase_x, axis=1)
    fy = np.sum(phase_y, axis=1)

    Sq_complex = phase_x @ phase_y.T
    Sq = (Sq_complex.real ** 2 + Sq_complex.imag ** 2) / N

    QX, QY = np.meshgrid(qx_vals, qy_vals, indexing='ij')
    mask = (QX ** 2 + QY ** 2) < q_min ** 2
    Sq[mask] = -1.0

    idx = np.argmax(Sq)
    i_best, j_best = np.unravel_index(idx, Sq.shape)

    return float(qx_vals[i_best]), float(qy_vals[j_best])

@njit
def compute_translationnal_correlation_grain(coords, bin_bounds, G, n_samples = 50_000):
    '''
    Compute the translational correlation function CG(r) for a single grain
    Inputs:
        coords : coordinates of the atoms
        bin_bounds : array of bin boundaries
        G : lattice vectors of the system
        n_samples : number of samples to use for the correlation function
    Outputs:
        CG : translational correlation function for each bin
    '''
    N = len(coords)
    num_bins = len(bin_bounds) - 1
    r_max = bin_bounds[num_bins]

    CG = np.zeros(num_bins, dtype=np.float64)
    count = np.zeros(num_bins, dtype=np.int64)

    Gx = G[0]
    Gy = G[1]

    for _ in range(n_samples):
        i = np.random.randint(0, N)
        j = np.random.randint(i, N)
        if i == j:
            continue

        dx = coords[i, 0] - coords[j, 0]
        dy = coords[i, 1] - coords[j, 1]
        r = np.sqrt(dx * dx + dy * dy)
        if r >= r_max:
            continue

        b = bin_index(r, bin_bounds)
        if b < 0:
            continue
        
        phi_i = Gx * coords[i, 0] + Gy * coords[i, 1]
        phi_j = Gx * coords[j, 0] + Gy * coords[j, 1]
        CG[b] += np.cos(phi_i - phi_j)
        count[b] += 1

    for b in range(num_bins):
        if count[b] > 0:
            CG[b] /= count[b]
    return CG

def compute_translationnal_correlation_total(atoms, grain_mask, bin_bounds, n_q = 200, q_max = 6.0, n_samples = 50_000):
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

        CG = compute_translationnal_correlation_grain(grain_atoms, bin_bounds, G, n_samples)
        curves.append(CG)

    curves = np.array(curves)
    CG_total = np.mean(curves, axis=0)
    return CG_total