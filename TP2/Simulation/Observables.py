import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree

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
def compute_psi6(i, atoms, neighbors, L):
    '''
    Compute the local orientational order parameter
    Inputs:
        i : index of the atom
        atoms : coordinates of the atoms
        neighbors : list of nearest neighbors for each atom
        L : size of the system
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
        dx -= L * np.round(dx / L)
        dy -= L * np.round(dy / L)
        theta = np.arctan2(dy, dx)
        sum_psi += np.exp(6j * theta)
        cnt += 1
    psi6 = sum_psi / cnt if cnt > 0 else 0.0 + 0.0j
    return psi6            

@njit
def compute_orientational_correlation(coords, neighbors, bin_bounds, L, n_samples = 10_000_000):
    '''
    Compute the orientational correlation function G6(r) 
    Inputs:
        coords : coordinates of the atoms
        neighbors : list of nearest neighbors for each atom
        bin_bounds : array of bin boundaries
        L : size of the system
        n_samples : number of samples to use for the correlation function
    Outputs:
        G6 : orientational correlation function for each bin
    '''
    N = len(coords)
    num_bins = len(bin_bounds) - 1

    G6_re = np.zeros(num_bins, dtype=np.float64)
    count = np.zeros(num_bins, dtype=np.int64)

    psi6_values = np.empty(N, dtype=np.complex128)
    for i in range(N):
        psi6_values[i] = compute_psi6(i, coords, neighbors, L)

    for _ in range(n_samples):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        
        dx = coords[i, 0] - coords[j, 0]
        dy = coords[i, 1] - coords[j, 1]
        dx -= L * np.round(dx / L)
        dy -= L * np.round(dy / L)
        r = np.sqrt(dx * dx + dy * dy)

        b = bin_index(r, bin_bounds)
        if b < 0:
            continue

        psi6_i = psi6_values[i]
        psi6_j = psi6_values[j]

        val_re = psi6_i.real * psi6_j.real + psi6_i.imag * psi6_j.imag

        G6_re[b] += val_re
        count[b] += 1

    G6 = np.zeros(num_bins, dtype=np.float64)
    for b in range(num_bins):
        if count[b] > 0:
            G6[b] = G6_re[b] / count[b]
    
    G6[0] = 1.0
    
    return G6

@njit
def build_reference_sites(cx, cy, cos_t, sin_t, L, a_CC = 1.42, sublattice = 'A'):
    '''
    Build the reference sites of the graphene lattice
    Inputs:
        cx, cy : coordinates of the center of the grain
        cos_t, sin_t : cosine and sine of the rotation angle
        L : size of the lattice
        a_CC : carbon-carbon bond length
        sublattice : sublattice type ('A' or 'B')
    Outputs:
        sites : coordinates of the reference sites
    '''
    a = a_CC * 1.7320508075688772
    nmax = int(L / a) + 3
    
    a1x, a1y = a, 0
    a2x, a2y = a / 2, a * 0.8660254037844386

    if sublattice == 'A':
        bx, by = 0.0, 0.0
    else:
        bx, by = a_CC, 0.0

    n = 2 * (2 * nmax + 1) ** 2
    sites = np.empty((n, 2), dtype=np.float64)
    idx = 0

    for i in range(-nmax, nmax + 1):
        for j in range(-nmax, nmax + 1):
            lx = i * a1x + j * a2x + bx
            ly = i * a1y + j * a2y + by

            sites[idx, 0] = cx + cos_t * lx - sin_t * ly
            sites[idx, 1] = cy + sin_t * lx + cos_t * ly
            idx += 1

    return sites[:idx]

def assign_sublattice(coords, neighbors):
    '''
    Assign the sublattice type (A or B) to each atom
    Inputs:
        coords : coordinates of the atoms
        neighbors : list of nearest neighbors for each atom
    Outputs:
        sublattice : array indicating the sublattice type for each atom (0 for A, 1 for B)
    '''
    N = len(coords)
    label = np.full(N, -1, dtype=np.int64)

    for start in range(N):
        if label[start] != -1:
            continue
        queue = [start]
        label[start] = 0
        head = 0

        while head < len(queue):
            i = queue[head]
            head += 1
            for j in neighbors[i]:
                if j < 0:
                    break
                if label[j] == -1:
                    label[j] = 1 - label[i]
                    queue.append(j)

    return label

def compute_reference_sites(coords, neighbors, grain_mask, grain_center, grain_theta, L, a_CC = 1.42):
    '''
    Compute the reference sites for the atoms in a grain
    Inputs:
        coords : coordinates of the atoms
        neighbors : list of nearest neighbors for each atom
        grain_mask : boolean array indicating which atoms belong to the grain
        grain_center : coordinates of the center of the grain
        grain_theta : orientation angle of the grain
        L : size of the system
        a_CC : carbon-carbon bond length
    Outputs:
        R : coordinates of the reference sites for the atoms in the grain
    '''
    N = len(coords)
    R = np.empty((N, 2), dtype=np.float64)

    sublattice = assign_sublattice(coords, neighbors)
    grain_ids = np.unique(grain_mask)
    grain_ids = grain_ids[grain_ids >= 0]

    for g in grain_ids:
        idx = np.where(grain_mask == g)[0]
        if len(idx) == 0:
            continue

        theta = grain_theta[g]
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        cx, cy = float(grain_center[g, 0]), float(grain_center[g, 1])

        idx_A = idx[sublattice[idx] == 0]
        idx_B = idx[sublattice[idx] == 1]

        refs_A = build_reference_sites(cx, cy, cos_t, sin_t, L, a_CC, sublattice='A')
        tree_A = cKDTree(refs_A)
        _, nn_A = tree_A.query(coords[idx_A, :2], k=1, workers=-1)

        refs_B = build_reference_sites(cx, cy, cos_t, sin_t, L, a_CC, sublattice='B')
        tree_B = cKDTree(refs_B)
        _, nn_B = tree_B.query(coords[idx_B, :2], k=1, workers=-1)

        R[idx_A] = refs_A[nn_A]
        R[idx_B] = refs_B[nn_B]

    return R

@njit(parallel=True)
def compute_GT(coords, R, grain_of_atoms, Gx_per_grain, Gy_per_grain, bin_bounds, L, n_samples = 10_000_000, n_threads = 8):
    '''
    Compute the translational correlation function GT(r) 
    Inputs:
        coords : coordinates of the atoms
        R : coordinates of the reference sites for each atom
        grain_of_atoms : array indicating the grain of each atom
        Gx_per_grain, Gy_per_grain : components of the reciprocal lattice vector for each grain
        bin_bounds : array of bin boundaries
        L : size of the lattice
        n_samples : number of samples to use for the correlation function
    Outputs:
        GT : translational correlation function
    '''
    N = len(coords)
    num_bins = len(bin_bounds) - 1
    chunk = n_samples // n_threads

    GT = np.zeros((n_threads, num_bins), dtype=np.float64)
    count = np.zeros((n_threads, num_bins), dtype=np.int64)

    for t in prange(n_threads):
        gt = np.zeros(num_bins, dtype=np.float64)
        cnt = np.zeros(num_bins, dtype=np.int64)

        for _ in range(chunk):
            i = np.random.randint(0, N)
            j = np.random.randint(0, N)

            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dx -= L * np.round(dx / L)
            dy -= L * np.round(dy / L)

            r = np.sqrt(dx * dx + dy * dy)

            b = bin_index(r, bin_bounds)
            if b < 0:
                continue

            g_i = grain_of_atoms[i]
            g_j = grain_of_atoms[j]

            ux_i = R[i, 0] - coords[i, 0]
            uy_i = R[i, 1] - coords[i, 1]

            ux_j = R[j, 0] - coords[j, 0]
            uy_j = R[j, 1] - coords[j, 1]

            phi_i = Gx_per_grain[g_i] * ux_i + Gy_per_grain[g_i] * uy_i
            phi_j = Gx_per_grain[g_j] * ux_j + Gy_per_grain[g_j] * uy_j

            gt[b] += np.cos(phi_i - phi_j)
            cnt[b] += 1

        GT[t] = gt
        count[t] = cnt

    GT_total = np.zeros(num_bins, dtype=np.float64)
    count_total = np.zeros(num_bins, dtype=np.int64)
    for t in range(n_threads):
        for b in range(num_bins):
            GT_total[b] += GT[t, b]
            count_total[b] += count[t, b]

    for b in range(num_bins):
        if count_total[b] > 0:
            GT_total[b] /= count_total[b]

    GT_total[0] = 1.0

    return GT_total

def compute_translational_correlation(coords, neighbors, grain_of_atoms, grain_centers, grain_thetas, bin_bounds, L, n_samples = 20_000_000, a_CC = 1.42):
    '''
    Compute the translational correlation function GT(r) 
    Inputs:
        coords : coordinates of the atoms
        neighbors : array of neighbor indices for each atom
        grain_of_atoms : array indicating the grain of each atom
        grain_centers : coordinates of the centers of the grains
        grain_thetas : orientation angles of the grains
        bin_bounds : array of bin boundaries
        L : size of the system
        n_samples : number of samples to use for the correlation function
        a_CC : carbon-carbon bond length (default: 1.42 Angstroms)
    Outputs:
        GT : translational correlation function for each bin
    '''
    a = a_CC * np.sqrt(3)
    G_base = (2 * np.pi / a) * np.array([1.0, -1.0 / np.sqrt(3)])

    n_grains = len(grain_centers)
    Gx = np.empty(n_grains, dtype=np.float64)
    Gy = np.empty(n_grains, dtype=np.float64)

    for g in range(n_grains):
        theta = grain_thetas[g]
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        Gx[g] = cos_t * G_base[0] - sin_t * G_base[1]
        Gy[g] = sin_t * G_base[0] + cos_t * G_base[1]

    R = compute_reference_sites(coords, neighbors, grain_of_atoms, grain_centers, grain_thetas, L, a_CC)

    return compute_GT(
        np.ascontiguousarray(coords[:, :2], dtype=np.float64),
        np.ascontiguousarray(R, dtype=np.float64),
        np.ascontiguousarray(grain_of_atoms, dtype=np.int64),
        np.ascontiguousarray(Gx, dtype=np.float64),
        np.ascontiguousarray(Gy, dtype=np.float64),
        np.ascontiguousarray(bin_bounds, dtype=np.float64),
        L, n_samples
    )