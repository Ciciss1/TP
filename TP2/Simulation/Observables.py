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
def compute_psi6(i, atoms, neighbors):
    '''
    Compute the local orientational order parameter
    Inputs:
        i : index of the atom
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
def compute_orientational_correlation(coords, neighbors, bin_bounds, n_samples = 5_000_000):
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

    G6_re = np.zeros(num_bins, dtype=np.float64)
    count = np.zeros(num_bins, dtype=np.int64)

    for _ in range(n_samples):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        
        dx = coords[i, 0] - coords[j, 0]
        dy = coords[i, 1] - coords[j, 1]
        r = np.sqrt(dx * dx + dy * dy)
        if r >= r_max:
            continue

        b = bin_index(r, bin_bounds)
        if b < 0:
            continue

        psi6_i = compute_psi6(i, coords, neighbors)
        psi6_j = compute_psi6(j, coords, neighbors)

        val_re = psi6_i.real * psi6_j.real + psi6_i.imag * psi6_j.imag

        G6_re[b] += val_re
        count[b] += 1

    G6 = np.zeros(num_bins, dtype=np.float64)
    for b in range(num_bins):
        if count[b] > 0:
            G6[b] = G6_re[b] / count[b]
    
    return G6


@njit
def build_reference_sites(cx, cy, cos_t, sin_t, L, a_CC = 1.42):
    '''
    Build the reference sites of the graphene lattice
    Inputs:
        cx, cy : coordinates of the center of the grain
        cos_t, sin_t : cosine and sine of the rotation angle
        L : size of the lattice
        a_CC : carbon-carbon bond length
    Outputs:
        sites : coordinates of the reference sites
    '''
    a = a_CC * 1.7320508075688772
    nmax = int(L / a) + 3
    
    a1x, a1y = a, 0
    a2x, a2y = a / 2, a * 0.8660254037844386

    bAx, bAy = 0, 0
    bBx, bBy = a1x / 2, a1y * 0.28867513459481287

    n = 2 * (2 * nmax + 1) ** 2
    sites = np.empty((n, 2), dtype=np.float64)
    idx = 0

    for i in range(-nmax, nmax + 1):
        for j in range(-nmax, nmax + 1):
            Rx = i * a1x + j * a2x
            Ry = i * a1y + j * a2y

            lx = Rx + bAx
            ly = Ry + bAy

            sites[idx, 0] = cx + cos_t * lx - sin_t * ly
            sites[idx, 1] = cy + sin_t * lx + cos_t * ly
            idx += 1

            lx = Rx + bBx
            ly = Ry + bBy

            sites[idx, 0] = cx + cos_t * lx - sin_t * ly
            sites[idx, 1] = cy + sin_t * lx + cos_t * ly
            idx += 1

    return sites[:idx]

def compute_reference_sites(atoms, grain_mask, grain_centers, grain_thetas, L, a_CC = 1.42):
    '''
    Compute the reference sites of each atom
    Inputs:
        atoms : coordinates of the atoms
        grain_mask : array indicating the grain of each atom
        grain_centers : coordinates of the centers of the grains
        grain_thetas : rotation angles of the grains
        a_CC : carbon-carbon bond length
    Outputs:
        R : coordinates of the reference sites for each atom
    '''
    N = len(atoms)
    R = np.empty((N, 2), dtype=np.float64)

    grain_ids = np.unique(grain_mask)
    grain_ids = grain_ids[grain_ids >= 0]

    for gid in grain_ids:
        idx = np.where(grain_mask == gid)[0]
        if len(idx) == 0:
            continue

        ref_sites = build_reference_sites(
            float(grain_centers[gid, 0]), float(grain_centers[gid, 1]),
            float(np.cos(grain_thetas[gid])), float(np.sin(grain_thetas[gid])),
            float(L), float(a_CC)
        )
        ref_sites += np.array([L / 2, L / 2])

        tree = cKDTree(ref_sites)
        _, nearest = tree.query(atoms[idx], k=1, workers=-1)
        R[idx] = ref_sites[nearest]

    return R

@njit(parallel=True, cache=True)
def compute_GT_kernel(R, grain_of_atoms, Gx_per_grain, Gy_per_grain, bin_bounds, n_samples, n_threads):
    '''
    Compute the translational correlation function GT(r) for a single grain
    Inputs:
        R : coordinates of the reference sites for each atom
        grain_of_atoms : array indicating the grain of each atom
        Gx_per_grain, Gy_per_grain : components of the reciprocal lattice vector for each grain
        bin_bounds : array of bin boundariesµ
        n_samples : number of samples to use for the correlation function
        n_threads : number of threads to use for parallelization
    Outputs:
        GT : translational correlation for each grain
    '''
    N = len(R)
    num_bins = len(bin_bounds) - 1
    r_max = bin_bounds[num_bins]

    GT_all = np.zeros((n_threads, num_bins), dtype=np.float64)
    count_all = np.zeros((n_threads, num_bins), dtype=np.int64)
    chunk = n_samples // n_threads

    for t in prange(n_threads):
        gt = np.zeros(num_bins, dtype=np.float64)
        count = np.zeros(num_bins, dtype=np.int64)
        
        for _ in range(chunk):
            i = np.random.randint(0, N)
            j = np.random.randint(0, N)
            
            dx = R[i, 0] - R[j, 0]
            dy = R[i, 1] - R[j, 1]
            r = np.sqrt(dx * dx + dy * dy)
            if r >= r_max:
                continue

            b = bin_index(r, bin_bounds)
            if b < 0:
                continue
            
            gi = grain_of_atoms[i]
            gj = grain_of_atoms[j]

            phi_i = Gx_per_grain[gi] * R[i, 0] + Gy_per_grain[gi] * R[i, 1]
            phi_j = Gx_per_grain[gj] * R[j, 0] + Gy_per_grain[gj] * R[j, 1]

            gt[b] += np.cos(phi_i - phi_j)
            count[b] += 1

        GT_all[t] = gt
        count_all[t] = count

    GT = np.zeros(num_bins, dtype=np.float64)
    count = np.zeros(num_bins, dtype=np.int64)
    for t in range(n_threads):
        for b in range(num_bins):
            GT[b] += GT_all[t, b]
            count[b] += count_all[t, b]

    for b in range(num_bins):
        if count[b] > 0:
            GT[b] /= count[b]

    return GT

def compute_translational_correlation(atoms, grain_mask, grain_centers, grain_thetas, bin_bounds, L, n_samples = 200_000, a_CC = 1.42, n_threads = 4):
    '''
    Compute the translational correlation function GT(r) 
    Inputs:
        atoms : coordinates of the atoms
        grain_mask : array indicating the grain of each atom
        grain_centers : coordinates of the centers of the grains
        grain_thetas : rotation angles of the grains
        bin_bounds : array of bin boundaries
        L : size of the lattice
        n_samples : number of samples to use for the correlation function
        a_CC : carbon-carbon bond length
        n_threads : number of threads to use for parallelization
    Outputs:
        GT : translational correlation function
    '''
    a = a_CC * np.sqrt(3)
    b1 = (2 * np.pi / a) * np.array([1, -1/np.sqrt(3)])
    b2 = (2 * np.pi / a) * np.array([0, 2/np.sqrt(3)])
    G_base = 2 * b1 + b2

    N = len(atoms)
    R = np.empty((N, 2), dtype=np.float64)
    Gx_per_grain = np.empty(len(grain_centers), dtype=np.float64)
    Gy_per_grain = np.empty(len(grain_centers), dtype=np.float64)

    grain_ids = np.unique(grain_mask)
    grain_ids = grain_ids[grain_ids >= 0]

    for gid in grain_ids:
        idx = np.where(grain_mask == gid)[0]
        theta = grain_thetas[gid]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        Gx_per_grain[gid] = G_base[0] * cos_t - G_base[1] * sin_t
        Gy_per_grain[gid] = G_base[0] * sin_t + G_base[1] * cos_t

        ref = build_reference_sites(
            float(grain_centers[gid, 0]), float(grain_centers[gid, 1]),
            float(np.cos(grain_thetas[gid])), float(np.sin(grain_thetas[gid])),
            float(L), float(a_CC)
        )
        _, nn = cKDTree(ref).query(atoms[idx], k=1, workers=-1)
        R[idx] = ref[nn]

    GT = compute_GT_kernel(
        np.ascontiguousarray(R, dtype=np.float64),
        np.ascontiguousarray(grain_mask, dtype=np.int64),
        np.ascontiguousarray(Gx_per_grain, dtype=np.float64),
        np.ascontiguousarray(Gy_per_grain, dtype=np.float64),
        np.ascontiguousarray(bin_bounds, dtype=np.float64),
        int(n_samples), int(n_threads)
    )

    return GT