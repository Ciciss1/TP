import numpy as np
from numba import njit, typed, types

@njit
def lower_bound(arr, x):
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
    if r < bin_bounds[0] or r >= bin_bounds[bin_bounds.size - 1]:
        return -1
    k = lower_bound(bin_bounds, r)
    return k - 1
    
@njit
def compute_psi6(atoms, neighbors):
    N = len(atoms)
    psi6 = np.zeros(N, dtype=np.complex128)

    for i in range(N):
        sum_psi = 0.0 + 0.0j
        for j in neighbors[i]:
            dx = atoms[j, 0] - atoms[i, 0]
            dy = atoms[j, 1] - atoms[i, 1]
            theta = np.arctan2(dy, dx)
            sum_psi += np.exp(6j * theta)
        psi6[i] = sum_psi / len(neighbors[i])
    return psi6            

@njit
def compute_orientational_correlation(psi6, coords, bin_bounds):
    N = len(psi6)
    num_bins = len(bin_bounds) - 1
    G6 = np.zeros(num_bins, dtype=np.float64)
    count = np.zeros(num_bins, dtype=np.int64)

    for i in range(N):
        xi = coords[i, 0]
        yi = coords[i, 1]
        psi6_i_re = psi6[i].real
        psi6_i_im = psi6[i].imag

        for j in range(i, N):
            dx = xi - coords[j, 0]
            dy = yi - coords[j, 1]
            r = np.sqrt(dx * dx + dy * dy)
            b = bin_index(r, bin_bounds)
            G6[b] += psi6_i_re * psi6[j].real + psi6_i_im * psi6[j].imag
            count[b] += 1

    for b in range(num_bins):
        if count[b] > 0:
            G6[b] /= count[b]
    return G6