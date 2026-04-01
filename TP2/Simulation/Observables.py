import numpy as np
from numba import njit

@njit
def lower_bound(arr, x):
    '''
    Find the index of the first element in arr that is greater than or equal to x
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
    '''
    if r < bin_bounds[0] or r >= bin_bounds[bin_bounds.size - 1]:
        return -1
    k = lower_bound(bin_bounds, r)
    return k - 1

@njit
def compute_psi6(atoms, neighbors):
    '''
    Compute the local orientational order parameter
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
            G6[b] += psi6[i].real * psi6[j].real + psi6[i].imag * psi6[j].imag
            count[b] += 1

    for b in range(num_bins):
        if count[b] > 0:
            G6[b] /= count[b]
    return G6
