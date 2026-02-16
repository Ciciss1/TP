import numpy as np
from numba import njit

PI = np.pi

@njit
def wrap_angle_pi_6(angle):
    '''
    Wrap angle between -pi/6 and pi/6.
    '''
    return (angle + PI / 6) % (PI / 3) - PI / 6

@njit
def energy(theta, adj_i, adj_j, adj_length, epsilon, rho, gamma):
    '''
    Compute the energy of the system.
    Inputs:
        theta : array of angles
        adj_i, adj_j : indices of adjacent points
        adj_length : length of the edge between adjacent points
        epsilon, rho, gamma : parameters
    Outputs:
        energy of the system
    '''
    H_0 = (epsilon / rho) * np.sum(theta**2)
    H_int = 0.0
    for k in range(len(adj_i)):
        i = adj_i[k]
        j = adj_j[k]
        length = adj_length[k]

        dtheta = np.abs(wrap_angle_pi_6(theta[i] - theta[j]))

        H_int += length * dtheta * (1 - np.log(max(dtheta, 1e-10)))
    H_int *= gamma / 3
    return H_0 + H_int

@njit
def dhamiltonian(i, theta, new_theta, adj_i, adj_j, adj_length, epsilon, rho, gamma):
    '''
    Compute the change in energy when i is changed
    Inputs:
        i : index of the changed point
        theta : array of angles
        new_theta : new angle at point i
        adj_i, adj_j : indices of adjacent points
        adj_length : length of the edge between adjacent points
        epsilon, rho, gamma : parameters
    Outputs:
        change in energy
    '''
    old_theta = theta[i]
    dH_0 = (epsilon / rho) * (new_theta**2 - old_theta**2)
    dH_int = 0.0
    for k in range(len(adj_i)):
        if adj_i[k] == i:
            j = adj_j[k]
            length = adj_length[k]
            dtheta_old = np.abs(wrap_angle_pi_6(old_theta - theta[j]))
            dtheta_new = np.abs(wrap_angle_pi_6(new_theta - theta[j]))
            dH_int += length * (dtheta_new * (1 - np.log(max(dtheta_new, 1e-10))) - dtheta_old * (1 - np.log(max(dtheta_old, 1e-10))))
        elif adj_j[k] == i:
            j = adj_i[k]
            length = adj_length[k]
            dtheta_old = np.abs(wrap_angle_pi_6(old_theta - theta[j]))
            dtheta_new = np.abs(wrap_angle_pi_6(new_theta - theta[j]))
            dH_int += length * (dtheta_new * (1 - np.log(max(dtheta_new, 1e-10))) - dtheta_old * (1 - np.log(max(dtheta_old, 1e-10))))
    dH_int *= gamma / 3
    return dH_0 + dH_int

@njit
def metropolis_sweep(theta, adj_i, adj_j, adj_length, beta, delta_theta, epsilon, rho, gamma):
    '''
    Perform a Metropolis sweep the lattice
    Inputs:
        theta : array of angles
        adj_i, adj_j : indices of adjacent points
        adj_length : length of the edge between adjacent points
        beta : inverse temperature
        delta_theta : maximum change in angle
        epsilon, rho, gamma : parameters
    Outputs:
        attempts : number of attempted moves
        accepts : number of accepted moves
    '''
    N = len(theta)
    attempts = 0
    accepts = 0

    for _ in range(N):
        i = np.random.randint(N)
        old_theta = theta[i]
        new_theta = wrap_angle_pi_6(old_theta + (np.random.rand() * 2 - 1) * delta_theta)
        dH = dhamiltonian(i, theta, new_theta, adj_i, adj_j, adj_length, epsilon, rho, gamma)
        attempts += 1
        if dH < 0.0 or np.random.rand() < np.exp(-beta * dH):
            theta[i] = new_theta
            accepts += 1
    return attempts, accepts

@njit
def overrelaxation_sweep(theta, adj_i, adj_j, adj_length, beta, epsilon, rho, gamma):
    '''
    Perform an overrelaxation sweep over the lattice
    Inputs:
        theta : array of angles
        adj_i, adj_j : indices of adjacent points
        adj_length : length of the edge between adjacent points
        epsilon, rho, gamma : parameters
    '''
    N = len(theta)
    for i in range(N):
        sum_sin = 0.0
        sum_cos = 0.0
        for k in range(len(adj_i)):
            if adj_i[k] == i:
                j = adj_j[k]
                sum_sin += np.sin(6 * theta[j]) 
                sum_cos += np.cos(6 * theta[j])
            elif adj_j[k] == i:
                j = adj_i[k]
                sum_sin += np.sin(6 * theta[j]) 
                sum_cos += np.cos(6 * theta[j])

        if sum_sin == 0.0 and sum_cos == 0.0:
            continue

        phi = np.arctan2(sum_sin, sum_cos)
        old_theta = theta[i]
        new_theta = wrap_angle_pi_6(2 * phi / 6 - old_theta)

        dH = dhamiltonian(i, theta, new_theta, adj_i, adj_j, adj_length, epsilon, rho, gamma)
        if dH < 0.0 or np.random.rand() < np.exp(-beta * dH):
            theta[i] = new_theta