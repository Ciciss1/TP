import numpy as np
from numba import njit
from tqdm import tqdm

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
    H_int *= gamma
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
    dH_int *= gamma
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

def adapt_delta(delta_theta, acceptance_rate, target_rate=0.5, adaptation_factor=1.05, min_delta=1e-3, max_delta=PI/6):
    '''
    Adapt the delta_theta to achieve the target acceptance rate
    Inputs:
        delta_theta : current delta_theta
        acceptance_rate : current acceptance rate
        target_rate : desired acceptance rate
        adaptation_factor : factor by which to increase/decrease delta_theta
    '''
    if acceptance_rate < target_rate - 0.05:
        delta_theta = max(delta_theta / adaptation_factor, min_delta)
    elif acceptance_rate > target_rate + 0.05:
        delta_theta = min(delta_theta * adaptation_factor, max_delta)
    return delta_theta

def monte_carlo(theta, adj_i, adj_j, adj_length, beta, epsilon, rho, gamma, n_sweeps = 1000, convergence_threshold=1e-3):

    delta_theta = 0.1
    attempts = 0
    accepts = 0

    energy_history = []
    energy_history.append(energy(theta, adj_i, adj_j, adj_length, epsilon, rho, gamma))

    counter = 0

    for sweep in tqdm(range(n_sweeps), desc="Monte Carlo Sweeps"):
        a, acc = metropolis_sweep(theta, adj_i, adj_j, adj_length, beta, delta_theta, epsilon, rho, gamma)
        attempts += a
        accepts += acc

        if sweep % 50 == 0 and sweep > 0:
            acceptance_rate = accepts / attempts if attempts > 0 else 0
            delta_theta = adapt_delta(delta_theta, acceptance_rate)
            attempts = 0
            accepts = 0

            current_energy = energy(theta, adj_i, adj_j, adj_length, epsilon, rho, gamma)
            energy_history.append(current_energy)

            if len(energy_history) > 2:
                energy_change = np.abs(energy_history[-1] - energy_history[-2])
                if energy_change < convergence_threshold:
                    counter += 1
                    if counter >= 10:
                        print(f"Convergence achieved after {sweep} sweeps.")
                        break
                else:
                    counter = 0
    return theta, energy_history