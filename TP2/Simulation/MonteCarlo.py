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
def energy(theta, adj_i, adj_j, adj_length, areas, epsilon, gamma, phi_s2, phi_s4, alpha, beta_RS):
    '''
    Compute the energy of the system.
    Inputs:
        theta : array of angles
        adj_i, adj_j : indices of adjacent points
        adj_length : length of the edge between adjacent points
        areas : array of grain areas
        epsilon, gamma, phi_s2, phi_s4, alpha, beta_RS : parameters
    Outputs:
        energy of the system
    '''
    theta2 = theta * theta
    theta4 = theta2 * theta2
    U_s = gamma * theta2 / (phi_s2 + theta2) + (1 - gamma) * theta4 / (phi_s4 + theta4)

    H_0 = epsilon * np.sum(areas * U_s)
    H_int = 0.0
    for k in range(len(adj_i)):
        i = adj_i[k]
        j = adj_j[k]
        length = adj_length[k]

        dtheta = np.abs(wrap_angle_pi_6(theta[i] - theta[j]))

        H_int += length * dtheta * (beta_RS - np.log(max(dtheta, 1e-10)))
    H_int *= alpha
    return H_0 + H_int

@njit
def dhamiltonian(i, theta, new_theta, adj_i, adj_j, adj_length, areas, epsilon, gamma, phi_s2, phi_s4, alpha, beta_RS):
    '''
    Compute the change in energy when i is changed
    Inputs:
        i : index of the changed point
        theta : array of angles
        new_theta : new angle at point i
        adj_i, adj_j : indices of adjacent points
        adj_length : length of the edge between adjacent points
        areas : array of grain areas
        epsilon, gamma, phi_s2, phi_s4, alpha, beta_RS : parameters
    Outputs:
        change in energy
    '''
    old_theta = theta[i]
    old_theta2 = old_theta * old_theta
    old_theta4 = old_theta2 * old_theta2

    U_s_old = gamma * old_theta2 / (phi_s2 + old_theta2) + (1 - gamma) * old_theta4 / (phi_s4 + old_theta4)
    U_s_new = gamma * new_theta**2 / (phi_s2 + new_theta**2) + (1 - gamma) * new_theta**4 / (phi_s4 + new_theta**4)

    dH_0 = epsilon * areas[i] * (U_s_new - U_s_old)
    dH_int = 0.0
    for k in range(len(adj_i)):
        if adj_i[k] == i:
            j = adj_j[k]
            length = adj_length[k]
        elif adj_j[k] == i:
            j = adj_i[k]
            length = adj_length[k]
        else:
            continue
        dtheta_old = np.abs(wrap_angle_pi_6(old_theta - theta[j]))
        dtheta_new = np.abs(wrap_angle_pi_6(new_theta - theta[j]))
        dH_int += length * (dtheta_new * (beta_RS - np.log(max(dtheta_new, 1e-10))) - dtheta_old * (beta_RS - np.log(max(dtheta_old, 1e-10))))
    dH_int *= alpha
    return dH_0 + dH_int

@njit
def metropolis_sweep(theta, adj_i, adj_j, adj_length, areas, beta, delta_theta, epsilon, gamma, phi_s2, phi_s4, alpha, beta_RS):
    '''
    Perform a Metropolis sweep the lattice
    Inputs:
        theta : array of angles
        adj_i, adj_j : indices of adjacent points
        adj_length : length of the edge between adjacent points
        areas : array of grain areas
        beta : inverse temperature
        delta_theta : maximum change in angle
        epsilon, gamma, phi_s2, phi_s4, alpha, beta_RS : parameters
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
        dH = dhamiltonian(i, theta, new_theta, adj_i, adj_j, adj_length, areas, epsilon, gamma, phi_s2, phi_s4, alpha, beta_RS)
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

def estimate_tau_int(obs, c=5.0, max_lag=None, max_iter=20):
    """
    Estimate the integrated autocorrelation time of a time series using the FFT method.
    Inputs:
        obs : time series data
        c : constant to determine the window size
        max_lag : maximum lag to consider for autocorrelation
        max_iter : maximum number of iterations
    Outputs:
        tau_int : estimated integrated autocorrelation time
    """
    x = np.asarray(obs, dtype=np.float64)
    n = len(x)

    if n < 2:
        return np.nan
    
    x = x - x.mean()
    var = np.dot(x, x)

    if var == 0:
        return 0.5
    
    if max_lag is None:
        max_lag = n // 2

    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conjugate(f))[:n]
    acf /= var
    tau, window = 0.5, 1
    for _ in range(max_iter):
        tau = 0.5 + np.sum(acf[1:window + 1])
        new_window = min(max_lag, int(c * tau))
        if new_window <= window:
            break
        window = new_window
    return max(tau, 0.5)

@njit
def mean_misorientation(theta, adj_i, adj_j, adj_length):
    total_l = 0.0
    total_dl = 0.0
    for k in range(len(adj_i)):
        dtheta = np.abs(wrap_angle_pi_6(theta[adj_i[k]] - theta[adj_j[k]]))
        total_dl += adj_length[k] * dtheta
        total_l += adj_length[k]
    return total_dl / total_l

def monte_carlo(theta, adj_i, adj_j, adj_length, areas, beta, epsilon, gamma, phi_s2, phi_s4, alpha, beta_RS, n_sweeps = 100_000, log_every=1000, thermodynamic_window=1000, min_eff_samples=100, use_tqdm=True):

    delta_theta = 0.1
    attempts = 0
    accepts = 0

    energy_history = []
    misor_history = []

    equilibrium_counter = 0
    mean_E_priv = None
    mean_misor_priv = None
    std_E_priv = 1.0
    std_misor_priv = 1.0
    window = thermodynamic_window * 2

    iterator = tqdm(range(n_sweeps), desc="Monte Carlo Sweeps") if use_tqdm else range(n_sweeps)
    for sweep in iterator:
        a, acc = metropolis_sweep(theta, adj_i, adj_j, adj_length, areas, beta, delta_theta, epsilon, gamma, phi_s2, phi_s4, alpha, beta_RS)
        attempts += a
        accepts += acc

        E = energy(theta, adj_i, adj_j, adj_length, areas, epsilon, gamma, phi_s2, phi_s4, alpha, beta_RS)
        misorientation = mean_misorientation(theta, adj_i, adj_j, adj_length)

        energy_history.append(E)
        misor_history.append(misorientation)

        if sweep % log_every == 0 and sweep > 0:
            acceptance_rate = accepts / attempts if attempts > 0 else 0
            delta_theta = adapt_delta(delta_theta, acceptance_rate)
            attempts = 0
            accepts = 0

            # tau_E = estimate_tau_int(energy_history[-window:])
            # tau_misor = estimate_tau_int(misor_history[-window:])

            # window = int(max(thermodynamic_window*2, 50*max(tau_E, tau_misor)) * 0.3 + window * 0.7)

            window = thermodynamic_window * 2
            tau_E = estimate_tau_int(energy_history[-window:])
            tau_misor = estimate_tau_int(misor_history[-window:])

            # eff_E = window / max(1e-12, 2 * tau_E)
            # eff_misor = window / max(1e-12, 2 * tau_misor)
            # print(f"sweep={sweep:>7}  tau_E={tau_E:7.1f}  tau_misor={tau_misor:7.1f}  window={window:>6}  eff_E={eff_E:6.1f}  eff_misor={eff_misor:6.1f}  eq_counter={equilibrium_counter}")

            mean_E = np.mean(energy_history[-thermodynamic_window:])
            mean_misor = np.mean(misor_history[-thermodynamic_window:])
            std_E = np.std(energy_history[-thermodynamic_window:]) * np.sqrt(2 * tau_E / thermodynamic_window)
            std_misor = np.std(misor_history[-thermodynamic_window:]) * np.sqrt(2 * tau_misor / thermodynamic_window)

            if mean_E_priv is not None and mean_misor_priv is not None:
                stable = (np.abs(mean_E - mean_E_priv) < 2 * np.sqrt(std_E**2 + std_E_priv**2)) and (np.abs(mean_misor - mean_misor_priv) < 2 * np.sqrt(std_misor**2 + std_misor_priv**2))
                equilibrium_counter = equilibrium_counter + 1 if stable else max(0, equilibrium_counter - 2)

            if equilibrium_counter >= 10:
                eff_E = window / max(2 * tau_E, 1e-10)
                eff_misor = window / max(2 * tau_misor, 1e-10)
                if eff_E >= min_eff_samples and eff_misor >= min_eff_samples:
                    if use_tqdm:
                        print(f"Equilibrium reached at sweep {sweep}.")
                    break

            mean_E_priv = mean_E
            mean_misor_priv = mean_misor
            std_E_priv = std_E
            std_misor_priv = std_misor

    return theta, energy_history, misor_history