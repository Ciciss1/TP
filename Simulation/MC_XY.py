import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from pathlib import Path
import os

@njit
def wrap_angle(x):
    '''
    Wrap angle between -pi and pi
    Inputs:
        x: angle
    Outputs:
        wrapped angle
    '''
    pi = np.pi
    return (x + pi) % (2 * pi) - pi

@njit
def dhamiltonien(theta, neigbors, idx, new_theta, J):
    '''
    Compute the change in Hamiltonian when one theta is changed
    Inputs:
        theta: 1D array of angles
        neighbors: list of neighbors for each index
        idx: index of the changed theta
        new_theta
        J: parameters
    Outputs:
        dH: Change in Hamiltonian
    '''

    old_theta = theta[idx]
    dH = 0.0
    z_max = neigbors.shape[1]

    for n in range(z_max):
        neighbor_idx = neigbors[idx, n]
        if neighbor_idx == -1:
            continue
        dtheta_old = wrap_angle(old_theta - theta[neighbor_idx])
        dtheta_new = wrap_angle(new_theta - theta[neighbor_idx])
        dH += -J * (np.cos(dtheta_new) - np.cos(dtheta_old))
    return dH

@njit
def energy(theta, neigbors, J):
    '''
    Compute the Hamiltonian of the system
    Inputs:
        theta: 1D array of angles
        neighbors: list of neighbors for each index
        J: parameters
    Outputs:
        H: Hamiltonian
    '''

    N = theta.size
    z_max = neigbors.shape[1]
    H = 0.0

    for idx in range(N):
        theta_i = theta[idx]
        for n in range(z_max):
            neighbor_idx = neigbors[idx, n]
            if neighbor_idx == -1:
                continue
            dtheta = wrap_angle(theta_i - theta[neighbor_idx])
            H += -J * np.cos(dtheta)
    return H * 0.5

@njit
def compute_psi_flat(theta_flat):
    '''
    Compute the complex order parameter psi in a flattened array
    Inputs:
        theta_flat: 1D array of angles
    Outputs:
        psi_flat: 1D array of complex order parameters
    '''

    N = theta_flat.size
    psi_flat = np.empty(N, dtype=np.complex128)

    for i in range(N):
        th = theta_flat[i]
        psi_flat[i] = np.cos(th) + 1j * np.sin(th)
    return psi_flat

@njit
def compute_magnetization(theta_flat):
    '''
    Compute the magnetization of the lattice
    Inputs:
        theta_flat: 1D array of angles
    Outputs:
        M: magnetization
    '''

    N = theta_flat.size
    sum_cos = 0.0
    sum_sin = 0.0

    for i in range(N):
        th = theta_flat[i]
        sum_cos += np.cos(th)
        sum_sin += np.sin(th)

    mx = sum_cos / N
    my = sum_sin / N
    M = np.sqrt(mx**2 + my**2)
    return M

@njit
def metropolis_sweep(theta, neigbors, beta, delta_theta, J):
    '''
    Perform a Metropolis sweep over the lattice
    Inputs:
        theta: 1D array of angles
        neighbors: list of neighbors for each index
        beta: inverse temperature
        delta_theta: maximum change in angle
        J: parameters
    Outputs:
        attempts: number of attempts
        accepts: number of accepted moves
    '''

    N = theta.size
    attempts = 0
    accepts = 0

    for _ in range(N):
        idx = np.random.randint(N)
        old_theta = theta[idx]
        new_theta = wrap_angle(old_theta + (np.random.rand() * 2 - 1) * delta_theta)
        dH = dhamiltonien(theta, neigbors, idx, new_theta, J)
        attempts += 1
        if dH <= 0.0 or np.random.rand() < np.exp(-beta * dH):
            theta[idx] = new_theta
            accepts += 1
    return attempts, accepts

@njit
def overrelaxation_sweep(theta, neigbors):
    '''
    Perform an overrelaxation sweep over the lattice
    Inputs:
        theta: 1D array of angles
        neighbors: list of neighbors for each index
    '''

    N = theta.size
    z_max = neigbors.shape[1]

    for idx in range(N):
        sum_sin = 0.0
        sum_cos = 0.0
        for n in range(z_max):
            neighbor_idx = neigbors[idx, n]
            if neighbor_idx == -1:
                continue
            sum_sin += np.sin(theta[neighbor_idx])
            sum_cos += np.cos(theta[neighbor_idx])
        
        if sum_sin == 0.0 and sum_cos == 0.0:
            continue

        phi = np.arctan2(sum_sin, sum_cos)
        old_theta = theta[idx]
        new_theta = wrap_angle(2 * phi - old_theta)
        theta[idx] = new_theta

@njit
def orientationnal_correlation(psi_flat, bin_indices):
    '''
    Compute the correlation function of the lattice
    Inputs:
        psi_flat: 1D array of complex order parameters
        bin_indices: bin indices for distance matrix
    Outputs:
        G_avg : correlation function
    '''

    N = psi_flat.size
    n_bins = bin_indices.max() + 1

    G_avg = np.zeros(n_bins, dtype=np.complex128)
    counts = np.zeros(n_bins, dtype=np.int64)

    for i in range(N):
        psi_i = psi_flat[i]
        for j in range(i, N):
            bin_idx = bin_indices[i, j]
            G_avg[bin_idx] += psi_i * np.conj(psi_flat[j])
            counts[bin_idx] += 1

    for b in range(n_bins):
        if counts[b] > 0:
            G_avg[b] /= counts[b]
        
    return G_avg.real

class Lattice:
    '''
    Lattice class to hold lattice properties
    Attributes:
        L: size of the lattice
        rho: density
        N: number of points
        indices: list of lattice indices
        index_map: mapping from (i, j) to index
        distance: distance between points
        d_x: x distance between points
        d_y: y distance between points
        xs: x coordinates
        ys: y coordinates
        coords: coordinates array
        neighbors: list of neighbors for each index
        r: distance matrix
        dr: distance bin width
        r_max: maximum distance
        n_bins: number of distance bins
        bin_edges: edges of distance bins
        bin_indices: bin indices for distance matrix
        theta: angle array
    '''

    def __init__(self, L, rho = 1.0):
        self.L = L
        self.rho = rho
        self.N = L * L

        self.indices = np.array([(i, j) for i in range(L) for j in range(L)], dtype=np.int32)
        self.index_map = np.zeros((L, L), dtype=np.int32)

        for idx, (i, j) in enumerate(self.indices):
            self.index_map[i, j] = idx

        area = self.N / self.rho
        self.distance = np.sqrt(area) / L
        self.d_x = self.distance
        self.d_y = self.distance * np.sqrt(3) / 2

        xs = []
        ys = []
        for (i, j) in self.indices:
            x = j * self.d_x + (i % 2) * (self.d_x / 2)
            y = i * self.d_y
            xs.append(x)
            ys.append(y)

        self.xs = np.array(xs)
        self.ys = np.array(ys)
        self.coords = np.vstack((self.xs, self.ys)).T

        z_max = 6
        neighbors = np.full((self.N, z_max), -1, dtype=np.int32)

        for idx, (i, j) in enumerate(self.indices):
            if i % 2 == 0:
                candidats = [
                    (i, j-1), (i, j+1),
                    (i-1, j), (i+1, j),
                    (i-1, j-1), (i+1, j-1)
                ]
            else:
                candidats = [
                    (i, j-1), (i, j+1),
                    (i-1, j), (i+1, j),
                    (i-1, j+1), (i+1, j+1)
                ]

            n_idx = 0
            for (ni, nj) in candidats:
                if 0 <= ni < L and 0 <= nj < L:
                    neighbors[idx, n_idx] = self.index_map[ni, nj]
                    n_idx += 1

        self.neighbors = neighbors

        dx = self.coords[:, 0][:, None] - self.coords[:, 0][None, :]
        dy = self.coords[:, 1][:, None] - self.coords[:, 1][None, :]
        self.r = np.sqrt(dx**2 + dy**2)

        self.dr = self.distance / 2
        self.r_max = self.distance * L
        self.n_bins = int(self.r_max / self.dr) + 1

        p = 2.0
        xs = np.linspace(0.0, 1.0, self.n_bins, endpoint=False)
        self.bin_edges = self.r_max * (1 - (1 - xs)**p)

        self.bin_indices = np.digitize(self.r, self.bin_edges) - 1

        self.theta = np.random.uniform(-np.pi, np.pi, size=(L, L))

    def compute_psi_flat(self):
        '''
        Compute the complex order parameter psi in a flattened array
        Outputs:
            psi_flat: 1D array of complex order parameters
        '''

        return np.exp(1j * self.theta.ravel())
    
    def compute_magnetization(self):
        '''
        Compute the magnetization of the lattice
        Outputs:
            M: magnetization
        '''

        theta_flat = self.theta.ravel()
        mx = np.mean(np.cos(theta_flat))
        my = np.mean(np.sin(theta_flat))
        M = np.sqrt(mx**2 + my**2)
        return M
class Simulation:
    '''
    Simulation class to run Monte Carlo simulations
    Attributes:
        lattice: Lattice object
        L: size of the lattice
        T: temperature
        J: interaction parameter
        beta: inverse temperature
        rho: density
        n_therm: number of thermalization steps
        n_meas: number of measurement steps
        overrelax_interval: interval for overrelaxation sweeps
        meas_interval: interval for measurements
        tune_interval: interval for delta_theta tuning
        delta_theta: maximum change in angle
        target_acceptance: target acceptance rate
        growth_factor: factor to increase delta_theta
        shrink_factor: factor to decrease delta_theta
        min_delta: minimum delta_theta
        max_delta: maximum delta_theta
        total_attempts: total number of Metropolis attempts
        total_accepts: total number of accepted moves
        energy: list of energy values during thermalization
        G: orientation correlation function
        G_err: error in orientation correlation function
        chi: magnetic susceptibility
        chi_err: error in magnetic susceptibility
    '''

    def __init__(
            self,
            L,
            T,
            J,
            rho = 1.0,
            n_therm = 10_000,
            n_meas = 10_000,
            overrelax_interval = 0,
            meas_interval = 10,
            tune_interval = 100,
            delta_init = 1.0,
            target_acceptance = 0.5,
            growth_factor = 1.05,
            shrink_factor = 0.95,
            min_delta = 0.001,
            max_delta = np.pi,
    ):
        self.lattice = Lattice(L, rho)
        self.L = L
        self.T = T
        self.J = J
        self.beta = 1.0 / T
        self.rho = rho
        
        self.n_therm = n_therm
        self.n_meas = n_meas
        self.overrelax_interval = overrelax_interval
        self.meas_interval = meas_interval

        self.tune_interval = tune_interval
        self.delta_theta = delta_init
        self.target_acceptance = target_acceptance
        self.growth_factor = growth_factor
        self.shrink_factor = shrink_factor
        self.min_delta = min_delta
        self.max_delta = max_delta
        self.total_attempts = 0
        self.total_accepts = 0

        self.energy = []
        self.G = None
        self.G_err = None
        self.chi = None
        self.chi_err = None

    def adapt_delta(self):
        '''
        Adapt the delta_theta based on acceptance rate
        '''

        if self.total_attempts == 0:
            return
        
        acceptance_rate = self.total_accepts / self.total_attempts
        if acceptance_rate < self.target_acceptance - 0.05:
            self.delta_theta = max(self.delta_theta * self.shrink_factor, self.min_delta)
        elif acceptance_rate > self.target_acceptance + 0.05:
            self.delta_theta = min(self.delta_theta * self.growth_factor, self.max_delta)

        self.total_attempts = 0
        self.total_accepts = 0

    def run(self, path):
        '''
        Run the Monte Carlo simulation
        Inputs:
            path: path to save results
        '''

        os.makedirs(path, exist_ok=True)

        theta_flat = self.lattice.theta.ravel()
        neighbors = self.lattice.neighbors
        bin_indices = self.lattice.bin_indices

        ## Thermalization
        for step in range(1, self.n_therm + 1):
            if self.overrelax_interval > 0 and step % self.overrelax_interval == 0:
                overrelaxation_sweep(theta_flat, neighbors)

            attempts, accepts = metropolis_sweep(
                theta_flat, neighbors, self.beta, self.delta_theta, self.J
            )

            self.total_attempts += attempts
            self.total_accepts += accepts

            if self.tune_interval > 0 and step % self.tune_interval == 0:
                self.adapt_delta()

            if step % (self.n_therm // 10) == 0:
                E = energy(theta_flat, neighbors, self.J)/self.lattice.N
                self.energy.append(E)

        ## Measurement
        Gs = []
        mags = []

        for step in range(1, self.n_meas + 1):
            metropolis_sweep(
                theta_flat, neighbors, self.beta, self.delta_theta, self.J
            )

            if step % 10 == 0:
                m = compute_magnetization(theta_flat)
                mags.append(m)

            if self.meas_interval > 0 and step % self.meas_interval == 0:
                psi_flat = compute_psi_flat(theta_flat)
                G = orientationnal_correlation(psi_flat, bin_indices)
                Gs.append(G)

        Gs = np.array(Gs)
        self.G = np.mean(Gs, axis=0)
        self.G_err = np.std(Gs, axis=0) / np.sqrt(Gs.shape[0])

        mags = np.array(mags)
        self.chi = self.lattice.N * (np.mean(mags**2) - np.mean(mags)**2)
        self.chi_err = np.std(mags**2) / np.sqrt(mags.size) * self.lattice.N

        self.lattice.theta = theta_flat.reshape((self.L, self.L))

        self.save_results(path)

    def save_results(self, path):
        '''
        Save the simulation results to files
        Inputs:
            path: path to save results
        '''

        theta_flat = self.lattice.theta.ravel()
        xs = self.lattice.xs
        ys = self.lattice.ys

        header = "x,y,theta"
        data = np.column_stack((xs, ys, theta_flat))
        np.savetxt(f"{path}/final_configuration_T{self.T:.3e}_L{self.L}.csv", data, delimiter=",", header=header, comments='')

        header = "index,energy_per_site"
        data = np.column_stack((np.arange(len(self.energy)), self.energy))
        np.savetxt(f"{path}/energy_T{self.T:.3e}_L{self.L}.csv", data, delimiter=",", header=header, comments='')

        header = "r,G(r),G_err(r)"
        data = np.column_stack((
            self.lattice.bin_edges,
            self.G,
            self.G_err
        ))
        np.savetxt(f"{path}/correlation_T{self.T:.3e}_L{self.L}.csv", data, delimiter=",", header=header, comments='')

        chi_file = Path(path).parent / f"chi_L{self.L}.csv"
        if not chi_file.exists():
            with open(chi_file, 'w') as f:
                f.write("T,chi,chi_err\n")
                f.write(f"{self.T:.8e},{self.chi:.8e},{self.chi_err:.8e}\n")
            return
        
        data = np.loadtxt(chi_file, delimiter=",", skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        T_vals = data[:, 0]

        mask = np.isclose(T_vals, self.T, atol=1e-8)
        if np.any(mask):
            data[mask, 1] = self.chi
            data[mask, 2] = self.chi_err
        else:
            new_row = np.array([[self.T, self.chi, self.chi_err]])
            data = np.vstack((data, new_row))

        sorted_indices = np.argsort(data[:, 0])
        data = data[sorted_indices]

        with open(chi_file, 'w') as f:
            f.write("T,chi,chi_err\n")
            for row in data:
                f.write(f"{row[0]:.8e},{row[1]:.8e},{row[2]:.8e}\n")

def plot_results(path, L, T, J, size=20):
    theta_data = np.loadtxt(
        f"{path}/final_configuration_T{T:.3e}_L{L}.csv", delimiter=",", skiprows=1
    )
    x_positions = theta_data[:, 0]
    y_positions = theta_data[:, 1]
    theta = theta_data[:, 2]

    energy_data = np.loadtxt(
        f"{path}/energy_T{T:.3e}_L{L}.csv", delimiter=",", skiprows=1
    )
    steps = energy_data[:, 0]
    energies = energy_data[:, 1]

    corr_data = np.loadtxt(
        f"{path}/correlation_T{T:.3e}_L{L}.csv", delimiter=",", skiprows=1
    )
    r = corr_data[:, 0]
    G = corr_data[:, 1]
    G_err = corr_data[:, 2]

    mask = np.abs(G) > 0
    r = r[mask]
    G = G[mask]
    G_err = G_err[mask]

    plt.figure(figsize=(8, 6))
    plt.scatter(
        x_positions,
        y_positions,
        c=theta,
        cmap="hsv",
        s=size,
        marker="h",
        vmin=-np.pi,
        vmax=np.pi,
    )
    plt.colorbar(label=r"$\theta$ (rad)")
    plt.title(rf"Hexagonal Lattice L={L}, $T={T:.2e}$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.tight_layout()
    plt.savefig(f"{path}/final_configuration_T{T:.3e}_L{L}.pdf")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(steps, energies, "-x")
    plt.title(rf"Energy per Site at $T={T:.3e}$, $L={L}$")
    plt.xlabel("index")
    plt.ylabel(r"$E/N$")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{path}/energy_T{T:.3e}_L{L}.pdf")
    plt.close()

    # Fit power-law + expo
    eta_th = T / (2 * np.pi * J)
    G_th = r[1:] ** (-eta_th)

    r_min = r[1]
    r_max = r[-1] / 2.0
    mask_fit = (r >= r_min) & (r <= r_max) & (G > 0)
    coef_power_decay, cov_power_decay = np.polyfit(
        np.log(r[mask_fit]), np.log(G[mask_fit]), 1, cov=True
    )
    a_power, b_power = coef_power_decay
    eta = -a_power
    eta_err = np.sqrt(cov_power_decay[0, 0])
    A_power = np.exp(b_power)
    G_power_fit = A_power * r[1:] ** (-eta)
    G_power_fit = np.insert(G_power_fit, 0, G[0])

    coef_expo_decay, cov_expo_decay = np.polyfit(
        r[mask_fit], np.log(G[mask_fit]), 1, cov=True
    )
    a_expo, b_expo = coef_expo_decay
    A = np.exp(b_expo)
    xi = -1.0 / a_expo
    xi_err = xi**2 * np.sqrt(cov_expo_decay[0, 0])
    G_expo_fit = A * np.exp(-r / xi)

    plt.figure(figsize=(8, 6))
    plt.errorbar(r, G, yerr=G_err, fmt="-x", label="Simulation Data")
    plt.plot(r, G_power_fit, "g--", label=rf"Power-law Fit, $\eta={eta:.3f}\pm{eta_err:.3f}$")
    plt.plot(r, G_expo_fit, "m--", label=rf"Exponential Fit, $\xi={xi:.3f}\pm{xi_err:.3f}$")
    plt.title(rf"Correlation Function at $T={T:.2e}$, $L={L}$")
    plt.xlabel(r"$r$")
    plt.ylabel(r"$G(r)$")
    plt.ylim(-0.1, 1.02)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{path}/correlation_T{T:.3e}_L{L}.pdf")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.errorbar(r, G, yerr=G_err, fmt="-x", label="Simulation Data")
    plt.xscale("log")
    plt.yscale("log")
    plt.title(rf"Correlation Function at $T={T:.2e}$, $L={L}$ (Log-Log Scale)")
    plt.xlabel(r"$r$")
    plt.ylabel(r"$G(r)$")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{path}/correlation_T{T:.3e}_L{L}_loglog.pdf")
    plt.close()


def plot_chi_vs_T(base_path, L):
    chi_data = np.loadtxt(f"{base_path}/chi_L{L}.csv", delimiter=",", skiprows=1)
    T_values = chi_data[:, 0]
    chi = chi_data[:, 1]
    chi_err = chi_data[:, 2]

    plt.figure(figsize=(8, 6))
    plt.errorbar(T_values, chi, yerr=chi_err, fmt="-x", label="Simulation Data")
    plt.title(rf"Susceptibility $\chi$ vs Temperature for $L={L}$")
    plt.xlabel(r"$T$")
    plt.ylabel(r"$\chi$")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{base_path}/chi_L{L}.pdf")
    plt.close()