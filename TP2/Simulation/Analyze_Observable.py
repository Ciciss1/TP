import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid
from pathlib import Path
from numba import njit, prange

from Graphene import GrapheneCrystal, load_crystal
import Observables as obs

def power_law(x, a, b):
    return a * x**(-b)

def exponential(x, a, b):
    return a * np.exp(-b * x)

def fractional(x, a):
    return a / x

def wrap_angle(x):
    return (x + np.pi / 6) % (np.pi / 3) - np.pi / 6

def system_energy(crystal, epsilon, gamma, phi_s2, phi_s4, alpha, beta_RS):
    vor = crystal.lattice
    theta = vor.theta
    adj_i, adj_j, adj_l, areas = vor.adj_i, vor.adj_j, vor.adj_length, vor.areas

    theta2 = theta * theta
    theta4 = theta2 * theta2

    U_s = gamma * theta2 / (phi_s2 + theta2) + (1 - gamma) * theta4 / (phi_s4 + theta4)
    H_0 = epsilon * np.sum(areas * U_s)

    H_int = 0.0
    for k in range(len(adj_i)):
        i = adj_i[k]
        j = adj_j[k]
        length = adj_l[k]

        dtheta = np.abs(wrap_angle(theta[i] - theta[j]))

        H_int += length * dtheta * (beta_RS - np.log(max(dtheta, 1e-10)))
    H_int *= alpha
    return H_0 + H_int

def shannon_entropy(angles, n_bins = 360, range = (-np.pi, np.pi)):
    hist, bin_edges = np.histogram(angles, bins=n_bins, range=range)
    probs = hist / np.sum(hist)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))

def analyze_observable(T_K, L, epsilon, rho, alpha, beta_RS, gamma, phi_s, a_CC = 1.42, n_bins = 100, fig_size = 6, dot_size = 1, lw = 0.5, results_dir = "results/"):

    phi_s2 = phi_s * phi_s
    phi_s4 = phi_s2 * phi_s2

    rho_path = Path(results_dir) / f"eps_{epsilon:.4f}" / f"L_{L}" / f"rho_{rho}"

    if not rho_path.exists():
        raise FileNotFoundError(f"Results directory {rho_path} does not exist.")
    
    T_dirs = sorted(rho_path.glob("T_*"), key=lambda x: float(x.name.split("_")[1]))
    if not T_dirs:
        raise FileNotFoundError(f"No T directories found in {rho_path}.")
    
    T_values = []
    G6_mean_list, Var_G6_mean_list = [], []
    S_Shannon_list = []

    r_min = a_CC
    r_max = L / 2.0

    bin_bounds = np.geomspace(r_min, r_max, n_bins + 1)
    bin_centers = np.sqrt(bin_bounds[:-1] * bin_bounds[1:])

    for T_dir in T_dirs:
        T = float(T_dir.name.split("_")[1])
        T_values.append(T)

        crystal_paths = sorted(T_dir.glob("Crystal_*.npz"),
                               key=lambda p: int(p.stem.split("_")[1]))
        if not crystal_paths:
            legacy = T_dir / "Crystal.npz"
            if legacy.exists():
                raise FileNotFoundError(f"No crystal files found in {T_dir}.")
            crystal_paths = [legacy]
        
        n_runs = len(crystal_paths)
        G6_runs, Var_G6_runs = [], []
        S_Shannon_runs = []

        for k, cpath in enumerate(crystal_paths):
            print(f"T={T:.3f} eV, run {k+1}/{n_runs}")
            crystal = load_crystal(cpath)

            G6, Var_G6 = obs.compute_orientational_correlation(crystal.atoms, crystal.neighbors, bin_bounds, crystal.L, n_ref=5000)
            G6_runs.append(G6)
            Var_G6_runs.append(Var_G6)

            angles = []
            psi_6_values = []
            for i in range(len(crystal.atoms)):
                psi_6_i = obs.compute_psi6(i, crystal.atoms, crystal.neighbors, crystal.L)
                psi_6_values.append(psi_6_i)
                angle = np.angle(psi_6_i)
                angles.append(angle)

            angles = np.array(angles)
            psi_6_values = np.array(psi_6_values)

            S_Shannon_runs.append(shannon_entropy(angles, n_bins=360, range=(-np.pi, np.pi)))

            if k == 0:
                # plot lattice
                crystal.plot_lattice()
                plt.savefig(T_dir / f"Lattice_rho_{rho}_T_{T}.pdf")
                plt.close()
                # plot crystal with angles as color
                crystal.plot_all(fig_size=fig_size, dot_size=dot_size, lw=lw, atom_phase=angles)
                plt.savefig(T_dir / f"Crystal_rho_{rho}_T_{T}.jpg", dpi=150, pil_kwargs={"quality": 85, "optimize": True})
                plt.close()
                # plot angle distribution
                plt.hist(angles, bins=360)
                plt.xlabel(r"$\theta$ (radians)"); plt.ylabel("Count")
                plt.tight_layout()
                plt.savefig(T_dir / "Angle_Distribution.pdf")
                plt.close()

        G6_runs = np.array(G6_runs)
        Var_G6_runs = np.array(Var_G6_runs)
        S_Shannon_runs = np.array(S_Shannon_runs)

        G6_mean_list.append(G6_runs.mean(axis=0))
        Var_G6_mean_list.append(Var_G6_runs.mean(axis=0))
        S_Shannon_list.append(S_Shannon_runs.mean())

    T_values = np.array(T_values)
    sort_idx = np.argsort(T_values)
    T_values = T_values[sort_idx]
    T_Kelvin = np.array(T_K)
    sort_K_idx = np.argsort(T_Kelvin)
    T_Kelvin = T_Kelvin[sort_K_idx]

    G6_mean_list = np.array(G6_mean_list)[sort_idx]
    Var_G6_mean_list = np.array(Var_G6_mean_list)[sort_idx]
    S_Shannon_list = np.array(S_Shannon_list)[sort_idx]

    rho_path.mkdir(parents=True, exist_ok=True)

    # plot G6 vs r/a_CC
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for i, T in enumerate(T_Kelvin):
        color = cmap(i / max(1, len(T_Kelvin) - 1))
        ax.loglog(bin_centers / a_CC, G6_mean_list[i], color=color, label=f"T={T:.3f} K")
    ax.loglog(bin_centers / a_CC, (bin_centers / a_CC)**(-1/4), color='black', linestyle='--', label=r"$r^{-1/4}$")
    ax.set_xlabel(r"$r / a_{\mathrm{CC}}$")
    ax.set_ylabel(r"$G_6(r)$")
    ax.set_title(f"Orientational Correlation Function $G_6(r)$ for $\\rho={rho:.4f}$")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(rho_path / f"G6_rho_{rho}.pdf")
    plt.close()

    # plot Var(G6) vs r/a_CC
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for i, T in enumerate(T_Kelvin):
        color = cmap(i / max(1, len(T_Kelvin) - 1))
        ax.loglog(bin_centers / a_CC, Var_G6_mean_list[i], color=color, label=f"T={T:.3f} K")
    ax.set_xlabel(r"$r / a_{\mathrm{CC}}$")
    ax.set_ylabel(r"$\mathrm{Var}(G_6(r))$")
    ax.set_title(f"Variance of Orientational Correlation Function $G_6(r)$ for $\\rho={rho:.4f}$")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(rho_path / f"Var_G6_rho_{rho}.pdf")
    plt.close()

    #plot Var(G6) vs T for fixes r/a_CC
    ratios = [1, 10, 50, 100]
    fig, ax = plt.subplots(figsize=(8, 6))
    for ratio in ratios:
        idx = np.argmin(np.abs(bin_centers / a_CC - ratio))
        ax.plot(T_Kelvin, Var_G6_mean_list[:, idx], 'x-', label=f"r/a_CC={ratio}")
    ax.set_xlabel(r"$T$ (K)")
    ax.set_ylabel(r"$\mathrm{Var}(G_6(r))$")
    ax.set_title(f"Variance of Orientational Correlation Function $G_6(r)$ for $\\rho={rho:.4f}$")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(rho_path / f"Var_G6_vs_T_rho_{rho}.pdf")
    plt.close()

    # plot Shannon entropy vs T
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(T_Kelvin, S_Shannon_list, 'x-')
    ax.set_xlabel(r"$T$ (K)")
    ax.set_ylabel(r"$S$ (a.u.)")
    ax.set_title(f"Shannon Entropy for $\\rho={rho:.4f}$")
    plt.tight_layout()
    plt.savefig(rho_path / f"S_Shannon_rho_{rho}.pdf")
    plt.close()

def find_transition(T_arr, phases, from_phase, to_phase):
    last_from, first_to = None, None
    for t, ph in zip(T_arr, phases):
        if ph == from_phase:
            last_from = t
        if ph == to_phase and last_from is not None and first_to is None:
            first_to = t
    if last_from is not None and first_to is not None:
        return last_from
    return None

def extract_transitions(phase_data: dict):
    rhos = np.array(sorted(phase_data.keys()))
    T_sh_arr, T_hl_arr = [], []

    for rho in rhos:
        T_arr = np.array(phase_data[rho]["T"])
        phases = phase_data[rho]["phases"]
        T_sh = find_transition(T_arr, phases, from_phase="s", to_phase="h")
        T_hl = find_transition(T_arr, phases, from_phase="h", to_phase="l")
        T_sl = find_transition(T_arr, phases, from_phase="s", to_phase="l")
        T_sh_arr.append(T_sh)
        T_hl_arr.append(T_hl if T_hl is not None else T_sl)

    return rhos, np.array(T_sh_arr), np.array(T_hl_arr)


def plot_phase_diagram(phase_data: dict, epsilon, results_dir = "results/"):

    rhos, Tsh, Thl = extract_transitions(phase_data)

    Tsh = np.array(Tsh)
    Thl = np.array(Thl)

    fig, ax = plt.subplots(figsize=(8, 6))
    T_max_plot = np.nanmax(Thl) * 1.1

    mask_hl = np.isfinite(Thl)
    mask_sh = np.isfinite(Tsh)
    mask_hex = mask_hl & mask_sh

    rho_hl = rhos[mask_hl]
    T_hl = Thl[mask_hl]
    rho_sh = rhos[mask_sh]
    T_sh = Tsh[mask_sh]

    if len(rho_hl) >= 2:
        poly_x = np.concatenate([T_hl, [T_max_plot, T_max_plot, T_hl[0]]])
        poly_y = np.concatenate([rho_hl, [rho_hl[-1], rho_hl[0], rho_hl[0]]])
        ax.fill(poly_x, poly_y, color='red', alpha=0.3, label='Liquid')

    T_left = np.where(np.isfinite(Tsh), Tsh, np.where(np.isfinite(Thl), Thl, np.nan))
    mask_sol = np.isfinite(T_left)
    rho_sol = rhos[mask_sol]
    T_sol = T_left[mask_sol]

    if len(rho_sol) >= 2:
        poly_x = np.concatenate([[0], T_sol, [0]])
        poly_y = np.concatenate([[rho_sol[0]], rho_sol, [rho_sol[-1]]])
        ax.fill(poly_x, poly_y, color='blue', alpha=0.3, label='Solid')

    if mask_hex.sum() >= 2:
        rho_hex = rhos[mask_hex]
        T_s = Tsh[mask_hex]
        T_l = Thl[mask_hex]
        poly_x = np.concatenate([T_s, T_l[::-1]])
        poly_y = np.concatenate([rho_hex, rho_hex[::-1]])
        ax.fill(poly_x, poly_y, color='purple', alpha=0.3, label='Hexatic')

    if len(rho_sh) >= 1:
        ax.plot(T_sh, rho_sh, color='blue', linestyle='--', linewidth=2, label=r'Solid-Hexatic')
    if len(rho_hl) >= 1:
        ax.plot(T_hl, rho_hl, color='red', linestyle='--', linewidth=2, label=r'Hexatic-Liquid')

    ax.set_xlabel(r"$T$ (eV)")
    ax.set_ylabel(r"$\rho$ ($N_{\mathrm{grains}}/\AA^2$)")
    ax.set_xlim(0, T_max_plot)
    ax.set_ylim(rhos.min(), rhos.max())
    ax.grid()
    ax.legend()
    plt.tight_layout()
    save_path = Path(results_dir) / f"eps_{epsilon}" / "phase_diagram.pdf"
    plt.savefig(save_path)
    plt.close()