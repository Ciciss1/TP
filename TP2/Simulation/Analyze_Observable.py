import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid
from pathlib import Path

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

def grain_energy(crystal, epsilon, alpha, beta_RS):
    vor  = crystal.lattice
    rho  = vor.rho
    theta = vor.theta
    adj_i, adj_j, adj_l = vor.adj_i, vor.adj_j, vor.adj_length
 
    H_0 = epsilon * np.sum(theta**2)
 
    dtheta = np.abs(wrap_angle(theta[adj_i] - theta[adj_j]))
    dtheta = np.maximum(dtheta, 1e-10)
    per_bond = alpha * adj_l * dtheta * (beta_RS - np.log(dtheta))
    H_int = per_bond.sum()
 
    N = len(theta)
    return (H_0 + H_int) / N, H_0 / N, H_int / N, per_bond

def analyze_observable(L, epsilon, rho, fig_size = 6, dot_size = 1, lw = 0.5, results_dir = "results/"):

    rho_path = Path(results_dir) / f"eps_{epsilon}" / f"L_{L}" / f"rho_{rho}"

    if not rho_path.exists():
        raise FileNotFoundError(f"Results directory {rho_path} does not exist.")
    
    T_dirs = sorted(rho_path.glob("T_*"), key=lambda x: float(x.name.split("_")[1]))
    if not T_dirs:
        raise FileNotFoundError(f"No T directories found in {rho_path}.")
    
    T_values = []

    chi6_values = []

    eta_values = []
    eta_err_values = []

    U_values = []
    U0_values = []
    Uint_values = []
    dU_mean = []
    dU_std = []

    for T_dir in T_dirs:
        T = float(T_dir.name.split("_")[1])
        T_values.append(T)

        Crystal_path = T_dir / "Crystal.npz"
        if not Crystal_path.exists():
            raise FileNotFoundError(f"Crystal file {Crystal_path} does not exist.")
        
        crystal = load_crystal(Crystal_path)

        # plot lattice
        crystal.plot_lattice()
        plt.savefig(T_dir / f"Lattice_rho_{rho}_T_{T}.pdf")
        plt.close()

        angles = []
        psi_6_values = []
        
        for i, atom in enumerate(crystal.atoms):
            psi_6_i = obs.compute_psi6(i, crystal.atoms, crystal.neighbors, crystal.L)
            psi_6_values.append(np.abs(psi_6_i))
            angle = np.angle(psi_6_i)
            angle = angle % (np.pi/3)
            angles.append(angle)

        angles = np.array(angles)
        psi_6_values = np.array(psi_6_values)
        mean_psi6 = np.mean(psi_6_values)

        chi6 = len(crystal.atoms) * (np.mean(psi_6_values**2) - mean_psi6**2)
        chi6_values.append(chi6)

        # plot crystal with angles as color
        crystal.plot_all(fig_size=fig_size, dot_size=dot_size, lw=lw, atom_phase=angles)

        plt.savefig(T_dir / f"Crystal_rho_{rho}_T_{T}.jpg", dpi=150, pil_kwargs={"quality": 85, "optimize": True})
        plt.close()
        
        # plot angle distribution
        plt.hist(angles, bins=60)
        plt.xlabel(r"$\theta$ (radians)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(T_dir / f"Angle_Distribution_rho_{rho}_T_{T}.pdf")
        plt.close()

        bin_centers, G6, GT = crystal.compute_observables()

        # Plot G6 and GT with fits
        a_CC = 1.42
        r_min = 5
        mask_fit = bin_centers >= r_min

        x_fit = bin_centers[mask_fit]
        y_fit = G6[mask_fit]

        coeffs, cov = curve_fit(power_law, x_fit, y_fit, p0=[1, 1/4], bounds=([0, 0], [10, 2]))
        a, b = coeffs  

        eta = b
        eta_err = np.sqrt(cov[1, 1])
        eta_values.append(eta)
        eta_err_values.append(eta_err)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.loglog(bin_centers, G6, label=r"$G_6(r)$")
        ax.loglog(x_fit, power_law(x_fit, *coeffs), 'r--', label=f"Fit: $\\eta_6$={eta:.3f}±{eta_err:.3f}")
        # ax.axhline(np.exp(-18 * T / epsilon), color='r', linestyle=':', label=r"Crystalline prediction")
        ax.set_ylim(1e-4, 1.05)
        ax.set_xlabel(r"$r (\AA)$")
        ax.set_ylabel(r"$G_6(r)$")
        ax.legend()
        ax.grid()
        plt.tight_layout()
        plt.savefig(T_dir / f"G6_fit_rho_{rho}_T_{T}.pdf")
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 6))

        coeffs_T, cov_T = curve_fit(power_law, x_fit, GT[mask_fit], p0=[1, 1/4], bounds=([0, 0], [10, 2]))
        eta_T, eta_T_err = coeffs_T[1], np.sqrt(cov_T[1, 1])

        ax.loglog(bin_centers, GT, label=r"$G_T(r)$")
        ax.loglog(x_fit, power_law(x_fit, *coeffs_T), 'r--', label=f"Fit: $\\eta_T$={eta_T:.3f}±{eta_T_err:.3f}")
        ax.set_ylim(1e-4, 1.05)
        ax.set_xlabel(r"$r (\AA)$")
        ax.set_ylabel(r"$G_T(r)$")
        ax.legend()
        ax.grid()
        plt.tight_layout()
        plt.savefig(T_dir / f"GT_rho_{rho}_T_{T}.pdf")
        plt.close()

        U, U0, Uint, per_bond = grain_energy(crystal, epsilon, alpha=1.70, beta_RS=-0.16)
        U_values.append(U)
        U0_values.append(U0)
        Uint_values.append(Uint)
        dU_mean.append(float(np.mean(per_bond)))
        dU_std.append(float(np.std(per_bond)))

    T_values = np.array(T_values)
    sort_idx = np.argsort(T_values)
    T_values = T_values[sort_idx]

    chi6_values = np.array(chi6_values)[sort_idx]

    eta_values = np.array(eta_values)[sort_idx]
    eta_err_values = np.array(eta_err_values)[sort_idx]

    U_values = np.array(U_values)[sort_idx]
    U0_values = np.array(U0_values)[sort_idx]
    Uint_values = np.array(Uint_values)[sort_idx]
    dU_mean = np.array(dU_mean)[sort_idx]
    dU_std = np.array(dU_std)[sort_idx]

    CV = np.gradient(U_values, T_values)

    integrand = np.where(T_values > 0, CV / T_values, 0)
    S = cumulative_trapezoid(integrand, T_values, initial=0)

    F = U_values - T_values * S

    # plot chi6 vs T
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(T_values, chi6_values, 'x-')
    ax.set_xlabel(r"$T (eV)$")
    ax.set_ylabel(r"$\chi_6$")
    ax.grid()
    plt.tight_layout()
    plt.savefig(rho_path / f"chi6_vs_T_rho_{rho}.pdf")
    plt.close()

    # plot eta vs T with error bars and horizontal line at 1/4
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(T_values, eta_values, yerr=eta_err_values, fmt='x-', label=r"Extracted $\eta_6$")
    ax.axhline(1/4, color='r', linestyle='--', label=r"$\eta_6 = 1/4$")
    ax.set_xlabel(r"$T (eV)$")
    ax.set_ylabel(r"$\eta_6$")
    ax.set_ylim(-0.05, 2)
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(rho_path / f"eta_vs_T_rho_{rho}.pdf")
    plt.close()

    # plot U, U0, Uint vs T
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(T_values, U_values, 'x-', label=r"$U$", alpha=0.7)
    ax.plot(T_values, U0_values, 'o-', label=r"$U_0$", alpha=0.7)
    ax.plot(T_values, Uint_values, 's-', label=r"$U_{\mathrm{int}}$", alpha=0.7)
    ax.set_xlabel(r"$T (eV)$")
    ax.set_ylabel(r"Energy per grain ($eV$)")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(rho_path / f"U_vs_T_rho_{rho}.pdf")
    plt.close()

    # plot CV vs T
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(T_values, CV, 'x-')
    ax.set_xlabel(r"$T (eV)$")
    ax.set_ylabel(r"Heat Capacity")
    ax.grid()
    plt.tight_layout()
    plt.savefig(rho_path / f"CV_vs_T_rho_{rho}.pdf")
    plt.close()

    # plot dU_mean vs T with error bars and line proportional to T
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(T_values, dU_mean, yerr=dU_std, fmt='x-', label=r"$\langle \Delta U_{ij}\rangle$")
    T_ax = np.linspace(T_values.min(), T_values.max(), 100)
    ax.plot(T_ax, T_ax, 'r--', label=r"$\Delta U \propto T$")
    ax.set_xlabel(r"$T (eV)$")
    ax.set_ylabel(r"Mean bond energy ($eV$)")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(rho_path / f"DeltaU_vs_T_rho_{rho}.pdf")
    plt.close()

    # plot dU_mean / T vs T
    ratio = dU_mean / T_values
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(T_values, ratio, 'x-', label=r"$\langle \Delta U_{ij}\rangle / T$")
    ax.set_xlabel(r"$T (eV)$")
    ax.set_ylabel(r"$\langle \Delta U_{ij}\rangle / T$")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(rho_path / f"DeltaU_over_T_vs_T_rho_{rho}.pdf")
    plt.close()

    # plot F vs T
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(T_values, F, 'x-')
    ax.set_xlabel(r"$T (eV)$")
    ax.set_ylabel(r"Free energy per grain ($eV$)")
    ax.grid()
    plt.tight_layout()
    plt.savefig(rho_path / f"F_vs_T_rho_{rho}.pdf")
    plt.close()

    # plot S vs T
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(T_values, S, 'x-')
    ax.set_xlabel(r"$T (eV)$")
    ax.set_ylabel(r"Entropy per grain")
    ax.grid()
    plt.tight_layout()
    plt.savefig(rho_path / f"S_vs_T_rho_{rho}.pdf")
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
    