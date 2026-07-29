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
    adj_i, adj_j, adj_l, areas = vor.adj_i, vor.adj_j, vor.adj_length, vor.areas
 
    H_0 = epsilon * np.sum(areas * theta**2)
 
    dtheta = np.abs(wrap_angle(theta[adj_i] - theta[adj_j]))
    dtheta = np.maximum(dtheta, 1e-10)
    per_bond = alpha * adj_l * dtheta * (beta_RS - np.log(dtheta))
    H_int = per_bond.sum()
 
    N = len(theta)
    return (H_0 + H_int) / N, H_0 / N, H_int / N, per_bond

def shannon_entropy(angles, n_bins=100, range=(-np.pi, np.pi)):
    counts, _ = np.histogram(angles, bins=n_bins, range=range)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))

def compute_chi6(Phi6, N, T):
    mean_Phi6 = np.mean(np.abs(Phi6))
    return N * (np.mean(np.abs(Phi6)**2) - mean_Phi6**2) / T

def analyze_observable(L, epsilon, rho, fig_size = 6, dot_size = 1, lw = 0.5, results_dir = "results/"):

    rho_path = Path(results_dir) / f"eps_{epsilon}" / f"L_{L}" / f"rho_{rho}"

    if not rho_path.exists():
        raise FileNotFoundError(f"Results directory {rho_path} does not exist.")
    
    T_dirs = sorted(rho_path.glob("T_*"), key=lambda x: float(x.name.split("_")[1]))
    if not T_dirs:
        raise FileNotFoundError(f"No T directories found in {rho_path}.")
    
    T_values = []

    Phi6 = []
    chi6_values = []

    eta_values = []
    eta_err_values = []

    U_values = []
    U_err_values = []
    U0_values = []
    U0_err_values = []
    Uint_values = []
    Uint_err_values = []
    dU_mean = []
    dU_std = []
    S_Shannon = []
    S_Shannon_err = []

    for T_dir in T_dirs:
        T = float(T_dir.name.split("_")[1])
        T_values.append(T)

        # crystal_paths = sorted(T_dir.glob("Crystal_*.npz"),
        #                        key=lambda p: int(p.stem.split("_")[1]))
        # if not crystal_paths:
        legacy = T_dir / "Crystal.npz"
        if not legacy.exists():
            raise FileNotFoundError(f"No Crystal file in {T_dir}")
        crystal_paths = [legacy]
        
        G6_runs, GT_runs, bin_centers = [], [], None
        Phi6_values, Ns = [], []
        Shannon_values = []
        Us, U0s, Uints, per_bonds = [], [], [], []

        for k, cpath in enumerate(crystal_paths):
            crystal = load_crystal(cpath)
            bin_centers, G6, GT = crystal.compute_observables()
            G6_runs.append(G6)
            GT_runs.append(GT)
            Ns.append(len(crystal.atoms))

            angles = []
            psi_6_values = []
            for i in range(len(crystal.atoms)):
                psi_6_i = obs.compute_psi6(i, crystal.atoms, crystal.neighbors, crystal.L)
                psi_6_values.append(psi_6_i)
                angle = np.angle(psi_6_i)
                angles.append(angle)

            angles = np.array(angles)
            psi_6_values = np.array(psi_6_values)

            Phi6_values.append(psi_6_values.mean())

            Shannon = shannon_entropy(angles)
            Shannon_values.append(Shannon)

            U, U0, Uint, per_bond = grain_energy(crystal, epsilon, alpha=1.70, beta_RS=-0.16)
            Us.append(U)
            U0s.append(U0)
            Uints.append(Uint)
            per_bonds.append(per_bond.mean())

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
                plt.hist(angles, bins=60)
                plt.xlabel(r"$\theta$ (radians)"); plt.ylabel("Count")
                plt.tight_layout()
                plt.savefig(T_dir / "Angle_Distribution.pdf")
                plt.close()

        G6_runs = np.array(G6_runs)
        GT_runs = np.array(GT_runs)
        n_runs = len(crystal_paths)

        G6_mean = G6_runs.mean(axis=0)
        GT_mean = GT_runs.mean(axis=0)
        sem = lambda x: x.std(axis=0, ddof=1) / np.sqrt(n_runs) if n_runs > 1 else np.zeros(x.shape[1])
        G6_err = sem(G6_runs)
        GT_err = sem(GT_runs)

        a_CC = 1.42
        r_min = a_CC
        mask_fit = bin_centers >= r_min
        x_fit = bin_centers[mask_fit]

        eta6_list, etaT_list = [], []

        for G6, GT in zip(G6_runs, GT_runs):
            coeffs, _ = curve_fit(power_law, x_fit, G6[mask_fit], p0=[1, 1/4], bounds=([0, 0], [10, 2]))
            eta6_list.append(coeffs[1])
            coeffs_T, _ = curve_fit(power_law, x_fit, GT[mask_fit], p0=[1, 1/4], bounds=([0, 0], [10, 2]))
            etaT_list.append(coeffs_T[1])

        eta6_list = np.array(eta6_list)
        etaT_list = np.array(etaT_list)
        eta = eta6_list.mean()
        eta_err = eta6_list.std(ddof=1) / np.sqrt(n_runs) if n_runs > 1 else 0
        eta_values.append(eta)
        eta_err_values.append(eta_err)
        eta_T = etaT_list.mean()
        eta_T_err = etaT_list.std(ddof=1) / np.sqrt(n_runs) if n_runs > 1 else 0

        # Plot G6 and GT with fits

        fig, ax = plt.subplots(figsize=(8, 6))

        coeffs, cov = curve_fit(power_law, x_fit, G6_mean[mask_fit], p0=[1, 1/4], bounds=([0, 0], [10, 2]))
        eta, eta_err = coeffs[1], np.sqrt(cov[1, 1])

        ax.loglog(bin_centers, G6_mean, label=r"$G_6(r)$")
        ax.fill_between(bin_centers, G6_mean - G6_err, G6_mean + G6_err, alpha=0.3)
        ax.loglog(x_fit, power_law(x_fit, *coeffs), 'r--', label=f"Fit: $\\eta_6$={eta:.3f}±{eta_err:.3f}")
        ax.set_ylim(1e-4, 1.05)
        ax.set_xlabel(r"$r (\AA)$")
        ax.set_ylabel(r"$G_6(r)$")
        ax.legend()
        ax.grid()
        plt.tight_layout()
        plt.savefig(T_dir / f"G6_rho_{rho}_T_{T}.pdf")
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 6))

        coeffs_T, cov_T = curve_fit(power_law, x_fit, GT_mean[mask_fit], p0=[1, 1/4], bounds=([0, 0], [10, 2]))
        eta_T, eta_T_err = coeffs_T[1], np.sqrt(cov_T[1, 1])

        ax.loglog(bin_centers, GT_mean, label=r"$G_T(r)$")
        ax.fill_between(bin_centers, GT_mean - GT_err, GT_mean + GT_err, alpha=0.3)
        ax.loglog(x_fit, power_law(x_fit, *coeffs_T), 'r--', label=f"Fit: $\\eta_T$={eta_T:.3f}±{eta_T_err:.3f}")
        ax.set_ylim(1e-4, 1.05)
        ax.set_xlabel(r"$r (\AA)$")
        ax.set_ylabel(r"$G_T(r)$")
        ax.legend()
        ax.grid()
        plt.tight_layout()
        plt.savefig(T_dir / f"GT_rho_{rho}_T_{T}.pdf")
        plt.close()

        Phi6_values = np.array(Phi6_values)
        Ns = np.array(Ns)
        N_atoms = Ns.mean()

        chi6 = compute_chi6(Phi6_values, N_atoms, T)
        chi6_values.append(chi6)
        Phi6.append(Phi6_values.mean())

        Shannon_values = np.array(Shannon_values)
        S_Shannon.append(Shannon_values.mean())
        S_Shannon_err.append(Shannon_values.std(ddof=1) / np.sqrt(n_runs) if n_runs > 1 else 0)

        Us = np.array(Us)
        U0s = np.array(U0s)
        Uints = np.array(Uints)
        per_bonds = np.array(per_bonds)

        U_values.append(Us.mean())
        U_err_values.append(Us.std(ddof=1) / np.sqrt(n_runs) if n_runs > 1 else 0)
        U0_values.append(U0s.mean())
        U0_err_values.append(U0s.std(ddof=1) / np.sqrt(n_runs) if n_runs > 1 else 0)
        Uint_values.append(Uints.mean())
        Uint_err_values.append(Uints.std(ddof=1) / np.sqrt(n_runs) if n_runs > 1 else 0)
        dU_mean.append(per_bonds.mean())
        dU_std.append(per_bonds.std(ddof=1) / np.sqrt(n_runs) if n_runs > 1 else 0)

    T_values = np.array(T_values)
    sort_idx = np.argsort(T_values)
    T_values = T_values[sort_idx]

    chi6_values = np.array(chi6_values)[sort_idx]
    Phi6 = np.array(Phi6)[sort_idx]

    eta_values = np.array(eta_values)[sort_idx]
    eta_err_values = np.array(eta_err_values)[sort_idx]

    U_values = np.array(U_values)[sort_idx]
    U_err_values = np.array(U_err_values)[sort_idx]
    U0_values = np.array(U0_values)[sort_idx]
    U0_err_values = np.array(U0_err_values)[sort_idx]
    Uint_values = np.array(Uint_values)[sort_idx]
    Uint_err_values = np.array(Uint_err_values)[sort_idx]
    dU_mean = np.array(dU_mean)[sort_idx]
    dU_std = np.array(dU_std)[sort_idx]
    S_Shannon = np.array(S_Shannon)[sort_idx]
    S_Shannon_err = np.array(S_Shannon_err)[sort_idx]

    CV = np.gradient(U_values, T_values)

    integrand = np.where(T_values > 0, CV / T_values, 0)
    S = cumulative_trapezoid(integrand, T_values, initial=0)

    F = U_values - T_values * S

    # plot chi6 vs T and phi6 vs T
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(T_values, chi6_values, 'x-')
    ax.set_xlabel(r"$T (eV)$")
    ax.set_ylabel(r"$\chi_6$")
    ax.grid()
    plt.tight_layout()
    plt.savefig(rho_path / f"chi6_vs_T_rho_{rho}.pdf")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(T_values, Phi6, 'x-')
    ax.set_xlabel(r"$T (eV)$")
    ax.set_ylabel(r"$\Phi_6$")
    ax.grid()
    plt.tight_layout()
    plt.savefig(rho_path / f"Phi6_vs_T_rho_{rho}.pdf")
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

    ax.errorbar(T_values, U_values, yerr=U_err_values, fmt='x-', label=r"$U$", alpha=0.7)
    ax.errorbar(T_values, U0_values, yerr=U0_err_values, fmt='o-', label=r"$U_0$", alpha=0.7)
    ax.errorbar(T_values, Uint_values, yerr=Uint_err_values, fmt='s-', label=r"$U_{\mathrm{int}}$", alpha=0.7)
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

    ax.plot(T_values, ratio, 'x-')
    ax.set_xlabel(r"$T (eV)$")
    ax.set_ylabel(r"$\langle \Delta U_{ij}\rangle / T$")
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

    # plot S_Shannon vs T
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(T_values, S_Shannon, yerr=S_Shannon_err, fmt='x-')
    ax.set_xlabel(r"$T (eV)$")
    ax.set_ylabel(r"Shannon entropy of angle distribution")
    ax.grid()
    plt.tight_layout()
    plt.savefig(rho_path / f"S_Shannon_vs_T_rho_{rho}.pdf")
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
    