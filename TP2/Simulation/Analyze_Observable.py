import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
from Graphene import GrapheneCrystal, load_crystal

def power_law(x, a, b):
    return a * x**(-b)

def analyze_observable(L, epsilon, rho, results_dir = "results/"):

    rho_path = Path(results_dir) / f"eps_{epsilon}" / f"L_{L}" / f"rho_{rho}"

    if not rho_path.exists():
        raise FileNotFoundError(f"Results directory {rho_path} does not exist.")
    
    T_dirs = sorted(rho_path.glob("T_*"), key=lambda x: float(x.name.split("_")[1]))
    if not T_dirs:
        raise FileNotFoundError(f"No T directories found in {rho_path}.")
    
    T_values = []
    eta_values = []
    eta_err_values = []

    for T_dir in T_dirs:
        T = float(T_dir.name.split("_")[1])
        T_values.append(T)

        final_crystal_path = T_dir / "final_crystal.npz"
        if not final_crystal_path.exists():
            raise FileNotFoundError(f"Final crystal file {final_crystal_path} does not exist.")
        
        crystal = load_crystal(final_crystal_path)
        
        bin_centers, G6 = crystal.compute_observables()

        

        coeffs, cov = curve_fit(power_law, bin_centers, G6, p0=[1, 1/4])
        a, b = coeffs  

        eta = b
        eta_err = np.sqrt(cov[1, 1])
        eta_values.append(eta)
        eta_err_values.append(eta_err)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(bin_centers, G6, label = r"$G_6(r)$")
        x_fit = np.linspace(bin_centers.min(), bin_centers.max(), 100)
        ax.plot(x_fit, power_law(x_fit, *coeffs), label = r"Fit: $G_6(r) \sim r^{-\eta}$", linestyle='--')
        ax.set_xlabel(r"$r$")
        ax.set_ylabel(r"$G_6(r)$")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"L={L}, epsilon={epsilon}, rho={rho}, T={T}")
        ax.legend()
        ax.grid()
        plt.tight_layout()
        plt.savefig(T_dir / "G6_fit.pdf")
        plt.close()

    T_values = np.array(T_values)
    eta_values = np.array(eta_values)
    eta_err_values = np.array(eta_err_values)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(T_values, eta_values, yerr=eta_err_values, fmt='x-', label=r"Extracted $\eta$")
    ax.axhline(1/4, color='r', linestyle='--', label=r"$\eta = 1/4$")
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$\eta$")
    ax.set_ylim(-0.05, 0.7)
    ax.set_title(f"L={L}, epsilon={epsilon}, rho={rho}")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(rho_path / "eta_vs_T.pdf")
    plt.close()

def plot_phase_diagram(phase_data: dict, epsilon, results_dir = "results/"):

    def find_transition(T_arr, phases, from_phase, to_phase):
        last_from, first_to = None, None
        for t, ph in zip(T_arr, phases):
            if ph == from_phase:
                last_from = t
            if ph == to_phase and last_from is not None and first_to is None:
                first_to = t
        if last_from is not None and first_to is not None:
            return (last_from + first_to) / 2
        return None

    rhos = np.array(sorted(phase_data.keys()))
    Tsh, Thl = [], []

    for rho in rhos:
        T = phase_data[rho]["T"]
        T_arr = np.array(T)
        phases = phase_data[rho]["phases"]
        T_sh = find_transition(T_arr, phases, from_phase="s", to_phase="h")
        T_hl = find_transition(T_arr, phases, from_phase="h", to_phase="l")
        T_sl = find_transition(T_arr, phases, from_phase="s", to_phase="l")
        Tsh.append(T_sh)
        Thl.append(T_hl if T_hl is not None else T_sl)

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

    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$\rho$")
    ax.set_title(f"Phase Diagram for epsilon={epsilon}")
    ax.set_xlim(0, T_max_plot)
    ax.set_ylim(rhos.min(), rhos.max())
    ax.set_yscale('log')
    ax.grid()
    ax.legend()
    plt.tight_layout()
    save_path = Path(results_dir) / f"eps_{epsilon}" / "phase_diagram.pdf"
    plt.savefig(save_path)
    plt.close()
    