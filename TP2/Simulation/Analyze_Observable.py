import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import make_interp_spline
from pathlib import Path
from Graphene import GrapheneCrystal, load_crystal

def power_law(x, a, b):
    return a * x**(-b)

def analyze_observable(L, epsilon, rho, results_dir = "results/"):

    rho_path = Path(results_dir) / f"L{L}" / f"epsilon_{epsilon}" / f"rho_{rho}"

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

def plot_phase_diagram(phase_data: dict, L, epsilon, results_dir = "results/"):

    def find_transition(T_arr, phases, from_phases, to_phase):
        T = np.array(T_arr)
        last_from, first_to = None, None
        for t, ph in zip(T, phases):
            if ph in from_phases:
                last_from = t
            if ph == to_phase and last_from is not None and first_to is None:
                first_to = t
        if last_from is not None and first_to is not None:
            return (last_from + first_to) / 2
        return None
    
    rhos = np.array(sorted(phase_data.keys()))
    Tsh, Thl = [], []

    for rho in rhos:
        T_arr = phase_data[rho]["T"]
        phases = phase_data[rho]["phases"]
        T_sh = find_transition(T_arr, phases, from_phases=["solid"], to_phase="hexatic")
        T_hl = find_transition(T_arr, phases, from_phases=["hexatic"], to_phase="liquid")
        T_sl = find_transition(T_arr, phases, from_phases=["solid"], to_phase="liquid")
        Tsh.append(T_sh)
        Thl.append(T_hl if T_hl is not None else T_sl)

    Tsh = np.array(Tsh)
    Thl = np.array(Thl)

    fig, ax = plt.subplots(figsize=(8, 6))
    T_max_plot = np.nanmax(Thl) * 1.2

    mask_liq = np.isfinite(Thl)
    if mask_liq.sum() >= 2:
        ax.fill_between(rhos[mask_liq], Thl[mask_liq], T_max_plot, color='red', alpha=0.3, label='Liquid')

    T_left = np.where(np.isfinite(Tsh), Tsh, np.where(np.isfinite(Thl), Thl, np.nan))
    mask_sol = np.isfinite(T_left)
    if mask_sol.sum() >= 2:
        ax.fill_between(rhos[mask_sol], 0, T_left[mask_sol], color='blue', alpha=0.3, label='Solid')

    mask_hex = np.isfinite(Tsh) & np.isfinite(Thl)
    if mask_hex.sum() >= 2:
        ax.fill_between(rhos[mask_hex], Tsh[mask_hex], Thl[mask_hex], color='green', alpha=0.3, label='Hexatic')

    def plot_line(mask, T_arr, color, ls, lw, label):
        r_pts, T_pts = rhos[mask], T_arr[mask]
        if len(r_pts) < 2:
            if len(r_pts) == 1:
                ax.plot(T_pts, r_pts, 'x', color=color, ms=6)
            return
        
        if len(r_pts) >= 4:
            r_fine = np.linspace(r_pts.min(), r_pts.max(), 100)
            T_fine = make_interp_spline(r_pts, T_pts, k=3)(r_fine)
            ax.plot(T_fine, r_fine, color=color, ls=ls, lw=lw, label=label)
        else:
            ax.plot(T_pts, r_pts, color=color, ls=ls, lw=lw, label=label)
        ax.plot(T_pts, r_pts, 'x', color=color, ms=5, zorder=5)

    plot_line(np.isfinite(Tsh), Tsh, 'blue', '--', 2, r'Solid-Hexatic')
    plot_line(np.isfinite(Thl), Thl, 'red', '--', 2, r'Hexatic-Liquid')

    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$\rho$")
    ax.set_title(f"Phase Diagram for L={L}, epsilon={epsilon}")
    ax.set_xlim(0, T_max_plot)
    ax.set_ylim(rhos.min() * 0.95, rhos.max() * 1.05)
    ax.grid()
    ax.legend()
    plt.tight_layout()
    save_path = Path(results_dir) / f"L{L}" / f"epsilon_{epsilon}" / "phase_diagram.pdf"
    plt.savefig(save_path)
    plt.close()
    