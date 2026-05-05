import os
import shutil
import subprocess
import numpy as np
from tqdm import tqdm

from Graphene import load_crystal
from LammpsWriter import LammpsWriter

LAMMPS_CMD_KOKKOS = (
    "mpirun -np 4 /home/alexi/lammps/build-gpu/lmp"
    " -sf omp"
    " -pk omp 2"
)

def lammps_run(lammps_cmd, inp_file, out_file, cwd):
    cmd = lammps_cmd.split() + ["-in", inp_file]
    with open(out_file, "w") as f_out:
        result = subprocess.run(cmd, cwd=cwd, stdout=f_out, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        with open(out_file) as f:
            print(f.read())
        raise RuntimeError(f"LAMMPS run failed with return code {result.returncode}")

def read_energy(out_path):
    with open(out_path) as f:
        lines = f.readlines()
    for line in reversed(lines):
        try:
            return float(line.split("=")[1].strip().split()[0])
        except (ValueError, IndexError):
            pass
    return None

def read_final_coords(traj_path):
    with open(traj_path) as f:
        lines = f.readlines()

    least_atoms_line = -1
    for idx, line in enumerate(lines):
        if line.startswith("ITEM: ATOMS"):
            least_atoms_line = idx

    if least_atoms_line < 0:
        raise ValueError("No ATOMS section found in trajectory file")
    
    xyz = []
    for line in lines[least_atoms_line + 1:]:
        if line.startswith("ITEM:"):
            break
        parts = line.split()
        if len(parts) >= 4:
            xyz.append([float(parts[2]), float(parts[3])])

    if not xyz:
        raise ValueError("No atom coordinates found in trajectory file")

    return np.array(xyz)

def run_anneal(
        unfreeze_dist = 1.5,
        T_start = 16500,
        T_end = 100,
        n_anneal = 5000,
        n_min = 5000,
        damping = 0.02,
        dump_every = 100,
        n_quench = 500,
        n_iter = 2,
        out_dir = "results",
        lammps_cmd = LAMMPS_CMD_KOKKOS,
        a = 1.42,
):
    output_anneal = os.path.abspath(out_dir)
    output_anneal = os.path.join(output_anneal, f"anneal/")
    os.makedirs(output_anneal, exist_ok=True)
    
    airebo_file = "CH.airebo"
    if os.path.exists(airebo_file):
        shutil.copy(airebo_file, output_anneal)
    else:
        raise FileNotFoundError("CH.airebo not found")
    
    graphene = load_crystal(os.path.join(out_dir, "initial_crystal.npz"))

    writer = LammpsWriter(graphene, unfreeze_dist=unfreeze_dist)
    n_unfreeze = writer.unfreeze_mask.sum()
    n_freeze = len(writer.unfreeze_mask) - n_unfreeze
    print(f"Unfreezing {n_unfreeze} atoms, freezing {n_freeze} atoms.")

    energies = []

    for i in tqdm(range(n_iter), desc="Annealing iterations"):
        name_2d = os.path.join(output_anneal, f"sim_iter_{i}_2d")
        basename_2d = os.path.basename(name_2d)
        inp_path_2d = os.path.join(output_anneal, f"{basename_2d}.inp")
        out_path_2d = os.path.join(output_anneal, f"{basename_2d}.out")
        trj_2d = os.path.join(output_anneal, f"{basename_2d}.lammpstrj")

        writer.write(name_2d)
        writer.write_input_2d(
            name_2d,
            T_start = T_start,
            T_end = T_end,
            n_anneal = n_anneal,
            n_min = n_min,
            damping = damping,
            dump_every = dump_every,
            seed = None
        )
            
        lammps_run(lammps_cmd, inp_path_2d, out_path_2d, cwd = output_anneal)
        
        if os.path.exists(trj_2d):
            writer.atoms = read_final_coords(trj_2d)
        else:
            raise FileNotFoundError(f"Trajectory file {trj_2d} not found after 2D anneal.")
        
        name_3d = os.path.join(output_anneal, f"sim_iter_{i}_3d")
        basename_3d = os.path.basename(name_3d)
        inp_path_3d = os.path.join(output_anneal, f"{basename_3d}.inp")
        out_path_3d = os.path.join(output_anneal, f"{basename_3d}.out")
        
        writer.write(name_3d)
        writer.write_input_3d(
            name_3d,
            T_quench = T_end,
            n_quench = n_quench,
            n_min = n_min,
            damping = damping,
            dump_every= dump_every,
            seed = None
        )
        
        lammps_run(lammps_cmd, inp_path_3d, out_path_3d, cwd = output_anneal)

        E = read_energy(out_path_3d)
        energies.append(E)

    E_vals = [e for e in energies if e is not None]
    E_mean = float(np.mean(E_vals)) if E_vals else None
    print(f"Mean Energy over {n_iter} iterations: {E_mean}")

    graphene.atoms = writer.atoms + np.array([writer.L/2, writer.L/2])
    graphene.build_graphene_bonds()

    graphene.save_crystal(os.path.join(out_dir, "final_crystal.npz"))