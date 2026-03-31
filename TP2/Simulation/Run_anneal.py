import os
import shutil
import subprocess
import numpy as np
import sys
from tqdm import tqdm

sys.path.insert(0, "TP2/Simulation")
from Voronoi import PeriodicVoronoi
from Graphene import GrapheneCrystal, load_crystal
from LammpsWriter import LammpsWriter

LAMMPS_CMD_KOKKOS = (
    "/home/alexi/lammps/build-kokkos/lmp"
    " -k on g 1 t 1"
    " -sf kk"
    " -pk kokkos neigh half"
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
        if "Energy" in line:
            try:
                return float(line.split()[-1])
            except ValueError:
                pass
    return None

def read_final_coords(traj_path):
    with open(traj_path) as f:
        lines = f.readlines()

    xyz = []
    i = -1
    while "ITEM" not in lines[i]:
        parts = lines[i].split()
        if len(parts) >= 5:
            xyz.append([float(parts[2]), float(parts[3])])
        i -= 1

    return np.array(xyz[::-1])

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
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    airebo_file = "CH.airebo"
    if os.path.exists(airebo_file):
        shutil.copy(airebo_file, out_dir)
    else:
        raise FileNotFoundError("CH.airebo not found")
    
    graphene = load_crystal(os.path.join(out_dir, "initial_crystal.npz"))

    writer = LammpsWriter(graphene, unfreeze_dist=unfreeze_dist)
    n_unfreeze = writer.unfreeze_mask.sum()
    n_freeze = len(writer.unfreeze_mask) - n_unfreeze
    print(f"Unfreezing {n_unfreeze} atoms, freezing {n_freeze} atoms.")

    energies = []

    for i in tqdm(range(n_iter), desc="Annealing iterations"):
        name_2d = os.path.join(out_dir, f"sim_iter_{i}_2d")
        basename_2d = os.path.basename(name_2d)
        inp_path = os.path.join(out_dir, f"{basename_2d}_2d.inp")
        out_path = os.path.join(out_dir, f"{basename_2d}_2d.out")
        trj_2d = os.path.join(out_dir, f"{basename_2d}.lammpstrj")

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
            
        lammps_run(lammps_cmd, inp_path, out_path, cwd = out_dir)
        
        if os.path.exists(trj_2d):
            writer.atoms = read_final_coords(trj_2d)
        else:
            raise FileNotFoundError(f"Trajectory file {trj_2d} not found after 2D anneal.")
        
        name_3d = os.path.join(out_dir, f"sim_iter_{i}_3d")
        basename_3d = os.path.basename(name_3d)
        inp_path_3d = os.path.join(out_dir, f"{basename_3d}_3d.inp")
        out_path_3d = os.path.join(out_dir, f"{basename_3d}_3d.out")
        
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
        
        lammps_run(lammps_cmd, inp_path_3d, out_path_3d, cwd = out_dir)

        E = read_energy(f"{name_3d}_3d.out")
        energies.append(E)

    E_vals = [e for e in energies if e is not None]
    E_mean = np.mean(E_vals) if E_vals else None
    print(f"Mean Energy over {n_iter} iterations: {E_mean}")

    graphene.atoms = writer.atoms
    graphene.atoms += writer.L / 2
    graphene.build_graphene_bonds()

    graphene.save_crystal(os.path.join(out_dir, "final_crystal.npz"))

if __name__ == "__main__":
    params = {}
    with open("parameters.txt") as f:
        exec(f.read(), {}, params)

    run_anneal(
        unfreeze_dist=params.get("unfreeze_dist"),
        T_start=params.get("T_start"),
        T_end=params.get("T_end"),
        n_anneal=params.get("n_anneal"),
        n_min=params.get("n_min"),
        damping=params.get("damping"),
        dump_every=params.get("dump_every"),
        n_quench=params.get("n_quench"),
        n_iter=params.get("n_iter"),
        out_dir=params.get("out_dir"),
    )
