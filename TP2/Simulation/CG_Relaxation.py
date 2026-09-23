import os
import shutil
import subprocess
import tempfile
import numpy as np
from scipy.spatial import cKDTree

N_THREADS = 6
Z_VACUUM = 10.0
C_MASS = 12.011

# Stage 1 (planar, only atoms near the grain boundaries move): REBO only (AIREBO with the LJ and
# torsion terms switched off). The LJ term is the expensive part of AIREBO and is irrelevant for
# an in-plane pre-relaxation of the bond network.
PAIR_2D = "airebo 3.0 0 0"
# Stage 2 (3D, all atoms): full AIREBO
PAIR_3D = "airebo 3.0 1 1"


def check_lammps_installation(lmp="lmp"):
    if shutil.which(lmp) is None:
        raise RuntimeError(f"LAMMPS executable '{lmp}' not found. Please ensure LAMMPS is installed and in your PATH.")


def atom_boundary_mask_from_generators(atoms, generators, generator_boundary_mask, L):
    '''
    Mask of the atoms whose nearest generator is a free (boundary) generator
    Inputs:
        atoms : atomic positions
        generators : positions of the generators (in [0, L))
        generator_boundary_mask : mask of the free generators
        L : box size
    Outputs:
        mask of the atoms that are free to move in the planar stage
    '''
    tree = cKDTree(np.mod(generators, L), boxsize=L)
    _, nearest = tree.query(np.mod(atoms[:, :2], L), k=1)
    return generator_boundary_mask[nearest]


def write_lammps_data(atoms, types, L, path):
    '''
    Write the atoms to a LAMMPS data file (type 1 = free, type 2 = fixed during the planar stage)
    '''
    N = len(atoms)
    pos = atoms.copy()
    pos[:, 0] = np.mod(pos[:, 0], L)
    pos[:, 1] = np.mod(pos[:, 1], L)
    with open(path, "w") as f:
        f.write("Graphene polycrystal\n\n")
        f.write(f"{N} atoms\n")
        f.write("2 atom types\n\n")
        f.write(f"0 {L:.6f} xlo xhi\n")
        f.write(f"0 {L:.6f} ylo yhi\n")
        f.write(f"-{Z_VACUUM} {Z_VACUUM} zlo zhi\n\n")
        f.write("Masses\n\n")
        f.write(f"1 {C_MASS:.6f}\n2 {C_MASS:.6f}\n\n")
        f.write("Atoms\n\n")
        table = np.column_stack([np.arange(1, N + 1), types, pos[:, 0], pos[:, 1], pos[:, 2]])
        np.savetxt(f, table, fmt="%d %d %.8f %.8f %.8f")


def write_lammps_input(data_file, dump_file, airebo_abs, ftol, max_steps_2d, max_steps_3d,
                       etol_2d, etol_3d, z_noise, seed, pair_2d, pair_3d, min_style, skin):
    '''
    LAMMPS script running both relaxation stages in a single process:
        1. planar minimisation, only the type-1 atoms move, z frozen
        2. random z displacement, then 3D minimisation of all atoms
    '''
    return f"""
units           metal
atom_style      atomic
boundary        p p p
read_data       {data_file}

neighbor        {skin} bin
neigh_modify    delay 0 every 1 check yes
thermo          0

group           fixed type 2

# ---------- Stage 1 : planar relaxation of the grain boundaries ----------
pair_style      {pair_2d}
pair_coeff      * * {airebo_abs} C C
fix             freeze fixed setforce 0.0 0.0 0.0
fix             planar all setforce NULL NULL 0.0
min_style       {min_style}
minimize        {etol_2d} {ftol} {max_steps_2d} {10 * max_steps_2d}
unfix           freeze
unfix           planar

# ---------- Stage 2 : out-of-plane relaxation of all atoms ----------
displace_atoms  all random 0.0 0.0 {z_noise} {seed} units box
pair_style      {pair_3d}
pair_coeff      * * {airebo_abs} C C
min_style       {min_style}
minimize        {etol_3d} {ftol} {max_steps_3d} {10 * max_steps_3d}

write_dump      all custom {dump_file} id x y z modify sort id
"""


def read_lammps_dump(dump_file, N, L):
    '''
    Read the positions written by write_dump (sorted by id)
    '''
    data = np.loadtxt(dump_file, skiprows=9)
    if data.ndim == 1:
        data = data[None, :]
    if len(data) != N:
        raise RuntimeError(f"LAMMPS dump contains {len(data)} atoms, expected {N}.")
    pos = np.empty((N, 3))
    ids = data[:, 0].astype(np.int64) - 1
    pos[ids] = data[:, 1:4]
    pos[:, 0] = np.mod(pos[:, 0], L)
    pos[:, 1] = np.mod(pos[:, 1], L)
    return pos


def parse_minimization_stats(log_file):
    '''
    Extract, for each minimize command, the stopping criterion, the number of iterations
    and the final force two-norm from the LAMMPS log
    '''
    stats = {"criterion": [], "iterations": [], "fnorm_final": [], "energy_final": []}
    if not os.path.isfile(log_file):
        return stats
    with open(log_file) as f:
        lines = f.readlines()
    for k, line in enumerate(lines):
        if "Stopping criterion =" in line:
            stats["criterion"].append(line.split("=", 1)[1].strip())
        elif "Iterations, force evaluations =" in line:
            stats["iterations"].append(int(line.split("=", 1)[1].split()[0]))
        elif "Force two-norm initial, final =" in line:
            stats["fnorm_final"].append(float(line.split("=", 1)[1].split()[1]))
        elif "Energy initial, next-to-last, final =" in line and k + 1 < len(lines):
            stats["energy_final"].append(float(lines[k + 1].split()[-1]))
    return stats


def minimize_CG(atoms, L, generators, generator_boundary_mask,
                ftol=1.0, max_steps_2d=500, max_steps_3d=500, etol_2d=0.0, etol_3d=1e-6,
                z_noise=0.05, seed=None, pair_2d=PAIR_2D, pair_3d=PAIR_3D, min_style="cg",
                skin=1.0, airebo_file="CH.airebo", n_threads=N_THREADS, lmp="lmp"):
    '''
    Relax a polycrystalline graphene sheet with LAMMPS (planar stage then 3D stage)
    Inputs:
        atoms : atomic positions (N, 3)
        L : box size
        generators, generator_boundary_mask : used to select the atoms free in the planar stage
        ftol : force tolerance (global force two-norm, eV/AA)
        max_steps_2d, max_steps_3d : iteration caps of the two stages
        etol_2d, etol_3d : relative energy tolerances of the two stages
        z_noise : amplitude of the random z displacement before the 3D stage (AA)
        seed : seed of the z displacement
        pair_2d, pair_3d : pair styles of the two stages
        min_style : LAMMPS minimiser ("cg" or "fire")
        skin : neighbour-list skin (AA)
        airebo_file : AIREBO parameter file
        n_threads : OpenMP threads for LAMMPS
    Outputs:
        relaxed atomic positions, dictionary with the minimisation statistics
    '''
    check_lammps_installation(lmp)
    airebo_abs = os.path.abspath(airebo_file)
    if not os.path.isfile(airebo_abs):
        raise FileNotFoundError(f"AIREBO potential file not found at {airebo_abs}")

    atoms = np.asarray(atoms, dtype=np.float64)
    if atoms.shape[1] == 2:
        atoms = np.hstack([atoms, np.zeros((len(atoms), 1))])

    free = atom_boundary_mask_from_generators(atoms, generators, generator_boundary_mask, L)
    types = np.where(free, 1, 2)
    if seed is None:
        seed = int(np.random.default_rng().integers(1, 2**31 - 1))
    seed = max(1, int(seed) % (2**31 - 1))

    with tempfile.TemporaryDirectory(prefix="lammps_relax_") as tmpdir:
        data_file = os.path.join(tmpdir, "input.data")
        input_file = os.path.join(tmpdir, "relax.in")
        dump_file = os.path.join(tmpdir, "relaxed.dump")
        log_file = os.path.join(tmpdir, "lammps.log")

        write_lammps_data(atoms, types, L, data_file)
        with open(input_file, "w") as f:
            f.write(write_lammps_input(data_file, dump_file, airebo_abs, ftol, max_steps_2d, max_steps_3d,
                                       etol_2d, etol_3d, z_noise, seed, pair_2d, pair_3d, min_style, skin))

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(n_threads)
        cmd = [lmp, "-in", input_file, "-log", log_file, "-screen", "none",
               "-pk", "omp", str(n_threads), "-sf", "omp"]
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        if result.returncode != 0 or not os.path.isfile(dump_file):
            log_content = open(log_file).read() if os.path.isfile(log_file) else ""
            raise RuntimeError(
                f"LAMMPS failed (return code {result.returncode}).\n"
                f"Stdout: {result.stdout}\nStderr: {result.stderr}\n"
                f"Log (last 3000 chars):\n{log_content[-3000:]}"
            )

        pos = read_lammps_dump(dump_file, len(atoms), L)
        stats = parse_minimization_stats(log_file)

    stats["n_free_2d"] = int(np.sum(free))
    stats["seed"] = seed
    return pos, stats


class CGRelaxation:

    def relaxation_CG(self, atoms, generators, generator_boundary_mask, **kwargs):
        pos, stats = minimize_CG(atoms=atoms, L=self.L, generators=generators,
                                 generator_boundary_mask=generator_boundary_mask, **kwargs)
        self.lammps_info = stats
        return pos