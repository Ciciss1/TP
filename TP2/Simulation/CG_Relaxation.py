import os
import subprocess
import tempfile
import numpy as np
from scipy.spatial import cKDTree

N_TRHEADS = 8
Z_VACUUM = 20.0
C_MASS = 12.011

def check_lammps_installation():
    import shutil
    if shutil.which("lammps") is None:
        raise RuntimeError("LAMMPS executable not found. Please ensure LAMMPS is installed and in your PATH.")

def atom_boundary_mask_from_generators(
        atoms: np.ndarray,
        generators: np.ndarray,
        generator_boundary_mask: np.ndarray,
    ) -> np.ndarray:
    '''
    Create a mask for atoms that are on the boundary of the Voronoi cells defined by the generators.
    Inputs:
        atoms: atomic positions
        generators: positions of the Voronoi generators
        generator_boundary_mask: mask for generators that are on the boundary
    Outputs:
        mask for atoms that are on the boundary
    '''
    tree = cKDTree(generators)
    _, nearest_gen = tree.query(atoms, k=1)
    return generator_boundary_mask[nearest_gen]

def write_lammps_data(
        atoms: np.ndarray,
        L: float,
        path: str
    ):
    '''
    Write atomic positions to a LAMMPS data file.
    Inputs:
        atoms: atomic positions
        L: box size
        path: path to the output LAMMPS data file
    '''
    N = len(atoms)
    with open(path, 'w') as f:
        f.write("Graphene polycrystal\n\n")
        f.write(f"{N} atoms\n")
        f.write("1 atom types\n\n")
        f.write(f"-1.0 {L + 1.0:.6f} xlo xhi\n")
        f.write(f"-1.0 {L + 1.0:.6f} ylo yhi\n")
        f.write(f"0.0 {Z_VACUUM:.6f} zlo zhi\n\n")
        f.write("Masses\n\n")
        f.write(f"1 {C_MASS:.6f}\n\n")
        f.write("Atoms\n\n")
        for i, (x, y) in enumerate(atoms):
            f.write(f"{i+1} 1 {x:.8f} {y:.8f} 0.0\n")

def write_lammps_input(
        data_file: str,
        dump_file: str,
        airebo_abs: str,
        free_indices: np.ndarray,
        ftol: float,
        max_steps: int,
        n_threads: int
    ) -> str:
    '''
    Write a LAMMPS input script for energy minimization using the AIREBO potential.
    Inputs:
        data_file: path to the LAMMPS data file
        dump_file: path to the LAMMPS dump file for output
        airebo_abs: absolute path to the AIREBO potential file
        free_indices: indices of atoms that should be free to move during minimization
        ftol: force tolerance for convergence
        max_steps: maximum number of minimization steps
        n_threads: number of threads to use for LAMMPS
    Outputs:
        path to the generated LAMMPS input script
    '''
    free_str = " ".join(str(i) for i in free_indices)
    script = f"""
    # Setup
    units metal
    atom_style atomic
    boundary s s p

    package omp {n_threads}
    read_data {data_file}

    group free id {free_str}
    group fixed subtract all free
    fix freeze fixed setforce 0.0 0.0 0.0

    # Define potential
    
    pair_style airebo/omp 3.0 1 1
    pair_coeff * * {airebo_abs} C

    # Minimization CG
    min_style cg
    minimize 0.0 {ftol} {max_steps} {max_steps * 10}

    # Output
    dump final all custom 1 {dump_file} id x y z
    dump_modify final sort id
    run 0
    undump final
    """
    return script

def read_lammps_dump(
        dump_file: str,
        atoms: np.ndarray,
        boundary_mask: np.ndarray,
        L: float
    ) -> np.ndarray:
    '''
    Read atomic positions from a LAMMPS dump file and update the positions
    Inputs:
        dump_file: path to the LAMMPS dump file
        atoms: original atomic positions
        boundary_mask: mask for atoms that are on the boundary
        L: box size
    Outputs:
        updated atomic positions
    '''
    pos = atoms.copy()

    with open(dump_file) as f:
        lines = f.readlines()
        
    data_start = None

    for i, line in enumerate(lines):
        if "ITEM: ATOMS" in line:
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError("Could not find atomic data in LAMMPS dump file.")
    
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        atom_id = int(parts[0]) - 1
        if boundary_mask[atom_id]:
            pos[atom_id, 0] = float(parts[1])
            pos[atom_id, 1] = float(parts[2])

    return pos

def parse_n_iterations(log_file: str) -> int:
    '''
    Parse the number of iterations taken for convergence from the LAMMPS log file.
    Inputs:
        log_file: path to the LAMMPS log file
    Outputs:
        number of iterations taken for convergence
    '''
    if not os.path.isfile(log_file):
        return -1
    with open(log_file) as f:
        for line in f:
            if "Iterations" in line and "=" in line:
                try:
                    return int(line.split("=")[-1].strip())
                except ValueError:
                    pass
    return -1

def minimize_CG(
        atoms: np.ndarray,
        L: float,
        generators: np.ndarray,
        generator_boundary_mask: np.ndarray,
        ftol: float = 0.1,
        max_steps: int = 150,
        airebo_file: str = "CH.airebo",
        n_threads: int = N_TRHEADS
    ) -> np.ndarray:
    '''
    Minimize the energy of a coarse-grained graphene structure using LAMMPS and ASE.
    Inputs:
        atoms: atomic positions
        generators: positions of the Voronoi generators
        generator_boundary_mask: mask for generators that are on the boundary
        L: box size
        ftol: tolerance for convergence
        max_steps: maximum number of optimization steps
        airebo_file: path to the AIREBO potential file
    Outputs:
        relaxed atomic positions
    '''
    check_lammps_installation()

    airebo_abs = os.path.abspath(airebo_file)
    if not os.path.isfile(airebo_abs):
        raise FileNotFoundError(f"AIREBO potential file not found at {airebo_abs}")

    boundary_mask = atom_boundary_mask_from_generators(atoms, generators, generator_boundary_mask)
    n_free = int(np.sum(boundary_mask))

    if n_free == 0:
        return atoms.copy()
    
    free_ids = np.where(boundary_mask)[0] + 1

    with tempfile.TemporaryDirectory(prefix="lammps_relax_") as tmpdir:
        data_file = os.path.join(tmpdir, "input.data")
        input_file = os.path.join(tmpdir, "input.in")
        dump_file = os.path.join(tmpdir, "relaxed.dump")
        log_file = os.path.join(tmpdir, "lammps.log")

        write_lammps_data(atoms, L, data_file)
        input_script = write_lammps_input(
            data_file=data_file,
            dump_file=dump_file,
            airebo_abs=airebo_abs,
            free_indices=free_ids,
            ftol=ftol,
            max_steps=max_steps,
            n_threads=n_threads
        )
        with open(input_file, 'w') as f:
            f.write(input_script)

        cmd = [
            "lammps",
            "-in", input_file,
            "-log", log_file,
            "-pk", "omp", str(n_threads),
            "-sf", "omp",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            log_content = ""
            if os.path.isfile(log_file):
                with open(log_file) as f:
                    log_content = f.read()
            raise RuntimeError(
                f"LAMMPS execution failed with return code {result.returncode}.\n"
                f"Stdout: {result.stdout}\n"
                f"Stderr: {result.stderr}\n"
                f"Log content:\n{log_content}"
                )
        
        if not os.path.isfile(dump_file):
            raise RuntimeError("LAMMPS did not produce the expected dump file.")
        
        pos_relaxed = read_lammps_dump(dump_file, atoms, boundary_mask, L)

    return pos_relaxed

class CGRelaxation:

    def relaxation_CG(
        self,
        atoms: np.ndarray,
        generators: np.ndarray,
        generator_boundary_mask: np.ndarray,
        ftol: float = 0.1,
        max_steps: int = 150,
        airebo_file: str = "CH.airebo",
        n_threads: int = N_TRHEADS
        ) -> np.ndarray:
        
        return minimize_CG(
            atoms = atoms,
            L = self.L,
            generators = generators,
            generator_boundary_mask = generator_boundary_mask,
            ftol = ftol,
            max_steps = max_steps,
            airebo_file = airebo_file,
            n_threads = n_threads
        )
