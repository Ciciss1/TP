import numpy as np
from numba import njit
import os

from Graphene import GrapheneCrystal

@njit
def dist_point_segments(px, py, ax, ay, bx, by):
    '''
    Perpendicular distance from point (px, py) to segment to segment (ax, ay) - (bx, by)
    '''
    abx = bx - ax
    aby = by - ay
    ab2 = abx * abx + aby * aby

    if ab2 < 1e-12:
        return np.sqrt((px - ax) ** 2 + (py - ay) ** 2)
    
    t = ((px - ax) * abx + (py - ay) * aby) / ab2
    t = max(0.0, min(1.0, t))

    cx = ax + t * abx
    cy = ay + t * aby
    return np.sqrt((px - cx) ** 2 + (py - cy) ** 2)

@njit
def find_gb_atoms(atoms, ridge_v1, ridge_v2, unfreeze_dist):
    '''
    Find atoms within unfreeze_dist of any Voronoi ridge
    '''
    N_atoms = atoms.shape[0]
    N_ridges = ridge_v1.shape[0]

    mask = np.zeros(N_atoms, dtype=np.bool_)

    for i in range(N_atoms):
        px = atoms[i, 0]
        py = atoms[i, 1]

        for j in range(N_ridges):
            d = dist_point_segments(px, py, ridge_v1[j, 0], ridge_v1[j, 1], ridge_v2[j, 0], ridge_v2[j, 1])

            if d < unfreeze_dist:
                mask[i] = True
                break

    return mask
    

class LammpsWriter:

    Z_THICKNESS = 20.0

    def __init__(self, graphene: GrapheneCrystal, unfreeze_dist = 2.0):
        self.crystal = graphene
        self.voronoi = graphene.lattice
        self.atoms = graphene.atoms
        self.L = graphene.lattice.L
        self.unfreeze_dist = unfreeze_dist
        self.unfreeze_mask = find_gb_atoms(self.atoms, self.voronoi.ridge_v1, self.voronoi.ridge_v2, unfreeze_dist)

        self.freeze_ids = np.where(~self.unfreeze_mask)[0] + 1
        self.unfreeze_ids = np.where(self.unfreeze_mask)[0] + 1

    def write(self, filename):
        path = filename + ".coord"
        N = len(self.atoms)
        Lx = self.L
        Ly = self.L
        Lz = self.Z_THICKNESS

        xy = self.atoms - np.array([Lx/2, Ly/2])

        with open(path, 'w') as f:
            f.write(f"LAMMPS data file\n\n")
            f.write(f"{N} atoms \n1 atom types\n\n")
            f.write(f"-{Lx/2:.6f} {Lx/2:.6f} xlo xhi\n")
            f.write(f"-{Ly/2:.6f} {Ly/2:.6f} ylo yhi\n")
            f.write(f"-{Lz/2:.6f} {Lz/2:.6f} zlo zhi\n\n")
            f.write("Masses\n\n1 12.0107\n\n")
            f.write("Atoms # atomic\n\n")
            
            lines = [f"{i+1} 1 {xy[i, 0]:.6f} {xy[i, 1]:.6f} 0.0\n" for i in range(N)]
            f.writelines(lines)

    def write_input_2d(self, filename,
                       T_start = 16500,
                       T_end = 100,
                       n_anneal = 5000,
                       n_min = 5000,
                       damping = 0.02,
                       dump_every = 100,
                       seed = None):
        
        if seed is None:
            seed = np.random.randint(1, int(1e6))

        basename = os.path.basename(filename)
        path = filename + "_2d.inp"

        freeze_group = "freeze"
        unfreeze_group = "unfreeze"

        with open(path, 'w') as f:
            # ---Header---
            f.write("units metal\n")
            f.write("dimension 2\n")
            f.write("atom_style atomic\n")
            f.write("newton on\n")
            f.write("boundary p p p\n")

            f.write(f"read_data {basename}.coord\n\n")

            # ---AIREBO potential---
            f.write("pair_style airebo 3.0\n")
            f.write("pair_coeff * * CH.airebo C\n\n")

            # ---dump---
            f.write(f"dump DDump all atom {dump_every} {basename}.lammpstrj\n")
            f.write("dump_modify DDump sort id\n")
            f.write("dump_modify DDump scale no\n\n")

            # ---timestep / thermo---
            f.write("timestep 0.0002\n")
            f.write("thermo_style multi\n")
            f.write("thermo 100\n\n")

            # ---groups freeze / unfreeze---
            unfreeze_ids_str = " ".join(map(str, self.unfreeze_ids))
            f.write(f"group {unfreeze_group} id {unfreeze_ids_str}\n")
            f.write(f"group {freeze_group} subtract all {unfreeze_group}\n\n")

            # ---freeze far atoms---
            f.write(f"fix zeroforce {freeze_group} setforce 0 0 0\n")
            f.write(f"compute myTemp {unfreeze_group} temp\n")
            f.write("thermo_modify temp myTemp\n\n")

            # ---enforce 2d---
            f.write("fix enforce2d all enforce2d\n\n")

            # ---1: minimize initial structure---
            f.write(f"minimize 1.0e-8 0 {n_min} 1000000\n\n")

            # ---2: anneal Langevin---
            f.write(f"fix lang {unfreeze_group} langevin {T_start} {T_end} {damping} {seed}\n")
            f.write(f"fix nve {unfreeze_group} nve\n\n")
            f.write(f"run {n_anneal}\n\n")

            # ---3: minimize final structure---
            f.write(f"minimize 1.0e-8 0 {n_min} 1000000\n\n")

    def write_input_3d(self, filename,
                       T_quench = 100,
                       n_quench = 500,
                       n_min = 5000,
                       damping = 0.02,
                       dump_every = 100,
                       seed = None):
        
        if seed is None:
            seed = np.random.randint(1, int(1e6))

        basename = os.path.basename(filename)
        path = filename + "_3d.inp"

        with open(path, 'w') as f:
            # ---Header---
            f.write("units metal\n")
            f.write("atom_style atomic\n")
            f.write("boundary p p p\n")
            f.write("newton on\n\n")

            f.write(f"read_data {basename}.coord\n\n")

            # ---AIREBO potential---
            f.write("pair_style airebo 3.0\n")
            f.write("pair_coeff * * CH.airebo C\n\n")

            # ---dump---
            f.write(f"dump DDump all atom {dump_every} {basename}_3d.lammpstrj\n")
            f.write("dump_modify DDump sort id\n")
            f.write("dump_modify DDump scale no\n\n")

            # ---timestep / thermo---
            f.write("timestep 0.0002\n")
            f.write("thermo_style multi\n")
            f.write("thermo 100\n\n")

            # ---1: minimize initial structure---
            f.write(f"minimize 1.0e-8 0 {n_min} 1000000\n\n")

            # ---2: small quench---
            f.write(f"fix lang all langevin {T_quench} 0 {damping} {seed}\n")
            f.write(f"fix nve all nve\n\n")
            f.write(f"run {n_quench}\n\n")

            # ---3: minimize final structure---
            f.write(f"minimize 1.0e-8 0 {n_min} 1000000\n\n")