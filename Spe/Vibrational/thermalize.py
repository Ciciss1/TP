from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

C_MASS = 12.011        # amu, carbon
Z_VACUUM = 10.0         # Angstrom
TIMESTEP_PS = 0.001     # 1 fs ; LAMMPS 'units metal' -> time in ps

# LAMMPS backend: "omp"/"gpu"/"kk"
_BACKENDS = {
    "omp": {"pair_suffix": "omp", "package": "package omp {n_threads}",
            "cmd_extra": ["-pk", "omp", "{n_threads}", "-sf", "omp"]},
    "gpu": {"pair_suffix": "gpu", "package": "package gpu 1",
            "cmd_extra": ["-pk", "gpu", "1", "-sf", "gpu"]},
    "kk":  {"pair_suffix": "kk", "package": "package kokkos",
            "cmd_extra": ["-k", "on", "g", "1", "-sf", "kk"]},
}


def _check_lammps():
    import shutil
    if shutil.which("lmp") is None:
        raise RuntimeError("LAMMPS executable ('lmp') not found in PATH.")


@dataclass
class ThermalizationResult:
    positions: np.ndarray      # (n_atoms, 3) -- final snapshot only, Angstrom
    velocities: np.ndarray     # (n_frames, n_atoms, 3), Angstrom/ps, float32
    times_ps: np.ndarray       # (n_frames,) production time, in ps
    equil_steps: np.ndarray    # NVT history: step
    equil_temp: np.ndarray     # NVT history: temperature (K)
    equil_pe: np.ndarray       # NVT history: potential energy (eV)
    T: float
    Lx: float
    Ly: float
    dt_ps: float
    seed: int
    n_atoms: int
    converged: bool = True     # did equilibration stop by PE convergence, or hit equil_max_ps?

    def save(self, path: str):
        np.savez_compressed(
            path,
            positions=self.positions, velocities=self.velocities,
            times_ps=self.times_ps,
            equil_steps=self.equil_steps, equil_temp=self.equil_temp,
            equil_pe=self.equil_pe,
            T=self.T, Lx=self.Lx, Ly=self.Ly, dt_ps=self.dt_ps, seed=self.seed,
            n_atoms=self.n_atoms, converged=self.converged,
        )

    @staticmethod
    def load(path: str) -> "ThermalizationResult":
        d = np.load(path)
        return ThermalizationResult(
            positions=d["positions"], velocities=d["velocities"],
            times_ps=d["times_ps"],
            equil_steps=d["equil_steps"], equil_temp=d["equil_temp"],
            equil_pe=d["equil_pe"],
            T=float(d["T"]), Lx=float(d["Lx"]), Ly=float(d["Ly"]), dt_ps=float(d["dt_ps"]),
            seed=int(d["seed"]), n_atoms=int(d["n_atoms"]),
            converged=bool(d["converged"]) if "converged" in d.files else True,
        )

    def is_equilibrated(self, tail_frac: float = 0.3, pe_rel_tol: float = 0.01) -> bool:
        """Independent check: PE drift on the last tail_frac of equilibration."""
        n = len(self.equil_pe)
        if n == 0:
            return False
        tail = self.equil_pe[int(n * (1 - tail_frac)):]
        if len(tail) < 2:
            return False
        drift = (tail.max() - tail.min()) / abs(tail.mean())
        return drift < pe_rel_tol


class Thermalizer:
    def __init__(
        self,
        airebo_file: str = "CH.airebo",
        n_threads: int = 6,
        backend: str = "omp",
        dt_ps: float = TIMESTEP_PS,
    ):
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {list(_BACKENDS)}")
        self.airebo_abs = os.path.abspath(airebo_file)
        if not os.path.isfile(self.airebo_abs):
            raise FileNotFoundError(f"AIREBO potential not found: {self.airebo_abs}")
        self.n_threads = n_threads
        self.backend = backend
        self.dt_ps = dt_ps
        _check_lammps()

    def run(
        self,
        atoms: np.ndarray,
        Lx: float,
        Ly: float,
        T: float = 300.0,
        equil_check_ps: float = 2.0,
        equil_max_ps: float = 20.0,
        pe_tol: float = 0.005,
        prod_ps: float = 5.0,
        dump_every_fs: float = 2.0,
        seed: Optional[int] = None,
    ) -> ThermalizationResult:
        """
        atoms: positions (N, 3)
        Lx, Ly: box dimensions
        T: temperature (K)
        equil_check_ps: NVT chunk length between convergence checks
        equil_max_ps: hard cap on total NVT time (safety net if it never converges)
        pe_tol: relative PE change between two consecutive chunks below which
                equilibration is considered converged and stops early
        prod_ps: NVE production duration (ps), feeds the VACF/VDOS
        dump_every_fs: velocity dump frequency during production, in fs
                       (keep < ~10 fs to resolve graphene's optical modes up
                       to ~1600 cm^-1)
        seed: random seed for velocity initialization
        """
        seed = seed if seed is not None else int(np.random.default_rng().integers(1, 900_000_000))
        n_atoms = len(atoms)
        chunk_steps = max(1, round(equil_check_ps / self.dt_ps))
        max_chunks = max(1, round(equil_max_ps / equil_check_ps))
        prod_steps = max(1, round(prod_ps / self.dt_ps))
        dump_every = max(1, round((dump_every_fs / 1000.0) / self.dt_ps))

        with tempfile.TemporaryDirectory(prefix="lammps_thermalize_") as tmp:
            tmp = Path(tmp)
            data_file = tmp / "input.data"
            restart_file = tmp / "equil.restart"

            self._write_data(atoms, Lx, Ly, data_file)

            # --- Chunked NVT equilibration, early stop on PE plateau ---
            equil_steps_chunks, equil_temp_chunks, equil_pe_chunks = [], [], []
            prev_chunk_pe_mean = None
            converged = False
            step_offset = 0

            for i in range(max_chunks):
                first_chunk = (i == 0)
                in_file = tmp / f"equil_{i}.in"
                in_file.write_text(self._build_equil_chunk_script(
                    data_file if first_chunk else restart_file,
                    restart_file, T, seed, chunk_steps, first_chunk,
                ))
                result = self._run_lmp(in_file)

                steps, temp, pe = self._parse_thermo(result.stdout)
                if len(steps) == 0:
                    raise RuntimeError(f"No thermo data parsed (chunk {i}).\n{result.stdout}")
                equil_steps_chunks.append(steps + step_offset)
                equil_temp_chunks.append(temp)
                equil_pe_chunks.append(pe)
                step_offset += chunk_steps

                chunk_pe_mean = pe.mean()
                if prev_chunk_pe_mean is not None:
                    rel_diff = abs(chunk_pe_mean - prev_chunk_pe_mean) / abs(chunk_pe_mean)
                    if rel_diff < pe_tol:
                        converged = True
                        break
                prev_chunk_pe_mean = chunk_pe_mean

            equil_steps_arr = np.concatenate(equil_steps_chunks)
            equil_temp = np.concatenate(equil_temp_chunks)
            equil_pe = np.concatenate(equil_pe_chunks)

            # --- NVE production, resumed from the last restart ---
            prod_in = tmp / "prod.in"
            dump_file = tmp / "prod.dump"
            final_data_file = tmp / "final.data"
            prod_in.write_text(self._build_production_script(
                restart_file, dump_file, final_data_file, prod_steps, dump_every
            ))
            result = self._run_lmp(prod_in)
            if not dump_file.is_file():
                raise RuntimeError(
                    f"No production dump generated by LAMMPS.\n"
                    f"Stdout:\n{result.stdout}\nStderr:\n{result.stderr}"
                )

            velocities = self._read_velocity_dump(dump_file, n_atoms)
            final_positions = self._read_final_positions(final_data_file, n_atoms)

        n_frames = velocities.shape[0]
        times_ps = np.arange(n_frames) * dump_every * self.dt_ps

        return ThermalizationResult(
            positions=final_positions, velocities=velocities, times_ps=times_ps,
            equil_steps=equil_steps_arr, equil_temp=equil_temp, equil_pe=equil_pe,
            Lx=Lx, Ly=Ly, T=T, dt_ps=self.dt_ps, seed=seed, n_atoms=n_atoms,
            converged=converged,
        )

    def _run_lmp(self, in_file: Path):
        backend = _BACKENDS[self.backend]
        cmd_extra = [a.format(n_threads=self.n_threads) for a in backend["cmd_extra"]]
        result = subprocess.run(
            ["lmp", "-in", str(in_file), "-log", "none", *cmd_extra],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"LAMMPS failed (code {result.returncode}) on {in_file.name}.\n"
                f"Stdout:\n{result.stdout}\nStderr:\n{result.stderr}"
            )
        return result

    @staticmethod
    def _write_data(atoms: np.ndarray, Lx: float, Ly: float, path: Path):
        n = len(atoms)
        lines = [
            "Graphene polycrystal", "",
            f"{n} atoms", "1 atom types", "",
            f"0 {Lx:.6f} xlo xhi",
            f"0 {Ly:.6f} ylo yhi",
            f"-{Z_VACUUM} {Z_VACUUM} zlo zhi", "",
            "Masses", "", f"1 {C_MASS:.6f}", "",
            "Atoms", "",
        ]
        lines += [f"{i + 1} 1 {x:.8f} {y:.8f} {z:.8f}" for i, (x, y, z) in enumerate(atoms)]
        path.write_text("\n".join(lines) + "\n")

    def _build_equil_chunk_script(self, input_source, restart_file, T, seed, chunk_steps, first_chunk: bool) -> str:
        """
        One NVT chunk. First chunk reads the data file and creates
        velocities (Maxwell-Boltzmann); later chunks resume from the
        previous restart (positions + velocities already thermalized, no
        velocity create). Writes a new restart at the end for the next
        chunk/production.
        """
        backend = _BACKENDS[self.backend]
        package_line = backend["package"].format(n_threads=self.n_threads)
        read_line = f"read_data {input_source}" if first_chunk else f"read_restart {input_source}"
        velocity_line = (
            f"velocity all create {T} {seed} mom yes rot no dist gaussian"
            if first_chunk else ""
        )
        return f"""
        units metal
        atom_style atomic
        boundary p p p

        {package_line}
        {read_line}

        pair_style airebo/{backend['pair_suffix']} 3.0 1 1
        pair_coeff * * {self.airebo_abs} C

        timestep {self.dt_ps}

        {velocity_line}

        thermo 50
        thermo_style custom step temp pe etotal
        fix nvt_equil all nvt temp {T} {T} $(100*dt)
        run {chunk_steps}
        unfix nvt_equil

        write_restart {restart_file}
        """

    def _build_production_script(self, restart_file, dump_file, final_data_file, prod_steps, dump_every) -> str:
        """
        NVE production, resumed from the last equilibration restart. Only
        velocities are dumped per frame (positions aren't needed for
        VACF/VDOS and doubling the trajectory doubles memory for nothing).
        A single final position snapshot is written via write_data.
        """
        backend = _BACKENDS[self.backend]
        package_line = backend["package"].format(n_threads=self.n_threads)
        return f"""
        units metal
        atom_style atomic
        boundary p p p

        {package_line}
        read_restart {restart_file}

        pair_style airebo/{backend['pair_suffix']} 3.0 1 1
        pair_coeff * * {self.airebo_abs} C

        timestep {self.dt_ps}

        fix nve_prod all nve
        dump prod all custom {dump_every} {dump_file} id vx vy vz
        dump_modify prod sort id
        run {prod_steps}
        undump prod

        write_data {final_data_file} nocoeff
        """

    @staticmethod
    def _parse_thermo(stdout: str):
        """Extracts step/temp/pe from the first 'Step Temp PotEng TotEng' table."""
        lines = stdout.splitlines()
        header_idx = next(
            (i for i, l in enumerate(lines) if re.match(r"^\s*Step\s+Temp\s+PotEng\s+TotEng", l)),
            None,
        )
        if header_idx is None:
            return np.array([]), np.array([]), np.array([])

        rows = []
        for line in lines[header_idx + 1:]:
            parts = line.split()
            if len(parts) != 4 or not re.match(r"^-?\d", parts[0]):
                break
            rows.append([float(p) for p in parts])
        if not rows:
            return np.array([]), np.array([]), np.array([])
        rows = np.array(rows)
        return rows[:, 0], rows[:, 1], rows[:, 2]

    @staticmethod
    def _read_velocity_dump(dump_file: Path, n_atoms: int) -> np.ndarray:
        """
        Streaming parse, two passes: 1) count frames, 2) fill a
        preallocated float32 array. Avoids loading the whole dump file
        into memory at once and avoids intermediate Python lists -- both
        matter a lot once n_atoms gets into the 100k+ range.
        """
        with open(dump_file) as f:
            n_frames = sum(1 for line in f if "ITEM: ATOMS" in line)

        velocities = np.empty((n_frames, n_atoms, 3), dtype=np.float32)

        with open(dump_file) as f:
            frame = -1
            atoms_left = 0
            for line in f:
                if "ITEM: ATOMS" in line:
                    frame += 1
                    atoms_left = n_atoms
                    continue
                if atoms_left > 0:
                    parts = line.split()
                    aid = int(parts[0]) - 1
                    velocities[frame, aid] = (float(parts[1]), float(parts[2]), float(parts[3]))
                    atoms_left -= 1

        return velocities

    @staticmethod
    def _read_final_positions(data_file: Path, n_atoms: int) -> np.ndarray:
        """Reads the final position snapshot written by write_data ('Atoms' section)."""
        positions = np.zeros((n_atoms, 3), dtype=np.float32)
        with open(data_file) as f:
            lines = f.readlines()

        start = next(i for i, l in enumerate(lines) if l.strip().startswith("Atoms")) + 2
        for line in lines[start:start + n_atoms]:
            parts = line.split()
            if len(parts) < 5:
                continue
            aid = int(parts[0]) - 1
            positions[aid] = (float(parts[2]), float(parts[3]), float(parts[4]))

        return positions


def compute_vacf(velocities: np.ndarray, batch_size: int = 5000) -> np.ndarray:
    """
    velocities: (n_frames, n_atoms, 3)
    Normalized VACF via FFT correlation (Wiener-Khinchin, O(N log N)).
    Processes atoms in batches so peak memory stays bounded regardless of
    n_atoms -- doing the FFT over all atoms at once blows up memory for
    large systems (100k+ atoms).
    """
    n_frames, n_atoms, _ = velocities.shape
    n_fft = 2 * n_frames
    acf_sum = np.zeros(n_frames)

    for start in range(0, n_atoms, batch_size):
        end = min(start + batch_size, n_atoms)
        v = velocities[:, start:end, :].reshape(n_frames, -1)
        fft_v = np.fft.fft(v, n=n_fft, axis=0)
        acf = np.fft.ifft(fft_v * np.conj(fft_v), axis=0).real[:n_frames]
        acf_sum += acf.sum(axis=1)

    acf_sum /= (n_frames - np.arange(n_frames)) * (n_atoms * 3)
    return acf_sum / acf_sum[0]


def compute_vdos(velocities: np.ndarray, dt_frame_ps: float, batch_size: int = 5000):
    """
    velocities: (n_frames, n_atoms, 3)
    dt_frame_ps: dump_every * dt_ps, in ps
    Same batching as compute_vacf, for the same memory reason.
    """
    n_frames, n_atoms, _ = velocities.shape
    window = np.hanning(n_frames).astype(np.float32)[:, None, None]
    power = np.zeros(n_frames // 2 + 1)

    for start in range(0, n_atoms, batch_size):
        end = min(start + batch_size, n_atoms)
        v_windowed = velocities[:, start:end, :] * window
        fft_v = np.fft.rfft(v_windowed, axis=0)
        power += (np.abs(fft_v) ** 2).sum(axis=(1, 2))

    freq_thz = np.fft.rfftfreq(n_frames, d=dt_frame_ps)  # 1/ps = THz
    return freq_thz, power


# -----------------------------------------------------------------------
# Usage example
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # atoms, Lx, Ly = ...
    # thermalizer = Thermalizer(airebo_file="CH.airebo", n_threads=6)
    # result = thermalizer.run(
    #     atoms, Lx, Ly, T=300.0,
    #     equil_check_ps=2.0, equil_max_ps=20.0, pe_tol=0.005,
    #     prod_ps=5.0,
    # )
    # print("Converged before equil_max_ps:", result.converged)
    # result.save("thermalized_300K.npz")
    #
    # freq, vdos = compute_vdos(result.velocities, dt_frame_ps=result.times_ps[1])
    pass