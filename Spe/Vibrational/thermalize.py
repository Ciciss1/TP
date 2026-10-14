"""
Thermalisation of a relaxed crystal with LAMMPS/AIREBO (OpenMP) and vibrational density of states.

NVT (Nose-Hoover) in chunks until the mean potential energy of two consecutive chunks agrees within
pe_tol, then NVE production during which only the velocities are dumped (VDOS from their spectrum).
"""
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

C_MASS = 12.011         # amu
Z_VACUUM = 30.0         # AA
THZ_TO_CM = 33.35641    # 1 THz in cm^-1


@dataclass
class ThermalizationResult:
    positions: np.ndarray      # (n_atoms, 3) final snapshot (AA)
    velocities: np.ndarray     # (n_frames, n_atoms, 3) production velocities (AA/ps), float32
    times_ps: np.ndarray       # (n_frames,)
    equil_steps: np.ndarray    # NVT thermo history
    equil_temp: np.ndarray
    equil_pe: np.ndarray
    T: float
    Lx: float
    Ly: float
    dt_ps: float
    seed: int
    converged: bool            # NVT stopped on the PE criterion (False: equil_max_ps reached)

    def save(self, path):
        np.savez_compressed(path, **asdict(self))

    @staticmethod
    def load(path):
        d = np.load(path)
        fields = ThermalizationResult.__dataclass_fields__
        return ThermalizationResult(**{k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in fields})


class Thermalizer:
    def __init__(self, airebo_file = "CH.airebo", n_threads = 6, dt_ps = 0.001, lmp = "lmp"):
        self.airebo = str(Path(airebo_file).resolve())
        if not Path(self.airebo).is_file():
            raise FileNotFoundError(f"AIREBO potential not found: {self.airebo}")
        if shutil.which(lmp) is None:
            raise RuntimeError(f"LAMMPS executable '{lmp}' not found")
        self.n_threads, self.dt_ps, self.lmp = n_threads, dt_ps, lmp

    def run(self, atoms, Lx, Ly, T = 300.0, equil_check_ps = 2.0, equil_max_ps = 20.0, pe_tol = 0.005,
            prod_ps = 5.0, dump_every_fs = 2.0, seed = None):
        '''
        Inputs:
            atoms : positions (N, 3), Lx, Ly : box size
            T : temperature (K)
            equil_check_ps : length of the NVT chunks, equil_max_ps : maximum NVT time
            pe_tol : relative change of the mean PE between two chunks below which NVT stops
            prod_ps : NVE production, dump_every_fs : velocity dump period (< ~10 fs for the G band)
        '''
        seed = seed or int(np.random.default_rng().integers(1, 900_000_000))
        N = len(atoms)
        chunk = max(1, round(equil_check_ps / self.dt_ps))
        every = max(1, round(dump_every_fs / 1000 / self.dt_ps))
        header = f"""units metal
atom_style atomic
boundary p p p
package omp {self.n_threads}
{{read}}
pair_style airebo/omp 3.0 1 1
pair_coeff * * {self.airebo} C
timestep {self.dt_ps}
"""
        with tempfile.TemporaryDirectory(prefix="thermalize_") as tmp:
            tmp = Path(tmp)
            self._write_data(atoms, Lx, Ly, tmp / "input.data")
            restart = tmp / "equil.restart"

            steps, temp, pe = [], [], []
            converged, previous = False, None
            for i in range(max(1, round(equil_max_ps / equil_check_ps))):
                read = f"read_data {tmp / 'input.data'}" if i == 0 else f"read_restart {restart}"
                velocity = f"velocity all create {T} {seed} mom yes rot no dist gaussian\n" if i == 0 else ""
                s, t, e = self._thermo(self._run(header.format(read=read) + velocity + f"""thermo 50
thermo_style custom step temp pe etotal
fix nvt all nvt temp {T} {T} $(100*dt)
run {chunk}
write_restart {restart}
""", tmp, f"equil_{i}"))
                steps.append(s + i * chunk)
                temp.append(t)
                pe.append(e)
                if previous is not None and abs(e.mean() - previous) < pe_tol * abs(e.mean()):
                    converged = True
                    break
                previous = e.mean()

            self._run(header.format(read=f"read_restart {restart}") + f"""fix nve all nve
dump prod all custom {every} {tmp / 'prod.dump'} id vx vy vz
dump_modify prod sort id
run {max(1, round(prod_ps / self.dt_ps))}
undump prod
write_data {tmp / 'final.data'} nocoeff
""", tmp, "prod")
            velocities = self._read_dump(tmp / "prod.dump", N)
            lines = (tmp / "final.data").read_text().splitlines()
            start = next(i for i, l in enumerate(lines) if l.startswith("Atoms")) + 2
            table = np.loadtxt(lines[start:start + N])
            positions = np.empty((N, 3))
            positions[table[:, 0].astype(int) - 1] = table[:, 2:5]

        return ThermalizationResult(positions, velocities, np.arange(len(velocities)) * every * self.dt_ps,
                                    np.concatenate(steps), np.concatenate(temp), np.concatenate(pe),
                                    T, Lx, Ly, self.dt_ps, seed, converged)

    def _run(self, script, tmp, name):
        path = tmp / f"{name}.in"
        path.write_text(script)
        r = subprocess.run([self.lmp, "-in", str(path), "-log", "none", "-sf", "omp"], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"LAMMPS failed on {name}:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
        return r.stdout

    @staticmethod
    def _thermo(stdout):
        '''
        step, temperature and potential energy of the thermo table
        '''
        lines = stdout.splitlines()
        i = next(i for i, l in enumerate(lines) if re.match(r"^\s*Step\s+Temp\s+PotEng", l))
        rows = []
        for l in lines[i + 1:]:
            p = l.split()
            if len(p) != 4 or not re.match(r"^-?\d", p[0]):
                break
            rows.append([float(x) for x in p])
        rows = np.array(rows)
        return rows[:, 0], rows[:, 1], rows[:, 2]

    @staticmethod
    def _write_data(atoms, Lx, Ly, path):
        N = len(atoms)
        with open(path, "w") as f:
            f.write(f"Graphene polycrystal\n\n{N} atoms\n1 atom types\n\n0 {Lx:.6f} xlo xhi\n0 {Ly:.6f} ylo yhi\n"
                    f"-{Z_VACUUM} {Z_VACUUM} zlo zhi\n\nMasses\n\n1 {C_MASS}\n\nAtoms\n\n")
            np.savetxt(f, np.column_stack([np.arange(1, N + 1), np.ones(N), atoms]), fmt="%d %d %.8f %.8f %.8f")

    @staticmethod
    def _read_dump(path, N):
        '''
        Velocities of every frame (n_frames, N, 3), float32, read frame by frame (bounded memory)
        '''
        with open(path) as f:
            n_frames = sum(1 for l in f if l.startswith("ITEM: ATOMS"))
        v = np.empty((n_frames, N, 3), dtype=np.float32)
        frame, left = -1, 0
        with open(path) as f:
            for l in f:
                if left:
                    p = l.split()
                    v[frame, int(p[0]) - 1] = (float(p[1]), float(p[2]), float(p[3]))
                    left -= 1
                elif l.startswith("ITEM: ATOMS"):
                    frame, left = frame + 1, N
        return v


def vdos(velocities, dt_ps, atoms = None, comp = "all", batch = 5000):
    '''
    Vibrational density of states: windowed power spectrum of the velocities (Wiener-Khinchin
    equivalent of the Fourier transform of the VACF), centre-of-mass drift removed, unit area
    Inputs:
        velocities : (n_frames, n_atoms, 3), dt_ps : time between frames (ps)
        atoms : indices or boolean mask of the atoms (None: all)
        comp : "all", "xy" or "z"
    Outputs:
        frequencies (cm^-1), spectrum
    '''
    n_frames, n_atoms, _ = velocities.shape
    idx = np.arange(n_atoms) if atoms is None else np.arange(n_atoms)[atoms]
    cols = {"all": [0, 1, 2], "xy": [0, 1], "z": [2]}[comp]
    v_com = velocities.mean(axis=1, keepdims=True, dtype=np.float64)
    window = np.hanning(n_frames)[:, None, None]
    power = 0.0
    for start in range(0, len(idx), batch):
        v = (velocities[:, idx[start:start + batch]] - v_com)[:, :, cols] * window
        power = power + (np.abs(np.fft.rfft(v, axis=0)) ** 2).sum(axis=(1, 2))
    freq = np.fft.rfftfreq(n_frames, dt_ps) * THZ_TO_CM
    return freq, power / np.trapezoid(power, freq)


def fit_G(freq, spec, lo = 1450, hi = 1950):
    '''
    Centre and half-width (cm^-1) of a Lorentzian fitted to the G band
    '''
    m = (freq > lo) & (freq < hi)
    f, s = freq[m], spec[m]
    lor = lambda x, a, x0, g, b: a * g ** 2 / ((x - x0) ** 2 + g ** 2) + b
    p, _ = curve_fit(lor, f, s, p0=[s.max(), f[np.argmax(s)], 30, s.min()], maxfev=20000)
    return p[1], abs(p[2])


def low_freq_weight(freq, spec, fmax = 30):
    '''
    Fraction of the spectral weight below fmax (cm^-1)
    '''
    m = freq < fmax
    return np.trapezoid(spec[m], freq[m])
