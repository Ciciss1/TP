"""
Correlation functions of the relaxed crystals (T=<T>K_k=<k>.npz written by Run_Simulation.py):
local bond-orientational order psi6, orientational correlation G6(r) and translational correlation GT(r).

All pair averages are over every pair of atoms (no sampling), from FFT autocorrelations of per-atom
fields binned on a fine grid: the phases are exact, only the pair distances are known to within the
grid spacing h.
"""
import numpy as np


def load_crystal(path):
    '''
    Relaxed positions (N, 3), bonds (Nb, 2) and box size Lx, Ly of a crystal
    '''
    d = np.load(path)
    Lx, Ly = (float(v) for v in d["lattice"])
    return np.asarray(d["atoms"], dtype=np.float64), np.asarray(d["bonds"], dtype=np.int64), Lx, Ly


def bond_vectors(xyz, bonds, Lx, Ly):
    '''
    In-plane vectors i -> j of the bonds (Nb, 2), minimum image
    '''
    d = xyz[bonds[:, 1], :2] - xyz[bonds[:, 0], :2]
    return d - np.array([Lx, Ly]) * np.round(d / [Lx, Ly])


def bond_average(N, bonds, z_ij, z_ji):
    '''
    Average over the bonds of each atom of a complex bond quantity (z_ij seen from i, z_ji from j)
    '''
    ends = np.r_[bonds[:, 0], bonds[:, 1]]
    z = np.r_[z_ij, z_ji]
    return (np.bincount(ends, z.real, N) + 1j * np.bincount(ends, z.imag, N)) / np.bincount(ends, minlength=N)


def psi6(xyz, bonds, Lx, Ly):
    '''
    Local bond-orientational order of each atom, psi6_i = <exp(6 i phi_ij)> over its bonds
    (|psi6| = 1 for a perfect honeycomb, arg(psi6) / 6 = local lattice orientation)
    '''
    d = bond_vectors(xyz, bonds, Lx, Ly)
    z = np.exp(6j * np.arctan2(d[:, 1], d[:, 0]))          # same for i -> j and j -> i
    return bond_average(len(xyz), bonds, z, z)


def translational_field(xyz, bonds, Lx, Ly, G):
    '''
    exp(i G.c) averaged over the centres c = 2 r_i - r_j of the three hexagons around atom i: one
    phase per site of the triangular lattice of hexagon centres, the same for both sublattices
    '''
    d = bond_vectors(xyz, bonds, Lx, Ly)
    ph = np.exp(-1j * d @ G)
    return np.exp(1j * xyz[:, :2] @ G) * bond_average(len(xyz), bonds, ph, ph.conj())


class PairGrid:
    '''
    Pair averages of per-atom fields as a function of the in-plane (periodic) distance
    '''
    def __init__(self, xy, Lx, Ly, r_bins, h = 0.5):
        self.nx, self.ny = int(round(Lx / h)), int(round(Ly / h))
        ix = np.floor(xy[:, 0] / Lx * self.nx).astype(np.int64) % self.nx
        iy = np.floor(xy[:, 1] / Ly * self.ny).astype(np.int64) % self.ny
        self.cell = ix * self.ny + iy
        kx = np.fft.fftfreq(self.nx, 1 / self.nx) * Lx / self.nx      # displacement of each grid offset
        ky = np.fft.fftfreq(self.ny, 1 / self.ny) * Ly / self.ny
        b = np.digitize(np.hypot(kx[:, None], ky[None, :]).ravel(), r_bins) - 1
        self.valid = (b >= 0) & (b < len(r_bins) - 1)
        self.bin = b[self.valid]
        self.n_bins = len(r_bins) - 1
        self.n_pairs = self._sum(self.autocorrelation(np.ones(len(xy))))

    def fft(self, w):
        g = np.bincount(self.cell, w.real, self.nx * self.ny) + 1j * np.bincount(self.cell, w.imag, self.nx * self.ny)
        return np.fft.fft2(g.reshape(self.nx, self.ny))

    def autocorrelation(self, w):
        F = self.fft(w)
        return np.fft.ifft2(F * F.conj()).real                # sum over pairs of Re(w_i conj(w_j)) per offset

    def _sum(self, c):
        return np.bincount(self.bin, c.ravel()[self.valid], self.n_bins)

    def average(self, w):
        '''
        <Re(w_i conj(w_j))> over the pairs at distance r, for each bin (NaN for empty bins)
        '''
        with np.errstate(invalid="ignore", divide="ignore"):
            return self._sum(self.autocorrelation(w)) / self.n_pairs


def bragg_vectors(grid, xy, Lx, Ly, n_atoms):
    '''
    The three reciprocal vectors of the first diffraction ring at 60 deg from each other: the
    brightest peak of |FFT(density)|^2 near |G| = 4 pi / (3 a_CC) (a_CC from the atom density), then
    the brightest peaks near its rotations by 60 and 120 deg. The grid wave vectors are exactly those
    of lattices commensurate with the periodic box, like the crystals of Graphene.py.
    '''
    S = np.abs(grid.fft(np.ones(n_atoms))) ** 2
    qx = 2 * np.pi * np.fft.fftfreq(grid.nx, Lx / grid.nx)
    qy = 2 * np.pi * np.fft.fftfreq(grid.ny, Ly / grid.ny)
    QX, QY = np.meshgrid(qx, qy, indexing="ij")
    a_cc = np.sqrt(Lx * Ly / n_atoms / (3 * np.sqrt(3) / 4))
    q_mag = 4 * np.pi / (3 * a_cc)

    def brightest(center, radius):
        near = np.hypot(QX - center[0], QY - center[1]) < radius
        i = np.argmax(np.where(near, S, -1))
        return np.array([QX.flat[i], QY.flat[i]])

    ring = np.abs(np.hypot(QX, QY) - q_mag) < 0.1 * q_mag
    G = np.array([QX[ring], QY[ring]])[:, np.argmax(S[ring])]
    out = [G]
    for angle in (np.pi / 3, 2 * np.pi / 3):
        c, s = np.cos(angle), np.sin(angle)
        out.append(brightest([c * G[0] - s * G[1], s * G[0] + c * G[1]], 0.05 * q_mag))
    return np.array(out)


def correlations(xyz, bonds, Lx, Ly, r_bins, h = 0.5):
    '''
    Correlation functions of one crystal
    Outputs:
        G6 : <Re(psi6_i conj(psi6_j))> at distance r
        GT : <Re(rho_i conj(rho_j))> at distance r with rho = translational_field, averaged over the
             three first-ring Bragg vectors G (1 at all r for a perfect crystal)
        psi : psi6 of each atom
        G : Bragg vectors used (3, 2)
    '''
    xy = xyz[:, :2]
    psi = psi6(xyz, bonds, Lx, Ly)
    grid = PairGrid(xy, Lx, Ly, r_bins, h)
    G = bragg_vectors(grid, xy, Lx, Ly, len(xy))
    GT = np.mean([grid.average(translational_field(xyz, bonds, Lx, Ly, g)) for g in G], axis=0)
    return grid.average(psi), GT, psi, G


def power_law_exponent(r, C, r_min, r_max):
    '''
    eta of C(r) ~ r^-eta, least squares in log-log over r_min <= r <= r_max (positive values only)
    '''
    m = (r >= r_min) & (r <= r_max) & (C > 0)
    return -np.polyfit(np.log(r[m]), np.log(C[m]), 1)[0] if m.sum() >= 3 else np.nan
