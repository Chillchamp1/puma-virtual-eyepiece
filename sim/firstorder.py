"""
Fast first-order (single-scattering) brightfield image formation, for interactive views.

Same discretisation as waveimage.simulate_wavelength (slices of thickness dz, Koehler source points on
the FFT lattice, objective pupil, angular-spectrum defocus), but every slice scatters the *unperturbed*
illumination only once:
    perturbation per slice   o_m(x) = (i k0 dn - mu/2) dz
    scattered field          u1_s(nu) = P(nu+s) e^{i2pi kappa (N dz + f)} sum_m O_m(nu) e^{-i2pi kappa m dz},
                             kappa = eta(nu+s) - eta(s),   nu = object frequency (shifted by the source s)
    Born:   I = sum_s w_s |P(s)|^2 (1 + 2 Re(u1/u0))
    Rytov:  I = sum_s w_s |P(s)|^2 exp(2 Re(u1/u0))      (u1/u0 read as a complex phase)
The slice sum costs one complex multiply-add per pixel and slice instead of two FFTs.

RESULT (scripts/compare_firstorder.py, GTX 1080): only ~2.6x faster than multi-slice on the GPU, and
NOT usable for the tardigrade. Rytov matches multi-slice on a weak copy of the animal (delta-n x 0.01:
12 % rms error relative to the image contrast) but fails at delta-n x 0.05 (59 %); the real animal
(~10 rad accumulated phase, cuticle delta-n 0.13) is far outside single scattering. Kept as a
documented negative result; all published views use multi-slice.
"""
import numpy as np
import waveimage as wi

_cp = wi._cp


def simulate_wavelength(specimen_slab, dz_um, grid, lam_um, n_med, na_obj, na_c, focus_um,
                        pupil_phase=None, n_rings=4, place=(0, 0), model="rytov"):
    xp = _cp if _cp is not None else np
    fft = xp.fft
    dn, mu = specimen_slab
    nz, sx, sy = dn.shape
    n = grid.n
    ox = (n - sx) // 2 + place[0]
    oy = (n - sy) // 2 + place[1]
    k0 = 2 * np.pi / lam_um
    # slice spectra O_m(nu)
    O = xp.zeros((nz, n, n), xp.complex64)
    O[:, ox:ox + sx, oy:oy + sy] = xp.asarray((1j * k0 * dn - 0.5 * mu) * dz_um, dtype=xp.complex64)
    O = fft.fft2(O, axes=(1, 2))
    fx, fy = xp.asarray(grid.fx), xp.asarray(grid.fy)
    eta2 = (n_med / lam_um) ** 2 - (fx ** 2 + fy ** 2)
    eta = xp.sqrt(xp.maximum(eta2, 0)).astype(xp.float32)
    rho_x, rho_y = grid.fx * lam_um / na_obj, grid.fy * lam_um / na_obj
    inside = (rho_x ** 2 + rho_y ** 2) <= 1.0
    phases = pupil_phase if isinstance(pupil_phase, (list, tuple)) else [pupil_phase] * len(focus_um)
    Ps = []
    for ph in phases:
        P = inside.astype(np.complex64)
        if ph is not None:
            P = (P * np.exp(2j * np.pi * ph(rho_x, rho_y) / lam_um)).astype(np.complex64)
        Ps.append(xp.asarray(P))
    out = [xp.zeros((n, n), xp.float32) for _ in focus_um]
    for ix, iy, w in wi.source_points(na_c, lam_um, grid, n_rings):
        # quantities at nu + s: roll the frequency grids by -s
        eta_s = float(eta[ix % n, iy % n])
        eta_q = xp.roll(eta, (-ix, -iy), (0, 1))
        kap = eta_q - eta_s
        r = xp.exp(-2j * np.pi * kap * dz_um).astype(xp.complex64)
        A = xp.zeros((n, n), xp.complex64)
        for m in range(nz - 1, -1, -1):           # Horner: sum_m O_m r^m
            A = A * r + O[m]
        for o, P, f in zip(out, Ps, focus_um):
            Pq = xp.roll(P, (-ix, -iy), (0, 1))
            P0 = P[ix % n, iy % n]
            if abs(complex(P0)) < 1e-6:
                continue
            u1 = fft.ifft2(Pq / P0 * xp.exp(2j * np.pi * kap * (nz * dz_um + f)) * A)
            # u1 is the scattered field relative to the unscattered wave (both after the pupil)
            if model == "born":                    # linear term only (|u1|^2 without u2 is inconsistent)
                o += xp.float32(w * abs(complex(P0)) ** 2) * (1 + 2 * u1.real)
            else:
                o += xp.float32(w * abs(complex(P0)) ** 2) * xp.exp(2 * u1.real)
    res = [o.get() if xp is not np else o for o in out]
    if xp is not np:
        del O
        xp.get_default_memory_pool().free_all_blocks()
    return res
