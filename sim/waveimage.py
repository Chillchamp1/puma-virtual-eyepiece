"""
Partially coherent brightfield image formation (scalar wave optics).

Abbe source-point integration with Koehler illumination:
  for every wavelength lambda and every condenser source point s:
    1. tilted plane wave enters the specimen from below
    2. multi-slice beam propagation through the 3D complex refractive-index volume
       (phase screen exp(i k0 dn dz) * amplitude exp(-mu dz / 2), then exact angular-spectrum
       propagation over dz in the surrounding water)
    3. angular-spectrum (back)propagation from the specimen top to the focal plane
    4. objective pupil: circ(NA) * exp(i 2 pi W(pupil; lambda, field) / lambda), W from ray trace
    5. |field|^2 added incoherently with weight S(lambda) * (source point weight)
Colour: spectral image -> CIE 1931 XYZ (Wyman et al. 2013 analytic fit) -> linear sRGB.
"""
import os
import numpy as np
import scipy.fft as sfft

WORKERS = os.cpu_count() or 6

# Optional GPU backend: set PUMA_BACKEND=gpu|cpu|auto (default auto = GPU if CuPy + CUDA device found).
_cp = None
if os.environ.get("PUMA_BACKEND", "auto") != "cpu":
    try:
        import cupy as _cupy
        if _cupy.cuda.runtime.getDeviceCount() > 0:
            _cp = _cupy
    except Exception:
        _cp = None
    if os.environ.get("PUMA_BACKEND") == "gpu" and _cp is None:
        raise RuntimeError("PUMA_BACKEND=gpu but CuPy/CUDA is not available")
BACKEND = "gpu" if _cp is not None else "cpu"


# ---------------------------------------------------------------- spectra / colour
def led_6500k(lam_um):
    """Phosphor-converted white LED (InGaN 450 nm pump + broad YAG:Ce emission), ~6500 K."""
    l = lam_um * 1000
    return 1.00 * np.exp(-0.5 * ((l - 452) / 10.5) ** 2) + 0.62 * np.exp(-0.5 * ((l - 555) / 52) ** 2) \
        + 0.18 * np.exp(-0.5 * ((l - 625) / 38) ** 2)


def cie_xyz(lam_um):
    """CIE 1931 2-deg colour matching functions, multi-lobe Gaussian fit (Wyman, Sloan, Shirley 2013)."""
    l = lam_um * 1000
    def g(x, mu, s1, s2):
        s = np.where(x < mu, s1, s2)
        return np.exp(-0.5 * ((x - mu) / s) ** 2)
    x = 1.056 * g(l, 599.8, 37.9, 31.0) + 0.362 * g(l, 442.0, 16.0, 26.7) - 0.065 * g(l, 501.1, 20.4, 26.2)
    y = 0.821 * g(l, 568.8, 46.9, 40.5) + 0.286 * g(l, 530.9, 16.3, 31.1)
    z = 1.217 * g(l, 437.0, 11.8, 36.0) + 0.681 * g(l, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], -1)


XYZ2RGB = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])


def spectral_to_srgb(images, lams, weights):
    """images: list of 2D intensity images (background ~1); returns linear sRGB, white-balanced
    to the illuminated empty slide (the eye adapts to the lamp's white)."""
    cmf = cie_xyz(np.asarray(lams))
    xyz = sum(w * im[..., None] * c for im, c, w in zip(images, cmf, weights))
    white = sum(w * c for c, w in zip(cmf, weights))
    rgb = xyz @ XYZ2RGB.T
    rgb_w = white @ XYZ2RGB.T
    return rgb / rgb_w


def encode_srgb(lin, exposure=1.0):
    x = np.clip(lin * exposure, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


# ---------------------------------------------------------------- geometry helpers
class Grid:
    def __init__(self, n, dx_um):
        self.n, self.dx = n, dx_um
        f = sfft.fftfreq(n, dx_um).astype(np.float32)
        self.fx, self.fy = np.meshgrid(f, f, indexing="ij")
        self.f2 = self.fx ** 2 + self.fy ** 2
        x = (np.arange(n) - n // 2) * dx_um
        self.x, self.y = np.meshgrid(x, x, indexing="ij")
        self.df = 1.0 / (n * dx_um)

    def propagator(self, dz_um, n_med, lam_um):
        kz2 = (n_med / lam_um) ** 2 - self.f2
        prop = kz2 > 0
        kz = np.sqrt(np.where(prop, kz2, 0.0))
        return np.where(prop, np.exp(2j * np.pi * dz_um * kz), 0).astype(np.complex64)


def source_points(na_c, lam_um, grid, n_rings=4):
    """Koehler condenser aperture sampled on concentric rings, snapped to the FFT frequency lattice
    (keeps the tilted waves periodic on the grid). Returns list of (ix, iy, weight)."""
    fmax = na_c / lam_um
    pts = [(0.0, 0.0, 1.0)]
    for r in range(1, n_rings + 1):
        rad = fmax * r / n_rings
        m = 6 * r
        area = np.pi * ((rad + fmax / (2 * n_rings)) ** 2 - (rad - fmax / (2 * n_rings)) ** 2)
        if r == n_rings:
            area = np.pi * (fmax ** 2 - (rad - fmax / (2 * n_rings)) ** 2)
        for k in range(m):
            a = 2 * np.pi * (k + 0.5 * (r % 2)) / m
            pts.append((rad * np.cos(a), rad * np.sin(a), area / m))
    pts[0] = (0.0, 0.0, np.pi * (fmax / (2 * n_rings)) ** 2)
    out = []
    for fx, fy, w in pts:
        out.append((int(round(fx / grid.df)), int(round(fy / grid.df)), w))
    tot = sum(p[2] for p in out)
    return [(a, b, w / tot) for a, b, w in out]


# ---------------------------------------------------------------- core
def simulate_wavelength(specimen_slab, dz_um, grid, lam_um, n_med, na_obj, na_c, focus_um,
                        pupil_phase=None, n_rings=4, place=(0, 0)):
    """specimen_slab: (dn[z, X, Y], mu[z, X, Y]) already sampled at grid.dx laterally and dz axially,
    in propagation order (first slice = side facing the condenser).
    focus_um: list of focal-plane heights measured from the TOP (last) slice, negative = inside.
    pupil_phase: callable(rho_x, rho_y) -> W in um (normalised pupil coords) or None (ideal lens),
                 or a list with one entry per focus plane.
    Returns list of intensity images (one per focus), normalised so the empty slide = 1."""
    dn, mu = specimen_slab
    nz, sx, sy = dn.shape
    n = grid.n
    ox = (n - sx) // 2 + place[0]
    oy = (n - sy) // 2 + place[1]
    k0 = 2 * np.pi / lam_um
    H = grid.propagator(dz_um, n_med, lam_um)
    # complex transmission of each slice (only the specimen footprint differs from 1)
    trans = (np.exp(1j * k0 * dn * dz_um) * np.exp(-0.5 * mu * dz_um)).astype(np.complex64)
    # pupil in frequency space
    rho_x = grid.fx * lam_um / na_obj
    rho_y = grid.fy * lam_um / na_obj
    inside = (rho_x ** 2 + rho_y ** 2) <= 1.0
    phases = pupil_phase if isinstance(pupil_phase, (list, tuple)) else [pupil_phase] * len(focus_um)
    Ps = []
    for ph in phases:
        P = inside.astype(np.complex64)
        if ph is not None:
            W = ph(rho_x, rho_y)
            P = (P * np.exp(2j * np.pi * W / lam_um)).astype(np.complex64)
        Ps.append(P)
    backs = [grid.propagator(fz, n_med, lam_um) * P for fz, P in zip(focus_um, Ps)]
    if _cp is not None:
        return _run_gpu(trans, H, backs, grid, lam_um, na_c, n_rings, nz, sx, sy, ox, oy, n)
    out = [np.zeros((n, n), np.float32) for _ in focus_um]
    for ix, iy, w in source_points(na_c, lam_um, grid, n_rings):
        fx, fy = ix * grid.df, iy * grid.df
        E = np.exp(2j * np.pi * (fx * grid.x + fy * grid.y)).astype(np.complex64)
        for j in range(nz):
            E[ox:ox + sx, oy:oy + sy] *= trans[j]
            E = sfft.ifft2(sfft.fft2(E, workers=WORKERS) * H, workers=WORKERS)
        Ehat = sfft.fft2(E, workers=WORKERS)
        for o, B in zip(out, backs):
            img = sfft.ifft2(Ehat * B, workers=WORKERS)
            o += w * (np.abs(img) ** 2).astype(np.float32)
    return out


def _run_gpu(trans, H, backs, grid, lam_um, na_c, n_rings, nz, sx, sy, ox, oy, n):
    """Same algorithm as the CPU loop, on a CUDA GPU via CuPy (complex64 cuFFT)."""
    cp = _cp
    trans_g = cp.asarray(trans)
    H_g = cp.asarray(H)
    backs_g = [cp.asarray(B) for B in backs]
    x_g = cp.asarray(grid.x, dtype=cp.float32)
    y_g = cp.asarray(grid.y, dtype=cp.float32)
    out = [cp.zeros((n, n), cp.float32) for _ in backs]
    for ix, iy, w in source_points(na_c, lam_um, grid, n_rings):
        fx, fy = ix * grid.df, iy * grid.df
        E = cp.exp(2j * np.pi * (fx * x_g + fy * y_g)).astype(cp.complex64)
        for j in range(nz):
            E[ox:ox + sx, oy:oy + sy] *= trans_g[j]
            E = cp.fft.ifft2(cp.fft.fft2(E) * H_g)
        Ehat = cp.fft.fft2(E)
        for o, B in zip(out, backs_g):
            img = cp.fft.ifft2(Ehat * B)
            o += cp.float32(w) * (cp.abs(img) ** 2).astype(cp.float32)
    res = [cp.asnumpy(o) for o in out]
    del trans_g, H_g, backs_g, out
    cp.get_default_memory_pool().free_all_blocks()
    return res
