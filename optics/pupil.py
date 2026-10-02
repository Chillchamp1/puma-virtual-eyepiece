"""
Pupil functions of the full PUMA beam path for the wave-optics image formation.

For a focus-knob setting (air gap between cover glass and objective) and each wavelength:
  1. d_lam  = depth in the water (below the cover glass) of the specimen plane that the system
              images sharply onto the retina at that wavelength  -> longitudinal colour in OBJECT space
  2. W_lam(rho) = residual wavefront for that plane (spherical aberration, spherochromatism, the
              eyepiece and the eye included), from transverse ray aberrations (optics/opd.py)
The wave simulation propagates the specimen field exactly (angular spectrum in water) to d_lam and
applies exp(i 2 pi W_lam / lam): defocus is never linearised.
The knob is set so that 555 nm (photopic peak) is sharp at the requested depth.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import griddata, RegularGridInterpolator
import system as S
import opd

EYE_RELIEF = 21.43 - 3.04   # cornea placed so the eye's entrance pupil sits in the Ramsden disc
LAM_FOCUS = 0.555
MAX_FIELD = 0.5             # mm object height = FN 20 field edge


def _sys(depth_mm, wd_mm, lam):
    return S.build(focus_depth_mm=depth_mm, wd_mm=wd_mm, eye_relief=EYE_RELIEF, wavelengths=(lam,),
                   fields_mm=(0.0, MAX_FIELD))


def _rms(W, rx, ry):
    A = np.stack([np.ones_like(rx), rx, ry], 1)
    c, *_ = np.linalg.lstsq(A, W, rcond=None)
    return float(np.sqrt(np.mean((W - A @ c) ** 2)))


def rms_at(depth_mm, wd_mm, lam, field_mm=0.0):
    rx, ry, W, _ = opd.wavefront(_sys(depth_mm, wd_mm, lam), field_mm / MAX_FIELD, lam, n_side=21, order=8)
    return _rms(W, rx, ry)


def _scan_refine(f, lo, hi, step):
    """Coarse scan + local bounded refinement (far from focus the RMS landscape bends over,
    so a global bounded search can be lured into a false minimum)."""
    xs = np.arange(lo, hi + step / 2, step)
    vals = [f(x) for x in xs]
    x0 = xs[int(np.argmin(vals))]
    r = minimize_scalar(f, bounds=(x0 - step, x0 + step), method="bounded", options={"xatol": 1e-7})
    return r.x, r.fun


def knob_for_depth(depth_mm, lam=LAM_FOCUS):
    # paraxial guess: 1.9233 mm focuses 5 um deep; deeper focus -> smaller air gap (~1/n per um)
    g = 1.9233 - (depth_mm - 0.005) / 1.334
    return _scan_refine(lambda wd: rms_at(depth_mm, wd, lam), g - 0.012, g + 0.012, 0.002)[0]


def sharp_depth(wd_mm, lam, guess_mm, field_mm=0.0):
    return _scan_refine(lambda d: rms_at(d, wd_mm, lam, field_mm), max(1e-6, guess_mm - 0.03), guess_mm + 0.03, 0.002)


class TracedPupil:
    def __init__(self, focus_depth_mm, lams, field_mm=0.0, n_grid=65):
        self.wd = knob_for_depth(focus_depth_mm)
        self.depth, self.maps, self.rms = {}, {}, {}
        g = np.linspace(-1, 1, n_grid)
        GX, GY = np.meshgrid(g, g, indexing="ij")
        self.g = g
        for lam in lams:
            d, r = sharp_depth(self.wd, lam, focus_depth_mm, field_mm)
            rx, ry, W, _ = opd.wavefront(_sys(d, self.wd, lam), field_mm / MAX_FIELD, lam, n_side=41, order=10)
            A = np.stack([np.ones_like(rx), rx, ry], 1)
            c, *_ = np.linalg.lstsq(A, W, rcond=None)
            W = W - c[0]          # keep tilt (lateral colour / distortion off-axis), drop piston
            Z = griddata((rx, ry), W, (GX, GY), method="cubic")
            Zn = griddata((rx, ry), W, (GX, GY), method="nearest")
            self.maps[lam] = np.where(np.isnan(Z), Zn, Z)
            self.depth[lam] = d
            self.rms[lam] = r

    def phase_fn(self, lam):
        key = min(self.maps, key=lambda k: abs(k - lam))
        itp = RegularGridInterpolator((self.g, self.g), self.maps[key], bounds_error=False, fill_value=0.0)

        def W(rx, ry):
            pts = np.stack([np.clip(rx, -1, 1).ravel(), np.clip(ry, -1, 1).ravel()], -1)
            return itp(pts).reshape(rx.shape).astype(np.float32)
        return W
