"""
Wavefront (OPD) of the traced system, reconstructed from TRANSVERSE RAY ABERRATIONS.

Why not path lengths: Optiland's built-in Wavefront and its per-ray optical-path output gave
defocus/spherical terms several times too large for this system (finite object in water,
near-telecentric stop, objectNA); both were fine on simple test lenses. The ray geometry itself is
verified (magnification -19.9, image shift = m^2 per knob step, NA' = NA/m, landing heights).

Method (Hopkins' canonical relation, exact for a reference point Q on the image surface):
    dW/dp = -dx ,  dW/dq = -dy
with p, q = n' * (L, M) the optical direction cosines of each ray in image space (relative to the
chief ray), and (dx, dy) the ray's landing offset from Q, measured in the plane through Q normal
to the chief ray. W is fitted as a polynomial in (p, q) whose gradient matches the ray data in a
least-squares sense, then expressed at each ray's OBJECT-space pupil coordinate
rho = n_obj * (L0, M0) / NA, which is exactly the coordinate of the angular-spectrum pupil.
Sign: W > 0 = longer optical path (wave lags), i.e. a positive W020 means focus beyond Q.
"""
import numpy as np


def _poly_terms(p, q, order):
    """Monomials p^i q^j (1 <= i+j <= order) and their derivatives."""
    T, Tp, Tq = [], [], []
    for n in range(1, order + 1):
        for i in range(n + 1):
            j = n - i
            T.append(p ** i * q ** j)
            Tp.append(i * p ** (i - 1) * q ** j if i > 0 else np.zeros_like(p))
            Tq.append(j * p ** i * q ** (j - 1) if j > 0 else np.zeros_like(p))
    return np.stack(T, 1), np.stack(Tp, 1), np.stack(Tq, 1)


def wavefront(o, field_norm, lam, n_side=31, na=0.40, order=10):
    g = np.linspace(-1, 1, n_side)
    PX, PY = np.meshgrid(g, g, indexing="ij")
    m = PX ** 2 + PY ** 2 <= 1.0
    px = np.concatenate([[0.0], PX[m]]); py = np.concatenate([[0.0], PY[m]])
    rays = o.trace_generic(Hx=0.0, Hy=field_norm, Px=px, Py=py, wavelength=lam)
    sg = o.surface_group
    n_obj = float(np.ravel(sg.surfaces[0].material_post.n(lam))[0])
    n_img = float(np.ravel(sg.surfaces[-2].material_post.n(lam))[0])
    L0, M0 = np.ravel(sg.L[0]), np.ravel(sg.M[0])
    x, y, z = np.ravel(rays.x), np.ravel(rays.y), np.ravel(rays.z)
    # NOTE: Optiland refracts AT the image surface into the image surface's own (default: air)
    # medium, so rays.L/M/N there are already n'*sin. Take the true image-space directions from
    # the last real refraction (surface -2) instead.
    D = np.stack([np.ravel(sg.L[-2]), np.ravel(sg.M[-2]), np.ravel(sg.N[-2])], 1)
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    ok = np.isfinite(x) & np.isfinite(D).all(1)
    if hasattr(rays, "i"):
        ok &= np.ravel(rays.i) > 0
    Q = np.array([x[0], y[0], z[0]])
    c = D[0]                                    # chief direction = reference plane normal
    # local frame: e1, e2 perpendicular to the chief ray
    e1 = np.cross([0.0, 1.0, 0.0], c); e1 /= np.linalg.norm(e1)
    e2 = np.cross(c, e1)
    P = np.stack([x, y, z], 1)
    # extend each ray from its image-surface point to the plane through Q normal to c
    t = ((Q - P) @ c) / (D @ c)
    Pp = P + t[:, None] * D
    dx = (Pp - Q) @ e1
    dy = (Pp - Q) @ e2
    p = n_img * (D @ e1)
    q = n_img * (D @ e2)
    sel = ok.copy(); sel[0] = True
    p_, q_, dx_, dy_ = p[sel], q[sel], dx[sel], dy[sel]
    s = max(np.abs(p_).max(), np.abs(q_).max())       # normalise for conditioning
    T, Tp, Tq = _poly_terms(p_ / s, q_ / s, order)
    A = np.concatenate([Tp / s, Tq / s], 0)
    b = np.concatenate([-dx_, -dy_])
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    W = (T @ coef) * 1000.0                             # mm -> um
    W -= W[0]                                           # chief ray = 0
    rho_x = n_obj * L0[sel] / na
    rho_y = n_obj * M0[sel] / na
    keep = np.ones(W.size, bool); keep[0] = False
    resid = np.sqrt(np.mean((A @ coef - b) ** 2)) * 1000
    return rho_x[keep], rho_y[keep], W[keep], resid


def fit_radial(rx, ry, W):
    r2 = rx ** 2 + ry ** 2
    A = np.stack([np.ones_like(r2), r2, r2 ** 2, r2 ** 3], 1)
    c, *_ = np.linalg.lstsq(A, W, rcond=None)
    return c
