"""
Walking tardigrade: articulated deformation of the 3D refractive-index volume.

Rig (built once from the rest volume, cached in data/nanoct/rig.npz):
  * trunk: midline along the stack z axis. Bending moves whole cross-sections rigidly ("plane sections
    stay plane", like a bent beam), so organs keep their shape instead of being smeared.
  * legs: the 8 lobopods are the protrusions an 8 um ball cannot enter (morphological opening of the
    body). Each leg gets a base pivot, a rest axis, a rest length and a skinning weight that is 1 in the
    leg and fades into the trunk wall over 4 um.

Motion (deterministic function of time t, seconds):
  * follow-the-leader: the head walks along a path in the substrate plane and every body section at arc
    length s behind the head sits on the same path, so a turn starts at the head and travels backwards.
  * superimposed head motions: lateral searching sweeps and a dorsal head lift (pitch), both confined to
    the region in front of the neck.
  * gait (Nirody et al. 2021 PNAS 118:e2107289118; Anderson et al. 2024 PLOS ONE 19:e0310738):
    legs I-III tetrapod-like, swing waves run posterior -> anterior with ipsilateral phase lag 1/3,
    contralateral legs in antiphase, duty factor ~0.72. Legs IV step alternately with a shorter stroke.
    Stance: the claw is fixed to the substrate in the world frame while the body moves over it.
    Swing: the claw lifts, is carried forward on a minimum-jerk path and set down ahead.
  * each leg is posed by inverse kinematics: rotate the rest axis onto base->claw (minimal rotation,
    lobopods have no joints and do not twist) and telescope the distal part to the required length.

Resampling is a backward warp. The trunk maps are inverted exactly by projection onto the posed midline
(2D problems, independent of the third axis); the leg map is inverted by Newton iteration, preferring the
leg solution where a leg has swung into water.
"""
import os
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

VOX = 0.27  # um

# optional GPU backend for the warp (same switch as waveimage.py: PUMA_BACKEND=gpu|cpu|auto)
_cp = None
if os.environ.get("PUMA_BACKEND", "auto") != "cpu":
    try:
        import cupy as _cupy
        import cupyx.scipy.ndimage as _cnd
        if _cupy.cuda.runtime.getDeviceCount() > 0:
            _cp = _cupy
    except Exception:
        _cp = None


def _xp(a):
    return _cp.get_array_module(a) if _cp is not None else np


def _ndi(xp):
    return _cnd if xp is not np else ndimage


def _to_np(a):
    return a.get() if hasattr(a, "get") else a

# --------------------------------------------------------------------------- motion parameters
BODY_LEN_UM = 152.0
SPEED = 0.20 * BODY_LEN_UM          # um/s (Anderson 2024: 0.23 +- 0.08 BL/s; Nirody 2021: 0.48 BL/s)
PERIOD = 1.0                        # s per stride
SWING = 0.28                        # swing fraction legs I-III (duty factor 0.72)
SWING_IV = 0.36                     # legs IV: shorter stance, longer swing (Anderson 2024)
STROKE_IV = 0.6                     # legs IV stroke relative to legs I-III
LIFT_UM = 2.5                       # claw lift during swing
LEG_EXT = 1.12                      # living legs are a little longer than in the dried specimen
NECK_Z, HEAD_RAMP = 440.0, 130.0    # voxels: head motions act in front of the neck (pharynx at z 483-550)


def _smoothstep(x):
    x = x.clip(0.0, 1.0) if hasattr(x, "clip") else min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def _minjerk(x):
    x = np.clip(x, 0.0, 1.0)
    return x ** 3 * (10 - 15 * x + 6 * x * x)


def _rot_between(a, b):
    """Minimal rotation matrix taking unit vector a onto unit vector b (Rodrigues)."""
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.eye(3)
    k = v / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


# --------------------------------------------------------------------------- 2D bending map
class CurveMap:
    """Bending in one plane. Rest: midline b = b0, arc length s = a. Posed: (s, u = b - b0) ->
    P(s) + u n(s) with tangent t = (cos th, sin th), normal n = (-sin th, cos th)."""

    def __init__(self, s, th, s_ref, p_ref):
        self.s, self.th = s, th
        t = np.stack([np.cos(th), np.sin(th)], -1)
        cum = np.concatenate([[[0.0, 0.0]], np.cumsum(0.5 * (t[1:] + t[:-1]) * np.diff(s)[:, None], 0)])
        self.P = cum - np.array([np.interp(s_ref, s, cum[:, 0]), np.interp(s_ref, s, cum[:, 1])]) + p_ref
        self._tree = None

    def _at(self, s):
        th = np.interp(s, self.s, self.th)
        return th, np.interp(s, self.s, self.P[:, 0]), np.interp(s, self.s, self.P[:, 1])

    def forward(self, a, b, b0):
        th, pa, pb = self._at(a)
        u = b - b0
        return pa - u * np.sin(th), pb + u * np.cos(th)

    def inverse(self, A, B, b0):
        if self._tree is None:
            self._tree = cKDTree(self.P)
        shp = np.shape(A)
        A = np.ravel(A).astype(np.float64); B = np.ravel(B).astype(np.float64)
        _, k = self._tree.query(np.stack([A, B], -1))
        s = self.s[k].copy()
        for _ in range(3):              # Newton on the projection condition (Q - P(s)) . t(s) = 0
            th, pa, pb = self._at(s)
            s = s + (A - pa) * np.cos(th) + (B - pb) * np.sin(th)
        th, pa, pb = self._at(s)
        u = -(A - pa) * np.sin(th) + (B - pb) * np.cos(th)
        # reject points where the projection is ambiguous (beyond the centre of curvature)
        fa, fb = self.forward(s, b0 + u, b0)
        bad = np.hypot(fa - A, fb - B) > 0.5
        return s.reshape(shp), (b0 + u).reshape(shp), bad.reshape(shp)


# --------------------------------------------------------------------------- rig
def build_rig(dn, labels, cache=None):
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    body = dn > 0.006
    nz, ny, nx = body.shape
    # trunk midline (top view x, side view y), smoothed
    cnt = body.sum((1, 2))
    ok = cnt > 0
    zz = np.arange(nz)
    cx = np.array([np.nonzero(body[k])[1].mean() if cnt[k] else 0 for k in range(nz)])
    cy = np.array([np.nonzero(body[k])[0].mean() if cnt[k] else 0 for k in range(nz)])
    cx0 = float(np.median(cx[ok])); cy0 = float(np.median(cy[ok]))
    z_tail, z_head = float(zz[ok].min()), float(zz[ok].max())
    # leg lobes: body minus its opening with an 8 um ball (computed at 2x downsampling)
    b2 = body[::2, ::2, ::2]
    r = 15
    core = ndimage.distance_transform_edt(b2) > r
    opened = ndimage.distance_transform_edt(~core) <= r
    lobe2 = b2 & ~opened
    lab, n = ndimage.label(lobe2)
    sizes = ndimage.sum(lobe2, lab, range(1, n + 1))
    keep = 1 + np.argsort(sizes)[::-1][:8]
    # claw glands (label 11) identify and order the legs
    gl, ng = ndimage.label(labels == 11)
    gc = ndimage.center_of_mass(labels == 11, gl, range(1, ng + 1))
    gs = ndimage.sum(labels == 11, gl, range(1, ng + 1))
    glands = [np.array(c) for c, s in sorted(zip(gc, gs), key=lambda t: -t[1])[:8]]
    legs = []
    for g in glands:
        lc = [np.array(ndimage.center_of_mass(lab == k)) * 2 for k in keep]
        k = keep[int(np.argmin([np.linalg.norm(c - g) for c in lc]))]
        m2 = lab == k
        # full-resolution lobe and its interface with the trunk
        sl = ndimage.find_objects(m2.astype(np.uint8))[0]
        pad = 20
        box = tuple(slice(max(0, s.start * 2 - pad), min(N, s.stop * 2 + pad)) for s, N in zip(sl, body.shape))
        lobe = np.zeros([b.stop - b.start for b in box], bool)
        up = np.repeat(np.repeat(np.repeat(m2[sl], 2, 0), 2, 1), 2, 2)
        o = [s.start * 2 - b.start for s, b in zip(sl, box)]
        lobe[o[0]:o[0] + up.shape[0], o[1]:o[1] + up.shape[1], o[2]:o[2] + up.shape[2]] = \
            up[:lobe.shape[0] - o[0], :lobe.shape[1] - o[1], :lobe.shape[2] - o[2]]
        lobe &= body[box]
        trunk = body[box] & ~lobe
        iface = lobe & ndimage.binary_dilation(trunk, iterations=2)
        org = np.array([b.start for b in box], float)
        B = np.array(np.nonzero(iface)).mean(1) + org
        P = np.array(np.nonzero(lobe)).T + org
        a = P.mean(0) - B
        a /= np.linalg.norm(a)
        ell = float(np.percentile((P - B) @ a, 99))
        legs.append(dict(gland=g, base=B, axis=a, length=ell, box=box, lobe=lobe, trunk=trunk))
    # order: pair 0 = leg IV (posterior) ... pair 3 = leg I; side +1 = larger x
    legs.sort(key=lambda L: L["gland"][0])
    rig = dict(cx0=cx0, cy0=cy0, z_tail=z_tail, z_head=z_head, shape=np.array(body.shape))
    for i, L in enumerate(legs):
        pair, side = i // 2, (1 if L["gland"][2] > cx0 else -1)
        # skinning weight: 1 in the leg (+ a water margin so its soft surface moves along),
        # fading to 0 over 4 um into the trunk wall
        d_out = ndimage.distance_transform_edt(~L["lobe"])
        w = np.where(L["trunk"], 1 - d_out / 15.0, 1 - (d_out - 4.0) / 4.0)
        w = ndimage.gaussian_filter(np.clip(w, 0, 1).astype(np.float32), 1.0)
        b = L["box"]
        rig[f"leg{i}_w"] = w
        rig[f"leg{i}_org"] = np.array([s.start for s in b], float)
        rig[f"leg{i}_base"] = L["base"]; rig[f"leg{i}_axis"] = L["axis"]
        rig[f"leg{i}_len"] = np.array(L["length"]); rig[f"leg{i}_pair"] = np.array(pair)
        rig[f"leg{i}_side"] = np.array(side)
    if cache:
        np.savez_compressed(cache, **rig)
    return rig


# --------------------------------------------------------------------------- walker
class Walker:
    """pose(t) -> trunk maps + leg transforms; warp(vols, t) -> deformed volumes on a canvas."""

    def __init__(self, dn, labels, cache=None, canvas=(720, 720), turn_deg=55.0, turn_at_um=40.0,
                 turn_len_um=55.0, sweep_deg=20.0, lift_deg=-12.0):
        self.rig = rig = build_rig(dn, labels, cache)
        self.shape = tuple(int(v) for v in rig["shape"])
        self.cx0, self.cy0 = float(rig["cx0"]), float(rig["cy0"])
        self.z_head = float(rig["z_head"]); self.z_tail = float(rig["z_tail"])
        self.canvas = canvas
        self.legs = []
        for i in range(8):
            self.legs.append(dict(w=rig[f"leg{i}_w"], org=rig[f"leg{i}_org"], B=rig[f"leg{i}_base"],
                                  a=rig[f"leg{i}_axis"], ell=float(rig[f"leg{i}_len"]),
                                  pair=int(rig[f"leg{i}_pair"]), side=int(rig[f"leg{i}_side"])))
        nz = self.shape[0]
        self.s = np.arange(-200.0, nz + 400.0, 0.5)
        self.v = SPEED / VOX                                    # voxels / s
        self.turn = (np.radians(turn_deg), turn_at_um / VOX, turn_len_um / VOX)
        self.sweep, self.lift = np.radians(sweep_deg), np.radians(lift_deg)
        for L in self.legs:
            # nominal claw position in the rest frame: extended leg, claw on the substrate
            tip = L["B"] + LEG_EXT * L["ell"] * L["a"]
            L["tip_nom"] = tip
            L["ground_y"] = tip[1]
            if L["pair"] == 0:
                L["swing"], L["stroke"], L["off"] = SWING_IV, STROKE_IV, 2 / 3 + (0.5 if L["side"] > 0 else 0)
            else:
                L["swing"], L["stroke"] = SWING, 1.0
                L["off"] = (L["pair"] - 1) / 3 + (0.5 if L["side"] > 0 else 0)
        self._cam_cache = {}
        # path table H(sigma), arc-length parameterised; H(sigma) = (sigma, cx0) before the turn
        self._sig = np.arange(-2000.0, 6000.0, 0.5)
        hh = self._path_heading(self._sig)
        t2 = np.stack([np.cos(hh), np.sin(hh)], -1)
        cum = np.concatenate([[[0.0, 0.0]], np.cumsum(0.5 * (t2[1:] + t2[:-1]) * 0.5, 0)])
        self._H = cum - cum[np.searchsorted(self._sig, 0.0)] + np.array([0.0, self.cx0])

    # ---- trunk pose
    def _path_heading(self, sig):
        ang, at, ln = self.turn
        x = np.clip((sig - (self.z_head + at)) / ln, 0, 1)
        return ang * _smoothstep(x)

    def head_sweep(self, t):
        return self.sweep * np.sin(2 * np.pi * t / 1.2)

    def head_lift(self, t):
        return self.lift * np.exp(-0.5 * ((t - 2.2) / 0.45) ** 2)

    def trunk(self, t):
        s = self.s
        sig_head = self.z_head + self.v * t
        sig = sig_head - (self.z_head - s)
        hr = _smoothstep((s - NECK_Z) / HEAD_RAMP)
        th = self._path_heading(sig) + self.head_sweep(t) * hr
        # world position of the mid-body on the path (path = straight along +z, then the turn)
        s_ref = 0.5 * (self.z_tail + NECK_Z)
        sig_ref = sig_head - (self.z_head - s_ref)
        p_ref = np.array([np.interp(sig_ref, self._sig, self._H[:, 0]), np.interp(sig_ref, self._sig, self._H[:, 1])])
        lat = CurveMap(s, th, s_ref, p_ref)
        psi = self.head_lift(t) * _smoothstep((s - NECK_Z) / HEAD_RAMP)
        pit = CurveMap(s, psi, 0.0, np.array([0.0, self.cy0]))
        return lat, pit

    def world(self, x, maps):
        """rest point(s) (..., 3) [after leg map] -> world (z, y, x)."""
        lat, pit = maps
        z1, y1 = pit.forward(x[..., 0], x[..., 1], self.cy0)
        zw, xw = lat.forward(z1, x[..., 2], self.cx0)
        return np.stack([zw, y1, xw], -1)

    def camera(self, t):
        """the observer follows the animal with the stage, lagging ~0.6 s behind the mid-body."""
        key = round(t, 4)
        if key not in self._cam_cache:
            ts = t - np.linspace(0, 1.2, 13)
            w = np.exp(-np.linspace(0, 1.2, 13) / 0.6)
            ps = []
            for tt in ts:
                lat, _ = self.trunk(tt)
                mid = 0.5 * (self.z_tail + self.z_head)
                ps.append(np.array(lat.forward(np.array(mid), np.array(self.cx0), self.cx0)))
            self._cam_cache[key] = (np.array(ps) * w[:, None]).sum(0) / w.sum()
        return self._cam_cache[key]

    # ---- legs
    def _claw_world(self, L, t_td):
        """claw position for the stance that starts at touchdown time t_td."""
        maps = self.trunk(t_td)
        nom = self.world(L["tip_nom"], maps)
        b0 = self.world(L["B"], maps)
        b1 = self.world(L["B"] + np.array([1.0, 0, 0]), maps)
        tang = (b1 - b0); tang[1] = 0; tang /= np.linalg.norm(tang)
        stance_len = self.v * PERIOD * (1 - L["swing"]) * L["stroke"]
        c = nom + 0.5 * stance_len * tang
        c[1] = L["ground_y"]
        return c

    def claw(self, L, t):
        ph = (t / PERIOD - L["off"]) % 1.0
        sw = L["swing"]
        t_cycle = t - ph * PERIOD                     # liftoff of the current cycle
        if ph < sw:                                    # swing
            tau = ph / sw
            c0 = self._claw_world(L, t_cycle - (1 - sw) * PERIOD)
            c1 = self._claw_world(L, t_cycle + sw * PERIOD)
            c = c0 + (c1 - c0) * _minjerk(tau)
            c[1] = L["ground_y"] - LIFT_UM / VOX * np.sin(np.pi * tau)
            return c, True
        return self._claw_world(L, t_cycle + sw * PERIOD), False

    def leg_pose(self, L, t, maps):
        """IK: find bend vector omega (perpendicular to the rest axis) and telescoping factor lam so
        that the bent leg's claw lands on the target claw position."""
        target, swinging = self.claw(L, t)
        a = L["a"]
        e1 = np.cross(a, [0.0, 1.0, 0.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(a, e1)
        tip0 = (L["B"] + L["ell"] * a)[None]

        def tip_world(p):
            om = p[0] * e1 + p[1] * e2
            return self.world(self._leg_apply(L, om, p[2], tip0)[0], maps)

        # start from the straight-chord solution, then Newton on the bent leg
        J = np.zeros((3, 3)); b0 = self.world(L["B"], maps)
        for k in range(3):
            e = np.zeros(3); e[k] = 1.0
            J[:, k] = self.world(L["B"] + e, maps) - b0
        d = np.linalg.solve(J, target - b0)
        R = _rot_between(a, d / np.linalg.norm(d))
        ang = np.arccos(np.clip(np.dot(a, d / np.linalg.norm(d)), -1, 1))
        k_ax = np.cross(a, d); k_ax = k_ax / (np.linalg.norm(k_ax) + 1e-12)
        p = np.array([ang * np.dot(k_ax, e1), ang * np.dot(k_ax, e2), np.linalg.norm(d) / L["ell"]])
        for _ in range(6):
            f0 = tip_world(p)
            Jp = np.zeros((3, 3))
            for k in range(3):
                dp = np.zeros(3); dp[k] = 1e-3
                Jp[:, k] = (tip_world(p + dp) - f0) / 1e-3
            p = p + np.linalg.lstsq(Jp, target - f0, rcond=None)[0]
            p[2] = np.clip(p[2], 0.75, 1.7)
        return p[0] * e1 + p[1] * e2, p[2], swinging

    # ---- warp
    def _bend(self, L, al):
        """fraction of the bend applied at axial position al: 0 at the base, 1 over the distal fifth
        (claw region stays rigid)."""
        return _smoothstep(al / (0.8 * L["ell"]))

    @staticmethod
    def _rotate(v, om, frac):
        xp = _xp(v)
        ang = float(np.linalg.norm(om))
        if ang < 1e-9:
            return v
        k = xp.asarray(om / ang, dtype=v.dtype)
        phi = frac * ang
        c, s = xp.cos(phi)[:, None], xp.sin(phi)[:, None]
        return v * c + xp.cross(k[None], v) * s + k[None] * (v @ k)[:, None] * (1 - c)

    def _leg_apply(self, L, om, lam, x):
        xp = _xp(x)
        B = xp.asarray(L["B"], dtype=x.dtype); a = xp.asarray(L["a"], dtype=x.dtype)
        r = x - B
        al = r @ a
        r = r + (lam - 1) * xp.maximum(al, 0)[:, None] * a[None]
        return B + self._rotate(r, om, self._bend(L, al))

    def _leg_unapply(self, L, om, lam, y):
        """approximate inverse (full bend) used as Newton start for the distal leg."""
        xp = _xp(y)
        B = xp.asarray(L["B"], dtype=y.dtype); a = xp.asarray(L["a"], dtype=y.dtype)
        v = self._rotate(y - B, -om, xp.ones(len(y), y.dtype))
        al = v @ a
        v = v + (xp.where(al > 0, al / lam, al) - al)[:, None] * a[None]
        return B + v

    def _w(self, L, x):
        xp = _xp(x)
        wv = L["w_gpu"] if xp is not np else L["w"]
        c = (x - xp.asarray(L["org"], dtype=x.dtype)).T
        return _ndi(xp).map_coordinates(wv, c, order=1, mode="constant", cval=0.0)

    def _leg_invert(self, L, R, lam, q):
        """solve F(x) = x + w(x) (A(x) - x) = q. Returns x and a mask of points that are 'empty'
        (vacated by the leg: no preimage)."""
        xp = _xp(q)

        def F(x):
            w = self._w(L, x)
            return x + w[:, None] * (self._leg_apply(L, R, lam, x) - x), w

        def jac_inv(x, fx):
            M = xp.empty((len(x), 3, 3), xp.float32)
            for k in range(3):                         # finite-difference Jacobian per point
                dx = xp.zeros(3, xp.float32); dx[k] = 0.5
                M[:, :, k] = (F(x + dx)[0] - fx) / 0.5
            return xp.linalg.inv(M + 1e-3 * xp.eye(3, dtype=xp.float32))   # regularised at folds

        def newton(x):
            # chord method: Jacobian refreshed only at iterations 0 and 3
            for it in range(7):
                fx, w = F(x)
                if it in (0, 3):
                    Mi = jac_inv(x, fx)
                x = x + (Mi @ (q_ - fx)[..., None])[..., 0]
            fx, w = F(x)
            return x, w, xp.linalg.norm(fx - q_, axis=1) < 0.35

        out = q.copy()
        empty = xp.zeros(len(q), bool)
        wq = self._w(L, q)
        xa0 = self._leg_unapply(L, R, lam, q)
        wa0 = self._w(L, xa0)
        ia = xp.nonzero(wa0 > 1e-3)[0]
        ib = xp.nonzero(wq > 1e-3)[0]
        q_ = q[ia]; xa, wa, oka = newton(xa0[ia])
        q_ = q[ib]; xb, wb, okb = newton(q[ib].copy())
        # B (identity start) candidates: keep converged, else the point was vacated by the leg
        out[ib[okb]] = xb[okb]
        empty[ib[~okb]] = True
        # A (leg start) candidates win where converged and the leg is there (the leg occludes)
        sel = oka & (wa > 0.5)
        out[ia[sel]] = xa[sel]
        empty[ia[sel]] = False
        return out, empty

    def warp(self, vols, t, y_chunk=12):
        """vols: list of rest volumes (z, y, x). Returns deformed volumes on the canvas
        (Cz, y, Cx) in the observer's (stage) frame."""
        maps = self.trunk(t)
        lat, pit = maps
        poses = [self.leg_pose(L, t, maps) for L in self.legs]
        Cz, Cx = self.canvas
        cam = self.camera(t)
        zc, xc = np.meshgrid(np.arange(Cz, dtype=np.float64), np.arange(Cx, dtype=np.float64), indexing="ij")
        Zw = zc - Cz / 2 + cam[0]
        Xw = xc - Cx / 2 + cam[1]
        z1, xr, bad = lat.inverse(Zw, Xw, self.cx0)
        z1[bad] = -1e4
        # pitch inverse tabulated on a 2D (z1, y) grid
        nz, ny, nx = self.shape
        zg = np.arange(-64.0, nz + 64.0)
        Zg, Yg = np.meshgrid(zg, np.arange(ny, dtype=np.float64), indexing="ij")
        pz, py, pbad = pit.inverse(Zg, Yg, self.cy0)
        pz[pbad] = -1e4
        # leg query boxes in the rest frame (rest extent + posed extent)
        boxes = []
        for L, (R, lam, _) in zip(self.legs, poses):
            org = L["org"]; shp = np.array(L["w"].shape)
            corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) * (shp - 1) + org
            posed = self._leg_apply(L, R, lam, corners)
            lo = np.minimum(corners.min(0), posed.min(0)) - 8
            hi = np.maximum(corners.max(0), posed.max(0)) + 8
            boxes.append((lo, hi))
        xp = _cp if _cp is not None else np
        nd = _ndi(xp)
        if xp is not np:
            for L in self.legs:
                if "w_gpu" not in L:
                    L["w_gpu"] = xp.asarray(L["w"])
            y_chunk = max(y_chunk, 32)
        vols_x = [xp.asarray(v, dtype=xp.float32) for v in vols]
        pz_x, py_x = xp.asarray(pz, dtype=xp.float32), xp.asarray(py, dtype=xp.float32)
        boxes = [(xp.asarray(lo, dtype=xp.float32), xp.asarray(hi, dtype=xp.float32)) for lo, hi in boxes]
        outs = [np.zeros((Cz, ny, Cx), np.float32) for _ in vols]
        inside = (z1 > -64) & (z1 < nz + 63) & (xr > -40) & (xr < nx + 40)
        zi, xi = np.nonzero(inside)
        z1i = xp.asarray(z1[zi, xi], dtype=xp.float32); xri = xp.asarray(xr[zi, xi], dtype=xp.float32)
        zi_x, xi_x = xp.asarray(zi), xp.asarray(xi)
        for y0 in range(0, ny, y_chunk):
            ys = xp.arange(y0, min(ny, y0 + y_chunk), dtype=xp.float32)
            Z1 = xp.repeat(z1i[None], len(ys), 0).ravel()
            Y = xp.repeat(ys[:, None], len(zi), 1).ravel()
            X = xp.repeat(xri[None], len(ys), 0).ravel()
            cz = Z1 + 64.0
            zr = nd.map_coordinates(pz_x, xp.stack([cz, Y]), order=1, mode="constant", cval=-1e4)
            yr = nd.map_coordinates(py_x, xp.stack([cz, Y]), order=1, mode="nearest")
            q = xp.stack([zr, yr, X], 1)
            empty = xp.zeros(len(q), bool)
            for L, (R, lam, _), (lo, hi) in zip(self.legs, poses, boxes):
                m = xp.nonzero(xp.all((q >= lo) & (q <= hi), 1))[0]
                if len(m):
                    x, e = self._leg_invert(L, R, lam, q[m])
                    q[m] = x
                    empty[m] |= e
            q[empty] = -1e4
            for o, v in zip(outs, vols_x):
                vals = nd.map_coordinates(v, q.T, order=1, mode="constant", cval=0.0)
                blk = xp.zeros((len(ys), Cz, Cx), xp.float32)
                blk[:, zi_x, xi_x] = vals.reshape(len(ys), len(zi))
                o[:, y0:y0 + len(ys), :] = _to_np(blk).transpose(1, 0, 2)
        if xp is not np:
            del vols_x, pz_x, py_x
            xp.get_default_memory_pool().free_all_blocks()
        return outs

    def diagnostics(self, t):
        """world-frame positions for plots: midline, leg bases and claws, swing flags."""
        maps = self.trunk(t)
        mid = np.stack([np.linspace(self.z_tail, self.z_head, 60), np.full(60, self.cy0),
                        np.full(60, self.cx0)], -1)
        out = dict(mid=self.world(mid, maps), cam=self.camera(t), legs=[])
        for L in self.legs:
            R, lam, sw = self.leg_pose(L, t, maps)
            c, _ = self.claw(L, t)
            out["legs"].append(dict(base=self.world(L["B"], maps), claw=c, swing=sw, lam=lam,
                                    pair=L["pair"], side=L["side"]))
        return out
