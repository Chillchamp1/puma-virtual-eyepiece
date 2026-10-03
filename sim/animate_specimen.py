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
V_WALK = 0.25                       # body lengths / s, first walking bout (Anderson 2024: 0.23 +- 0.08)
V_WALK2 = 0.28                      # second bout after the stop (cadence within ~15 % of the first)
STEP_UM = 40.0                      # stride length legs I-III, ~0.26 body lengths; period = STEP / speed
STEP_VAR = 0.15                     # stride-to-stride variation of the step length
PHASE_JITTER = 0.07                 # per-leg phase irregularity (cycles)
SWING = 0.28                        # swing fraction legs I-III (duty factor 0.72)
SWING_IV = 0.15                     # legs IV: mostly holding on (duty factor ~0.85)
STROKE_IV = 0.5                     # legs IV stroke relative to legs I-III
LIFT_UM = 5.0                       # claw lift during swing
REACH_OUT_UM = 5.0                  # claw path bows outward during swing
LEG_EXT = 1.3                       # living legs are longer than in the dried specimen (telescoped out)
HEAD_FRAC = 0.34                    # head motions are spread over the front third of the body
LENGTH_OSC = 0.05                   # global trunk length change at stride frequency
STOP_SHORTEN = 0.04                 # the trunk contracts a little while the animal stands
HEAD_RETRACT = 0.33                 # ... and the head retracts (front 30 % shortens by a third ~ 10 % BL)
HEAD_TELE = 0.07                    # anterior telescoping (head extends / retracts), ~0.8 s period
BULGE = 0.15                        # segmental widening between leg pairs, phased with their stance
WANDER_DEG = 18.0                   # slow path wander -> C / S body shapes (wavelength 1.3 body lengths)
STOP_SCAN_DEG = 32.0                # fast lateral head scan while standing (front 30 %)
SWING_RETRACT = 0.35                # legs shorten (telescope in) during swing
YAW_DEG = 7.0                       # lateral trunk yaw at stride frequency
TAIL_DEG = 5.0                      # caudal end swings with each leg-IV step
SEARCH_DEG = 24.0                   # head search sweep amplitude (irregular 1.5-3 s period)
LIFT_DEG = -28.0                    # dorsal head lift while the animal stops and probes
TURN_DEG = 45.0                     # head-led turn after the stop (radius ~1.3 body lengths)


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

    def __init__(self, s, th, s_ref, p_ref, stretch=1.0, width=1.0):
        """stretch: arc length scale of the posed midline; width: scale of the cross-section offset u."""
        self.s, self.th, self.w, self.st = s, th, width, stretch
        t = np.stack([np.cos(th), np.sin(th)], -1)
        st = np.broadcast_to(np.asarray(stretch, float), s.shape)
        ds = (0.5 * (st[1:] + st[:-1]) * np.diff(s))[:, None]
        cum = np.concatenate([[[0.0, 0.0]], np.cumsum(0.5 * (t[1:] + t[:-1]) * ds, 0)])
        self.P = cum - np.array([np.interp(s_ref, s, cum[:, 0]), np.interp(s_ref, s, cum[:, 1])]) + p_ref
        self._tree = None

    def _at(self, s):
        th = np.interp(s, self.s, self.th)
        return th, np.interp(s, self.s, self.P[:, 0]), np.interp(s, self.s, self.P[:, 1])

    def forward(self, a, b, b0):
        th, pa, pb = self._at(a)
        u = (b - b0) * self._width(a)
        return pa - u * np.sin(th), pb + u * np.cos(th)

    def _width(self, s):
        return np.interp(s, self.s, self.w) if np.ndim(self.w) else self.w

    def inverse(self, A, B, b0):
        if self._tree is None:
            self._tree = cKDTree(self.P)
        shp = np.shape(A)
        A = np.ravel(A).astype(np.float64); B = np.ravel(B).astype(np.float64)
        _, k = self._tree.query(np.stack([A, B], -1))
        s = self.s[k].copy()
        for _ in range(3):              # Newton on the projection condition (Q - P(s)) . t(s) = 0
            th, pa, pb = self._at(s)
            st = np.interp(s, self.s, self.st) if np.ndim(self.st) else self.st
            s = s + ((A - pa) * np.cos(th) + (B - pb) * np.sin(th)) / st
        th, pa, pb = self._at(s)
        u = (-(A - pa) * np.sin(th) + (B - pb) * np.cos(th)) / self._width(s)
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

    def __init__(self, dn, labels, cache=None, canvas=(720, 720)):
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
        blen = self.z_head - self.z_tail
        self.z_neck = self.z_head - HEAD_FRAC * blen
        self.z_mid = 0.5 * (self.z_tail + self.z_neck)
        self.step = STEP_UM / VOX
        for L in self.legs:
            a = L["a"]
            if L["pair"] == 0:                                # legs IV reach backwards (~40 deg)
                a = a + np.array([-0.6, 0.0, 0.0]); a = a / np.linalg.norm(a)
            tip = L["B"] + LEG_EXT * L["ell"] * a             # nominal claw position (rest frame)
            tip[2] = L["B"][2] + 0.65 * (tip[2] - L["B"][2])   # legs come out ventrolaterally, partly under the body
            L["tip_nom"] = tip
            L["ground_y"] = tip[1]
            if L["pair"] == 0:
                L["swing"], L["stroke"], L["off"] = SWING_IV, STROKE_IV, 2 / 3 + (0.5 if L["side"] > 0 else 0)
            else:
                L["swing"], L["stroke"] = SWING, 1.0
                L["off"] = (L["pair"] - 1) / 3 + (0.5 if L["side"] > 0 else 0)
            L["holds"] = {}
            L["seed"] = 1.7 * L["pair"] + (0.9 if L["side"] > 0 else 0.0)
        # ---- head trajectory, tabulated in time: speed, heading, position, arc length, gait phase
        dt = 1 / 240
        tg = np.arange(-15.0, 15.0, dt)
        v = self.speed(tg) * BODY_LEN_UM / VOX                        # voxels / s
        i0 = np.searchsorted(tg, 0.0)

        def cum(x):
            c = np.concatenate([[0.0], np.cumsum(0.5 * (x[1:] + x[:-1]) * dt)])
            return c - c[i0]
        self._t = tg
        self._sig = cum(v) + self.z_head
        self._sig_inc = self._sig + 1e-6 * np.arange(len(tg))            # strictly increasing for inversion
        # the path's direction is a function of distance walked, not of time: while the animal stands
        # still its path cannot bend (a time-based heading would put a kink into the path there)
        th = self.heading(self._sig - self.z_head, np.interp(5.0, tg, self._sig) - self.z_head)
        self._th = th
        self._stopped = 1 - np.clip(v / (V_WALK * BODY_LEN_UM / VOX), 0, 1)
        self.debris = self._make_debris()
        self._H = np.stack([cum(v * np.cos(th)) + self.z_head, cum(v * np.sin(th)) + self.cx0], -1)
        self._phi = (self._sig - self.z_head) / self.step
        self._phi_inc = self._phi + 1e-7 * np.arange(len(tg))
        self._v0 = V_WALK * BODY_LEN_UM / VOX

    # ---- behaviour timeline (seconds)
    @staticmethod
    def speed(t):
        """body lengths / s: walk, slow to a stop at ~3 s, probe, walk off faster at ~4 s."""
        ss = lambda x: _smoothstep(np.clip(x, 0, 1))
        base = V_WALK * (1 - ss((t - 2.5) / 0.5)) + V_WALK2 * ss((t - 5.0) / 1.5)
        return base * (1 + 0.08 * np.sin(2 * np.pi * t / 2.3))

    @staticmethod
    def heading(d, d_turn):
        """direction of the head's path vs distance walked d (voxels): slow wander (wavelength 1.3 body
        lengths) plus a head-led turn that starts where the animal walks off after the stop (d_turn) and
        is completed over 0.75 body lengths (radius ~0.6 body lengths)."""
        bl = BODY_LEN_UM / VOX
        # the wander fades out around the turn so that the turn reads as its own event
        quiet = 1 - 0.7 * _smoothstep(np.clip((d - d_turn + 0.5 * bl) / (0.5 * bl), 0, 1))
        return np.radians(WANDER_DEG * quiet * np.sin(2 * np.pi * d / (1.3 * bl) + 0.7)
                          + TURN_DEG * _smoothstep(np.clip((d - d_turn) / (1.0 * bl), 0, 1)))

    @staticmethod
    def stop_env(t):
        """0 while walking, 1 while standing (smooth)."""
        return _smoothstep(np.clip((t - 2.8) / 0.4, 0, 1)) * (1 - _smoothstep(np.clip((t - 4.9) / 0.4, 0, 1)))

    @classmethod
    def search(cls, t):
        """lateral head sweep relative to the path: irregular 1.5-3 s sweep with dwells while walking,
        a fast +-32 deg scan (~1 s) while standing."""
        g = t - 0.5 * 3.1 / (2 * np.pi) * np.sin(2 * np.pi * t / 3.1)      # period wanders, dwells
        walk = SEARCH_DEG * np.sin(2 * np.pi * g / 2.3) + 5.0 * np.sin(2 * np.pi * t / 1.3 + 1.0)
        w = cls.stop_env(t)
        return np.radians((1 - w) * walk + w * STOP_SCAN_DEG * np.sin(2 * np.pi * (t - 2.95) / 0.7))

    @classmethod
    def lift(cls, t):
        """dorsal head lift and leg-I wave weight (0..1): only while the animal stands."""
        return _smoothstep(np.clip((t - 3.05) / 0.3, 0, 1)) * (1 - _smoothstep(np.clip((t - 4.6) / 0.3, 0, 1)))

    def _make_debris(self):
        """sparse detritus on the slide, fixed in the world: (z, y, x, radius, delta-n, absorption)
        in voxels. Lies on the substrate (ventral side of the animal)."""
        rng = np.random.default_rng(7)
        ground = max(L["ground_y"] for L in self.legs)
        z0, z1 = self.z_tail - 600, self.z_head + 3000
        n = int((z1 - z0) * 1600 / (80.0 / VOX) ** 2)          # ~1 particle per (80 um)^2
        parts = []
        for _ in range(n):
            z, x = rng.uniform(z0, z1), self.cx0 + rng.uniform(-800, 800)
            r = rng.choice([1.0, 1.5, 2.5, 4.0], p=[0.45, 0.3, 0.17, 0.08]) / VOX
            for _ in range(int(rng.integers(1, 4)) if r > 2 / VOX else 1):          # small clumps
                parts.append((z + rng.normal(0, r), ground - r * rng.uniform(0.6, 1.0), x + rng.normal(0, r),
                              r * rng.uniform(0.6, 1.0), rng.uniform(0.02, 0.07), rng.uniform(0.3, 1.5)))
        return np.array(parts)

    def add_debris(self, dn, gut, t):
        """draw the debris into canvas volumes (Cz, y, Cx); the animal hides what lies beneath it."""
        Cz, Cx = self.canvas
        cam = self.camera(t)
        for z, y, x, r, d, ab in self.debris:
            zc, xc = z - cam[0] + Cz / 2, x - cam[1] + Cx / 2
            if not (-r < zc < Cz + r and -r < xc < Cx + r and 0 <= y < dn.shape[1]):
                continue
            lo = [max(0, int(c - r - 2)) for c in (zc, y, xc)]
            hi = [min(n, int(c + r + 3)) for c, n in zip((zc, y, xc), dn.shape)]
            Z, Y, X = np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing="ij")
            f = np.clip(r + 0.5 - np.sqrt((Z - zc) ** 2 + (Y - y) ** 2 + (X - xc) ** 2), 0, 1).astype(np.float32)
            sl = tuple(slice(a, b) for a, b in zip(lo, hi))
            dn[sl] = np.maximum(dn[sl], d * f)
            gut[sl] = np.maximum(gut[sl], ab * f)

    def _interp_t(self, t, arr):
        return np.interp(t, self._t, arr)

    def gait_phase(self, t):
        return np.interp(t, self._t, self._phi)

    @staticmethod
    def _jitter(L, p):
        """smooth per-leg phase irregularity (a function of the leg's own phase, so it stays continuous)."""
        s = L["seed"]
        return PHASE_JITTER * (0.6 * np.sin(2 * np.pi * p / 1.7 + s) + 0.4 * np.sin(2 * np.pi * p / 2.9 + 2.3 * s))

    def leg_phase(self, L, t):
        p = self.gait_phase(t) - L["off"]
        return p - self._jitter(L, p)

    def _phase_time(self, L, q):
        """time at which leg_phase(L, t) == q (monotonic: the jitter slope is << 1)."""
        p = q
        for _ in range(4):
            p = q + self._jitter(L, p)
        return np.interp(p + L["off"], self._phi_inc, self._t)

    # ---- trunk pose
    def trunk(self, t, local=True):
        """follow-the-leader: the section at rest position s sits where the head was when the head had
        walked the (stretched) arc length between s and the head less. local=False omits head search,
        yaw, telescoping and lift (used for foothold planning)."""
        s = self.s
        blen = self.z_head - self.z_tail
        walking = float(np.clip(self.speed(t) / V_WALK, 0, 1))
        ph = self.gait_phase(t)
        front = np.clip((s - self.z_neck) / (self.z_head - self.z_neck), 0, 1)
        if local:
            eps = LENGTH_OSC * np.sin(2 * np.pi * ph) * walking - STOP_SHORTEN * self.stop_env(t)
            fr30 = _smoothstep(np.clip((s - (self.z_head - 0.3 * blen)) / (0.3 * blen), 0, 1))
            stretch = 1 + eps + HEAD_TELE * np.sin(2 * np.pi * t / 0.8 + 0.4) * _smoothstep(front) \
                - HEAD_RETRACT * self.stop_env(t) * fr30
        else:
            eps, stretch = 0.0, np.ones_like(s)
        # arc length between s and the head tip along the posed (stretched) midline
        seg = 0.5 * (stretch[1:] + stretch[:-1]) * np.diff(s)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        d = np.interp(self.z_head, s, cum) - cum
        sig_h = np.interp(t, self._t, self._sig)
        sig = sig_h - d
        tau = np.interp(sig, self._sig_inc, self._t)
        th = np.interp(tau, self._t, self._th)
        th_h = np.interp(t, self._t, self._th)
        th = np.where(sig > sig_h, th_h, th)                       # beyond the head tip: straight on
        if local:
            # head search spread evenly over the front 45 % (linear angle ramp = constant curvature)
            ramp = np.clip((s - (self.z_head - 0.45 * blen)) / (0.45 * blen), 0, 1)
            ramp_scan = np.clip((s - (self.z_head - 0.30 * blen)) / (0.30 * blen), 0, 1)
            w = self.stop_env(t)
            srch = self.search(t)
            # while turning, the head sweep is damped so path + sweep stay within ~50 deg head-to-tail
            turning = abs(np.interp(t, self._t, self._th) - np.interp(t - 1.5, self._t, self._th))
            srch = srch * (1 - 0.85 * np.clip(2 * turning / np.radians(TURN_DEG), 0, 1))
            th = th + srch * ((1 - w) * ramp + w * ramp_scan)
            th = th + np.radians(YAW_DEG) * walking * np.sin(2 * np.pi * ph) * \
                np.cos(2 * np.pi * (s - self.z_mid) / blen) * (1 - ramp)
            tail = _smoothstep((self.z_tail + 0.25 * blen - s) / (0.25 * blen))
            L4 = next(L for L in self.legs if L["pair"] == 0)
            th = th + np.radians(TAIL_DEG) * walking * np.sin(2 * np.pi * self.leg_phase(L4, t)) * tail
        s_ref = self.z_mid
        tau_ref = np.interp(sig_h - np.interp(s_ref, s, d), self._sig_inc, self._t)
        p_ref = np.array([np.interp(tau_ref, self._t, self._H[:, 0]), np.interp(tau_ref, self._t, self._H[:, 1])])
        width = np.full_like(s, (1 + eps) ** -0.5)
        if local:     # segments widen between leg pairs while those legs are in stance
            for pair in (1, 2, 3):
                Ls = [L for L in self.legs if L["pair"] == pair]
                zc = np.mean([L["B"][0] for L in Ls])
                st = np.mean([np.cos(2 * np.pi * self.leg_phase(L, t)) for L in Ls])
                width = width * (1 + BULGE * walking * st * np.exp(-0.5 * ((s - zc) / 45.0) ** 2))
        lat = CurveMap(s, th, s_ref, p_ref, stretch=stretch, width=width)
        psi = (np.radians(LIFT_DEG) * self.lift(t) if local else 0.0) * _smoothstep(front)
        pit = CurveMap(s, np.zeros_like(s) + psi, 0.0, np.array([0.0, self.cy0]))
        return lat, pit

    def world(self, x, maps):
        """rest point(s) (..., 3) [after leg map] -> world (z, y, x)."""
        lat, pit = maps
        z1, y1 = pit.forward(x[..., 0], x[..., 1], self.cy0)
        zw, xw = lat.forward(z1, x[..., 2], self.cx0)
        return np.stack([zw, y1, xw], -1)

    def camera(self, t):
        """the observer follows the animal with the stage, lagging ~0.3 s behind the body centre."""
        ts = t - np.linspace(0, 0.8, 25)
        w = np.exp(-np.linspace(0, 0.8, 25) / 0.3)
        sig = np.interp(ts, self._t, self._sig) - 0.5 * (self.z_head - self.z_tail)    # body centre
        tau = np.interp(sig, self._sig_inc, self._t)
        p = np.stack([np.interp(tau, self._t, self._H[:, 0]), np.interp(tau, self._t, self._H[:, 1])], -1)
        return (p * w[:, None]).sum(0) / w.sum()

    # ---- legs
    def _foothold(self, L, k):
        """claw position for stance number k (phase k + swing .. k + 1): planned on the path pose at
        touchdown, without head search, so legs I do not follow the head's sweeps."""
        if k not in L["holds"]:
            t_td = self._phase_time(L, k + L["swing"])
            maps = self.trunk(t_td, local=False)
            nom = self.world(L["tip_nom"], maps)
            b0 = self.world(L["B"], maps)
            b1 = self.world(L["B"] + np.array([1.0, 0, 0]), maps)
            tang = b1 - b0; tang[1] = 0; tang /= np.linalg.norm(tang)
            var = 1 + STEP_VAR * np.sin(12.9898 * k + 78.233 * L["seed"])
            c = nom + 0.5 * self.step * var * (1 - L["swing"]) * L["stroke"] * tang
            c[1] = L["ground_y"]
            L["holds"][k] = c
        return L["holds"][k].copy()

    def claw(self, L, t):
        p = self.leg_phase(L, t)
        k = int(np.floor(p))
        fr = p - k
        sw = L["swing"]
        walking = np.clip(self.speed(t) / V_WALK, 0, 1)
        if fr < sw:                                    # swing: lift, carry forward, bow outward
            # if the animal stopped since liftoff, the swing is completed and the claw set down
            i0 = np.searchsorted(self._t, self._phase_time(L, k)); i1 = np.searchsorted(self._t, t)
            m = float(self._stopped[i0:i1 + 1].max()) if i1 >= i0 else 0.0
            tau = fr / sw
            tau = tau + (1 - tau) * m
            walking = walking * (1 - m)
            c0, c1 = self._foothold(L, k - 1), self._foothold(L, k)
            c = c0 + (c1 - c0) * _minjerk(tau)
            d = c1 - c0; d[1] = 0
            out = np.array([-d[2], 0.0, d[0]]) * L["side"]    # horizontal normal, pointing to the leg's side
            n = np.linalg.norm(out)
            if n > 1e-6:
                c = c + out / n * REACH_OUT_UM / VOX * np.sin(np.pi * tau) * walking
            c[1] = L["ground_y"] - LIFT_UM / VOX * np.sin(np.pi * tau) * walking   # set down when stopping
            if walking > 0:                            # telescoping in during swing: the leg's projection shrinks
                b = self.world(L["B"], self.trunk(t, local=False))
                c = c + (b - c) * SWING_RETRACT * np.sin(np.pi * tau) * walking * np.array([1.0, 0.0, 1.0])
            swinging = tau < 0.999 and walking > 0.05
        else:
            c, swinging = self._foothold(L, k), False
        if L["pair"] == 3:                             # legs I leave the ground, reach and wave while the head probes
            wl = self.lift(t)
            if wl > 1e-3:
                maps = self.trunk(t)
                d = L["a"] + np.array([1.6, 0.0, 0.0]); d /= np.linalg.norm(d)   # forward and sideways
                ang = np.radians(20.0) * np.sin(2 * np.pi * (t - 3.0) / 0.6)    # 2-3 waving cycles
                ca, sa = np.cos(ang), np.sin(ang)
                d = np.array([ca * d[0] - sa * d[2], d[1], sa * d[0] + ca * d[2]])
                reach = self.world(L["B"] + 1.3 * L["ell"] * d, maps)
                tap = np.clip(np.cos(2 * np.pi * (t - 3.1) / 0.7) * 4 - 3, 0, 1)    # brief touch-downs
                reach[1] = L["ground_y"] - 10.0 / VOX * (1 - tap)
                c = (1 - wl) * c + wl * reach
                swinging = not (tap > 0.5 and wl > 0.9)
        if L["pair"] == 2 and fr >= sw:                # legs II re-grip once during the stop
            if k == int(np.floor(self.leg_phase(L, 3.6))):
                g = float(np.clip((t - 3.6) / 0.25, 0, 1))
                if g > 0:
                    maps = self.trunk(t, local=False)
                    b0 = self.world(L["B"], maps); b1 = self.world(L["B"] + np.array([1.0, 0, 0]), maps)
                    tang = b1 - b0; tang[1] = 0; tang /= np.linalg.norm(tang)
                    c = c + tang * 5.0 / VOX * _minjerk(g)
                    c[1] -= 4.0 / VOX * np.sin(np.pi * g)
                    swinging = 0 < g < 1
        return c, swinging

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
            res = xp.linalg.norm(fx - q_, axis=1)
            return x, w, res < 0.35, res

        out = q.copy()
        empty = xp.zeros(len(q), bool)
        wq = self._w(L, q)
        xa0 = self._leg_unapply(L, R, lam, q)
        wa0 = self._w(L, xa0)
        ia = xp.nonzero(wa0 > 1e-3)[0]
        ib = xp.nonzero(wq > 1e-3)[0]
        q_ = q[ia]; xa, wa, oka, _ = newton(xa0[ia])
        q_ = q[ib]; xb, wb, okb, resb = newton(q[ib].copy())
        # B (identity start) candidates: keep converged ones. A point that has no preimage (large residual)
        # was vacated by the leg -> water; a small residual means a fold in the base transition, where the
        # best estimate is kept (avoids cracks in the trunk wall without leaving a ghost of the leg)
        out[ib] = xb
        empty[ib[~okb & (resb > 1.0)]] = True
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
            if "pts" not in L:      # weighted leg voxels (every 2nd), whose posed extent bounds the leg
                L["pts"] = np.argwhere(L["w"][::2, ::2, ::2] > 1e-3) * 2.0 + L["org"]
            org = L["org"]; shp = np.array(L["w"].shape)
            posed = self._leg_apply(L, R, lam, L["pts"])
            lo = np.minimum(org, posed.min(0)) - 6
            hi = np.maximum(org + shp - 1, posed.max(0)) + 6
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
