"""Quantitative body kinematics of a walking tardigrade: real brightfield video vs. simulated animation.

  python scripts/measure_kinematics.py              # extract (cached) + analyse both inputs
  python scripts/measure_kinematics.py --reextract  # ignore the per-frame cache

Inputs
  real : data/reference_video/Tardigrade_in_real_time.ogv  (800x600, 20 fps, brightfield, colour)
  sim  : renders/motion_v9/proj_###.png                     (832x832, 8 fps, bright animal on black)

Pipeline (identical shape analysis for both inputs, only the segmentation differs)
  1. Segmentation
     real: Lab colour rules relative to the per-frame background (L, a*, b* medians of bright pixels).
           animal pixel  = yellow/brown (db > 7, db > 0.08*darkness), not near-black (darkness < 175),
                           not green (da < -4 & db > 3 = algae);
           debris pixel  = dark (darkness > 40) but not animal/green (black, low-chroma clumps).
           The animal component is the largest plausible one (preferring overlap with the previous
           mask), closed (disk 15), hole-filled (debris lying ON the body becomes body), then grown by
           up to 14 px into non-background pixels to recover the dark, low-chroma blurred outline.
     sim : intensity > 0.25*p99.5 of the frame, largest component, hole-filled.
  2. QC (frame rejected if any): no plausible component; border contact > 8 % of the contour;
     area jump > 30 % vs. centred running median (+-1 s); > 30 % of the outline within 12 px of a
     compact debris blob, or > 35 % of the body area classified as debris (merged/hidden); compact
     debris inside the animal's convex hull > 10 % of the body area (clump overlapping / notching the body); sharpness (variance of Laplacian inside the
     body) < 0.45 x the median of all segmented frames (very blurry); plus the manual behaviour windows
     REAL_EXCLUDE (curled up / hidden / out of view, judged from contact sheets, documented below);
     finally midline length deviating > 30 % from the median of the remaining frames (mask split).
  3. Midline: legs and small protrusions removed by a morphological opening with radius
     0.45 x half-width (max of distance transform). Tips = geodesic extremes of the opened mask;
     ridge path = Dijkstra with cost 1/dt^2 (follows the distance-transform ridge); the path is
     extended along its end tangents to the boundary of the opened mask (so leg IV never captures a tip;
     both tips are rounded off by ~0.45 half-widths, identically for real and sim),
     smoothed (Gaussian, sigma = 3 % of the path) and resampled to 21 equidistant points.
  4. Head/tail: temporal continuity (head = end closer to the previous head, after stage-motion
     compensation) inside runs of trackable frames (gaps <= 0.5 s). sim: head is the right-hand end at
     t=0. real: per run, head = the narrower (tapered, lighter) end -- morphology, because the net travel
     of the real animal per run is too small vs. centroid jitter to decide reliably; the agreement of
     the travel direction with this rule is reported as a diagnostic.
  5. Per frame: body length (rel. to median), head-to-tail bend (angle between anterior 0-25 % and
     posterior 75-100 % chords), head yaw (0-15 % chord vs 15-40 % chord), max lateral deviation from
     the head-tail chord (% of length), mean |curvature| x length (total absolute turning, rad) and a
     10-bin |curvature| profile, centroid speed relative to the substrate (BL/s).
     Substrate motion: real = median Lucas-Kanade flow of background features outside the dilated
     animal + attached debris; sim = median displacement of the free-floating debris specks
     (they are fixed in the substrate frame, the virtual camera follows the animal).
     Speed = |X(t+1 s) - X(t-1 s)| / 2 s, X = centroid in substrate coordinates, on "trackable" frames
     (relaxed QC tier, see qc()).
Outputs: renders/kinematics/<name>_timeseries.png, <name>_stats.json, <name>_overlay_XXXX.jpg,
         <name>_frames.npz (per-frame cache) and compare.png.
"""
import argparse, glob, json, os, sys
os.environ.setdefault("PUMA_BACKEND", "cpu")
import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.signal import lombscargle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "renders", "kinematics")
REAL_VIDEO = os.path.join(ROOT, "data", "reference_video", "Tardigrade_in_real_time.ogv")
SIM_DIR = os.path.join(ROOT, "renders", "motion_v9")
NPTS = 21
STOP_BLS = 0.03
SPEED_HALFWIN_S = 1.0

# Manual behaviour selection for the real video (frame ranges, inclusive, excluded from analysis).
# Judged from contact sheets every 10-40 frames. Everything else is still subject to automatic QC.
REAL_EXCLUDE = [
    (0, 49),       # out of focus, stage moving fast, animal mostly outside the field
    (250, 330),    # animal buried under / merged with the large debris clump
    (930, 1065),   # C-shaped, wrapped around the debris clump, almost no locomotion
    (1110, 1470),  # C-shaped / curled, largely out of focus, partly hidden by debris
    (2255, 2455),  # curled C-shape around the debris clump
]
REAL_EXCLUDE_NOTES = ("Manual windows from contact sheets (every 10 frames): excluded stretches where the animal is "
                      "curled into a stationary C around the debris clump it carries, buried in debris, or out of "
                      "focus. All remaining frames still pass the automatic QC (border, area jump, debris "
                      "merge, blur).")


# ----------------------------------------------------------------------------------------- helpers
def disk(r):
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def largest_cc(mask, prefer=None, min_area=0):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return None
    areas = st[1:, cv2.CC_STAT_AREA].astype(float)
    score = areas.copy()
    if prefer is not None and prefer.any():
        ov = ndi.sum(prefer, lab, index=np.arange(1, n)).astype(float)
        score = areas * 0.25 + ov * 4.0
    k = int(np.argmax(score))
    if areas[k] < min_area:
        return None
    return lab == (k + 1)


def fill_holes(m):
    return ndi.binary_fill_holes(m)


def contour_border_fraction(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return 1.0
    c = max(cs, key=cv2.contourArea)[:, 0, :]
    h, w = mask.shape
    onb = (c[:, 0] <= 1) | (c[:, 1] <= 1) | (c[:, 0] >= w - 2) | (c[:, 1] >= h - 2)
    return float(onb.mean())


# ----------------------------------------------------------------------------------------- segmentation
def real_pixel_classes(frame):
    f = cv2.GaussianBlur(frame, (7, 7), 0)
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    Lbg = np.percentile(L, 75)
    bright = L > Lbg - 12
    Abg, Bbg = np.median(A[bright]), np.median(B[bright])
    D = Lbg - L
    db, da = B - Bbg, A - Abg
    green = (da < -4) & (db > 3)
    # light/translucent tissue: yellow-brown relative to the lilac background
    light = (db > 7) & (db > 0.08 * D) & (D < 175) & ~green
    # dark pigmented tissue (gut, posterior): still orange hue with G > B; black debris is
    # purple-tinted (G < B) or grey (low saturation)
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    H, Sa = hsv[..., 0].astype(np.int16), hsv[..., 1].astype(np.int16)
    b_, g_ = f[..., 0].astype(np.int16), f[..., 1].astype(np.int16)
    dark = (D >= 60) & ((H <= 28) | (H >= 176)) & (Sa > 70) & (g_ > b_ + 1) & ~green
    animal = light | dark
    debris = (D > 40) & ~animal & ~green
    fg = (D > 22) | (db > 7) | green
    return animal, debris, green, fg, debris & (D > 150)


def segment_real(frame, prev=None):
    animal, debris, green, fg, black = real_pixel_classes(frame)
    a = cv2.morphologyEx(animal.astype(np.uint8), cv2.MORPH_OPEN, disk(2))
    cand = largest_cc(a > 0, prefer=prev, min_area=6000)
    if cand is None:
        return None, debris, fg
    m = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, disk(15)) > 0
    m = fill_holes(m)
    # recover the dark, low-chroma rim of the body (blurred cuticle outline, out-of-focus legs):
    # grow up to 14 px into non-background, non-green pixels
    # (but not into near-black debris, darkness > 150, which the blurred rim never reaches)
    m = m | ((cv2.dilate(m.astype(np.uint8), disk(14)) > 0) & fg & ~green & ~black)
    m = fill_holes(cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, disk(3)) > 0)
    m = largest_cc(m)
    return m, debris, fg


def debris_contact(m, debris):
    """fraction of the animal outline lying within 12 px of a compact debris blob (>= 16 px thick)."""
    blobs = (cv2.morphologyEx((debris & ~m).astype(np.uint8), cv2.MORPH_OPEN, disk(8)) > 0)
    near = cv2.dilate(blobs.astype(np.uint8), disk(12)) > 0
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)[:, 0, :]
    return float(near[c[:, 1], c[:, 0]].mean())


def debris_in_hull(m, debris):
    """area of compact debris inside the convex hull of the animal but outside its mask, / animal area.
    Large when a debris clump overlaps the body and cuts a notch into the mask."""
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.fillPoly(np.zeros(m.shape, np.uint8), [cv2.convexHull(max(cs, key=cv2.contourArea))], 1) > 0
    blobs = cv2.morphologyEx((debris & ~m).astype(np.uint8), cv2.MORPH_OPEN, disk(5)) > 0
    return float((blobs & hull).sum() / m.sum())


def segment_sim(img):
    g = cv2.GaussianBlur(img.astype(np.float32), (5, 5), 0)
    thr = 0.25 * np.percentile(g, 99.5)
    m = largest_cc(g > thr, min_area=2000)
    if m is None:
        return None, g, thr
    m = fill_holes(cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, disk(4)) > 0)
    return m, g, thr


# ----------------------------------------------------------------------------------------- midline
def _graph(mask_ds, weight):
    h, w = mask_ds.shape
    idx = -np.ones((h, w), np.int64)
    ys, xs = np.nonzero(mask_ds)
    idx[ys, xs] = np.arange(len(ys))
    rows, cols, vals = [], [], []
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        y2, x2 = ys + dy, xs + dx
        ok = (y2 >= 0) & (y2 < h) & (x2 >= 0) & (x2 < w)
        ok[ok] &= mask_ds[y2[ok], x2[ok]]
        a, b = idx[ys[ok], xs[ok]], idx[y2[ok], x2[ok]]
        step = np.hypot(dy, dx)
        wv = step * 0.5 * (weight[ys[ok], xs[ok]] + weight[y2[ok], x2[ok]])
        rows += [a, b]; cols += [b, a]; vals += [wv, wv]
    n = len(ys)
    G = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    return G, ys, xs


def extract_midline(mask, ds=3):
    """Return (pts[NPTS,2] in xy image px, length_px, halfwidth_px) or None."""
    m8 = mask.astype(np.uint8)
    dt = cv2.distanceTransform(m8, cv2.DIST_L2, 5)
    hw = float(dt.max())
    if hw < 4:
        return None
    heavy = cv2.morphologyEx(m8, cv2.MORPH_OPEN, disk(0.45 * hw)) > 0
    heavy = largest_cc(heavy)
    if heavy is None:
        return None
    h, w = mask.shape
    hs = cv2.resize(heavy.astype(np.uint8), (w // ds, h // ds), interpolation=cv2.INTER_NEAREST) > 0
    hs = largest_cc(hs)
    if hs is None or hs.sum() < 20:
        return None
    dts = cv2.distanceTransform(hs.astype(np.uint8), cv2.DIST_L2, 5)
    G1, ys, xs = _graph(hs, np.ones_like(dts))
    c0 = int(np.argmax(dts[ys, xs]))
    d0 = dijkstra(G1, indices=c0)
    a = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
    da = dijkstra(G1, indices=a)
    b = int(np.argmax(np.where(np.isfinite(da), da, -1)))
    G2, _, _ = _graph(hs, 1.0 / (dts + 0.5) ** 2)
    _, pred = dijkstra(G2, indices=a, return_predecessors=True)
    path = [b]
    while path[-1] != a and pred[path[-1]] >= 0:
        path.append(pred[path[-1]])
    path = path[::-1]
    P = np.stack([xs[path], ys[path]], 1).astype(float) * ds + (ds - 1) / 2.0
    if len(P) < 6:
        return None
    P = _resample(P, 200)
    P = np.stack([ndi.gaussian_filter1d(P[:, i], 4, mode="nearest") for i in range(2)], 1)
    # extend both ends along the end tangent to the boundary of the opened (leg-free) mask
    def extend(P, end):
        Q = P if end == 1 else P[::-1]
        t = Q[-1] - Q[-12]
        t /= np.linalg.norm(t) + 1e-9
        p = Q[-1].copy(); ext = []
        for _ in range(int(3 * hw)):
            p = p + t
            xi, yi = int(round(p[0])), int(round(p[1]))
            if not (0 <= xi < w and 0 <= yi < h) or not heavy[yi, xi]:
                break
            ext.append(p.copy())
        if ext:
            Q = np.vstack([Q, ext])
        return Q if end == 1 else Q[::-1]
    P = extend(extend(P, 1), 0)
    P = _resample(P, 200)
    P = np.stack([ndi.gaussian_filter1d(P[:, i], 6, mode="nearest") for i in range(2)], 1)
    # re-pin the tips (smoothing pulls them inwards) by resampling the curve, keep the true extent
    L = _arclen(P)
    Q = _resample(P, NPTS)
    xi = np.clip(np.round(Q[:, 0]).astype(int), 0, w - 1); yi = np.clip(np.round(Q[:, 1]).astype(int), 0, h - 1)
    return Q, L, hw, dt[yi, xi].astype(float)


def _arclen(P):
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))


def _resample(P, n):
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    if s[-1] <= 0:
        return np.repeat(P[:1], n, 0)
    u = np.linspace(0, s[-1], n)
    return np.stack([np.interp(u, s, P[:, 0]), np.interp(u, s, P[:, 1])], 1)


# ----------------------------------------------------------------------------------------- shape metrics
def _ang(u, v):
    """signed angle from u to v in degrees"""
    return float(np.degrees(np.arctan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1])))


def _at(P, frac):
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    u = frac * s[-1]
    return np.array([np.interp(u, s, P[:, 0]), np.interp(u, s, P[:, 1])])


def shape_metrics(P):
    """P: NPTS x 2 midline, P[0] = head. Returns dict of per-frame metrics."""
    L = _arclen(P)
    h0, h15, h25, h40, t75, t100 = (_at(P, f) for f in (0, 0.15, 0.25, 0.40, 0.75, 1.0))
    ant = h0 - h25            # anterior chord, pointing forward
    post = t75 - t100         # posterior chord, pointing forward
    bend = _ang(post, ant)
    yaw = _ang(h15 - h40, h0 - h15)
    chord = P[0] - P[-1]
    cn = np.linalg.norm(chord) + 1e-9
    lat = np.abs((P[:, 0] - P[-1, 0]) * chord[1] - (P[:, 1] - P[-1, 1]) * chord[0]) / cn
    # curvature on a finely resampled, smoothed curve
    F = _resample(P, 101)
    F = np.stack([ndi.gaussian_filter1d(F[:, i], 2, mode="nearest") for i in range(2)], 1)
    th = np.unwrap(np.arctan2(np.diff(F[:, 1]), np.diff(F[:, 0])))
    ds = L / 100.0
    kap = np.abs(np.diff(th)) / ds                      # 99 values at s = 0.01..0.99
    kprof = np.array([kap[i * 10:(i + 1) * 10].mean() for i in range(10)]) * L
    return dict(length=L, bend=bend, yaw=yaw, latdev=100 * lat.max() / L,
                curv=float(kap[3:-3].mean() * L), curv_prof=kprof, chord_ratio=cn / L)


# ----------------------------------------------------------------------------------------- stage motion
def stage_shift_lk(g0, g1, excl):
    """median translation of background between gray frames g0->g1 (image px)."""
    sc = 0.5
    a = cv2.resize(g0, None, fx=sc, fy=sc); b = cv2.resize(g1, None, fx=sc, fy=sc)
    msk = (cv2.resize((~excl).astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST) * 255)
    msk[:6] = 0; msk[-6:] = 0; msk[:, :6] = 0; msk[:, -6:] = 0
    p0 = cv2.goodFeaturesToTrack(a, 300, 0.005, 6, mask=msk, blockSize=7)
    if p0 is None or len(p0) < 8:
        return np.array([np.nan, np.nan]), 0
    p1, st, err = cv2.calcOpticalFlowPyrLK(a, b, p0, None, winSize=(21, 21), maxLevel=4)
    ok = st[:, 0] == 1
    if ok.sum() < 8:
        return np.array([np.nan, np.nan]), int(ok.sum())
    d = (p1 - p0)[ok, 0, :] / sc
    med = np.median(d, 0)
    inl = np.linalg.norm(d - med, axis=1) < 2.0 + 0.1 * np.linalg.norm(med)
    if inl.sum() < 6:
        return np.array([np.nan, np.nan]), int(inl.sum())
    return np.median(d[inl], 0), int(inl.sum())


def speck_positions(g, animal, thr):
    m = (g > 0.15 * thr) & ~(cv2.dilate(animal.astype(np.uint8), disk(25)) > 0)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    keep = [(cen[i]) for i in range(1, n) if 6 <= st[i, cv2.CC_STAT_AREA] <= 4000]
    return np.array(keep).reshape(-1, 2)


def match_shift(p0, p1, maxd=40.0):
    if len(p0) == 0 or len(p1) == 0:
        return np.array([np.nan, np.nan]), 0
    d = p1[None, :, :] - p0[:, None, :]
    dist = np.linalg.norm(d, axis=2)
    j = np.argmin(dist, 1)
    ok = dist[np.arange(len(p0)), j] < maxd
    if ok.sum() == 0:
        return np.array([np.nan, np.nan]), 0
    v = d[np.arange(len(p0)), j][ok]
    return np.median(v, 0), int(ok.sum())


# ----------------------------------------------------------------------------------------- extraction
def blank_rec():
    return dict(seg=False, area=np.nan, border=np.nan, occl_ring=np.nan, occl_body=np.nan, occl_hull=np.nan, sharp=np.nan,
                cx=np.nan, cy=np.nan, pts=np.full((NPTS, 2), np.nan), length=np.nan, hw=np.nan,
                shift=np.array([np.nan, np.nan]), nfeat=0, width=np.full(NPTS, np.nan))


def extract_real(path, workers=4):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    edges = np.linspace(0, n, workers + 1).astype(int)
    from multiprocessing import Pool
    with Pool(workers) as pool:
        parts = pool.starmap(_extract_real_range, [(path, int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])])
    recs = [r for p in parts for r in p]
    return recs, fps


def _extract_real_range(path, start, stop):
    """frames [start, stop); frame start-1 is decoded as 'previous' for tracking/stage motion."""
    cap = cv2.VideoCapture(path)
    recs = []; prev = None; prev_gray = None; prev_excl = None; i = -1
    while True:
        ok, fr = cap.read()
        i += 1
        if not ok or i >= stop:
            break
        if i < start - 1:
            continue
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        r = blank_rec()
        m, debris, fg = segment_real(fr, prev)
        if m is not None:
            r["seg"] = True
            r["area"] = float(m.sum())
            r["border"] = contour_border_fraction(m)
            r["occl_ring"] = debris_contact(m, debris)
            r["occl_hull"] = debris_in_hull(m, debris)
            r["occl_body"] = float(debris[m].mean())
            er = cv2.erode(m.astype(np.uint8), disk(5)) > 0
            lap = cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_32F)
            r["sharp"] = float(lap[er].var()) if er.any() else np.nan
            ys, xs = np.nonzero(m)
            r["cx"], r["cy"] = float(xs.mean()), float(ys.mean())
            ml = extract_midline(m)
            if ml is not None:
                r["pts"], r["length"], r["hw"], r["width"] = ml
            prev = m
            # exclusion for stage motion: animal + every foreground blob touching it, dilated
            n, lab = cv2.connectedComponents((fg | m).astype(np.uint8), connectivity=8)
            ids = np.unique(lab[cv2.dilate(m.astype(np.uint8), disk(8)) > 0]); ids = ids[ids > 0]
            excl = cv2.dilate(np.isin(lab, ids).astype(np.uint8), disk(25)) > 0
        else:
            excl = cv2.dilate(fg.astype(np.uint8), disk(15)) > 0
        if prev_gray is not None:
            ex = excl | prev_excl
            r["shift"], r["nfeat"] = stage_shift_lk(prev_gray, gray, ex)
        prev_gray, prev_excl = gray, excl
        if i >= start:
            recs.append(r)
        if i % 200 == 0:
            print(f"  real frame {i}", flush=True)
    return recs


def extract_sim(folder):
    files = sorted(glob.glob(os.path.join(folder, "proj_*.png")))
    meta = json.load(open(os.path.join(folder, "meta.json")))
    fps = float(meta.get("fps", 8.0))
    recs = []; prev_sp = None
    for f in files:
        img = np.array(cv2.imread(f, cv2.IMREAD_GRAYSCALE))
        r = blank_rec()
        m, g, thr = segment_sim(img)
        sp = np.zeros((0, 2))
        if m is not None:
            r["seg"] = True; r["area"] = float(m.sum()); r["border"] = contour_border_fraction(m)
            r["occl_ring"] = 0.0; r["occl_body"] = 0.0; r["occl_hull"] = 0.0
            er = cv2.erode(m.astype(np.uint8), disk(5)) > 0
            lap = cv2.Laplacian(cv2.GaussianBlur(img, (3, 3), 0), cv2.CV_32F)
            r["sharp"] = float(lap[er].var())
            ys, xs = np.nonzero(m); r["cx"], r["cy"] = float(xs.mean()), float(ys.mean())
            ml = extract_midline(m)
            if ml is not None:
                r["pts"], r["length"], r["hw"], r["width"] = ml
            sp = speck_positions(g, m, thr)
        if prev_sp is not None:
            r["shift"], r["nfeat"] = match_shift(prev_sp, sp)
        prev_sp = sp
        recs.append(r)
    return recs, fps


def save_cache(name, recs, fps):
    keys = [k for k in recs[0]]
    d = {k: np.array([r[k] for r in recs]) for k in keys}
    d["fps"] = fps
    np.savez_compressed(os.path.join(OUT, f"{name}_frames.npz"), **d)


def load_cache(name):
    p = os.path.join(OUT, f"{name}_frames.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------------------- analysis
# QC thresholds (identical for real and sim)
QC = dict(border=0.08, contact=0.30, hull_debris=0.10, body_debris=0.35, area_jump=0.30, sharp_rel=0.45, length_dev=0.30)


def qc(D, name):
    n = len(D["seg"])
    seg = D["seg"].astype(bool) & np.isfinite(D["length"])
    border = seg & (D["border"] > QC["border"])
    # sudden area change vs. centred running median (+-1 s) of the segmented area
    fps = float(D["fps"]); w = int(2 * round(fps)) + 1
    A = np.where(seg, D["area"], np.nan)
    Af = pd_nanmedian_filter(A, w)
    jump = seg & (np.abs(A / Af - 1) > QC["area_jump"])
    merged = seg & ((D["occl_ring"] > QC["contact"]) | (D["occl_body"] > QC["body_debris"]))
    # debris clump overlapping the body (cuts a notch into the mask)
    notched = seg & ~(D["occl_hull"] <= QC["hull_debris"])
    sharp_med = np.nanmedian(D["sharp"][seg]) if seg.any() else np.nan
    blurry = seg & (D["sharp"] < QC["sharp_rel"] * sharp_med)
    manual = np.zeros(n, bool)
    if name == "real":
        for a, b in REAL_EXCLUDE:
            manual[a:b + 1] = True
    ok = seg & ~border & ~jump & ~merged & ~notched & ~blurry & ~manual
    # implausible midline length (part of the body lost under debris / out of frame): outside
    # [0.75, 1.20] x the 80th percentile of otherwise good frames (the p80 is used because partial masks
    # only ever shorten the midline)
    Lref = np.nanpercentile(D["length"][ok], 80) if ok.any() else np.nan
    rel = D["length"] / Lref
    badlen = ok & ~((rel >= 0.75) & (rel <= 1.20))
    ok &= ~badlen
    # relaxed "trackable" tier, used only to carry head/tail identity across short gaps and to define the
    # time windows for spectra (shape statistics always use the strict set `ok`)
    track = seg & ~manual & ~blurry & ~jump & ~(D["border"] > 0.25) & ~(D["occl_hull"] > 0.25)
    track &= (rel >= 0.65) & (rel <= 1.30)      # drop half-animal masks (posterior lost under debris)
    track |= ok
    reasons = dict(not_segmented=int((~seg).sum()), border=int(border.sum()), area_jump=int(jump.sum()),
                   merged_or_hidden=int(merged.sum()), notched_by_debris=int(notched.sum()), blurry=int(blurry.sum()), manual_behaviour=int(manual.sum()),
                   implausible_length_after_other_checks=int(badlen.sum()), n_trackable=int(track.sum()),
                   note="counts overlap; a frame can fail several checks")
    return ok, track, reasons


def pd_nanmedian_filter(x, w):
    h = w // 2; out = np.full_like(x, np.nan)
    for i in range(len(x)):
        v = x[max(0, i - h):i + h + 1]; v = v[np.isfinite(v)]
        if len(v):
            out[i] = np.median(v)
    return out


def substrate_track(D):
    sh = D["shift"].copy()
    bad = ~np.isfinite(sh).all(1)
    sh[0] = 0
    # interpolate missing stage shifts
    for k in range(2):
        v = sh[:, k]; idx = np.arange(len(v)); good = np.isfinite(v)
        v[~good] = np.interp(idx[~good], idx[good], v[good]) if good.any() else 0
    cum = np.cumsum(sh, 0)
    X = np.stack([D["cx"], D["cy"]], 1) - cum            # centroid in substrate coordinates
    return X, cum, bad


def runs(ok, maxgap):
    idx = np.nonzero(ok)[0]
    if len(idx) == 0:
        return []
    out = [[idx[0]]]
    for i in idx[1:]:
        if i - out[-1][-1] <= maxgap + 1:
            out[-1].append(i)
        else:
            out.append([i])
    return [np.array(r) for r in out]


def orient(D, ok, track, X, cum, fps, name, BL):
    """head/tail identity. Continuity inside runs of trackable frames (gaps <= 0.5 s); per run the
    orientation follows the direction of travel (head leads). Runs with < 0.15 BL net forward travel
    are flagged. sim: head = right-hand end at t=0. real: see comment below (morphological rule)."""
    pts = D["pts"].copy(); W = D["width"].copy()
    n = len(pts)
    source = np.array([""] * n, dtype=object)
    rr = runs(track, int(round(0.5 * fps)))
    info = []
    for r in rr:
        for j in range(1, len(r)):
            a, b = r[j - 1], r[j]
            hp = pts[a][0] - cum[a]
            h0, h1 = pts[b][0] - cum[b], pts[b][-1] - cum[b]
            if np.linalg.norm(h1 - hp) < np.linalg.norm(h0 - hp):
                pts[b] = pts[b][::-1]; W[b] = W[b][::-1]
        if name == "sim":
            if pts[r[0]][0, 0] < pts[r[0]][-1, 0]:     # head is the right-hand end at t=0
                pts[r] = pts[r][:, ::-1]; W[r] = W[r][:, ::-1]
            source[r] = "given"
            info.append(dict(start=int(r[0]), end=int(r[-1])))
            continue
        fwd = 0.0
        for j in range(1, len(r)):
            a, b = r[j - 1], r[j]
            u = pts[b][0] - pts[b][-1]; u /= np.linalg.norm(u) + 1e-9
            fwd += float(np.dot(X[b] - X[a], u))
        if fwd < 0:
            pts[r] = pts[r][:, ::-1]; W[r] = W[r][:, ::-1]
        conf = abs(fwd) / BL > 0.15 and (r[-1] - r[0]) >= fps
        taper = float(np.nanmean(W[r][:, 2:6]) - np.nanmean(W[r][:, -6:-2]))   # <0: head end narrower
        if conf:
            source[r] = "travel"
        info.append(dict(start=int(r[0]), end=int(r[-1]), n=int(len(r)), forward_BL=round(abs(fwd) / BL, 3),
                         travel_confident=bool(conf), head_end_narrower=bool(taper < 0)))
    agree = None
    if name == "real":
        # Travel direction is NOT reliable in this video (net displacement per run 0.1-0.4 BL, comparable
        # to centroid jitter from debris, plus large stage pans), so it is only reported as a diagnostic.
        # Primary rule from morphology: head = narrower, lighter, tapered end (buccal apparatus at the
        # tip, legs I-III as lateral bumps 20-40 % behind it); the broad dark end is the posterior.
        cf = [(ri["n"], ri["head_end_narrower"]) for ri in info if ri["travel_confident"]]
        if cf:
            agree = sum(nn for nn, t in cf if t) / sum(nn for nn, _ in cf)
        for r, ri in zip(rr, info):
            if not ri["head_end_narrower"]:
                pts[r] = pts[r][:, ::-1]; W[r] = W[r][:, ::-1]
            source[r] = "taper"
    return pts, source, info, rr, agree


def periodogram(series, ok, fps, rr, fmin=0.2, fmax=3.5, minlen_s=3.0):
    freqs = np.linspace(fmin, fmax, 300)
    acc = np.zeros_like(freqs); wsum = 0
    for r in rr:
        if (r[-1] - r[0]) / fps < minlen_s:
            continue
        sel = r[np.isfinite(series[r])]
        if len(sel) < 20 or (sel[-1] - sel[0]) / fps < minlen_s:
            continue
        t = sel / fps; y = series[sel] - np.mean(series[sel])
        y = y - np.polyval(np.polyfit(t, y, 1), t)
        if np.std(y) == 0:
            continue
        p = lombscargle(t, y, 2 * np.pi * freqs, normalize=False) / len(sel)
        w = len(sel)
        acc += p * w; wsum += w
    if wsum == 0:
        return freqs, np.full_like(freqs, np.nan), np.nan
    acc /= wsum
    return freqs, acc, float(1.0 / freqs[np.argmax(acc)])


def pct(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(median=None, p10=None, p90=None, n=0)
    return dict(median=round(float(np.median(x)), 4), p10=round(float(np.percentile(x, 10)), 4),
                p90=round(float(np.percentile(x, 90)), 4), n=int(len(x)))


def analyse(name, D):
    fps = float(D["fps"])
    n = len(D["seg"])
    ok, track, reasons = qc(D, name)
    X, cum, badshift = substrate_track(D)
    BL = float(np.nanmedian(D["length"][ok])) if ok.any() else np.nan
    pts, source, runinfo, rr, agree = orient(D, ok, track, X, cum, fps, name, BL)
    confident = source != ""
    M = dict(length=np.full(n, np.nan), bend=np.full(n, np.nan), yaw=np.full(n, np.nan),
             latdev=np.full(n, np.nan), curv=np.full(n, np.nan), chord_ratio=np.full(n, np.nan))
    prof = np.full((n, 10), np.nan)
    for i in np.nonzero(track & np.isfinite(D["length"]))[0]:
        s = shape_metrics(pts[i])
        for k in M:
            M[k][i] = s[k]
        prof[i] = s["curv_prof"]
    Lrel = M["length"] / BL
    yaw_c = np.where(confident, M["yaw"], np.nan)
    # speed relative to substrate: centroid displacement over a 2 s window (+-1 s) between trackable
    # frames (long window because segmentation jitter of the real centroid is ~0.02-0.05 BL per frame)
    k = max(1, int(round(SPEED_HALFWIN_S * fps)))
    speed = np.full(n, np.nan); img_speed = np.full(n, np.nan)
    C = np.stack([D["cx"], D["cy"]], 1)
    for i in range(k, n - k):
        if track[i - k] and track[i + k] and track[i]:
            speed[i] = np.linalg.norm(X[i + k] - X[i - k]) / (2 * k / fps) / BL
            img_speed[i] = np.linalg.norm(C[i + k] - C[i - k]) / (2 * k / fps) / BL
    # spectra: Lomb-Scargle over the trackable tier (gaps allowed, runs spanning >= 3 s)
    fq, py, Tyaw = periodogram(yaw_c, track, fps, rr)
    _, pl, Tlen = periodogram(Lrel, track, fps, rr)
    _, pb, Tbend = periodogram(M["bend"], track, fps, rr)
    def tier(sel):
        g = lambda x: np.where(sel, x, np.nan)
        return dict(n=int(sel.sum()), bend_abs_deg=pct(np.abs(g(M["bend"]))), head_yaw_abs_deg=pct(np.abs(g(yaw_c))),
                    lateral_dev_pct_BL=pct(g(M["latdev"])), length_rel=pct(g(Lrel)),
                    length_cv=round(float(np.nanstd(g(Lrel))), 4), curvature_total_abs_rad=pct(g(M["curv"])))
    sp = speed[np.isfinite(speed)]
    G = lambda x: np.where(ok, x, np.nan)      # strict-QC frames only
    stats = dict(
        name=name, fps=fps, n_frames=n, n_usable=int(ok.sum()), usable_fraction=round(float(ok.mean()), 4),
        rejections=reasons, body_length_px=round(BL, 1),
        body_halfwidth_px=round(float(np.nanmedian(D["hw"][ok])), 1) if ok.any() else None,
        n_runs_ge_3s=int(sum((r[-1] - r[0]) / fps >= 3 for r in rr)),
        usable_seconds=round(ok.sum() / fps, 2),
        bend_abs_deg=pct(np.abs(G(M["bend"]))), bend_signed_deg=pct(G(M["bend"])),
        head_yaw_abs_deg=pct(np.abs(G(yaw_c))), head_yaw_signed_deg=pct(G(yaw_c)),
        head_yaw_frames_orientation_confident=int(np.isfinite(G(yaw_c)).sum()),
        lateral_dev_pct_BL=pct(G(M["latdev"])), length_rel=pct(G(Lrel)),
        length_cv=round(float(np.nanstd(G(Lrel))), 4),
        curvature_total_abs_rad=pct(G(M["curv"])), chord_over_length=pct(G(M["chord_ratio"])),
        curvature_profile_head_to_tail=[round(float(v), 3) for v in np.nanmean(prof[ok], 0)] if ok.any() else None,
        dominant_period_s=dict(head_yaw=round(Tyaw, 2) if np.isfinite(Tyaw) else None,
                               length=round(Tlen, 2) if np.isfinite(Tlen) else None,
                               bend=round(Tbend, 2) if np.isfinite(Tbend) else None),
        speed_BL_per_s=pct(sp), speed_mean_BL_per_s=round(float(sp.mean()), 4) if len(sp) else None,
        stop_fraction=round(float((sp < STOP_BLS).mean()), 4) if len(sp) else None,
        image_centroid_speed_BL_per_s=pct(img_speed),
        stage_shift_missing_frames=int(badshift.sum()),
        orientation=dict(frames_by_travel=int((ok & (source == "travel")).sum()),
                         frames_by_taper_rule=int((ok & (source == "taper")).sum()),
                         frames_ambiguous=int((ok & (source == "")).sum()),
                         taper_rule_agreement_with_travel=None if agree is None else round(agree, 3),
                         note="real: head = narrower tapered end (morphology); travel direction only diagnostic"),
        n_trackable=int(track.sum()),
        relaxed_tier_trackable_frames=tier(track),
        spectra_note="dominant periods from Lomb-Scargle on the trackable tier (runs >= 3 s, 0.2-3.5 Hz; periods > 5 s are not resolvable in the 10.5 s simulation)",
        runs=runinfo if name == "sim" else [ri for ri in runinfo if ri["end"] - ri["start"] >= fps],
    )
    if name == "real":
        stats["manual_exclusions"] = REAL_EXCLUDE
        stats["manual_exclusion_notes"] = REAL_EXCLUDE_NOTES
    else:
        stats["speed_note"] = ("substrate-relative speed from centroid motion minus median displacement of the "
                               "free-floating debris specks (assumed fixed to the substrate); "
                               "image_centroid_speed is the raw centroid speed in the camera frame")
        stats["body_length_um"] = round(BL * 0.27, 1)
    series = dict(t=np.arange(n) / fps, ok=ok, track=track, Lrel=Lrel, bend=M["bend"], yaw=yaw_c, yaw_all=M["yaw"],
                  latdev=M["latdev"], curv=M["curv"], speed=speed, img_speed=img_speed, prof=prof,
                  fq=fq, py=py, pl=pl, pb=pb, pts=pts, confident=confident)
    return stats, series


# ----------------------------------------------------------------------------------------- plots
def plot_timeseries(name, S, st):
    t = S["t"]; ok = S["ok"]; tr = S["track"] & ~ok
    fig, ax = plt.subplots(7, 1, figsize=(14, 15), sharex=True)
    rows = [("Lrel", "length / median"), ("bend", "bend (deg)"), ("yaw", "head yaw (deg)"),
            ("latdev", "lat. dev. (% BL)"), ("curv", "mean|k| x L (rad)")]
    for a, (k, lab) in zip(ax, rows):
        a.plot(t[tr], S[k][tr], ".", ms=2, color="0.65", label="trackable tier (relaxed QC)")
        a.plot(t[ok], S[k][ok], ".", ms=4, color="tab:red", label="strict QC")
        a.set_ylabel(lab); a.grid(alpha=0.3)
    ax[0].legend(fontsize=8, loc="upper right")
    ax[5].plot(t, S["speed"], ".-", ms=2, lw=0.6, label="speed rel. substrate (2 s window)")
    ax[5].plot(t, S["img_speed"], ".", ms=1.5, color="0.6", label="image-frame centroid speed")
    ax[5].axhline(STOP_BLS, color="r", lw=0.8, ls="--", label="stop threshold"); ax[5].legend(fontsize=8)
    ax[5].set_ylabel("speed (BL/s)"); ax[5].grid(alpha=0.3)
    ax[6].fill_between(t, 0, S["track"].astype(float), step="mid", alpha=0.4, color="0.5", label="trackable")
    ax[6].fill_between(t, 0, ok.astype(float), step="mid", alpha=0.8, color="tab:red", label="strict")
    ax[6].legend(fontsize=8); ax[6].set_ylabel("usable"); ax[6].set_xlabel("time (s)")
    fig.suptitle(f"{name}: strict {st['n_usable']}/{st['n_frames']} frames ({100*st['usable_fraction']:.1f} %), "
                 f"trackable {st['n_trackable']}, BL = {st['body_length_px']} px")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{name}_timeseries.png"), dpi=80)
    plt.close(fig)


def plot_compare(R, S):
    groups = [("real strict", "real", "ok", "tab:red"), ("real relaxed", "real", "track", "tab:orange"),
              ("sim", "sim", "ok", "tab:blue")]
    fig, ax = plt.subplots(3, 3, figsize=(17, 13))
    def boxes(a, key, f, title):
        data = []
        for lab, nm, sel, c in groups:
            v = f(S[nm][key]); data.append(v[S[nm][sel] & np.isfinite(v)])
        b = a.boxplot(data, tick_labels=[f"{g[0]}\n(n={len(d)})" for g, d in zip(groups, data)], whis=(10, 90),
                      showfliers=False, patch_artist=True)
        for p_, g in zip(b["boxes"], groups):
            p_.set_facecolor(g[3]); p_.set_alpha(0.5)
        for j, d in enumerate(data):
            x = np.random.default_rng(0).normal(j + 1, 0.06, len(d))
            a.plot(x, d, ".", ms=1.5, color="k", alpha=0.2)
        a.set_title(title + " (box 25-75 %, whiskers 10-90 %)", fontsize=9); a.grid(alpha=0.3)
    boxes(ax[0, 0], "bend", np.abs, "|head-tail bend| (deg)")
    boxes(ax[0, 1], "yaw", np.abs, "|head yaw| (deg)")
    boxes(ax[0, 2], "latdev", lambda x: x, "max lateral deviation (% BL)")
    boxes(ax[1, 0], "Lrel", lambda x: x, "body length / median")
    for nm, c in [("real", "tab:orange"), ("sim", "tab:blue")]:
        for a, key, tk in [(ax[1, 1], "py", "head_yaw"), (ax[1, 2], "pl", "length")]:
            if np.isfinite(S[nm][key]).any():
                a.plot(S[nm]["fq"], S[nm][key] / np.nanmax(S[nm][key]), color=c,
                       label=f"{nm}: peak {R[nm]['dominant_period_s'][tk]} s")
    for a, tt in [(ax[1, 1], "head-yaw power (Lomb-Scargle, norm.; real = trackable tier)"),
                  (ax[1, 2], "body-length power (norm.; real = trackable tier)")]:
        a.set_xlabel("frequency (Hz)"); a.set_title(tt, fontsize=9); a.legend(fontsize=8); a.grid(alpha=0.3)
    bins = np.linspace(0, 0.5, 26)
    for nm, c in [("real", "tab:orange"), ("sim", "tab:blue")]:
        sp = S[nm]["speed"][np.isfinite(S[nm]["speed"])]
        if len(sp):
            ax[2, 0].hist(np.clip(sp, 0, 0.5), bins=bins, density=True, alpha=0.5, color=c,
                          label=f"{nm}: median {np.median(sp):.3f} BL/s, stopped {100*np.mean(sp<STOP_BLS):.0f} %")
    ax[2, 0].axvline(STOP_BLS, color="r", ls="--", lw=0.8)
    ax[2, 0].set_xlabel("speed rel. substrate (BL/s, 2 s window)"); ax[2, 0].legend(fontsize=8)
    ax[2, 0].set_title("speed", fontsize=9)
    xs = np.arange(10) * 10 + 5
    for lab, nm, sel, c in groups:
        p_ = S[nm]["prof"][S[nm][sel]]
        if len(p_):
            ax[2, 1].plot(xs, np.nanmedian(p_, 0), "o-", color=c, label=lab)
            ax[2, 1].fill_between(xs, np.nanpercentile(p_, 25, 0), np.nanpercentile(p_, 75, 0), color=c, alpha=0.15)
    ax[2, 1].set_xlabel("position along body (% from head)"); ax[2, 1].set_ylabel("|curvature| x L")
    ax[2, 1].set_title("curvature profile (median, IQR)", fontsize=9); ax[2, 1].legend(fontsize=8); ax[2, 1].grid(alpha=0.3)
    bins = np.linspace(-120, 120, 41)
    for lab, nm, sel, c in groups:
        b_ = S[nm]["bend"][S[nm][sel] & np.isfinite(S[nm]["bend"])]
        ax[2, 2].hist(b_, bins=bins, density=True, histtype="step", lw=1.5, color=c, label=lab)
    ax[2, 2].set_xlabel("signed bend (deg)"); ax[2, 2].legend(fontsize=8); ax[2, 2].set_title("signed bend distribution", fontsize=9)
    fig.suptitle("Tardigrade body kinematics: real video vs simulation " + os.path.basename(SIM_DIR) + " - identical definitions "
                 f"(real strict n={R['real']['n_usable']}, trackable n={R['real']['n_trackable']}; sim n={R['sim']['n_usable']})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "compare.png"), dpi=80)
    plt.close(fig)


def frame_reader_real():
    cap = cv2.VideoCapture(REAL_VIDEO)
    def get(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        return f
    return get


def overlay(name, i, S, D, img=None, write=True, reader=None):
    if name == "real":
        img = (reader or frame_reader_real())(i)
        m, _, _ = segment_real(img, None)
    else:
        g = cv2.imread(os.path.join(SIM_DIR, f"proj_{i:03d}.png"), cv2.IMREAD_GRAYSCALE)
        m, _, _ = segment_sim(g)
        img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    o = img.copy()
    if m is not None:
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(o, cs, -1, (0, 255, 255), 2)
    P = S["pts"][i]
    if np.isfinite(P).all():
        cv2.polylines(o, [np.round(P).astype(np.int32)], False, (255, 0, 255), 2)
        for p in P:
            cv2.circle(o, tuple(int(v) for v in p), 3, (255, 0, 255), -1)
        cv2.circle(o, tuple(int(v) for v in P[0]), 10, (0, 0, 255) if S["confident"][i] else (0, 165, 255), 3)
        cv2.putText(o, "H", tuple(int(v) + 12 for v in P[0]), 0, 0.8, (0, 0, 255), 2)
        cv2.circle(o, tuple(int(v) for v in P[-1]), 7, (255, 128, 0), 2)
    txt = f"{name} f{i} ok={bool(S['ok'][i])} bend={S['bend'][i]:.0f} yaw={S['yaw_all'][i]:.0f} lat={S['latdev'][i]:.1f}%"
    cv2.putText(o, txt, (8, 24), 0, 0.6, (0, 0, 0), 3); cv2.putText(o, txt, (8, 24), 0, 0.6, (255, 255, 255), 1)
    if write:
        cv2.imwrite(os.path.join(OUT, f"{name}_overlay_{i:04d}.jpg"), o, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return o


def qc_sheet(name, S, D, n=30, seed=1):
    """random sample of accepted frames as a 6-column contact sheet (segmentation proof)."""
    okidx = np.nonzero(S["ok"])[0]
    if len(okidx) == 0:
        return
    pick = np.sort(np.random.default_rng(seed).choice(okidx, min(n, len(okidx)), replace=False))
    rd = frame_reader_real() if name == "real" else None
    tiles = [cv2.resize(overlay(name, int(i), S, D, write=False, reader=rd), (400, 300 if name == "real" else 400))
             for i in pick]
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[k:k + 6]) for k in range(0, len(tiles), 6)])
    cv2.imwrite(os.path.join(OUT, f"{name}_qc_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reextract", action="store_true")
    ap.add_argument("--only", choices=["real", "sim"], default=None)
    ap.add_argument("--overlays", type=int, default=8, help="overlay images per input")
    ap.add_argument("--sim-dir", default=None, help="folder with proj_###.png of a simulated run (default motion_v9)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    global SIM_DIR
    if a.sim_dir:
        SIM_DIR = os.path.abspath(a.sim_dir)
    R, S, Dd = {}, {}, {}
    for name in ["sim", "real"]:
        D = None if a.reextract else load_cache(name)
        if D is None or (a.only and a.only != name and False):
            print(f"extracting {name} ...", flush=True)
            recs, fps = extract_real(REAL_VIDEO) if name == "real" else extract_sim(SIM_DIR)
            save_cache(name, recs, fps)
            D = load_cache(name)
        st, se = analyse(name, D)
        R[name], S[name], Dd[name] = st, se, D
        plot_timeseries(name, se, st)
        json.dump(jsonable(st), open(os.path.join(OUT, f"{name}_stats.json"), "w"), indent=1)
        okidx = np.nonzero(se["ok"])[0]
        if len(okidx) and a.overlays:
            pick = okidx[np.linspace(0, len(okidx) - 1, a.overlays).astype(int)]
            for i in pick:
                overlay(name, int(i), se, D)
            qc_sheet(name, se, D)
        print(json.dumps(jsonable({k: st[k] for k in ["n_usable", "usable_fraction", "rejections", "body_length_px",
              "bend_abs_deg", "head_yaw_abs_deg", "lateral_dev_pct_BL", "length_rel", "dominant_period_s",
              "speed_BL_per_s", "stop_fraction"]}), indent=None))
    plot_compare(R, S)


if __name__ == "__main__":
    main()
