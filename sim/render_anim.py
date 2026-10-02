"""Walking tardigrade through the PUMA optics, one focus plane, a sequence of frames.
  python render_anim.py --out ../renders/anim --focus=-30 --seconds 3.2 --fps 12 --rings 2 --nlam 5
  python render_anim.py --out ../renders/preview --preview          # 1 wavelength, 7 source points, ideal lens
Each frame: pose the articulated rig (animate_specimen.Walker) at time t, warp the 3D RI volume, then the
same wave-optics pipeline as render_color.py (traced pupils are computed once - the optics do not move).
The observer follows the animal with the stage, so the view is a lagging camera, not the animal's frame."""
import argparse, os, sys, time, json
import numpy as np
from PIL import Image
from specimen import Tardigrade, n_water
from animate_specimen import Walker
import waveimage as wi

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--focus", default="-30")
ap.add_argument("--seconds", type=float, default=3.2); ap.add_argument("--fps", type=float, default=12)
ap.add_argument("--t0", type=float, default=0.0)
ap.add_argument("--rings", type=int, default=2); ap.add_argument("--nlam", type=int, default=5)
ap.add_argument("--n", type=int, default=768); ap.add_argument("--pupil", default="traced")
ap.add_argument("--preview", action="store_true", help="fast look: 3 wavelengths, 19 source points, ideal lens")
ap.add_argument("--frames", default="", help="comma-separated frame indices to render (default: all)")
ap.add_argument("--proj", action="store_true", help="also save a projection of the deformed volume per frame")
a = ap.parse_args()
if a.preview:
    a.nlam, a.rings, a.pupil = 3, 2, "ideal"
os.makedirs(a.out, exist_ok=True)
f = float(a.focus)
here = os.path.dirname(os.path.abspath(__file__))
nano = os.path.join(here, "../data/nanoct")
T = Tardigrade(os.path.join(nano, "labels.npy"), os.path.join(nano, "grey.npy"))
walker = Walker(T.dn589, np.load(os.path.join(nano, "labels.npy")), cache=os.path.join(nano, "rig.npz"))
g = wi.Grid(a.n, T.voxel_um)
lams = np.array([0.55]) if a.nlam == 1 else np.linspace(0.42, 0.68, a.nlam)
weights = wi.led_6500k(lams) * ((lams[1] - lams[0]) if len(lams) > 1 else 1.0)
H0 = 3.0
planes = {l: [f] for l in lams}
phases = {l: [None] for l in lams}
if a.pupil == "traced":
    sys.path.insert(0, os.path.join(here, "..", "optics"))
    import pupil as PUP
    tp = PUP.TracedPupil((H0 - f) / 1000.0, list(lams))
    planes = {l: [-(tp.depth[l] * 1000 - H0)] for l in lams}
    phases = {l: [tp.phase_fn(l)] for l in lams}
dn0, gut0 = T.dn589, T.gut
nfr = int(round(a.seconds * a.fps))
todo = [int(k) for k in a.frames.split(",")] if a.frames else range(nfr)
t0 = time.time()
for k in todo:
    t = a.t0 + k / a.fps
    T.dn589, T.gut = walker.warp([dn0, gut0], t)
    if a.proj:   # line integral of delta-n along the optical axis (motion check, not an optical image)
        p = T.dn589.sum(1).T                     # same orientation as the rendered frames
        Image.fromarray((255 * np.clip(p / (np.percentile(p, 99.9) + 1e-9), 0, 1)).astype(np.uint8)).save(
            os.path.join(a.out, f"proj_{k:03d}.png"))
    ims = []
    for lam in lams:
        dn, mu = T.slab(lam)
        nz = dn.shape[0] // 2
        dn2 = 0.5 * (dn[0:2 * nz:2] + dn[1:2 * nz:2]); mu2 = 0.5 * (mu[0:2 * nz:2] + mu[1:2 * nz:2])
        ims += wi.simulate_wavelength((dn2, mu2), 2 * T.voxel_um, g, lam, float(n_water(lam)), 0.40, 0.30,
                                      planes[lam], phases[lam], n_rings=a.rings)
    if len(lams) == 1:
        img = (wi.encode_srgb(np.repeat(ims[0][..., None], 3, -1), exposure=0.85) * 255).astype(np.uint8)
    else:
        rgb = wi.spectral_to_srgb(ims, lams, weights)
        img = (wi.encode_srgb(rgb, exposure=0.85) * 255).astype(np.uint8)
    Image.fromarray(np.transpose(img, (1, 0, 2))).save(os.path.join(a.out, f"frame_{k:03d}.png"))
    print(f"frame {k + 1}/{nfr} t={t:.2f}s {time.time() - t0:.0f}s", flush=True)
json.dump({"focus_um": f, "seconds": a.seconds, "fps": a.fps, "t0": a.t0, "rings": a.rings, "nlam": int(len(lams)),
           "pupil": a.pupil, "dx_um": g.dx}, open(os.path.join(a.out, "meta.json"), "w"))
if not a.frames:
    frames = [Image.open(os.path.join(a.out, f"frame_{k:03d}.png")) for k in range(nfr)]
    frames[0].save(os.path.join(a.out, "walk.gif"), save_all=True, append_images=frames[1:],
                   duration=int(round(1000 / a.fps)), loop=0)
print("ANIM_DONE")
