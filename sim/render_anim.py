"""Walking tardigrade through the traced PUMA optics, one focus plane, N frames.
  python render_anim.py --out ../renders/anim --focus=-30 --frames 12 --rings 2 --nlam 5
Each frame: deform the 3D RI volume (animate_specimen.Walker), then the same wave-optics pipeline
as render_color.py (traced pupils are computed once - the optics do not move)."""
import argparse, os, sys, time, json
import numpy as np
from PIL import Image
from specimen import Tardigrade, n_water
from animate_specimen import Walker
import waveimage as wi

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--focus", default="-30")
ap.add_argument("--frames", type=int, default=12); ap.add_argument("--rings", type=int, default=2)
ap.add_argument("--nlam", type=int, default=5); ap.add_argument("--n", type=int, default=768)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
f = float(a.focus)
here = os.path.dirname(os.path.abspath(__file__))
T = Tardigrade(os.path.join(here, "../data/nanoct/labels.npy"), os.path.join(here, "../data/nanoct/grey.npy"))
L = np.load(os.path.join(here, "../data/nanoct/labels.npy"))
walker = Walker(L)
g = wi.Grid(a.n, T.voxel_um)
lams = np.linspace(0.42, 0.68, a.nlam); weights = wi.led_6500k(lams) * (lams[1] - lams[0])
sys.path.insert(0, os.path.join(here, "..", "optics"))
import pupil as PUP
H0 = 3.0
tp = PUP.TracedPupil((H0 - f) / 1000.0, list(lams))
planes = {l: [-(tp.depth[l] * 1000 - H0)] for l in lams}
phases = {l: [tp.phase_fn(l)] for l in lams}
dn0, gut0 = T.dn589.copy(), T.gut.copy()
t0 = time.time()
for k in range(a.frames):
    ph = k / a.frames
    T.dn589 = walker.warp(dn0, ph); T.gut = walker.warp(gut0, ph)
    ims = []
    for lam in lams:
        dn, mu = T.slab(lam)
        nz = dn.shape[0] // 2
        dn2 = 0.5 * (dn[0:2 * nz:2] + dn[1:2 * nz:2]); mu2 = 0.5 * (mu[0:2 * nz:2] + mu[1:2 * nz:2])
        ims += wi.simulate_wavelength((dn2, mu2), 2 * T.voxel_um, g, lam, float(n_water(lam)), 0.40, 0.30,
                                      planes[lam], phases[lam], n_rings=a.rings)
    rgb = wi.spectral_to_srgb(ims, lams, weights)
    img = (wi.encode_srgb(rgb, exposure=0.85) * 255).astype(np.uint8)
    Image.fromarray(np.transpose(img, (1, 0, 2))).save(os.path.join(a.out, f"frame_{k:02d}.png"))
    print(f"frame {k+1}/{a.frames} {time.time()-t0:.0f}s", flush=True)
json.dump({"focus_um": f, "frames": a.frames, "rings": a.rings, "nlam": a.nlam, "dx_um": g.dx},
          open(os.path.join(a.out, "meta.json"), "w"))
print("ANIM_DONE")
