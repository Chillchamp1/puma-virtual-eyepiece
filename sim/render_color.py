"""Spectral brightfield render -> sRGB.  usage:
  python render_color.py --out ../renders/ideal --focus -36,-24,-12 [--pupil ideal|traced] [--rings 2] [--nlam 7]
"""
import argparse, time, os, json
import numpy as np
from PIL import Image
from specimen import Tardigrade, n_water
import waveimage as wi

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--focus", default="-36")
ap.add_argument("--pupil", default="ideal")
ap.add_argument("--rings", type=int, default=2)
ap.add_argument("--nlam", type=int, default=7)
ap.add_argument("--n", type=int, default=768)
ap.add_argument("--na", type=float, default=0.40)
ap.add_argument("--nac", type=float, default=0.30)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
focus = [float(f) for f in a.focus.split(",")]

T = Tardigrade(os.path.join(os.path.dirname(__file__), "../data/nanoct/labels.npy"),
               os.path.join(os.path.dirname(__file__), "../data/nanoct/grey.npy"))
g = wi.Grid(a.n, T.voxel_um)
lams = np.linspace(0.42, 0.68, a.nlam)
dlam = lams[1] - lams[0]
weights = wi.led_6500k(lams) * dlam

H0_UM = 3.0          # water film between the animal's dorsal surface and the cover glass
planes = {lam: list(focus) for lam in lams}         # ideal lens: every colour sharp at the knob plane
phases = {lam: [None] * len(focus) for lam in lams}
if a.pupil == "traced":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "optics"))
    import pupil as PUP
    tps = []
    for f in focus:
        tp = PUP.TracedPupil((H0_UM - f) / 1000.0, list(lams))
        tps.append(tp)
        print(f"focus {f:+.0f} um: knob {tp.wd:.5f} mm; sharp planes " +
              ", ".join(f"{l*1000:.0f}:{(tp.depth[l]*1000 - H0_UM):.1f}" for l in lams), flush=True)
    for lam in lams:
        planes[lam] = [-(tp.depth[lam] * 1000 - H0_UM) for tp in tps]
        phases[lam] = [tp.phase_fn(lam) for tp in tps]

t0 = time.time()
stack = {f: [] for f in focus}
for lam in lams:
    dn, mu = T.slab(lam)
    nz = dn.shape[0] // 2
    dn2 = 0.5 * (dn[0:2 * nz:2] + dn[1:2 * nz:2]); mu2 = 0.5 * (mu[0:2 * nz:2] + mu[1:2 * nz:2])
    ims = wi.simulate_wavelength((dn2, mu2), 2 * T.voxel_um, g, lam, float(n_water(lam)), a.na, a.nac,
                                 planes[lam], phases[lam], n_rings=a.rings)
    for f, im in zip(focus, ims):
        stack[f].append(im)
    print(f"lambda {lam*1000:.0f} nm done, {time.time()-t0:.0f}s", flush=True)

meta = {"lams_um": lams.tolist(), "weights": weights.tolist(), "focus_um": focus, "na": a.na, "nac": a.nac,
        "rings": a.rings, "dx_um": g.dx, "pupil": a.pupil}
json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
for f in focus:
    np.save(os.path.join(a.out, f"spectral_f{f:+.0f}.npy"), np.stack(stack[f]).astype(np.float16))
    rgb = wi.spectral_to_srgb(stack[f], lams, weights)
    img = (wi.encode_srgb(rgb, exposure=0.85) * 255).astype(np.uint8)
    Image.fromarray(np.transpose(img, (1, 0, 2))).save(os.path.join(a.out, f"rgb_f{f:+.0f}.png"))
print("DONE", time.time() - t0)
