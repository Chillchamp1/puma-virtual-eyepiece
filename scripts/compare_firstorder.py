"""Compare the fast first-order models (Born, Rytov) with the multi-slice reference.

  python scripts/compare_firstorder.py [--rings 4] [--nac 0.30]

One wavelength (550 nm), ideal lens, three focus planes of the tardigrade. Prints time and the error
relative to multi-slice and writes renders/compare_firstorder.png (rows: multi-slice, Rytov, Born).
"""
import argparse, os, sys, time
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim"))
from specimen import Tardigrade, n_water
import waveimage as wi
import firstorder as fo

ap = argparse.ArgumentParser()
ap.add_argument("--rings", type=int, default=4); ap.add_argument("--nac", type=float, default=0.30)
a = ap.parse_args()
T = Tardigrade(os.path.join(ROOT, "data/nanoct/labels.npy"), os.path.join(ROOT, "data/nanoct/grey.npy"))
g = wi.Grid(768, T.voxel_um)
lam = 0.55
dn, mu = T.slab(lam)
nz = dn.shape[0] // 2
dn2 = 0.5 * (dn[0:2 * nz:2] + dn[1:2 * nz:2]); mu2 = 0.5 * (mu[0:2 * nz:2] + mu[1:2 * nz:2])
focus = [-12.0, -30.0, -48.0]
args = ((dn2, mu2), 2 * T.voxel_um, g, lam, float(n_water(lam)), 0.40, a.nac, focus, None)
res = {}
for name, fn in [("multi-slice", lambda: wi.simulate_wavelength(*args, n_rings=a.rings)),
                 ("rytov", lambda: fo.simulate_wavelength(*args, n_rings=a.rings, model="rytov")),
                 ("born", lambda: fo.simulate_wavelength(*args, n_rings=a.rings, model="born"))]:
    fn()                                   # warm-up (kernel compilation, FFT plans)
    t = time.time(); ims = fn(); dt = time.time() - t
    res[name] = (ims, dt)
ref = res["multi-slice"][0]
c = slice(768 // 2 - 330, 768 // 2 + 330)
for name, (ims, dt) in res.items():
    err = [float(np.sqrt(np.mean((im[c, c] - r[c, c]) ** 2)) / float(np.std(r[c, c]))) for im, r in zip(ims, ref)]
    print(f"{name:12s} {dt:6.2f} s   rms error / image contrast: " + ", ".join(f"{e:.2f}" for e in err))
rows = []
for name in ("multi-slice", "rytov", "born"):
    rows.append(np.concatenate([np.clip(im[c, c].T * 0.85, 0, 1) for im in res[name][0]], 1))
img = (np.concatenate(rows, 0) ** (1 / 2.2) * 255).astype(np.uint8)
os.makedirs(os.path.join(ROOT, "renders"), exist_ok=True)
Image.fromarray(img).save(os.path.join(ROOT, "renders", "compare_firstorder.png"))
print("wrote renders/compare_firstorder.png")
