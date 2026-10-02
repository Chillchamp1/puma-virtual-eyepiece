"""Render the tardigrade rolled about its long axis (other viewing angles) and publish into the viewer.

  python scripts/render_views.py                      # rolls 30..330 deg in 30 deg steps
  python scripts/render_views.py --rolls 90,180 --force

Roll 0 is the standard view (docs/img/k030). For every other roll the 3D refractive-index volume is
rotated about the body axis, shifted so the animal's top surface keeps the same water margin as in the
standard view, and rendered exactly like the k030 preset (multi-slice, 21 focus planes x 9 wavelengths,
61 condenser points, ray-traced PUMA optics and ideal lens). The traced pupils do not depend on the
specimen, so they are computed once for all rolls. Output: docs/img/views/r<roll>/<optics>/ and the
"views" entry of docs/img/manifest.json, which the viewer reads to enable the roll control.
"""
import argparse, json, os, subprocess, sys, time
import numpy as np
from scipy import ndimage
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim"))
sys.path.insert(0, os.path.join(ROOT, "optics"))
from specimen import Tardigrade, n_water
import waveimage as wi

NAC, RINGS, NLAM, PLANES = 0.30, 4, 9, 21
H0_UM = 3.0
BODY = 0.006          # delta-n threshold for "inside the animal"


def rotate(vol, deg):
    """rotate (z, y, x) about the z (body) axis; GPU when available."""
    if wi._cp is not None:
        import cupyx.scipy.ndimage as cnd
        r = cnd.rotate(wi._cp.asarray(vol), deg, axes=(1, 2), reshape=True, order=1)
        out = r.get(); del r
        wi._cp.get_default_memory_pool().free_all_blocks()
        return out
    return ndimage.rotate(vol, deg, axes=(1, 2), reshape=True, order=1)


def window(vol, y0, ny, xc, nx):
    """crop/zero-pad vol[:, y0:y0+ny, xc-nx/2 : xc+nx/2]."""
    out = np.zeros((vol.shape[0], ny, nx), vol.dtype)
    xs = int(round(xc - nx / 2))
    ys_src = slice(max(0, y0), min(vol.shape[1], y0 + ny))
    xs_src = slice(max(0, xs), min(vol.shape[2], xs + nx))
    out[:, ys_src.start - y0:ys_src.stop - y0, xs_src.start - xs:xs_src.stop - xs] = vol[:, ys_src, xs_src]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rolls", default=",".join(str(r) for r in range(30, 360, 30)))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    manifest_path = os.path.join(ROOT, "docs", "img", "manifest.json")
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    views = manifest.setdefault("views", {"illum": "k030", "rolls": [0]})
    rolls = [int(r) for r in a.rolls.split(",") if (int(r) not in views["rolls"] or a.force) and int(r) % 360]
    if not rolls:
        print("nothing to do"); return
    T = Tardigrade(os.path.join(ROOT, "data/nanoct/labels.npy"), os.path.join(ROOT, "data/nanoct/grey.npy"))
    dn0, gut0 = T.dn589, T.gut
    nz, ny, nx = dn0.shape
    body0 = dn0 > BODY
    top_margin = int(np.nonzero(body0.any((0, 2)))[0].min())          # water above the dorsal surface
    g = wi.Grid(768, T.voxel_um)
    lams = np.linspace(0.42, 0.68, NLAM)
    weights = wi.led_6500k(lams) * (lams[1] - lams[0])
    focus = [-3.0 * (i + 1) for i in range(PLANES)]
    import pupil as PUP
    t0 = time.time()
    tps = [PUP.TracedPupil((H0_UM - f) / 1000.0, list(lams)) for f in focus]
    traced = ({l: [-(tp.depth[l] * 1000 - H0_UM) for tp in tps] for l in lams},
              {l: [tp.phase_fn(l) for tp in tps] for l in lams})
    ideal = ({l: list(focus) for l in lams}, {l: [None] * PLANES for l in lams})
    print(f"traced pupils: {time.time() - t0:.0f}s", flush=True)
    NXW = 420
    for roll in rolls:
        t1 = time.time()
        dn = rotate(dn0, roll); gut = rotate(gut0, roll)
        body = dn > BODY
        ymin = int(np.nonzero(body.any((0, 2)))[0].min())
        xc = float(np.nonzero(body.any((0, 1)))[0].mean())
        T.dn589 = window(dn, ymin - top_margin, ny, xc, NXW)
        T.gut = window(gut, ymin - top_margin, ny, xc, NXW)
        del dn, gut, body
        # one propagation through the animal serves both optics: the field behind the specimen is the
        # same, only the pupil differs, so traced and ideal planes are evaluated in the same pass
        stacks = {o: {f: [] for f in focus} for o in ("traced", "ideal")}
        for lam in lams:
            d, mu = T.slab(lam)
            h = d.shape[0] // 2
            d2 = 0.5 * (d[0:2 * h:2] + d[1:2 * h:2]); mu2 = 0.5 * (mu[0:2 * h:2] + mu[1:2 * h:2])
            ims = wi.simulate_wavelength((d2, mu2), 2 * T.voxel_um, g, lam, float(n_water(lam)), 0.40, NAC,
                                         traced[0][lam] + ideal[0][lam], traced[1][lam] + ideal[1][lam],
                                         n_rings=RINGS)
            for f, it, ii in zip(focus, ims[:PLANES], ims[PLANES:]):
                stacks["traced"][f].append(it); stacks["ideal"][f].append(ii)
        for optics in ("traced", "ideal"):
            stack = stacks[optics]
            tmp = os.path.join(ROOT, "r", f"v{roll}{optics[0]}")          # short path: Windows MAX_PATH
            os.makedirs(tmp, exist_ok=True)
            json.dump({"lams_um": lams.tolist(), "focus_um": focus, "na": 0.40, "nac": NAC, "rings": RINGS,
                       "dx_um": g.dx, "pupil": optics, "roll_deg": roll}, open(os.path.join(tmp, "meta.json"), "w"))
            for f in focus:
                rgb = wi.spectral_to_srgb(stack[f], lams, weights)
                img = (wi.encode_srgb(rgb, exposure=0.85) * 255).astype(np.uint8)
                Image.fromarray(np.transpose(img, (1, 0, 2))).save(os.path.join(tmp, f"rgb_f{f:+.0f}.png"))
            subprocess.run([sys.executable, "eyepiece_view.py", tmp,
                            os.path.join(ROOT, "docs", "img", "views", f"r{roll:03d}", optics)],
                           cwd=os.path.join(ROOT, "sim"), check=True, capture_output=True)
        views["rolls"] = sorted(set(views["rolls"]) | {roll})
        json.dump(manifest, open(manifest_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(f"built roll {roll} in {time.time() - t1:.0f}s", flush=True)


if __name__ == "__main__":
    main()
