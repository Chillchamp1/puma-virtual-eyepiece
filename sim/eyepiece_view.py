"""
Composite "look into the eyepiece" frames from the simulated patches.

Physics-based part: the central 207 um patch (wave-optics simulation through the traced system).
Composited part (stated openly): the rest of the FN 20 field (1.0 mm in the object) is empty slide;
it is filled with the simulated empty-slide background colour of the same render, and the field stop
(sharp, it lies in the intermediate image) bounds it. Outside: black (inside of the ocular).

  python eyepiece_view.py <render_dir> <out_dir> [--size 1100] [--detail 900]
"""
import sys, os, glob, json
import numpy as np
from PIL import Image

src, dst = sys.argv[1], sys.argv[2]
size = int(sys.argv[sys.argv.index("--size") + 1]) if "--size" in sys.argv else 1100
detail = int(sys.argv[sys.argv.index("--detail") + 1]) if "--detail" in sys.argv else 900
os.makedirs(dst, exist_ok=True)
meta = json.load(open(os.path.join(src, "meta.json")))
dx = meta["dx_um"]
FIELD_UM = 1000.0

for f in sorted(glob.glob(os.path.join(src, "rgb_f*.png"))):
    tag = os.path.basename(f)[4:-4]
    im = np.asarray(Image.open(f)).astype(np.float32) / 255
    n = im.shape[0]
    patch_um = n * dx
    bg = np.median(np.concatenate([im[:20].reshape(-1, 3), im[-20:].reshape(-1, 3)]), 0)
    # full eyepiece field
    scale = size / FIELD_UM                       # px per um
    psz = int(round(patch_um * scale))
    patch = np.asarray(Image.fromarray((im * 255).astype(np.uint8)).resize((psz, psz), Image.LANCZOS)).astype(np.float32) / 255
    view = np.ones((size, size, 3), np.float32) * bg
    o = (size - psz) // 2
    # feather the patch border into the background (the patch edge is empty slide anyway)
    yy, xx = np.mgrid[0:psz, 0:psz]
    edge = np.minimum.reduce([xx, yy, psz - 1 - xx, psz - 1 - yy]) / max(4, psz * 0.04)
    a = np.clip(edge, 0, 1)[..., None]
    view[o:o + psz, o:o + psz] = a * patch + (1 - a) * bg
    Y, X = np.mgrid[0:size, 0:size]
    r = np.hypot(X - size / 2 + 0.5, Y - size / 2 + 0.5) / (size / 2)
    stop = np.clip((1.0 - r) * size / 2 / 1.2, 0, 1)[..., None]   # ~1 px soft field-stop edge
    view = view * stop
    Image.fromarray((np.clip(view, 0, 1) * 255).astype(np.uint8)).save(os.path.join(dst, f"eye_{tag}.jpg"), quality=92)
    # detail view: centre patch, magnified (digital zoom of the same simulated data)
    c = n // 2; h = int(n * 0.42)
    crop = im[c - h:c + h, c - h:c + h]
    Image.fromarray((crop * 255).astype(np.uint8)).resize((detail, detail), Image.LANCZOS).save(
        os.path.join(dst, f"detail_{tag}.jpg"), quality=92)
    print("wrote", tag)
