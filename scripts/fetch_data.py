"""Download the third-party inputs that are NOT redistributed in this repository.

  python scripts/fetch_data.py            # nanoCT tardigrade + PUMA FreeCAD files

* Tardigrade nanoCT (Gross et al. 2019, Zoological Letters 5:14, CC BY 4.0), Springer Nature figshare:
    Additional file 3  (whole-body scan, 3D TIFF, 50 MB)   article 8116028
    Additional file 4  (segmentation labels, 1.7 MB)       article 8116031
  converted to data/nanoct/{grey,labels}.npy (axes z, y, x; 270 nm voxels)
* PUMA microscope CAD (Dr Paul J. Tadrous, GPL-3.0): FreeCAD files from github.com/TadPath/PUMA
"""
import json, os, urllib.request
import numpy as np
from PIL import Image, ImageSequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NANO = os.path.join(ROOT, "data", "nanoct")
CAD = os.path.join(ROOT, "data", "puma_freecad")
FIGSHARE = {"grey": 8116028, "labels": 8116031}
PUMA_FILES = ["Stage", "Focus_Gears", "Legs", "Monocular", "QuickRelease_v2.0", "FilterBlock",
              "Dominus_part1", "Dominus_part2", "Dominus_part3", "Z_Motor", "Stabiliser", "PUMALite"]


def get(url, dst):
    if os.path.exists(dst):
        return
    print("download", url)
    urllib.request.urlretrieve(url, dst)


def main():
    os.makedirs(NANO, exist_ok=True)
    os.makedirs(CAD, exist_ok=True)
    for key, art in FIGSHARE.items():
        meta = json.load(urllib.request.urlopen(f"https://api.figshare.com/v2/articles/{art}"))
        tif = os.path.join(NANO, f"{key}.tif")
        get(meta["files"][0]["download_url"], tif)
        npy = os.path.join(NANO, f"{key}.npy")
        if not os.path.exists(npy):
            vol = np.stack([np.array(f) for f in ImageSequence.Iterator(Image.open(tif))])
            np.save(npy, vol)
            print(key, vol.shape, vol.dtype)
    for name in PUMA_FILES:
        get(f"https://raw.githubusercontent.com/TadPath/PUMA/main/FreeCAD/{name}.FCStd",
            os.path.join(CAD, f"{name}.FCStd"))
    print("done")


if __name__ == "__main__":
    main()
