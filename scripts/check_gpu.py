"""Verify the optional GPU backend against the CPU reference on one real test image.

  python scripts/check_gpu.py

Renders one focus plane (1 wavelength, 7 condenser points, ideal lens) of the tardigrade twice,
with PUMA_BACKEND=cpu and PUMA_BACKEND=gpu, and prints the max difference and the speed-up.
Needs data/nanoct (run scripts/fetch_data.py first) and CuPy matching your CUDA version,
e.g. pip install cupy-cuda12x
"""
import os, subprocess, sys, time, tempfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHILD = r'''
import sys, time, numpy as np
sys.path.insert(0, r"{sim}")
from specimen import Tardigrade, n_water
import waveimage as wi
T = Tardigrade(r"{nano}/labels.npy", r"{nano}/grey.npy")
g = wi.Grid(768, T.voxel_um)
dn, mu = T.slab(0.55)
nz = dn.shape[0] // 2
dn2 = 0.5 * (dn[0:2*nz:2] + dn[1:2*nz:2]); mu2 = 0.5 * (mu[0:2*nz:2] + mu[1:2*nz:2])
t = time.time()
im = wi.simulate_wavelength((dn2, mu2), 2 * T.voxel_um, g, 0.55, float(n_water(0.55)), 0.40, 0.30, [-30.0], None, n_rings=1)[0]
dt = time.time() - t
np.save(r"{out}", im)
print(wi.BACKEND, dt)
'''


def run(backend, out):
    env = dict(os.environ, PUMA_BACKEND=backend)
    code = CHILD.format(sim=os.path.join(ROOT, "sim"), nano=os.path.join(ROOT, "data", "nanoct"), out=out)
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        sys.exit(f"{backend} run failed")
    name, dt = r.stdout.split()[-2:]
    return name, float(dt)


def main():
    tmp = tempfile.mkdtemp()
    a, b = os.path.join(tmp, "cpu.npy"), os.path.join(tmp, "gpu.npy")
    n1, t1 = run("cpu", a)
    n2, t2 = run("gpu", b)
    A, B = np.load(a), np.load(b)
    diff = float(np.abs(A - B).max())
    print(f"cpu: {t1:.1f} s   gpu ({n2}): {t2:.1f} s   speed-up {t1 / t2:.1f}x")
    print(f"max |cpu - gpu| = {diff:.2e}  (image mean {A.mean():.3f})")
    if diff < 1e-3:
        print("OK: GPU backend matches the CPU reference")
    else:
        sys.exit("MISMATCH: do not use the GPU backend")


if __name__ == "__main__":
    main()
