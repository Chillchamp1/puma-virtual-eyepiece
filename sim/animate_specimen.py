"""
Walking tardigrade: deforms the 3D refractive-index volume per animation phase.

Motion model (plausible, not motion-captured):
  * 4 leg pairs step in a posterior-to-anterior metachronal wave (as described for tardigrade
    walking, e.g. Nirody et al. 2021 PNAS "Tardigrades exhibit robust interlimb coordination"):
    each leg swings forward along the body axis and lifts slightly, contralateral legs in antiphase.
  * the head region sways sideways (the animal "searches"), the trunk follows with a small lag.
Leg positions come from the segmented claw glands (label 11, two per leg pair side).
Displacements are smooth Gaussian-weighted fields; the volume is resampled (backward warp).
"""
import numpy as np
from scipy import ndimage

VOX = 0.27  # um


def leg_sites(labels):
    lab, n = ndimage.label(labels == 11)
    cents = ndimage.center_of_mass(labels == 11, lab, range(1, n + 1))
    sizes = ndimage.sum(labels == 11, lab, range(1, n + 1))
    cents = [c for c, s in sorted(zip(cents, sizes), key=lambda t: -t[1])[:8]]
    return sorted(cents, key=lambda c: c[0])          # sorted posterior -> anterior (stack z)


class Walker:
    def __init__(self, labels, body_axis_center_x=None):
        self.shape = labels.shape
        self.legs = leg_sites(labels)
        zc = np.array([c[0] for c in self.legs])
        self.z_head = labels.shape[0] * 0.80
        self.x_mid = np.mean([c[2] for c in self.legs])
        nz, ny, nx = labels.shape
        self.Z, self.Y, self.X = np.meshgrid(np.arange(nz, dtype=np.float32), np.arange(ny, dtype=np.float32),
                                             np.arange(nx, dtype=np.float32), indexing="ij")

    def displacement(self, phase):
        """phase in [0, 1): one full stride cycle. Returns (dz, dy, dx) in voxels."""
        dz = np.zeros(self.shape, np.float32); dy = np.zeros_like(dz); dx = np.zeros_like(dz)
        stride = 4.5 / VOX          # 4.5 um leg swing along the axis
        lift = 1.6 / VOX            # 1.6 um lift (towards the objective = -y, dorsal)
        rad = 13.0 / VOX
        for k, (cz, cy, cx) in enumerate(self.legs):
            pair = k // 2
            side = 1 if cx > self.x_mid else -1
            ph = 2 * np.pi * (phase - 0.25 * pair) + (np.pi if side > 0 else 0.0)
            # leg tip sits laterally outside the claw gland
            tip_x = cx + side * 6.0 / VOX
            w = np.exp(-(((self.Z - cz) ** 2 + (self.X - tip_x) ** 2) / (2 * rad ** 2)
                         + (self.Y - cy) ** 2 / (2 * (2 * rad) ** 2)))
            dz += w * stride * np.sin(ph)
            dy += w * (-lift) * np.clip(np.cos(ph), 0, None)
        # head sway (lateral, x) growing towards the anterior end, slower than the stride
        sway = 3.5 / VOX * np.sin(2 * np.pi * phase * 0.5)
        ramp = np.clip((self.Z - self.z_head * 0.7) / (self.shape[0] - self.z_head * 0.7), 0, 1) ** 2
        dx += sway * ramp
        return dz, dy, dx

    def warp(self, vol, phase, order=1):
        dz, dy, dx = self.displacement(phase)
        coords = np.stack([self.Z - dz, self.Y - dy, self.X - dx])
        return ndimage.map_coordinates(vol, coords, order=order, mode="constant", cval=0.0).astype(vol.dtype)
