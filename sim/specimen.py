"""
Tardigrade specimen as a 3D complex refractive-index volume.

Source data: Gross et al. 2019, "X-ray imaging of a water bear offers a new look at
tardigrade internal anatomy", Zoological Letters 5:14 (CC BY 4.0), nanoCT of
Hypsibius exemplaris, 270 nm voxels, 152 um body length. Additional files 3 (grey) + 4 (labels).

Stack axes (z, y, x) = (anterior-posterior, dorso-ventral, left-right). The animal lies on
its ventral side on the slide, so the microscope's optical axis is the stack's y axis.

Refractive indices are MODEL PARAMETERS (no measured tardigrade values exist):
literature ranges for cytoplasm, lipid/glycogen-rich storage cells, yolk and chitinous cuticle.
"""
import numpy as np
from scipy import ndimage

VOXEL_UM = 0.27

# label id -> (name, delta-n relative to the surrounding water at 589 nm)
LABELS = {
    1: ("pharynx (muscular bulb)", 0.040),
    2: ("oesophagus", 0.030),
    3: ("midgut (with ingested algae)", 0.030),
    4: ("buccal tube (cuticular)", 0.120),
    5: ("hindgut / cloaca", 0.030),
    6: ("ovary (yolk-rich oocytes)", 0.050),
    7: ("ventral nerve cord + trunk ganglia", 0.030),
    8: ("storage cells (lipid/glycogen)", 0.060),
    9: ("Malpighian tubules", 0.030),
    10: ("salivary glands", 0.030),
    11: ("claw glands", 0.035),
}
DN_TISSUE = 0.022          # generic body tissue / cytoplasm (~1.355)
DN_TISSUE_TEXTURE = 0.012  # extra delta-n scaled by OsO4-stained nanoCT grey value (membranes, lipids)
DN_CUTICLE = 0.130         # hydrated chitin-protein cuticle (~1.46)
DN_HAEMOLYMPH = 0.010      # body-cavity fluid (~1.343)


def n_water(lam_um):
    """Pure water, Cauchy fit to Daimon & Masumura (2007) at 20 C, 0.4-0.75 um."""
    return 1.3240 + 3.08e-3 / lam_um**2 - 1.0e-5 / lam_um**4


def tissue_dispersion(lam_um):
    """Proteins/lipids are more dispersive than water: scale delta-n mildly with 1/lambda^2."""
    return 1.0 + 0.035 * (0.5893**2 / lam_um**2 - 1.0)


def algae_absorption_per_um(lam_um):
    """Chlorophyll-like absorption of gut content (H. exemplaris is fed Chlorella).
    Two Soret/Q-band lobes; amplitude chosen so ~30 um of gut transmits ~55 % at 435 nm."""
    l = lam_um * 1000
    a = 0.020 * np.exp(-((l - 435) / 28) ** 2) + 0.012 * np.exp(-((l - 670) / 18) ** 2) \
        + 0.004 * np.exp(-((l - 480) / 40) ** 2)
    return a


class Tardigrade:
    def __init__(self, labels_npy, grey_npy):
        L = np.load(labels_npy)
        G = np.load(grey_npy).astype(np.float32)
        # body mask: stained tissue is brighter than the mounting background
        bg = np.median(G[:, :10, :10])
        sm = ndimage.gaussian_filter(G, 1.5)
        tissue = sm > bg + 0.20 * (np.percentile(sm, 99.5) - bg)
        tissue |= L > 0
        # outer envelope: the body cavity between organs is filled with haemolymph, not water
        env = ndimage.gaussian_filter(tissue.astype(np.float32), 4.0) > 0.18
        env = ndimage.binary_fill_holes(env)
        for ax in range(3):                       # fill cavities open in 3D but closed per section
            env = np.stack([ndimage.binary_fill_holes(s) for s in np.moveaxis(env, ax, 0)], ax)
        lab, n = ndimage.label(env)
        if n > 1:
            sizes = ndimage.sum(env, lab, range(1, n + 1))
            env = lab == (1 + int(np.argmax(sizes)))
        tissue &= ndimage.binary_dilation(env, iterations=2)   # drop debris specks outside the animal
        body = env | tissue
        cuticle = body & ~ndimage.binary_erosion(body, iterations=3)   # ~0.8 um outer shell
        g = np.clip((sm - bg) / (np.percentile(sm[tissue], 99) - bg + 1e-6), 0, 1)

        dn = np.zeros(L.shape, np.float32)
        dn[body] = DN_HAEMOLYMPH
        dn[tissue] = DN_TISSUE + DN_TISSUE_TEXTURE * g[tissue]
        for k, (_, v) in LABELS.items():
            m = L == k
            dn[m] = v + 0.5 * DN_TISSUE_TEXTURE * g[m]
        dn[cuticle] = DN_CUTICLE
        # soften voxel staircase (real interfaces are not voxel-sharp at 270 nm)
        dn = ndimage.gaussian_filter(dn, 0.6)
        self.dn589 = dn                      # (z, y, x)
        self.gut = ndimage.gaussian_filter((L == 3).astype(np.float32), 0.8)
        self.shape = L.shape
        self.voxel_um = VOXEL_UM

    def slab(self, lam_um):
        """Return (delta-n, absorption coeff [1/um]) at wavelength lam, axes (y_optical, X, Y)
        where X = anterior-posterior (stack z) and Y = left-right (stack x)."""
        dn = self.dn589 * tissue_dispersion(lam_um)
        mu = self.gut * algae_absorption_per_um(lam_um)
        # (z, y, x) -> (y, z, x): optical axis first; flip so dorsal side faces the objective
        return np.ascontiguousarray(dn.transpose(1, 0, 2)[::-1]), \
               np.ascontiguousarray(mu.transpose(1, 0, 2)[::-1])
