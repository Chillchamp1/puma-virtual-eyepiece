"""
End-to-end sequential model of the PUMA brightfield microscope (DIN 160 mm, finite):

  object point in water  ->  0.17 mm cover glass  ->  air (focus knob = working distance)
  ->  objective: Olympus plan achromat 20x, US 4,212,515 Emb. 1 (Itaya 1980), f = 7.96 mm,
      aperture stop in the rear focal plane, object-space NA 0.40
  ->  intermediate image (field stop FN 20)
  ->  10x eyepiece: US 3,888,567 (Shoemaker, American Optical 1975), f = 24.92 mm
  ->  eye: Navarro et al. 1985 schematic eye (relaxed), media dispersion per Navarro
  ->  retina (curved image surface, R = -12 mm)

Prescriptions and verification: optics/prescriptions_research.md
Units: mm, wavelengths in um.
"""
import warnings
import numpy as np
from optiland import optic
from optiland.materials import Material, AbbeMaterial, BaseMaterial

warnings.filterwarnings("ignore")


class FunctionMaterial(BaseMaterial):
    """Material with an analytic n(lambda) (lambda in um)."""

    def __init__(self, fn, label):
        super().__init__()
        self.fn, self.label = fn, label

    def _calculate_n(self, wavelength, **kwargs):
        return self.fn(np.asarray(wavelength, dtype=float))

    def _calculate_k(self, wavelength, **kwargs):
        return np.zeros_like(np.asarray(wavelength, dtype=float))

    def to_dict(self):
        return {"type": "FunctionMaterial", "label": self.label}


# ---------------------------------------------------------------- media
def n_water(lam):
    return 1.3240 + 3.08e-3 / lam ** 2 - 1.0e-5 / lam ** 4


# Navarro (1985) dispersion: n = a1 n** + a2 nF + a3 nC + a4 n*,  ai = Ai0 + Ai1 l^2 + Pi/(l^2-l0^2) + Ri/(l^2-l0^2)^2
_NAV_C = np.array([[0.66147196, -0.40352796, -0.28046790, 0.03385979],
                   [-4.20146383, 2.73508956, 1.50543784, -0.11593235],
                   [6.29834237, -4.69409935, -1.57508650, 0.10293038],
                   [-1.75835059, 2.36253794, 0.35011657, -0.02085782]])
_NAV_REF = {"cornea": (1.3975, 1.3807, 1.37405, 1.3668), "aqueous": (1.3593, 1.3422, 1.3354, 1.3278),
            "lens": (1.4492, 1.4263, 1.4175, 1.4097), "vitreous": (1.3565, 1.3407, 1.3341, 1.3273)}


def navarro_n(medium):
    ref = np.array(_NAV_REF[medium])
    def fn(lam):
        l2 = lam ** 2
        d = l2 - 0.028
        a = _NAV_C[:, 0][:, None] + _NAV_C[:, 1][:, None] * l2 + _NAV_C[:, 2][:, None] / d + _NAV_C[:, 3][:, None] / d ** 2
        return (ref[:, None] * a).sum(0).reshape(np.shape(lam))
    return fn


WATER = FunctionMaterial(n_water, "water")
EYE = {m: FunctionMaterial(navarro_n(m), "navarro_" + m) for m in _NAV_REF}


def glass(nd, vd, catalog_name=None, cat="ohara"):
    """Patent gives nd/vd only. Use the nearest real catalog glass (measured Sellmeier dispersion,
    so partial dispersion / secondary spectrum are realistic). Optiland's nd/vd-only model was
    tested and rejected: it misses nd by 4e-4 and mis-models the blue."""
    return Material(catalog_name, catalog=cat) if catalog_name else AbbeMaterial(nd, vd)


# ---------------------------------------------------------------- prescriptions
F_OBJ = 7.96  # mm, scale of the normalised patent table (f = 1)
# (R, thickness after, material) ; patent US 4,212,515 Embodiment 1, normalised to f = 1
OBJECTIVE_F1 = [
    (-0.3728, 0.6951, glass(1.7725, 49.6, "S-LAH66")),
    (-0.7076, 0.0113, None),
    (30.7899, 0.2105, glass(1.497, 81.6, "S-FPL51")),
    (-1.0993, 0.0100, None),
    (3.6804, 0.1155, glass(1.7618, 27.11, "S-TIH14")),   # nearest catalog: 1.76182/26.52
    (1.1203, 0.3145, glass(1.497, 81.6, "S-FPL51")),
    (-4.9268, 3.1525, None),                      # stop inserted in this air space (see below)
    (4.9513, 0.3014, glass(1.497, 81.6, "S-FPL51")),
    (-3.0944, 0.0628, None),
    (2.5059, 0.5401, glass(1.6779, 55.33, "S-LAL12")),
    (-86.2812, 0.3111, glass(1.6134, 43.84, "S-NBM51")),  # 1.61340/44.27
    (1.0903, None, None),
]
STOP_BEFORE_R8 = 0.2600  # f-units: rear focal plane lies 0.26 f in front of r8

EYEPIECE = [  # US 3,888,567, F = 24.919 mm, from the field-stop side
    (-169.084, 8.000, glass(1.588, 61.19, "N-SK5", "schott")),   # 1.58913/61.27
    (-36.657, 0.10, None),
    (52.630, 7.700, glass(1.588, 61.19, "N-SK5", "schott")),
    (-24.750, 3.000, glass(1.786, 25.52, "S-TIH11")),          # 1.78472/25.68
    (-169.084, 0.10, None),
    (20.930, 6.000, glass(1.588, 61.19, "N-SK5", "schott")),
    (77.059, None, None),
]
FP_TO_EYEPIECE = 16.90


def build(focus_depth_mm=0.0, wd_mm=1.928, tube_mm=146.67, eye=True, eye_relief=None, na=0.40,
          wavelengths=(0.55,), fields_mm=(0.0,), pupil_eye_mm=3.0, until="retina"):
    """focus_depth_mm: water between object plane and cover glass underside.
    wd_mm: air gap between cover glass and front lens (the focus knob)."""
    o = optic.Optic()
    s = o.surfaces
    k = 0
    s.add(index=k, radius=np.inf, thickness=max(focus_depth_mm, 1e-6), material=WATER); k += 1
    s.add(index=k, radius=np.inf, thickness=0.17, material=Material("D263TECO", catalog="schott")); k += 1
    s.add(index=k, radius=np.inf, thickness=wd_mm); k += 1
    for i, (r, t, m) in enumerate(OBJECTIVE_F1):
        R = r * F_OBJ
        if i == 6:      # r7: split the big air gap and insert the aperture stop
            gap = t * F_OBJ
            to_stop = gap - STOP_BEFORE_R8 * F_OBJ
            s.add(index=k, radius=R, thickness=to_stop); k += 1
            s.add(index=k, radius=np.inf, thickness=gap - to_stop, is_stop=True, comment="aperture stop"); k += 1
            continue
        if t is None:
            s.add(index=k, radius=R, thickness=tube_mm, comment="objective rear"); k += 1
        else:
            s.add(index=k, radius=R, thickness=t * F_OBJ, material=m if m else "air"); k += 1
    if until == "fs":
        s.add(index=k, radius=np.inf, comment="intermediate image"); k += 1
        o.set_aperture(aperture_type="objectNA", value=na)
        o.fields.set_type(field_type="object_height")
        for f in fields_mm:
            o.fields.add(y=f)
        for i, w in enumerate(wavelengths):
            o.wavelengths.add(value=w, is_primary=(i == 0))
        return o
    s.add(index=k, radius=np.inf, thickness=FP_TO_EYEPIECE, comment="field stop FN20"); k += 1
    fs_index = k - 1
    if eye_relief is None:
        eye_relief = 19.25
    for i, (r, t, m) in enumerate(EYEPIECE):
        if t is None:
            s.add(index=k, radius=r, thickness=eye_relief if eye else 50.0); k += 1
        else:
            s.add(index=k, radius=r, thickness=t, material=m if m else "air"); k += 1
    if eye:
        s.add(index=k, radius=7.72, conic=-0.26, thickness=0.55, material=EYE["cornea"], comment="cornea"); k += 1
        s.add(index=k, radius=6.50, thickness=3.05, material=EYE["aqueous"]); k += 1
        s.add(index=k, radius=10.20, conic=-3.1316, thickness=4.00, material=EYE["lens"], comment="eye lens / iris"); k += 1
        s.add(index=k, radius=-6.00, conic=-1.0, thickness=16.40398, material=EYE["vitreous"]); k += 1
        s.add(index=k, radius=-12.0, comment="retina"); k += 1
    else:
        s.add(index=k, radius=np.inf); k += 1
    o.set_aperture(aperture_type="objectNA", value=na)
    o.fields.set_type(field_type="object_height")
    for f in fields_mm:
        o.fields.add(y=f)
    for i, w in enumerate(wavelengths):
        o.wavelengths.add(value=w, is_primary=(i == 0))
    o.fs_index = fs_index
    return o
