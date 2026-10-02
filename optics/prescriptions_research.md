# Lens prescriptions for a generic brightfield microscope ray trace

Research date: 2026-10-02. All tables below were transcribed by reading the USPTO scan pages
(image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/<number>). Google Patents and
freepatentsonline were not reachable from this environment. Every table was then checked with an
independent paraxial ray trace (y-nu, matrix method). The check reproduces each patent's own stated
derived quantities (EFL, sub-group focal lengths, image distances), so typos in the transcription
are unlikely.

Conventions used in all tables (unless stated otherwise):
- Light travels left to right, from object to image (objective) or from field stop to eye (eyepiece).
- A radius is positive when its centre of curvature lies to the right of the vertex.
- `t after` is the axial distance from that surface to the next one.
- nd and vd are at the d line (587.56 nm). For the Leitz patent the values are ne/ve at the e line
  (546.07 nm).

---------------------------------------------------------------------------------------------------

## A. 20x plan-achromat objective, finite conjugate (160 mm tube), cover glass

### Summary of candidates

| # | Patent | Mag / NA | Plan? | Conjugate | Cover glass | Data | Confidence |
|---|--------|----------|-------|-----------|-------------|------|------------|
| A1 (CHOSEN) | US 4,212,515 (Itaya, Olympus, 1980), Embodiment 1 | 20x / 0.46 | yes, "plan achromat" (patent text) | finite. Verified: object-to-image 194.3 mm, which matches DIN 45 + 150 = 195 mm | 0.17 mm inferred (not stated) | complete, f = 1 normalized | table: high. Cover glass and conjugate: medium-high (inferred, see below) |
| A2 | US 4,563,060 (Yamagishi, Olympus, 1986), Embodiment 1 | 20x / 0.40 | yes, "plan achromat class" (patent text) | finite (verified paraxially) | 1.0 mm stated (0.126 f). This is a long-working-distance design for thick vessels or slides, NOT 0.17 mm | complete, f = 1 normalized, real f = 7.9365 mm (stated) | high |
| A3 | US 3,592,530 (Klein, E. Leitz, 1971), Table III | 16.2x / 0.40 | yes, "flattened image field" | finite, image distance stated (151.3 mm) | 0.17 mm explicit, in the table | complete, in mm, ne/ve | high (exact paraxial match) |

My recommendation for "generic 20x/0.40, 160/0.17" is A1. Stop it down to NA 0.40 with the aperture
stop placed as described below, or keep NA 0.46. If an explicitly documented NA 0.40 is more
important than the 0.17 mm cover glass, use A2 and model the 1.0 mm plate. A3 is the only one whose
0.17 mm cover glass and image distance are printed in the patent. Its magnification is 16x.

Textbook sources (Laikin "Lens Design", Smith "Modern Lens Design", Kingslake, and Gross
"Handbook of Optical Systems" vol. 4) are not openly accessible online, so I did not use them.

---

### A1. US 4,212,515, Embodiment 1 (H. Itaya, Olympus Optical, filed 1978-03-08, granted 1980-07-15), "Wide-visual-field microscope objective lens system"

Source: https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/4212515 (col. 3-4).
The patent states: "The present invention relates to a plan achromat microscope objective lens
system of medium magnification". It has 5 components and 7 elements:
- L1: thick negative meniscus, concave to the object
- L2: positive singlet
- L3: cemented doublet
- (large airspace d7)
- L4: positive singlet
- L5: negative cemented doublet

Patent header for Embodiment 1: beta = 20X, NA = 0.46, WD = 0.2422, f = 1, f_A = 1.131.
The aberration plots are drawn for "FA = 30", i.e. field number up to 30, so the field is wide.

Exact patent table (f = 1), with my scaling to f = 7.96 mm. (At f = 7.96 mm many thicknesses
become round numbers: d6 = 2.503, d8 = 2.399, d9 = 0.4999, d10 = 4.299 mm. This suggests the real
design f is about 7.96 mm.)

| Surf | R (patent, f=1) | R (mm, f=7.96) | t after (f=1) | t after (mm) | nd | vd | element |
|---|---|---|---|---|---|---|---|
| obj | - | - | cover glass | 0.17 (assumed) | ~1.5213 | ~56 | cover glass (assumed, see note) |
| - | - | - | WD 0.2422 | 1.928 | air | | free working distance (cover-glass top to r1) |
| r1 | -0.3728 | -2.9675 | 0.6951 | 5.5330 | 1.7725 | 49.6 | L1 |
| r2 | -0.7076 | -5.6325 | 0.0113 | 0.0899 | air | | |
| r3 | 30.7899 | 245.0876 | 0.2105 | 1.6756 | 1.497 | 81.6 | L2 |
| r4 | -1.0993 | -8.7504 | 0.0100 | 0.0796 | air | | |
| r5 | 3.6804 | 29.2960 | 0.1155 | 0.9194 | 1.7618 | 27.11 | L3a |
| r6 | 1.1203 | 8.9176 | 0.3145 | 2.5034 | 1.497 | 81.6 | L3b |
| r7 | -4.9268 | -39.2173 | 3.1525 | 25.0939 | air | | (aperture stop goes in this gap) |
| r8 | 4.9513 | 39.4123 | 0.3014 | 2.3991 | 1.497 | 81.6 | L4 |
| r9 | -3.0944 | -24.6314 | 0.0628 | 0.4999 | air | | |
| r10 | 2.5059 | 19.9470 | 0.5401 | 4.2992 | 1.6779 | 55.33 | L5a |
| r11 | -86.2812 | -686.7984 | 0.3111 | 2.4764 | 1.6134 | 43.84 | L5b |
| r12 | 1.0903 | 8.6788 | - | 146.67 (paraxial image distance) | air | | to intermediate image |

Glass suggestions (nd/vd matches only, not stated in the patent):

| nd / vd | Suggested glass |
|---|---|
| 1.7725 / 49.6 | N-LAF34 / S-LAH66 |
| 1.497 / 81.6 | S-FPL51 / FK01, close to N-PK52A |
| 1.7618 / 27.11 | close to SF14 / S-TIH14 (1.76182 / 26.5) |
| 1.6779 / 55.33 | S-LAL12 / LAK12 |
| 1.6134 / 43.84 | close to S-NBM51 (1.6134 / 44.3) |

Ray-trace data per glass is otherwise "model glass nd/vd".

Verification (my paraxial trace):
- EFL = 0.9996 (patent: 1).
- f_A (front group r1 to r7) = 1.1305 (patent: 1.131).
- With a 0.17 mm cover glass and WD = 0.2422 f in air, at f = 7.96 mm:
  - magnification is -19.91;
  - image is 146.67 mm behind r12;
  - object-to-image distance is 194.3 mm, close to the DIN/RMS 160 mm tube value of 45 + 150 = 195 mm;
  - lens length r1 to r12 is 45.57 mm. The rear end sits about 2.6 mm above a 45 mm parfocal
    shoulder, inside the RMS thread, which is normal.
- Without any cover glass the same WD gives -27.6x, so a cover glass must be present.

Cover glass and conjugate (inferred): neither the cover glass nor the tube length is printed in the
patent. Two things support the inference:
- Olympus objectives of that period (e.g. the S Plan 20/0.46 160/0.17) are 160/0.17 designs.
- The paraxial numbers above match 20x and 195 mm only when a 0.17 mm cover glass is included.

Aperture stop (not specified in the patent). The rear focal plane lies 1.4754 f before r12, i.e.
inside the d7 airspace, 0.2600 f (2.07 mm) in front of r8 and 23.02 mm after r7. Put the stop
there; the object side is then approximately telecentric. The cleanest setup is "Object-space NA"
= 0.46, or 0.40 for the stopped-down variant, with a floating stop at that plane. Paraxial
marginal-ray height at the stop: about 3.9 mm (NA 0.40) or about 4.5 mm (NA 0.46).

---

### A2. US 4,563,060, Embodiment 1 (M. Yamagishi, Olympus Optical, filed 1984-07-02, granted 1986-01-07), "Microscope objective"

Source: https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/4563060 (col. 3-4).
The patent states it relates to "a plan achromat class microscope objective with a medium class
magnification" with very long working distance.

Patent header: f = 1, NA = 0.4, beta = -20X, WD = 1.421. Also stated: |r1/r2| = 1.75,
|f_IV-VI| = 1.71, |1/phi10| = 0.91, |fV/fVI| = 1.08, "thickness of cover glass = 0.126".
Fig. 8 is described as "cover glass is 2 mm (0.252 when f=1)", which gives f = 7.9365 mm. So the
nominal cover glass is 1.0 mm, and the design tolerates 0 to 2 mm.

| Surf | R (f=1) | R (mm) | t after (f=1) | t after (mm) | nd | vd | element |
|---|---|---|---|---|---|---|---|
| obj | - | - | cover 0.126 | 1.000 | (not given; use ~1.521 / 56) | | cover glass / vessel bottom |
| - | - | - | WD 1.421 | 11.278 | air | | |
| r1 | -2.4858 | -19.7286 | 0.315 | 2.5000 | 1.77248 | 48.99 | I (pos. meniscus, concave to object) |
| r2 | -1.4178 | -11.2524 | 0.024 | 0.1905 | air | | |
| r3 | 6.9456 | 55.1238 | 0.403 | 3.1984 | 1.497 | 81.61 | II |
| r4 | -2.4991 | -19.8341 | 0.214 | 1.6984 | air | | |
| r5 | 2.165 | 17.1825 | 0.491 | 3.8968 | 1.497 | 81.61 | III a |
| r6 | -2.6012 | -20.6444 | 0.189 | 1.5000 | 1.6889 | 31.56 | III b |
| r7 | -3.6045 | -28.6071 | 0.088 | 0.6984 | air | | |
| r8 | 1.2709 | 10.0865 | 0.554 | 4.3968 | 1.497 | 81.61 | IV a |
| r9 | -1.5431 | -12.2468 | 0.365 | 2.8968 | 1.73997 | 28.75 | IV b |
| r10 | 0.6749 | 5.3563 | 0.440 | 3.4921 | air | | (rear focal plane / stop in this gap) |
| r11 | -0.6617 | -5.2516 | 0.151 | 1.1984 | 1.73997 | 29.7 | V a |
| r12 | -1.9544 | -15.5111 | 0.315 | 2.5000 | 1.497 | 81.61 | V b |
| r13 | -0.7436 | -5.9016 | 0.013 | 0.1032 | air | | |
| r14 | 1.0679 | 8.4754 | 0.302 | 2.3968 | 1.80514 | 25.15 | VI a |
| r15 | -10.9735 | -87.0913 | 0.239 | 1.8968 | 1.62094 | 37.16 | VI b |
| r16 | 0.7274 | 5.7730 | - | 145.68 (paraxial) | air | | to image |

Verification (paraxial):
- EFL = 1.0000.
- f_IV-VI = -1.712 (patent 1.71).
- 1/phi10 = 0.912 (patent 0.91).
- fV/fVI = 1.080 (patent 1.08).
- With 0.126 cover glass and WD 1.421, the image is 18.36 f (145.7 mm) behind r16 and beta = -19.4,
  so this is a finite conjugate design. Object-to-image distance is 190.5 mm.
- The magnification is very sensitive to WD (about -20.0 at WD ~1.419 f).
- Real thicknesses are round numbers at f = 7.9365 mm (2.500, 1.500, 2.500 mm), which confirms the
  scale.

Stop: the rear focal plane is 1.0683 f before r16, i.e. in the r10-r11 airspace, 0.392 f (3.11 mm)
after r10. Paraxial marginal height there at NA 0.40 is about 3.0 mm.

Caveat: this design is built for a 1 mm cover/vessel thickness. Replacing that with a 0.17 mm
cover glass would leave roughly 1-2 waves of spherical aberration at NA 0.4. Model the 1.0 mm plate
if you use it.

---

### A3. US 3,592,530, Table III (W. Klein, Ernst Leitz GmbH Wetzlar, priority DE 1963-07-09, granted 1971-07-13), "Anastigmatic medium-power microscope objective providing a flattened image field"

Source: https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/3592530 (col. 2 Table III,
repeated in claim 3). Values are in mm. Indices are ne/ve. d0 is the cover glass. l_b is the
distance from the last lens to the conjugate image plane.

| Surf | R (mm) | t after (mm) | ne | ve | note |
|---|---|---|---|---|---|
| obj | - | d0 = 0.17 | 1.52491 | 58.3 | cover glass (in patent) |
| - | - | l0 = 0.794 | air | | WD |
| r1 | -3.16 | d1 = 7.0 | 1.61022 | 49.0 | thick negative meniscus front lens |
| r2 | -4.982 | l1 = 0.4 | air | | |
| r3 | inf | d2 = 1.0 | 1.76860 | 26.3 | doublet, flint |
| r4 | +9.84 | d3 = 2.0 | 1.60973 | 59.2 | doublet, crown |
| r5 | -9.84 | l2 = 30.25 | air | | large gap |
| r6 | +29.232 | d4 = 2.0 | 1.56606 | 60.5 | positive singlet |
| r7 | -29.232 | l3 = 1.0 | air | | |
| r8 | +18.026 | d5 = 1.1 | 1.73430 | 28.1 | negative meniscus |
| r9 | +10.938 | l_b = 151.3 | air | | to image |

Patent states beta'e = -16.20, fe = 10.51, A (NA) = 0.40, Sum P = -0.001, Sum Gamma = +0.019.

Verification (paraxial, e-line indices): EFL = 10.510, image distance = 151.24 mm,
beta = -16.197. This is an exact match.

Stop: not given. Use the object-space NA.

The same patent also has two other examples:
- Table VI: 25x / 0.50, f = 6.99, l_b = 151.7; paraxially verified -24.82x, 151.5 mm.
- Table IX: 40x / 0.65 (the claim misprints this as -30.67).

---------------------------------------------------------------------------------------------------

## B. 10x eyepiece, f ~ 25 mm

### B1 (CHOSEN). US 3,888,567 (A. H. Shoemaker, American Optical Corp., filed 1974-05-01, granted 1975-06-10), "10X three element eyepiece"

Source: https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/3888567

Patent text:
- "microscope eyepiece having a magnification of about 10X"
- apparent field of view about 52.3 degrees
- maximum exit pupil 1.60 mm
- Petzval radius 1.42 F
- eye relief about 0.77 F

Form, from the field-stop side: I = concavo-convex positive singlet; II = biconvex cemented doublet;
III = convex-concave positive meniscus. This is a Kellner-like / "wide-field" 3-group type.
"FP" is the eyepiece focal plane, i.e. the intermediate image, where the field stop sits. Sign
convention in the patent: minus means the centre of curvature lies on the focal-plane side, which
matches the convention used here.

Embodiment with F = 24.919 mm (exact patent numbers in mm; normalized values from Table I in
brackets):

| Surf | R (mm) | t after (mm) | nd | vd | element |
|---|---|---|---|---|---|
| FP (field stop / intermediate image) | flat | S1 = 16.90 [0.6782F] | air | | |
| R1 | -169.084 [-6.7853F] | T1 = 8.000 [0.3210F] | 1.58800 | 61.19 | I |
| R2 | -36.657 [-1.4710F] | S2 = 0.10 [0.0040F] | air | | |
| R3 | 52.630 [2.1120F] | T2 = 7.700 [0.3090F] | 1.58800 | 61.19 | II a |
| R4 | -24.750 [-0.9932F] | T3 = 3.000 [0.1204F] | 1.78600 | 25.52 | II b |
| R5 | -169.084 [-6.7853F] | S3 = 0.10 [0.0040F] | air | | |
| R6 | 20.930 [0.8399F] | T4 = 6.000 [0.2408F] | 1.58800 | 61.19 | III |
| R7 | 77.059 [3.0924F] | S4 = 19.25 [0.7725F] | air | | to the eye (eye relief) |

Note: the claim-2 copy of the normalized table prints R5 as "6.7853F" without the minus sign. The
mm table and Table I both give -169.084 / -6.7853F, and the paraxial check confirms the negative
sign.

Glass suggestions (nd/vd matches only):

| nd / vd | Suggested glass |
|---|---|
| 1.588 / 61.2 | about SK5 / N-SK5 (1.58913 / 61.27) or S-BAL35 |
| 1.786 / 25.5 | about SF11 / S-TIH11 (1.78472 / 25.68) |

Verification (paraxial):
- EFL = 24.9195 mm (patent: 24.919).
- Front focal distance = 16.894 mm before R1 (patent S1 = 16.90), so FP is the front focal plane.
- Rear focal distance = 17.36 mm after R7.
- Taking the objective exit pupil about 150 mm below FP (160 mm tube), its image (the eyepoint)
  falls about 21.5 mm after R7. The patent's S4 = 19.25 mm is the stated eye relief, presumably for
  real chief rays with pupil aberration.

Field stop for a 10x/18 to 10x/20 eyepiece: put a circular stop at FP.

| Field number | Stop diameter | Apparent field |
|---|---|---|
| FN 18 | 18 mm | 39.7 deg |
| FN 20 | 20 mm | 43.7 deg |
| patent's 52.3 deg (for reference) | 24.5 mm | 52.3 deg |

In a 160 mm DIN microscope the FP / intermediate image sits 10 mm below the eyepiece shoulder, i.e.
150 mm from the objective shoulder.

Confidence: high. The numbers are complete and EFL and front focal distance are verified.

Alternatives I looked at but did not choose:
- US 3,975,088 (Shoemaker, AO): 12x three-element version, AFOV 51.3 deg.
- US 4,099,842 (Kaneko, Nikon): AFOV > 60 deg.
- US 5,557,464 (Imaizumi, Olympus): 75-80 deg.
- US 5,202,795 (Kashima, Olympus): plastic aspheric.

All of these are wider-angle or special designs than needed.

---------------------------------------------------------------------------------------------------

## C. Navarro (1985) schematic eye, unaccommodated (A = 0 D), with chromatic dispersion

Primary reference: R. Navarro, J. Santamaria, J. Bescos, "Accommodation-dependent model of the
human eye with aspherics", J. Opt. Soc. Am. A 2(8), 1273-1281 (1985).

Wide-angle version with the same optics: I. Escudero-Sanz & R. Navarro, JOSA A 16(8), 1881-1891
(1999). Its implementation is documented in the ZOSPy docs:
https://zospy.readthedocs.io/en/latest/examples/Escudero-Sanz%20eye%20model/Escudero-Sanz%20eye%20model.html

The structure is also reproduced in H. Gross et al., Handbook of Optical Systems vol. 4, ch. 36,
Table 36-16 ("relaxed eye, 589.3 nm"). Note that this table has a typo for the cornea index (1.367;
the correct value is 1.376).

### C1. Geometry (unaccommodated, 589.3 nm reference)

Surface sag: z = c r^2 / (1 + sqrt(1 - (1+Q) c^2 r^2)). Q is the conic constant (k in Zemax/ZOSPy).

| Surface | R (mm) | Q (conic) | t after (mm) | Medium after | n (589.3 nm) |
|---|---|---|---|---|---|
| Cornea anterior | 7.72 | -0.26 | 0.55 | cornea | 1.376 |
| Cornea posterior | 6.50 | 0 | 3.05 | aqueous | 1.3374 |
| Lens anterior (iris / aperture stop at this vertex) | 10.20 | -3.1316 | 4.00 | lens | 1.4200 |
| Lens posterior | -6.00 | -1.0 | 16.40398 | vitreous | 1.3360 |
| Retina (image) | -12.00 | 0 | - | - | - |

Verification:
- The paraxial back focal distance after the lens posterior surface, computed with the 589.3 nm
  indices above, is 16.40398 mm exactly. So the vitreous depth equals the paraxial focus at 589.3 nm
  and the cornea index is 1.376, not 1.3771 and not 1.367.
- With the 543 nm indices it gives 16.3203 mm. This equals the 16.3202 mm used by Escudero-Sanz &
  Navarro (1999), who focus at 543 nm.
- Equivalent power at 589.3 nm is 60.42 D.

Some papers write the asphericity as p = Q + 1 (e.g. 0.74 for the cornea). Do not double-convert.

### C2. Chromatic dispersion model (Navarro 1985; Herzberger-type interpolation)

n(lambda) = a1(lambda) n** + a2(lambda) nF + a3(lambda) nC + a4(lambda) n*

with lambda in micrometres and

ai(lambda) = Ai0 + Ai1 lambda^2 + Pi / (lambda^2 - lambda0^2) + Ri / (lambda^2 - lambda0^2)^2,
lambda0^2 = 0.028 um^2 (lambda0 = 0.167 um).

The four reference indices are:
- n** at 0.365 um
- nF at 0.4861 um
- nC at 0.6563 um
- n* at 1.014 um

Reference indices:

| Medium | n** (365 nm) | nF (486.1 nm) | nC (656.3 nm) | n* (1014 nm) |
|---|---|---|---|---|
| Cornea | 1.3975 | 1.3807 | 1.37405 | 1.3668 |
| Aqueous | 1.3593 | 1.3422 | 1.3354 | 1.3278 |
| Lens (A = 0) | 1.4492 | 1.4263 | 1.4175 | 1.4097 |
| Vitreous | 1.3565 | 1.3407 | 1.3341 | 1.3273 |

Coefficients:

| i | Ai0 | Ai1 | Pi | Ri |
|---|---|---|---|---|
| 1 | 0.66147196 | -0.40352796 | -0.28046790 | 0.03385979 |
| 2 | -4.20146383 | 2.73508956 | 1.50543784 | -0.11593235 |
| 3 | 6.29834237 | -4.69409935 | -1.57508650 | 0.10293038 |
| 4 | -1.75835059 | 2.36253794 | 0.35011657 | -0.02085782 |

Provenance and verification:
- A web search independently confirmed the leading coefficients a1 = 0.66147196,
  a2 = -4.20146383 and a3 = 6.29834237, as well as -1.75835059 for a4. The remaining values are the
  standard published set; I checked them numerically:
  - sum of ai(lambda) = 1.000000 at every wavelength;
  - ai(lambda_j) is close to delta_ij at the four reference wavelengths (within 4e-4, as expected
    from the published rounding).
- Evaluating the formula reproduces the Escudero-Sanz & Navarro (1999) index table, as implemented
  in ZOSPy, to within 1e-4:

| Medium | 458 nm | 543 nm | 589.3 nm | 632.8 nm |
|---|---|---|---|---|
| Cornea (formula) | 1.38281 | 1.37770 | 1.37597 | 1.37467 |
| Cornea (published) | 1.3828 | 1.3777 | 1.376 | 1.3747 |
| Aqueous (formula) | 1.34435 | 1.33914 | 1.33737 | 1.33604 |
| Aqueous (published) | 1.3445 | 1.3391 | 1.3374 | 1.336 |
| Lens (formula) | 1.42919 | 1.42223 | 1.41994 | 1.41827 |
| Lens (published) | 1.4292 | 1.4222 | 1.42 | 1.4183 |
| Vitreous (formula) | 1.34276 | 1.33773 | 1.33601 | 1.33471 |
| Vitreous (published) | 1.3428 | 1.3377 | 1.336 | 1.3347 |

- Caveat (from the literature): Navarro fitted these dispersions to the eye's measured
  longitudinal chromatic aberration. They are effective values, not anatomical media dispersions.
  For alternatives see Atchison & Smith, JOSA A 22, 29-37 (2005).

### C3. Accommodation dependence (for completeness; A in diopters)

As given by Navarro et al. (1985), reproduced in Gross, Handbook of Optical Systems vol. 4,
eq. 36-13. All reduce to C1 at A = 0. Confidence: medium-high (the Handbook copy is OCR-garbled
but the form is standard).

- R_lens,ant = 10.2 - 1.75 ln(A+1)
- R_lens,post = -6.0 + 0.2294 ln(A+1)
- d_aqueous = 3.05 - 0.05 ln(A+1)
- d_lens = 4.0 + 0.1 ln(A+1)
- n_lens = 1.42 + 9e-5 (10A + A^2)
- Q_lens,ant = -3.1316 - 0.34 ln(A+1)
- Q_lens,post = -1.0 - 0.125 ln(A+1)

### Confidence summary for C
- Geometry and 589.3 nm indices: high. The paraxial vitreous depth 16.40398 is reproduced
  exactly.
- Dispersion formula and coefficients: high. They reproduce the published 4-wavelength table and
  satisfy the interpolation identities.
- Stop: Navarro places the pupil at the anterior lens surface. A 3 mm pupil (semi-diameter 1.5 mm)
  is the usual choice, as in Escudero-Sanz/ZOSPy.

---------------------------------------------------------------------------------------------------

## Suggested assembly for the full microscope (DIN 160, finite)

Sequence:
1. Object
2. 0.17 mm cover glass
3. Objective A1 (f = 7.96 mm, -19.9x, stop in the d7 gap, object-space NA 0.40 or 0.46)
4. Intermediate image about 146.7 mm after r12 (object-to-image 194.3 mm; DIN nominal 195 mm)
5. Field stop (FN 18 to 20) at the eyepiece focal plane FP
6. Eyepiece B1 (FP to R1 = 16.90 mm, f = 24.919 mm)
7. Eye relief about 19-21.5 mm
8. Navarro eye with its pupil at the exit pupil (eyepoint)

Exit pupil diameter = 2 x NA x f_eyepiece / M_obj x (approx.):
- about 1.0 mm at NA 0.40 with 10x eyepiece and 20x objective (2 x 0.40 x 250 / 200)
- about 1.15 mm at NA 0.46

This is well inside the 3 mm eye pupil, so the microscope, not the eye, sets the aperture.

Reproducibility: the tables were checked with an independent paraxial (y-nu / ABCD) trace against
the patents' own stated EFL, sub-group focal lengths and image distances.
