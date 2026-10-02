"""Render the focus stacks for every illumination preset and publish them into the viewer.

  python scripts/render_illumination.py                       # all presets that are not built yet
  python scripts/render_illumination.py --presets k015,k040   # selected presets
  python scripts/render_illumination.py --planes 21 --force   # re-render even if present

For each preset this runs sim/render_color.py twice (ray-traced PUMA optics and ideal lens),
converts the result with sim/eyepiece_view.py into docs/img/<preset>/<optics>/ and records the
preset in docs/img/manifest.json, which the viewer reads to enable the illumination buttons.

Rough CPU time on a 6-core laptop (Ryzen 5 4500U), 21 focus planes x 9 wavelengths, both optics:
  mirror  ~10 min   k015  ~35 min   k030  ~45 min   k040  ~70 min
"""
import argparse, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRESETS = {
    # key: (label, condenser NA, source rings)  -- rings: 1 -> 7, 2 -> 19, 3 -> 37, 4 -> 61, 5 -> 91 points
    "mirror": ("Plane mirror, no condenser (PUMA Foundation scope)", 0.05, 1),
    "k015": ("Köhler, aperture stop closed", 0.15, 3),
    "k030": ("Köhler, standard setting", 0.30, 4),
    "k040": ("Köhler, aperture stop fully open (= objective NA)", 0.40, 5),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", default=",".join(PRESETS))
    ap.add_argument("--planes", type=int, default=21)
    ap.add_argument("--nlam", type=int, default=9)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    manifest_path = os.path.join(ROOT, "docs", "img", "manifest.json")
    manifest = json.load(open(manifest_path, encoding="utf-8")) if os.path.exists(manifest_path) else {"presets": {}}
    focus = ",".join(str(-3 * (i + 1)) for i in range(a.planes))
    for key in a.presets.split(","):
        label, nac, rings = PRESETS[key]
        if key in manifest["presets"] and not a.force:
            print("skip", key, "(already built)")
            continue
        for optics in ("traced", "ideal"):
            out = os.path.join(ROOT, "renders", "illum", f"{key}_{optics}")
            subprocess.run([sys.executable, "render_color.py", "--out", out, f"--focus={focus}", "--rings", str(rings),
                            "--nlam", str(a.nlam), "--nac", str(nac), "--pupil", optics],
                           cwd=os.path.join(ROOT, "sim"), check=True)
            subprocess.run([sys.executable, "eyepiece_view.py", out, os.path.join(ROOT, "docs", "img", key, optics)],
                           cwd=os.path.join(ROOT, "sim"), check=True)
        manifest["presets"][key] = {"label": label, "na_condenser": nac, "planes": a.planes}
        json.dump(manifest, open(manifest_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print("built", key)


if __name__ == "__main__":
    main()
