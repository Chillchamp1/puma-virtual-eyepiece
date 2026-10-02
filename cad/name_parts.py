"""Pick the PUMA parts used by assemble_puma.py from the FreeCAD export and give them short names.
Run after fc_toplevel.py:   python cad/name_parts.py <stl_export_dir> cad/named
"""
import json, os, shutil, sys

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
summary = json.load(open(os.path.join(src, "_summary.json")))
WANT = {
    "Stage.FCStd": ("ST", ["Base_plate", "Focus_plate", "Articulation"]),
    "Focus_Gears.FCStd": ("FG", ["Fine_gear", "Pulley_coarse", "Intermedius", "Eccentric_Tensioner_Top", "Focus_spacer", "Pulley"]),
    "Legs.FCStd": ("LG", ["Short_leg"]),
    "Monocular.FCStd": ("MN", ["Monocular_tube_c_adhesin", "Ocular_extension_c_adhesin", "Ocular_cap"]),
    "QuickRelease_v2.0.FCStd": ("QR", ["QR_Male_C_extn_1mm", "Base_thread", "C-RMS_Thread"]),
    "FilterBlock.FCStd": ("FB", ["Filter_block_simple", "Filter_slider", "Filter_slot_bottom", "Stopper"]),
    "Dominus_part1.FCStd": ("D1", ["Mirror_holder_plain", "Mirror_suspend_plain", "Mirror_to_baseplate"]),
    "Dominus_part2.FCStd": ("D2", ["Condenser_23_30", "Cnd_gripper", "Cnd_to_UC"]),
}
names = {}
for fcstd, (pre, labels) in WANT.items():
    for row in summary.get(fcstd, []):
        for w in labels:
            key = pre + "_" + w[:16].replace("-", "_")
            if row["label"] == w and row["ok"] is True and key not in names:
                shutil.copy(os.path.join(src, row["stl"]), os.path.join(dst, key + ".stl"))
                names[key] = {"file": fcstd, "label": w, "bb": row["bb"]}
json.dump(names, open(os.path.join(dst, "_map.json"), "w"), indent=1)
print(len(names), "parts")
