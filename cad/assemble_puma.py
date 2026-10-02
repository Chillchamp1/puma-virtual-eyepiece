"""
Blender: assemble the PUMA microscope from the ORIGINAL PUMA CAD parts (FreeCAD -> STL, GPL-3.0,
github.com/TadPath/PUMA) in the configuration used by the optical simulation:
  * Z-stage (base plate + focus platform on 3 sprung posts, belt/gear focus drive)
  * PUMA Abbe condenser (Condenser_23_30) under the stage, LED below
  * quick-release objective holder with C-RMS thread + generic 20x plan achromat (RMS, 45 mm parfocal)
  * filter block (simple), monocular tube, ocular cap, WF 10x/20 eyepiece  (DIN 160 mm tube)
  * glass slide + cover glass with the tardigrade
Part positions: the base plate and focus plate share one CAD frame (optical axis at x=-0.1,
y=-88.1 mm in both files). Heights follow the optics (parfocal 45 mm, ~160-170 mm tube).
Everything else is placed by hand from the PUMA Quick Start Guide photos -> APPROXIMATE assembly.

  blender -b --factory-startup --python assemble_puma.py -- --out <dir> [--res 1600] [--samples 128] [--engine cycles|eevee]
"""
import bpy, sys, os, math, importlib.util
from mathutils import Vector, Matrix, Euler

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
A = {"out": "out", "res": "1400", "samples": "96", "engine": "cycles", "view": "hero"}
for i in range(0, len(argv), 2):
    A[argv[i].lstrip("-")] = argv[i + 1]
HERE = os.path.dirname(os.path.abspath(__file__))
STL = os.path.join(HERE, "named")
MM = 0.001
AX = Vector((-0.1, -88.1, 0.0))        # optical axis in the CAD frame (mm)

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene


# ---------------------------------------------------------------- materials
def principled(name, color, rough, metal=0.0, coat=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    p = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    p.inputs["Base Color"].default_value = (*color, 1)
    p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = metal
    p.inputs["Coat Weight"].default_value = coat
    return m, p


def pla_black():
    m, p = principled("PLA_black", (0.018, 0.018, 0.02), 0.42)
    nt = m.node_tree
    p.inputs["Specular IOR Level"].default_value = 0.45
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ"); nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])
    mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = 2 * math.pi / 0.2          # 0.2 mm layer height (object units = mm)
    nt.links.new(sep.outputs["Z"], mul.inputs[0])
    sn = nt.nodes.new("ShaderNodeMath"); sn.operation = "SINE"; nt.links.new(mul.outputs[0], sn.inputs[0])
    bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.25
    bump.inputs["Distance"].default_value = 0.00004
    nt.links.new(sn.outputs[0], bump.inputs["Height"]); nt.links.new(bump.outputs["Normal"], p.inputs["Normal"])
    return m


PLA = pla_black()
STEEL, _ = principled("Steel", (0.75, 0.76, 0.78), 0.22, metal=1.0)
ZINC, _ = principled("Zinc_nut", (0.80, 0.80, 0.78), 0.28, metal=1.0)
SPRING, _ = principled("Spring_steel", (0.62, 0.63, 0.66), 0.18, metal=1.0)
RUBBER, _ = principled("Rubber", (0.02, 0.02, 0.02), 0.7)
EYE_BODY, _ = principled("Eyepiece_black", (0.012, 0.012, 0.012), 0.35)


def glass_mat(name, tint=(1, 1, 1), rough=0.0):
    m, p = principled(name, tint, rough)
    p.inputs["Transmission Weight"].default_value = 1.0
    p.inputs["IOR"].default_value = 1.52
    return m


GLASS = glass_mat("Slide_glass", (0.93, 0.98, 0.96))
COVER = glass_mat("Cover_glass", (0.97, 0.99, 0.99))
LENS = glass_mat("Lens_glass")
WATER = glass_mat("Water", (0.95, 0.98, 1.0)); next(n for n in WATER.node_tree.nodes if n.type == "BSDF_PRINCIPLED").inputs["IOR"].default_value = 1.333
LEDM = bpy.data.materials.new("LED"); LEDM.use_nodes = True
_nt = LEDM.node_tree; _nt.nodes.clear()
_e = _nt.nodes.new("ShaderNodeEmission"); _e.inputs["Color"].default_value = (1.0, 0.98, 0.95, 1); _e.inputs["Strength"].default_value = 25
_o = _nt.nodes.new("ShaderNodeOutputMaterial"); _nt.links.new(_e.outputs[0], _o.inputs[0])


# ---------------------------------------------------------------- helpers
def strip_brim(me, z_cut):
    """Remove the printing brim ('_c_adhesin' parts): cut the mesh just above the build plate."""
    import bmesh
    bm = bmesh.new(); bm.from_mesh(me)
    geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
    bmesh.ops.bisect_plane(bm, geom=geom, plane_co=(0, 0, z_cut), plane_no=(0, 0, 1), clear_inner=True)
    bm.to_mesh(me); bm.free()


def strip_radius(me, r_max):
    """Remove faces whose vertices all lie outside radius r_max around the part axis (print brim)."""
    import bmesh
    bm = bmesh.new(); bm.from_mesh(me)
    kill = [f for f in bm.faces if all((v.co.x ** 2 + v.co.y ** 2) ** 0.5 > r_max for v in f.verts)]
    bmesh.ops.delete(bm, geom=kill, context="FACES")
    bm.to_mesh(me); bm.free()


def part(name, loc_mm, rot_deg=(0, 0, 0), mat=PLA, pivot=None, brim_z=None, r_max=None):
    """Import a PUMA STL (mm, CAD frame) and place it: CAD point `pivot` (default origin) -> loc_mm."""
    bpy.ops.wm.stl_import(filepath=os.path.join(STL, name + ".stl"))
    ob = bpy.context.selected_objects[0]
    ob.name = name
    me = ob.data
    if brim_z is not None:
        strip_brim(me, brim_z)
    if r_max is not None:
        strip_radius(me, r_max)
    pv = Vector(pivot) if pivot is not None else Vector((0, 0, 0))
    me.transform(Matrix.Translation(-pv))
    me.transform(Euler([math.radians(a) for a in rot_deg]).to_matrix().to_4x4())
    ob.location = Vector(loc_mm)
    ob.scale = (MM, MM, MM)
    ob.location = ob.location * MM
    for poly in me.polygons:
        poly.use_smooth = False
    # auto smooth via modifier-free approach: mark smooth + sharp edges by angle
    bpy.context.view_layer.objects.active = ob
    try:
        bpy.ops.object.shade_smooth_by_angle(angle=math.radians(35))
    except Exception:
        pass
    me.materials.clear(); me.materials.append(mat)
    return ob


def cyl(name, r_mm, h_mm, loc_mm, mat, verts=96, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=r_mm * MM, depth=h_mm * MM,
                                        location=Vector(loc_mm) * MM, rotation=[math.radians(a) for a in rot])
    ob = bpy.context.active_object; ob.name = name
    ob.data.materials.append(mat)
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(35))
    return ob


def box(name, size_mm, loc_mm, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=Vector(loc_mm) * MM)
    ob = bpy.context.active_object; ob.name = name
    ob.scale = Vector(size_mm) * MM
    ob.data.materials.append(mat)
    bev = ob.modifiers.new("bevel", "BEVEL"); bev.width = 0.0004; bev.segments = 3
    return ob


def hex_nut(name, loc_mm, s=10.0, h=5.0):
    bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=s / math.sqrt(3) * MM, depth=h * MM, location=Vector(loc_mm) * MM)
    ob = bpy.context.active_object; ob.name = name; ob.data.materials.append(ZINC)
    bev = ob.modifiers.new("bevel", "BEVEL"); bev.width = 0.0004; bev.segments = 2
    return ob


def spring(name, center_mm, z0, z1, r=5.2, wire=0.8, turns=9):
    cu = bpy.data.curves.new(name, "CURVE"); cu.dimensions = "3D"
    sp = cu.splines.new("POLY"); n = turns * 48
    sp.points.add(n - 1)
    for i in range(n):
        t = i / (n - 1)
        a = 2 * math.pi * turns * t
        sp.points[i].co = ((center_mm[0] + r * math.cos(a)) * MM, (center_mm[1] + r * math.sin(a)) * MM,
                           (z0 + (z1 - z0) * t) * MM, 1)
    cu.bevel_depth = wire / 2 * MM; cu.bevel_resolution = 4
    ob = bpy.data.objects.new(name, cu); sc.collection.objects.link(ob)
    cu.materials.append(SPRING)
    return ob


# ---------------------------------------------------------------- heights (mm)
LEG = 26.0
Z_BASE_BOTTOM = LEG                     # base plate STL spans z 2..16 -> shift so bottom sits on legs
Z_STAGE_TOP = Z_BASE_BOTTOM + 14.0      # 40
Z_SPECIMEN = Z_STAGE_TOP + 1.0 + 0.05   # slide 1.0 mm, tardigrade just under the cover glass
Z_SHOULDER = Z_SPECIMEN + 45.0          # RMS parfocal 45 mm (incl. 0.17 cover glass)
Z_PLATE = Z_SHOULDER + 18.0             # C-RMS ring (5) inside QR adapter (14) + base thread (8, 3 mm engaged)
Z_PLATE_TOP = Z_PLATE + 11.0
Z_FB_TOP = Z_PLATE_TOP + 40.0              # filter block STL z 16..56

# ---------------------------------------------------------------- stage, legs
part("ST_Base_plate", (0, 0, Z_BASE_BOTTOM - 2.0))
for (x, y) in ((51.88, -110.8), (-52.12, -110.8), (2.75, 5.02)):
    part("LG_Short_leg", (x, y, 0.0), rot_deg=(0, -90, 0), pivot=(0, 5.0, 5.3))
    cyl("Foot", 6.5, 4.0, (x, y, -2.0), RUBBER)
# glass slide, cover glass, water film with the specimen
box("Slide", (75, 25, 1.0), (AX.x, AX.y, Z_STAGE_TOP + 0.5), GLASS)
box("Cover_glass", (22, 22, 0.17), (AX.x, AX.y, Z_STAGE_TOP + 1.0 + 0.085 + 0.08), COVER)
box("Mount_water", (20, 20, 0.08), (AX.x, AX.y, Z_STAGE_TOP + 1.04), WATER)
# stage clips (spring steel strips)
for sx in (-1, 1):
    box("Stage_clip", (28, 5.5, 0.4), (AX.x + sx * 30, AX.y + 6, Z_STAGE_TOP + 1.25), STEEL)
    cyl("Clip_peg", 2.0, 6.0, (AX.x + sx * 42, AX.y + 6, Z_STAGE_TOP + 2.0), PLA)

# ---------------------------------------------------------------- condenser + LED under the stage
part("D2_Condenser_23_30", (AX.x, AX.y, Z_BASE_BOTTOM - 25.1), pivot=(0, 0, 0))
cyl("Condenser_lens_30", 14.6, 3.5, (AX.x, AX.y, Z_BASE_BOTTOM - 4.0), LENS)
cyl("Condenser_lens_23", 11.0, 4.0, (AX.x, AX.y, Z_BASE_BOTTOM - 14.0), LENS)
cyl("LED_board", 10.0, 1.6, (AX.x, AX.y, 1.2), RUBBER)
cyl("LED_COB", 3.0, 0.6, (AX.x, AX.y, 2.2), LEDM)

# ---------------------------------------------------------------- focus posts, springs, nuts
POSTS = ((39.87, -32.14), (-40.13, -32.14), (-0.14, -118.24))
for (x, y) in POSTS:
    cyl("Focus_post_M6", 3.0, Z_PLATE_TOP + 32 - Z_STAGE_TOP, (x, y, (Z_STAGE_TOP + Z_PLATE_TOP + 32) / 2), STEEL, verts=32)
    spring("Focus_spring", (x, y), Z_PLATE_TOP, Z_PLATE_TOP + 20)
    hex_nut("Nyloc", (x, y, Z_PLATE_TOP + 22.5))
    hex_nut("Nut_lower", (x, y, Z_STAGE_TOP + 2.5))
part("ST_Focus_plate", (0, 0, Z_PLATE))

# ---------------------------------------------------------------- focus drive (gears) at the back
part("FG_Pulley_coarse", (39.87, -32.14, Z_STAGE_TOP + 9.0))
part("FG_Intermedius", (2.75, 5.02, Z_STAGE_TOP + 2.0))
part("FG_Fine_gear", (40.0, 15.0, Z_STAGE_TOP + 4.0))
part("FG_Eccentric_Tensio", (-40.13, -32.14, Z_STAGE_TOP + 1.0))

# ---------------------------------------------------------------- objective holder + objective
part("QR_C_RMS_Thread", (AX.x, AX.y, Z_SHOULDER - 3.0))          # STL z 3..8: sits on the shoulder
part("QR_QR_Male_C_extn_1", (AX.x, AX.y, Z_SHOULDER + 1.0))     # STL z -2..12 -> shoulder-1 .. shoulder+13
part("QR_Base_thread", (AX.x, AX.y, Z_PLATE + 7.0))             # STL z -15..-7 -> plate-8 .. plate

# generic 20x plan achromat: reuse the procedural RMS objective, 20x colour code = green
spec = importlib.util.spec_from_file_location("objb", os.path.join(HERE, "objective.py"))
objb = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0], "--", "--out", A["out"]]
spec.loader.exec_module(objb)
M = {
    "chrome": objb.mat_metal("Obj_Chrome", (0.90, 0.90, 0.91), 0.17, aniso=0.7, rough_var=0.03, rings=True),
    "gunmetal": objb.mat_metal("Obj_Gunmetal", (0.20, 0.205, 0.22), 0.30, aniso=0.3),
    "brass": objb.mat_metal("Obj_Brass", (0.98, 0.70, 0.32), 0.22, aniso=0.2),
    "blue": objb.mat_dielectric("Obj_Green20x", (0.05, 0.55, 0.12), 0.28, coat=0.3, coat_rough=0.08),
    "black": objb.mat_dielectric("Obj_Black", (0.014, 0.014, 0.016), 0.30),
    "black_matte": objb.mat_dielectric("Obj_BlackMatte", (0.004, 0.004, 0.004), 0.9),
    "glass": objb.mat_glass("Obj_FrontLens", 140.0),
    "glass_rear": objb.mat_glass("Obj_RearLens", 140.0),
}
parts = objb.build_objective(M)
rig = bpy.data.objects.new("Objective_20x_PlanAchromat", None); sc.collection.objects.link(rig)
for p in parts:
    p.parent = rig
rig.location = Vector((AX.x, AX.y, Z_SHOULDER - 44.3)) * MM      # model: z=0 front tip, shoulder at 44.3

# ---------------------------------------------------------------- filter block, monocular, eyepiece
part("FB_Filter_block_sim", (AX.x, AX.y, Z_PLATE_TOP), pivot=(0, 0, 16.0))
part("MN_Monocular_tube_c", (AX.x, AX.y, Z_FB_TOP + 0.0), pivot=(0, 0, -29.0), r_max=17.5)   # STL z -29..54
Z_TUBE_TOP = Z_FB_TOP + 0.0 + 83.0
part("MN_Ocular_cap", (AX.x, AX.y, Z_TUBE_TOP - 6.0))
Z_SEAT = Z_TUBE_TOP - 6.0 + 13.0                   # ocular cap STL z 0..13
# WF 10x/20 eyepiece (generic, 23.2 mm barrel)
cyl("Eyepiece_barrel", 11.6, 20.0, (AX.x, AX.y, Z_SEAT - 10.0), EYE_BODY)
cyl("Eyepiece_body", 16.5, 34.0, (AX.x, AX.y, Z_SEAT + 17.0), EYE_BODY)
cyl("Eyepiece_grip", 17.0, 8.0, (AX.x, AX.y, Z_SEAT + 12.0), RUBBER, verts=48)
cyl("Eyepiece_eyelens", 9.5, 1.2, (AX.x, AX.y, Z_SEAT + 33.8), LENS)
print("STACK mm: specimen %.1f shoulder %.1f plate %.1f seat %.1f -> mech. tube %.1f" %
      (Z_SPECIMEN, Z_SHOULDER, Z_PLATE, Z_SEAT, Z_SEAT - Z_SHOULDER))

# ---------------------------------------------------------------- studio
bm_floor = bpy.ops.mesh.primitive_plane_add(size=3.0, location=(AX.x * MM, AX.y * MM, -0.004))
floor = bpy.context.active_object; floor.name = "Studio_floor"
fm, fp = principled("Studio_floor", (0.42, 0.43, 0.46), 0.6)
floor.data.materials.append(fm)
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
bg = next(n for n in w.node_tree.nodes if n.type == "BACKGROUND")
bg.inputs["Color"].default_value = (0.78, 0.79, 0.82, 1); bg.inputs["Strength"].default_value = 0.12


def area(name, loc, target, size, energy, color=(1, 1, 1)):
    ld = bpy.data.lights.new(name, "AREA"); ld.shape = "RECTANGLE"; ld.size, ld.size_y = size
    ld.energy = energy; ld.color = color
    ob = bpy.data.objects.new(name, ld); sc.collection.objects.link(ob)
    ob.location = loc
    ob.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    return ob


C = Vector((AX.x * MM, AX.y * MM, 0.14))
area("Key", C + Vector((-0.45, -0.35, 0.40)), C, (0.5, 0.3), 16)
area("Fill", C + Vector((0.45, -0.25, 0.15)), C, (0.4, 0.4), 5, (0.95, 0.97, 1.0))
area("Rim", C + Vector((0.15, 0.45, 0.35)), C, (0.15, 0.5), 14)
area("Top", C + Vector((0.0, 0.0, 0.6)), C, (0.6, 0.6), 5)

cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera")); sc.collection.objects.link(cam)
cam.data.lens = 55; cam.data.sensor_width = 36
if A["view"] == "hero":
    cam.location = C + Vector((-0.42, -0.58, 0.16))
    aim = C + Vector((0.0, 0.0, -0.005))
else:  # close-up on objective / stage
    cam.location = C + Vector((-0.16, -0.20, -0.02))
    aim = Vector((AX.x * MM, AX.y * MM, Z_SPECIMEN * MM + 0.02))
cam.rotation_euler = (aim - cam.location).to_track_quat("-Z", "Y").to_euler()
cam.data.dof.use_dof = A["view"] != "hero"
cam.data.dof.focus_distance = (aim - cam.location).length; cam.data.dof.aperture_fstop = 8
sc.camera = cam

if A["engine"] == "cycles":
    sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.cycles.samples = int(A["samples"])
    sc.cycles.use_denoising = True
    try:
        sc.cycles.denoiser = "OPENIMAGEDENOISE"
    except TypeError:
        pass
else:
    try:
        sc.render.engine = "BLENDER_EEVEE"
    except TypeError:
        pass
try:
    sc.view_settings.view_transform = "AgX"; sc.view_settings.look = "AgX - Medium High Contrast"
except TypeError:
    pass
res = int(A["res"]); sc.render.resolution_x = res; sc.render.resolution_y = int(res * 1.25) if A["view"] == "hero" else int(res * 0.75)
out = os.path.abspath(A["out"]); os.makedirs(out, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "puma_microscope.blend"))
sc.render.filepath = os.path.join(out, f"puma_{A['view']}.png")
bpy.ops.render.render(write_still=True)
print("PUMA_DONE", sc.render.filepath)
