"""
Olympus PLN 40X (Plan N 40x / 0.65 / inf / 0.17 / FN22) - procedural Blender build.

Run headless:
  blender -b --factory-startup --python build_objective.py -- --out rounds/r2 --res 1800 --samples 256
  blender -b --factory-startup --python build_objective.py -- --out final --variant face --face ref/face.jpg

All geometry is built at real-world scale (1 unit = 1 m, modelled in mm).
Reference specs (Edmund Optics #86-815):
  max diameter 24 mm, length w/o thread 44.3 mm, parfocal 45 mm, WD 0.6 mm,
  RMS thread 0.797" (20.32 mm) x 36 TPI (pitch 0.7056 mm), 40X colour code = light blue.
"""
import bpy, bmesh, math, sys, os, time
from mathutils import Vector, Matrix

MM = 0.001
AXIS = Vector((0, 0, 1))      # optical axis (nose -> thread) in world space, set in main()
FACE_CAM = (0.0, -0.095, 0.008)  # viewer's eye position for the face-reflection variant
HERO_CAM = (0.0, -0.275, 0.07)

# ---------------------------------------------------------------- CLI
def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    a = {"out": "out", "res": 1200, "samples": 128, "variant": "hero", "face": "", "save": True,
         "chrome_rough": "", "chrome_aniso": ""}
    i = 0
    while i < len(argv):
        k = argv[i].lstrip("-")
        if k == "nosave":
            a["save"] = False; i += 1; continue
        a[k] = argv[i + 1]; i += 2
    a["res"] = int(a["res"]); a["samples"] = int(a["samples"])
    return a

ARGS = parse_args()

# ---------------------------------------------------------------- helpers
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def link(obj, coll=None):
    (coll or bpy.context.scene.collection).objects.link(obj)
    return obj

def finish_mesh(bm, name, mat, sharp_deg=None, smooth=True):
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    for f in bm.faces:
        f.smooth = smooth
    if sharp_deg is not None:
        lim = math.radians(sharp_deg)
        for e in bm.edges:
            if len(e.link_faces) == 2 and e.calc_face_angle(0) > lim:
                e.smooth = False
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    if mat:
        me.materials.append(mat)
    return link(bpy.data.objects.new(name, me))

def lathe(name, profile, mat, segs=256, sharp_deg=28):
    """Revolve a CLOSED (r, z) cross-section polygon (in mm) around Z."""
    bm = bmesh.new()
    vs = [bm.verts.new((r * MM, 0.0, z * MM)) for r, z in profile]
    es = [bm.edges.new((vs[i], vs[(i + 1) % len(vs)])) for i in range(len(vs))]
    bmesh.ops.spin(bm, geom=vs + es, cent=(0, 0, 0), axis=(0, 0, 1),
                   angle=2 * math.pi, steps=segs, use_merge=True)
    return finish_mesh(bm, name, mat, sharp_deg)

def smoothstep(e0, e1, x):
    t = max(0.0, min(1.0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)

# ---------------------------------------------------------------- materials
def new_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    p = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    return m, nt, p

def mat_metal(name, color, rough, aniso=0.0, rough_var=0.0, scratch=0.0, rings=False):
    m, nt, p = new_mat(name)
    p.inputs["Base Color"].default_value = (*color, 1)
    p.inputs["Metallic"].default_value = 1.0
    p.inputs["Roughness"].default_value = rough
    if aniso:
        p.inputs["Anisotropic"].default_value = aniso
        tg = nt.nodes.new("ShaderNodeTangent")
        tg.direction_type = "RADIAL"; tg.axis = "Z"      # circumferential = lathe-turned
        nt.links.new(tg.outputs["Tangent"], p.inputs["Tangent"])
    if rough_var:
        # low-frequency roughness variation -> breaks up CG-perfect highlights
        tc = nt.nodes.new("ShaderNodeTexCoord")
        nz = nt.nodes.new("ShaderNodeTexNoise")
        nz.inputs["Scale"].default_value = 180.0
        nz.inputs["Detail"].default_value = 4.0
        if rings:
            # stretch the noise around the axis -> faint concentric lathe banding along Z
            mp = nt.nodes.new("ShaderNodeMapping")
            mp.inputs["Scale"].default_value = (0.03, 0.03, 1.6)
            nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
            nt.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
        else:
            nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["To Min"].default_value = rough - rough_var
        mr.inputs["To Max"].default_value = rough + rough_var
        nt.links.new(nz.outputs["Fac"], mr.inputs["Value"])
        rough_out = mr.outputs["Result"]
        if rings:
            # fine turning marks (~0.2 mm pitch, irregular): roughness rings, no normal aliasing
            mp2 = nt.nodes.new("ShaderNodeMapping")
            mp2.inputs["Scale"].default_value = (0.01, 0.01, 25.0)
            nt.links.new(tc.outputs["Object"], mp2.inputs["Vector"])
            nz2 = nt.nodes.new("ShaderNodeTexNoise")
            nz2.inputs["Scale"].default_value = 180.0
            nz2.inputs["Detail"].default_value = 2.0
            nt.links.new(mp2.outputs["Vector"], nz2.inputs["Vector"])
            mr2 = nt.nodes.new("ShaderNodeMapRange")
            mr2.inputs["To Min"].default_value = -0.035
            mr2.inputs["To Max"].default_value = 0.035
            nt.links.new(nz2.outputs["Fac"], mr2.inputs["Value"])
            add = nt.nodes.new("ShaderNodeMath"); add.operation = "ADD"
            nt.links.new(rough_out, add.inputs[0]); nt.links.new(mr2.outputs["Result"], add.inputs[1])
            rough_out = add.outputs[0]
        nt.links.new(rough_out, p.inputs["Roughness"])
    if scratch:
        # lathe turning marks: very fine circumferential ridges along Z
        tc = nt.nodes.new("ShaderNodeTexCoord")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])
        mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = 2 * math.pi / (0.02 * MM)   # 20 um feed
        nt.links.new(sep.outputs["Z"], mul.inputs[0])
        sn = nt.nodes.new("ShaderNodeMath"); sn.operation = "SINE"
        nt.links.new(mul.outputs[0], sn.inputs[0])
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = scratch
        bump.inputs["Distance"].default_value = 0.00002
        nt.links.new(sn.outputs[0], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], p.inputs["Normal"])
    return m

def mat_dielectric(name, color, rough, coat=0.0, coat_rough=0.03):
    m, nt, p = new_mat(name)
    p.inputs["Base Color"].default_value = (*color, 1)
    p.inputs["Roughness"].default_value = rough
    p.inputs["Coat Weight"].default_value = coat
    p.inputs["Coat Roughness"].default_value = coat_rough
    return m

def mat_glass(name, film_nm=320.0):
    m, nt, p = new_mat(name)
    p.inputs["Base Color"].default_value = (1, 1, 1, 1)
    p.inputs["Transmission Weight"].default_value = 1.0
    p.inputs["Roughness"].default_value = 0.0
    p.inputs["IOR"].default_value = 1.62
    p.inputs["Thin Film Thickness"].default_value = film_nm   # MgF2-ish AR coat -> violet/magenta
    p.inputs["Thin Film IOR"].default_value = 1.38
    return m

def mat_emit_softbox(name, strength, color=(1, 1, 1), soft=0.22):
    """Emission plane with soft feathered edges (like a diffused softbox)."""
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*color, 1)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["UV"], sep.inputs["Vector"])
    def edge(sock):
        a = nt.nodes.new("ShaderNodeMath"); a.operation = "PINGPONG"; a.inputs[1].default_value = 0.5
        nt.links.new(sock, a.inputs[0])            # 0 at border -> 0.5 at centre
        mr = nt.nodes.new("ShaderNodeMapRange"); mr.interpolation_type = "SMOOTHSTEP"
        mr.inputs["From Min"].default_value = 0.0; mr.inputs["From Max"].default_value = soft
        nt.links.new(a.outputs[0], mr.inputs["Value"])
        return mr.outputs["Result"]
    mx = nt.nodes.new("ShaderNodeMath"); mx.operation = "MULTIPLY"
    nt.links.new(edge(sep.outputs["X"]), mx.inputs[0])
    nt.links.new(edge(sep.outputs["Y"]), mx.inputs[1])
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    front = nt.nodes.new("ShaderNodeMath"); front.operation = "SUBTRACT"; front.inputs[0].default_value = 1.0
    nt.links.new(geo.outputs["Backfacing"], front.inputs[1])
    mx2 = nt.nodes.new("ShaderNodeMath"); mx2.operation = "MULTIPLY"
    nt.links.new(mx.outputs[0], mx2.inputs[0]); nt.links.new(front.outputs[0], mx2.inputs[1])
    st = nt.nodes.new("ShaderNodeMath"); st.operation = "MULTIPLY"; st.inputs[1].default_value = strength
    nt.links.new(mx2.outputs[0], st.inputs[0])
    nt.links.new(st.outputs[0], em.inputs["Strength"])
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    return m

# ---------------------------------------------------------------- geometry
def build_thread(name, mat, z0, z1, r_relief, r_major, depth, pitch, r_inner,
                 z_thread_start, chamfer, segs=384, rows_per_pitch=28):
    """Real helical RMS thread as a closed tube (outer threaded skin + bore)."""
    rows = int((z1 - z0) / pitch * rows_per_pitch) + 1
    bm = bmesh.new()
    def R(theta, z):
        if z < z_thread_start:
            return r_relief
        u = (z / pitch - theta / (2 * math.pi)) % 1.0
        tri = 1.0 - abs(2.0 * u - 1.0)                 # 0 root .. 1 crest
        prof = 0.65 * tri + 0.35 * tri * tri * (3 - 2 * tri)   # straight flanks, slightly rounded crest/root
        ramp = smoothstep(z_thread_start, z_thread_start + 0.9 * pitch, z)
        r = (r_major - depth) + depth * prof * ramp
        r = max(r, r_relief) if ramp < 0.999 else r
        r = min(r, r_major - max(0.0, z - (z1 - chamfer)))   # 45 deg lead-in chamfer
        return r
    outer, inner = [], []
    for j in range(rows):
        z = z0 + (z1 - z0) * j / (rows - 1)
        ro, ri = [], []
        for i in range(segs):
            th = 2 * math.pi * i / segs
            r = R(th, z)
            ro.append(bm.verts.new((r * math.cos(th) * MM, r * math.sin(th) * MM, z * MM)))
            ri.append(bm.verts.new((r_inner * math.cos(th) * MM, r_inner * math.sin(th) * MM, z * MM)))
        outer.append(ro); inner.append(ri)
    for j in range(rows - 1):
        for i in range(segs):
            k = (i + 1) % segs
            bm.faces.new((outer[j][i], outer[j][k], outer[j + 1][k], outer[j + 1][i]))
            bm.faces.new((inner[j][i], inner[j + 1][i], inner[j + 1][k], inner[j][k]))
    for i in range(segs):
        k = (i + 1) % segs
        bm.faces.new((outer[0][i], inner[0][i], inner[0][k], outer[0][k]))
        bm.faces.new((outer[-1][i], outer[-1][k], inner[-1][k], inner[-1][i]))
    return finish_mesh(bm, name, mat, sharp_deg=None)

def build_objective(M):
    parts = []
    # --- main barrel + nose cone (satin chrome).  (r, z) mm, z = 0 at front of tip
    body = [
        (5.02, 3.00), (7.08, 3.00), (7.30, 3.05), (7.38, 3.16),   # flat nose face, crisp edge break
        (7.40, 3.40), (7.47, 3.50),                               # short land
        (11.80, 7.83),                                            # straight 45 deg cone mantle
        (11.93, 7.98), (11.985, 8.10), (12.00, 8.22),             # small bevel catching a light edge
        (12.00, 15.60), (11.97, 15.625), (11.66, 15.645),         # colour-band groove
        (11.66, 17.575), (11.97, 17.595), (12.00, 17.62),
        (12.00, 43.85), (11.94, 44.12), (11.80, 44.24), (11.62, 44.30),  # top edge radius
        (10.60, 44.30), (10.45, 44.26),                           # shoulder face
        (8.62, 44.26), (8.62, 14.0), (5.02, 9.0),
    ]
    parts.append(lathe("Body_Chrome", body, M["chrome"], segs=360, sharp_deg=22))
    # --- light-blue 40X colour band: flat, crisp-edged, flush in its groove
    band = [(11.66, 15.725), (11.91, 15.725), (11.94, 15.75), (11.94, 17.47),
            (11.91, 17.495), (11.66, 17.495)]          # 0.08 mm seams, 0.06 mm below the barrel
    parts.append(lathe("ColourBand_Blue", band, M["blue"], segs=360, sharp_deg=20))
    # --- spring-loaded front tip (polished steel), ~10 mm diameter
    tip_face = [(2.45, 0.14), (2.58, 0.0), (4.74, 0.0), (4.85, 0.04), (4.88, 0.26), (2.45, 0.26)]
    parts.append(lathe("TipFace_Gunmetal", tip_face, M["gunmetal"], segs=288, sharp_deg=22))
    sleeve = [(4.30, 0.26), (4.86, 0.26), (4.90, 0.31), (4.90, 6.0), (4.30, 6.0)]
    parts.append(lathe("TipSleeve_Chrome", sleeve, M["chrome"], segs=288, sharp_deg=22))
    # --- matte black lens retaining ring, slightly recessed
    ring = [(1.90, 0.17), (2.50, 0.11), (2.50, 1.5), (1.90, 1.5)]
    parts.append(lathe("FrontRetainer_Black", ring, M["black"], segs=192))
    # --- front lens: ~3.9 mm clear aperture, domed, almost flush with the tip face
    lens = [(0.0, 0.0)]
    for i in range(1, 13):
        r = 1.93 * i / 12
        lens.append((r, 0.0 + 0.22 * (r / 1.93) ** 2))
    lens += [(1.93, 1.4), (0.0, 1.4)]
    parts.append(lathe("FrontLens_Glass", lens, M["glass"], segs=256, sharp_deg=50))
    # --- dark interior (so the lenses look into a blackened tube, not into void)
    interior = [(0.0, 1.6), (1.95, 1.6), (1.95, 6.2), (4.90, 9.4), (8.45, 14.2), (8.45, 44.0), (0.0, 44.0)]
    parts.append(lathe("Interior_Black", interior, M["black_matte"], segs=128))
    # --- rear lens cell (black anodized, stepped baffle) + large rear lens
    rear_cell = [(7.05, 44.05), (8.49, 44.05), (8.49, 48.90), (8.22, 48.90),
                 (8.22, 47.30), (7.60, 47.10), (7.60, 46.40), (7.20, 46.30), (7.05, 46.1)]
    parts.append(lathe("RearCell_Black", rear_cell, M["black"], segs=192, sharp_deg=25))
    rlens = [(0.0, 44.2), (7.0, 44.2), (7.0, 45.6)]
    for i in range(10, -1, -1):
        r = 7.0 * i / 10
        rlens.append((r, 45.6 + 0.85 * (1 - (r / 7.0) ** 2)))
    parts.append(lathe("RearLens_Glass", rlens, M["glass_rear"], segs=192, sharp_deg=50))
    # --- RMS thread (brass), real helix 36 TPI
    parts.append(build_thread("RMS_Thread_Brass", M["brass"], z0=44.26, z1=49.0,
                              r_relief=9.55, r_major=10.16, depth=0.42, pitch=25.4 / 36,
                              r_inner=8.50, z_thread_start=45.20, chamfer=0.40))
    return parts

# ---------------------------------------------------------------- studio
def softbox(name, size, loc, target, strength, color=(1, 1, 1), soft=0.22, align=None):
    """Feathered emissive card facing `target`; its long (local Y) side follows `align`."""
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=loc)
    ob = bpy.context.active_object; ob.name = name
    loc, target = Vector(loc), Vector(target)
    z = (target - loc).normalized()          # +Z (emitting front face) looks at the target
    up = Vector(align) if align is not None else Vector((0, 0, 1))
    y = (up - z * up.dot(z)).normalized()
    x = y.cross(z)
    ob.rotation_euler = Matrix((x, y, z)).transposed().to_euler()
    ob.scale = (size[0], size[1], 1)
    ob.data.materials.append(mat_emit_softbox(name + "_M", strength, color, soft))
    ob.visible_camera = False
    ob.visible_shadow = False
    return ob

def mirror_dir(psi_deg, cam_loc):
    """World direction (from the object) whose reflection shows up on the barrel at
    angle psi/2 from the camera-facing line. psi=0 -> straight behind the camera,
    +psi -> upper flank, -psi -> lower flank, +-180 -> grazing silhouette."""
    A = AXIS
    v = (-Vector(cam_loc)).normalized()               # camera -> object
    a = v.dot(A)                                      # axial part survives the reflection
    c = -v
    e1 = (c - A * c.dot(A)).normalized()
    e2 = A.cross(e1)
    if e2.z < 0:
        e2 = -e2
    p = math.radians(psi_deg)
    return (math.sqrt(1 - a * a) * (math.cos(p) * e1 + math.sin(p) * e2) + a * A).normalized()

def strip(name, psi, dist, size, strength, cam_loc, color=(1, 1, 1), soft=0.28):
    d = mirror_dir(psi, cam_loc)
    return softbox(name, size, tuple(d * dist), (0, 0, 0), strength, color, soft, align=AXIS)

def kicker(name, normal, dist, size, strength, cam_loc, soft=0.4, color=(1, 1, 1)):
    """Small card placed exactly where a surface with `normal` (at the origin) mirrors it
    into the camera -> guarantees a highlight on e.g. the front lens or the cone."""
    v = (-Vector(cam_loc)).normalized()
    n = Vector(normal).normalized()
    r = v - 2 * v.dot(n) * n
    return softbox(name, size, tuple(r * dist), (0, 0, 0), strength, color, soft)

def build_cyclorama():
    bm = bmesh.new()
    prof = []
    floor_z, back_y, R = -0.20, 0.50, 0.30
    for i in range(0, 40):
        y = -1.2 + (back_y - R + 1.2) * i / 39
        prof.append((y, floor_z))
    for i in range(1, 25):
        a = math.pi / 2 * i / 24
        prof.append((back_y - R + R * math.sin(a), floor_z + R - R * math.cos(a)))
    for i in range(1, 12):
        prof.append((back_y, floor_z + R + 0.9 * i / 11))
    rows = []
    for x in (-1.4, 1.4):
        rows.append([bm.verts.new((x, y, z)) for y, z in prof])
    for k in range(len(prof) - 1):
        bm.faces.new((rows[0][k], rows[1][k], rows[1][k + 1], rows[0][k + 1]))
    cyc = finish_mesh(bm, "Cyclorama", None)
    m, nt, p = new_mat("Cyc_Charcoal")
    p.inputs["Base Color"].default_value = (0.022, 0.023, 0.026, 1)   # near-black sweep
    p.inputs["Roughness"].default_value = 0.7
    p.inputs["Specular IOR Level"].default_value = 0.2
    cyc.data.materials.append(m)
    return cyc

def add_backdrop_glow(cam, radius, strength):
    cyc = bpy.data.objects["Cyclorama"]
    ray = (-cam).normalized()
    dg = bpy.context.evaluated_depsgraph_get()
    start = cam + ray * (cam.length + 0.06)
    for _ in range(20):                          # skip the (camera-invisible) light cards
        hit, P, _n, _i, ob, _m = bpy.context.scene.ray_cast(dg, start, ray)
        if not hit or ob.name == "Cyclorama":
            break
        start = P + ray * 1e-4
    if not hit:
        return
    print("BACKDROP_GLOW_AT", tuple(round(c, 3) for c in P))
    nt = cyc.data.materials[0].node_tree
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    dist = nt.nodes.new("ShaderNodeVectorMath"); dist.operation = "DISTANCE"
    dist.inputs[1].default_value = P
    nt.links.new(tc.outputs["Object"], dist.inputs[0])
    mr = nt.nodes.new("ShaderNodeMapRange"); mr.interpolation_type = "SMOOTHSTEP"
    mr.inputs["From Min"].default_value = radius; mr.inputs["From Max"].default_value = 0.0
    mr.inputs["To Max"].default_value = strength
    nt.links.new(dist.outputs["Value"], mr.inputs["Value"])
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (0.92, 0.95, 1.0, 1)
    nt.links.new(mr.outputs["Result"], em.inputs["Strength"])
    addsh = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(bsdf.outputs[0], addsh.inputs[0]); nt.links.new(em.outputs[0], addsh.inputs[1])
    nt.links.new(addsh.outputs[0], out.inputs["Surface"])

def build_studio(variant):
    sc = bpy.context.scene
    build_cyclorama()
    cam = Vector(FACE_CAM if variant == "face" else HERO_CAM)
    # Strips are placed by mirror geometry (see mirror_dir) and run parallel to the axis:
    # bright upper line | dark band | broad soft centre | dark band | thin lower line | rims
    strip("S_Upper", 62, 0.24, (0.07, 0.55), 13.0, cam)
    strip("S_Centre", -22, 0.34, (0.16, 0.70), 2.6, cam, soft=0.45)
    strip("S_Lower", -80, 0.17, (0.05, 0.45), 8.0, cam)
    strip("S_RimA", 156, 0.22, (0.03, 0.40), 2.5, cam)
    strip("S_RimB", -158, 0.22, (0.03, 0.40), 2.0, cam, color=(1.0, 0.97, 0.92))
    # kickers: front lens (coating reflex) and the camera-facing / upper flank of the 45 deg cone
    v = (-cam).normalized()
    e1 = (-v - AXIS * (-v).dot(AXIS)).normalized()
    e2 = AXIS.cross(e1)
    if e2.z < 0:
        e2 = -e2
    kicker("K_Lens", -AXIS, 0.20, (0.08, 0.08), 1.6, cam)
    kicker("K_ConeFront", (-AXIS + e1), 0.24, (0.20, 0.28), 7.0, cam, soft=0.45)
    kicker("K_ConeTop", (-AXIS + e2), 0.24, (0.08, 0.14), 8.0, cam)
    # small soft key from the left: lights the thread end face and the cone flank
    softbox("SB_KeyL", (0.20, 0.25), (-0.28, -0.10, 0.10), (0, 0, 0), 3.0, soft=0.4)
    # backdrop glow: a soft light field painted exactly where the view ray behind the objective lands
    add_backdrop_glow(cam, radius=0.17, strength=0.09)

    if variant == "face" and ARGS["face"]:
        # the viewer's face, standing where the camera is: only seen via reflections
        # The camera IS the viewer's eye, so the photo sits right behind it. Physically, a
        # convex cylinder squeezes the reflection horizontally; it only reads when the face is close.
        img = bpy.data.images.load(os.path.abspath(ARGS["face"]))
        asp = img.size[0] / img.size[1]
        h = 0.40                                        # photo frame height at face scale (face ~0.2 m)
        cx, cy, cz = FACE_CAM
        # photo: eyes' midpoint sits at u=0.39, v=0.56 from top -> put that point exactly at the camera
        w = h * asp
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(cx - (0.5 - 0.39) * w, cy - 0.006, cz + (0.56 - 0.5) * h))
        fp = bpy.context.active_object; fp.name = "Viewer_Face"
        fp.scale = (h * asp, h, 1)
        fp.rotation_euler = (math.radians(90), 0, math.radians(180))   # upright, front faces +Y
        m = bpy.data.materials.new("Face_Photo"); m.use_nodes = True
        nt = m.node_tree; nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        em = nt.nodes.new("ShaderNodeEmission"); em.inputs["Strength"].default_value = 1.5
        tx = nt.nodes.new("ShaderNodeTexImage"); tx.image = img
        nt.links.new(tx.outputs["Color"], em.inputs["Color"])
        nt.links.new(em.outputs[0], out.inputs["Surface"])
        fp.data.materials.append(m)
        fp.visible_camera = False
        fp.visible_shadow = False

    w = bpy.data.worlds.new("World"); sc.world = w
    w.use_nodes = True
    bg = next(n for n in w.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs["Color"].default_value = (0.02, 0.02, 0.022, 1)
    bg.inputs["Strength"].default_value = 1.0

def setup_camera_and_render(variant):
    sc = bpy.context.scene
    cd = bpy.data.cameras.new("Cam")
    cam = link(bpy.data.objects.new("Cam_Hero", cd))
    cd.sensor_width = 36
    cd.clip_start = 0.005; cd.clip_end = 10
    if variant == "face":
        cd.lens = 52
        cam.location = FACE_CAM
        aim = Vector((0.0, 0.0, 0.0015))
    else:
        cd.lens = 115
        cam.location = HERO_CAM
        aim = Vector((0.0, 0.0, -0.002))
    cam.rotation_euler = (aim - cam.location).normalized().to_track_quat("-Z", "Y").to_euler()
    cd.dof.use_dof = False            # product shots are focus-stacked: everything crisp
    sc.camera = cam

    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = ARGS["samples"]
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.01
    sc.cycles.use_denoising = True
    try:
        sc.cycles.denoiser = "OPENIMAGEDENOISE"
    except TypeError:
        pass
    sc.cycles.max_bounces = 16
    sc.cycles.glossy_bounces = 8
    sc.cycles.transmission_bounces = 12
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.cycles.blur_glossy = 0.5
    sc.render.resolution_x = ARGS["res"]
    sc.render.resolution_y = int(ARGS["res"] * (5 / 4 if variant == "face" else 2 / 3))
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "PNG"
    vs = sc.view_settings
    try:
        vs.view_transform = "AgX"
    except TypeError as e:
        print("VTFAIL", e)
    for cand in ("AgX - Medium High Contrast", "Medium High Contrast"):
        try:
            vs.look = cand; break
        except TypeError as e:
            print("LOOKFAIL", cand, e)
    vs.exposure = 0.0
    print("VIEW", vs.view_transform, vs.look)
    return cam

def main():
    global AXIS
    t0 = time.time()
    reset_scene()
    M = {
        "chrome": mat_metal("Chrome_Satin", (0.90, 0.90, 0.91),
                            float(ARGS["chrome_rough"] or 0.17), aniso=float(ARGS["chrome_aniso"] or 0.70),
                            rough_var=0.03 if ARGS["variant"] != "face" else 0.008,
                            scratch=0.0, rings=True),
        "gunmetal": mat_metal("Gunmetal_TipFace", (0.20, 0.205, 0.22), 0.30, aniso=0.3, rough_var=0.02),
        "brass": mat_metal("Brass_Thread", (0.98, 0.70, 0.32), 0.22, aniso=0.2, rough_var=0.03),
        "blue": mat_dielectric("Paint_Blue40X", (0.03, 0.26, 0.82), 0.28, coat=0.3, coat_rough=0.08),
        "black": mat_dielectric("Black_Anodized", (0.014, 0.014, 0.016), 0.30),
        "black_matte": mat_dielectric("Black_Matte", (0.004, 0.004, 0.004), 0.9),
        "glass": mat_glass("Glass_FrontLens", 140.0),
        "glass_rear": mat_glass("Glass_RearLens", 140.0),
    }
    parts = build_objective(M)
    rig = link(bpy.data.objects.new("PLN40X", None))
    rig.empty_display_size = 0.02
    for p in parts:
        p.parent = rig
        p.location = (0, 0, -24.5 * MM)          # pivot at mid-length
    if ARGS["variant"] == "face":
        # upright, thread up, slight lean back: the barrel faces the viewer like a convex mirror
        AXIS = Vector((0.0, 0.0, 1.0))
    else:
        # hero pose: floating, nose toward camera-right-down (like a catalogue shot)
        AXIS = -Vector((0.75, -0.58, -0.22)).normalized()
    rig.rotation_euler = AXIS.to_track_quat("Z", "X").to_euler()
    build_studio(ARGS["variant"])
    setup_camera_and_render(ARGS["variant"])

    out = os.path.abspath(ARGS["out"]); os.makedirs(out, exist_ok=True)
    blend = "olympus_pln40x.blend" if ARGS["variant"] == "hero" else f"olympus_pln40x_{ARGS['variant']}.blend"
    if ARGS["save"]:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, blend))
    sc = bpy.context.scene
    sc.render.filepath = os.path.join(out, f"{ARGS['variant']}.png")
    print("BUILD_DONE", round(time.time() - t0, 1), "s")
    bpy.ops.render.render(write_still=True)
    print("RENDER_DONE", round(time.time() - t0, 1), "s ->", sc.render.filepath)

if __name__ == "__main__":
    main()
