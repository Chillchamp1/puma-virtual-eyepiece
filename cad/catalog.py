"""Blender: import selected PUMA STLs side by side (each normalised to its bbox) and render a labelled
parts catalogue so parts can be identified for assembly.
  blender -b --factory-startup --python catalog.py -- <stl_dir> <out.png> name1,name2,...
"""
import bpy, sys, os, math
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
src, out, names = argv[0], argv[1], argv[2].split(",")
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
cols = 6
cell = 0.16
mat = bpy.data.materials.new("PLA"); mat.use_nodes = True
p = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
p.inputs["Base Color"].default_value = (0.05, 0.05, 0.055, 1); p.inputs["Roughness"].default_value = 0.45
for i, nm in enumerate(names):
    path = os.path.join(src, nm + ".stl")
    if not os.path.exists(path):
        print("MISSING", nm); continue
    bpy.ops.wm.stl_import(filepath=path)
    ob = bpy.context.selected_objects[0]
    ob.scale = (0.001, 0.001, 0.001)
    bpy.context.view_layer.update()
    bb = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
    mn = Vector((min(v.x for v in bb), min(v.y for v in bb), min(v.z for v in bb)))
    mx = Vector((max(v.x for v in bb), max(v.y for v in bb), max(v.z for v in bb)))
    size = max(mx - mn)
    s = 0.12 / size
    ob.scale = (0.001 * s,) * 3
    bpy.context.view_layer.update()
    bb = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
    ctr = sum(bb, Vector()) / 8
    gx, gy = i % cols, i // cols
    ob.location += Vector((gx * cell, -gy * cell * 1.1, 0)) - ctr
    ob.data.materials.append(mat)
    for poly in ob.data.polygons:
        poly.use_smooth = False
    t = bpy.data.curves.new("t%d" % i, "FONT"); t.body = nm.split("__")[-1][:22]; t.size = 0.012
    to = bpy.data.objects.new("t%d" % i, t); sc.collection.objects.link(to)
    to.location = (gx * cell - 0.07, -gy * cell * 1.1 - 0.075, 0)
    tm = bpy.data.materials.new("txt"); tm.use_nodes = True
    next(n for n in tm.node_tree.nodes if n.type == "BSDF_PRINCIPLED").inputs["Base Color"].default_value = (0.9, 0.3, 0.1, 1)
    t.materials.append(tm)
rows = (len(names) + cols - 1) // cols
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam)
cam.data.type = "ORTHO"; cam.data.ortho_scale = max(cols * cell, rows * cell * 1.1) * 1.05
cx, cy = (cols - 1) * cell / 2, -(rows - 1) * cell * 1.1 / 2
cam.location = (cx + 0.25, cy - 0.35, 0.45)
cam.rotation_euler = (Vector((cx, cy, 0)) - cam.location).to_track_quat("-Z", "Y").to_euler()
sc.camera = cam
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
next(n for n in w.node_tree.nodes if n.type == "BACKGROUND").inputs["Color"].default_value = (0.85, 0.86, 0.88, 1)
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sc.collection.objects.link(sun)
sun.data.energy = 3.0; sun.rotation_euler = (0.6, 0.3, 0.8)
try:
    sc.render.engine = "BLENDER_EEVEE"
except TypeError:
    pass
sc.render.resolution_x = 1800; sc.render.resolution_y = int(1800 * rows * 1.1 / cols)
sc.render.filepath = out
bpy.ops.render.render(write_still=True)
print("CATALOG_DONE")
