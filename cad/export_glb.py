"""Blender: export the assembled PUMA (puma_microscope.blend) as a compact glTF binary for the web viewer.
  blender -b puma_microscope.blend --python export_glb.py -- <out.glb> [max_tris_per_object]
Studio objects (floor, lights, camera) are dropped, curves become meshes, dense meshes are decimated."""
import bpy, sys, os

argv = sys.argv[sys.argv.index("--") + 1:]
out = argv[0]
max_tris = int(argv[1]) if len(argv) > 1 else 12000
for ob in list(bpy.data.objects):
    if ob.type in {"LIGHT", "CAMERA"} or ob.name.startswith("Studio_floor"):
        bpy.data.objects.remove(ob, do_unlink=True)
bpy.ops.object.select_all(action="DESELECT")
for ob in list(bpy.data.objects):
    if ob.type == "CURVE":
        bpy.context.view_layer.objects.active = ob
        ob.select_set(True)
        bpy.ops.object.convert(target="MESH")
        ob.select_set(False)
tot = 0
for ob in bpy.data.objects:
    if ob.type != "MESH":
        continue
    tris = sum(len(p.vertices) - 2 for p in ob.data.polygons)
    if tris > max_tris:
        m = ob.modifiers.new("dec", "DECIMATE")
        m.ratio = max(0.04, max_tris / tris)
        tris = int(tris * m.ratio)
    tot += tris
print("TOTAL_TRIS", tot)
bpy.ops.export_scene.gltf(filepath=out, export_format=("GLTF_EMBEDDED" if out.endswith(".json") or out.endswith(".gltf") else "GLB"), export_apply=True, use_selection=False,
                          export_cameras=False, export_lights=False)
print("GLB_DONE", os.path.getsize(out) / 1e6, "MB")
