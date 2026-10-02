# Run with FreeCADCmd. For every PUMA .FCStd in PUMA_DIR: list top-level solids (objects that no
# other object consumes), with bounding boxes, and export each as STL (no recompute).
import os, json
import FreeCAD, Mesh, MeshPart

src = os.environ["PUMA_DIR"]
dst = os.environ["PUMA_OUT"]
files = os.environ.get("PUMA_FILES", "").split(";")
summary = {}
for fn in files:
    if not fn:
        continue
    doc = FreeCAD.openDocument(os.path.join(src, fn))
    rows = []
    for o in doc.Objects:
        if not hasattr(o, "Shape") or o.Shape.isNull() or not o.Shape.Solids:
            continue
        consumers = [p for p in o.InList if hasattr(p, "Shape")]
        if consumers:
            continue
        bb = o.Shape.BoundBox
        name = "%s__%s" % (fn.replace(".FCStd", ""), o.Name)
        stl = os.path.join(dst, name + ".stl")
        try:
            m = MeshPart.meshFromShape(Shape=o.Shape, LinearDeflection=0.08, AngularDeflection=0.35)
            m.write(stl)
            ok = True
        except Exception as e:
            ok = str(e)
        rows.append({"obj": o.Name, "label": o.Label, "type": o.TypeId, "vol": round(o.Shape.Volume, 1),
                     "bb": [round(v, 1) for v in (bb.XMin, bb.YMin, bb.ZMin, bb.XMax, bb.YMax, bb.ZMax)],
                     "stl": os.path.basename(stl), "ok": ok})
    summary[fn] = rows
    FreeCAD.closeDocument(doc.Name)
    print("DONE", fn, len(rows), flush=True)
json.dump(summary, open(os.path.join(dst, "_summary.json"), "w"), indent=1)
print("ALL_DONE")
