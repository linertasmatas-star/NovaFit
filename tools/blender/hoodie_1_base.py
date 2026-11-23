"""NovaFit džemperis Blenderyje, 1 žingsnis: kūno sričių duomenys ir džemperio pagrindas (apkirptas kūnas su visomis formomis)."""
import bpy, bmesh, base64, zlib, numpy as np
NAME = globals().get("NAME", "hoodie")   # tas pats žingsnis tinka ir kitiems drabužiams (pvz. NAME="jeans")
raw = np.frombuffer(zlib.decompress(base64.b64decode(BLOB)), np.uint8)   # BLOB pateikia paleidėjas (hoodie_data.b64)
n = 13380
f = np.frombuffer(raw[:2*n].tobytes(), np.float16).astype(np.float32)
part = raw[2*n:3*n].astype(np.int32)
apos = np.frombuffer(raw[3*n:5*n].tobytes(), np.float16).astype(np.float32)
av = bpy.data.objects["avatar"]
me = av.data
for name, data, kind in (("hd_f", f, "FLOAT"), ("part", part, "INT"), ("apos", apos, "FLOAT")):
    if name in me.attributes: me.attributes.remove(me.attributes[name])
    a = me.attributes.new(name, kind, "POINT")
    a.data.foreach_set("value", data)
# kitus objektus paslepiam
for o in bpy.data.objects:
    if o.name.startswith("bell-"): o.hide_set(True); o.hide_render = True
# Džemperis = kūno kopija (su visomis formomis), apkirpta pagal lauką
for nm in (NAME, NAME + "_src"):
    o = bpy.data.objects.get(nm)
    if o: bpy.data.objects.remove(o, do_unlink=True)
old = None
if old: bpy.data.objects.remove(old, do_unlink=True)
hd = av.copy(); hd.data = me.copy(); hd.name = NAME; hd.data.name = NAME
bpy.context.collection.objects.link(hd)
bm = bmesh.new(); bm.from_mesh(hd.data)
fl = bm.verts.layers.float.get("hd_f")
shapes = list(bm.verts.layers.shape.values())
bm.faces.ensure_lookup_table()
dead = [fc for fc in bm.faces if all(v[fl] <= 0 for v in fc.verts)]
bmesh.ops.delete(bm, geom=dead, context="FACES")
loose = [v for v in bm.verts if not v.link_faces]
bmesh.ops.delete(bm, geom=loose, context="VERTS")
# Kraštai: išorinius taškus perkeliam į vietą, kur laukas = 0 (tiesus kraštas)
moves = []
for v in bm.verts:
    if v[fl] >= 0: continue
    best = None
    for e in v.link_edges:
        o = e.other_vert(v)
        if o[fl] > 0 and (best is None or o[fl] > best[fl]): best = o
    if best is None: continue
    t = v[fl] / (v[fl] - best[fl])
    moves.append((v, best, t))
for v, o, t in moves:
    v.co = v.co.lerp(o.co, t)
    for L in shapes: v[L] = v[L].lerp(o[L], t)
bm.to_mesh(hd.data); bm.free()
# Kūno šešėliavimo kryptys (normalės) drabužiui netinka – jos „piešia“ spenelius ir bambą ant lygaus audinio
if "custom_normal" in hd.data.attributes:
    hd.data.attributes.remove(hd.data.attributes["custom_normal"])
hd.data.update()
av.select_set(False); hd.select_set(True); bpy.context.view_layer.objects.active = hd
result = {"hoodie_verts": len(hd.data.vertices), "faces": len(hd.data.polygons), "snapped": len(moves),
          "keys": len(hd.data.shape_keys.key_blocks)}
