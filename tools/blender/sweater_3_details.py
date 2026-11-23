"""NovaFit megztinis (be kapišono) Blenderyje, 3 žingsnis: apvali briaunuota apykaklė.

Leisti po hoodie_1_base.py ir hoodie_2_shape.py su NAME="sweater" (forma, juostos apačioje ir rankogaliai – iš ten).
Apykaklė – storas, į vidų užlenktas žiedas aplink kaklą; jo padėtis skaičiuojama kiekvienai iš 14 formų.
"""
import bpy, bmesh, numpy as np, math
from mathutils import Vector

sw = bpy.data.objects["sweater"]
me = sw.data
kb = me.shape_keys.key_blocks
nv = len(me.vertices)
me.calc_loop_triangles()
tri = np.empty(len(me.loop_triangles) * 3, np.int32)
me.loop_triangles.foreach_get("vertices", tri); tri = tri.reshape(-1, 3)
edges = np.empty(len(me.edges) * 2, np.int32); me.edges.foreach_get("vertices", edges); edges = edges.reshape(-1, 2)

def getco(k):
    a = np.empty(nv * 3, np.float32); k.data.foreach_get("co", a); return a.reshape(-1, 3)
K = {k.name: getco(k) for k in kb}
B = K["Basis"]

# --- Kaklo kraštas ---
cnt = {}
for t in tri:
    for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
        k = (min(a, b), max(a, b)); cnt[k] = cnt.get(k, 0) + 1
bverts = np.unique([v for (a, b), c in cnt.items() if c == 1 for v in (a, b)])
neck = bverts[B[bverts, 2] > 1.30]
c0 = B[neck, :2].mean(0)
neck = neck[np.argsort(np.arctan2(B[neck, 1] - c0[1], B[neck, 0] - c0[0]))]
NN = len(neck)

# Apykaklės profilis (spindulio dalis nuo kaklo krašto, aukštis m): kyla, apsivynioja ir grįžta į vidų
PROF = [(1.000, 0.000), (1.008, 0.005), (1.002, 0.010), (0.985, 0.014), (0.962, 0.015),
        (0.945, 0.011), (0.938, 0.004)]
NR = len(PROF)

def build(P):
    ring = P[neck]
    c = ring[:, :2].mean(0)
    zc = ring[:, 2]
    pts = []
    for a in range(NN):
        d = ring[a, :2] - c
        for f, h in PROF:
            xy = c + d * f
            pts.append((xy[0], xy[1], zc[a] + h))
    return np.array(pts, np.float32)

bm = bmesh.new(); bm.from_mesh(me)
bm.verts.ensure_lookup_table()
new = [bm.verts.new(Vector(p)) for p in build(B)]
def V(a, j): return new[a * NR + j]
for a in range(NN):
    b = (a + 1) % NN
    for j in range(NR - 1):
        bm.faces.new((V(a, j), V(b, j), V(b, j + 1), V(a, j + 1))).smooth = True
bm.verts.ensure_lookup_table()
weld = {new[a * NR]: bm.verts[int(neck[a])] for a in range(NN)}      # prisiuvam prie kaklo krašto
bmesh.ops.weld_verts(bm, targetmap=weld)
bm.to_mesh(me); bm.free()
me.update()

n_all = len(me.vertices)
keep = np.ones(NN * NR, bool); keep[[a * NR for a in range(NN)]] = False
for k in kb:
    full = np.empty(n_all * 3, np.float32); k.data.foreach_get("co", full); full = full.reshape(-1, 3)
    full[nv:] = build(K[k.name])[keep]
    k.data.foreach_set("co", full.ravel())
    if k.name == "Basis":
        me.vertices.foreach_set("co", full.ravel())
me.update()
sw.color = (0.78, 0.72, 0.62, 1)
result = {"neck_verts": NN, "added": int(n_all - nv), "faces": len(me.polygons)}
