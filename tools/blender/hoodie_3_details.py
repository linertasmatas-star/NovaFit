"""NovaFit džemperis Blenderyje, 3 žingsnis: kapišonas (nuleistas), kengūros kišenė, raišteliai.

Leisti iškart po 2 žingsnio (jis sukuria naują „hoodie“). Detalės pridedamos prie to paties modelio,
o jų padėtis kiekvienai kūno formai apskaičiuojama spinduliais ant tos formos džemperio paviršiaus.
"""
import bpy, bmesh, numpy as np, math
from mathutils import Vector
from mathutils.bvhtree import BVHTree

hd = bpy.data.objects["hoodie"]
me = hd.data
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

# --- Kaklo kraštas (kapišonas prisiūtas prie jo) ---
cnt = np.zeros(len(edges), int)
ekey = {tuple(sorted(e)): i for i, e in enumerate(edges)}
for t in tri:
    for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
        cnt[ekey[tuple(sorted((a, b)))]] += 1
bnd = edges[cnt == 1]
bverts = np.unique(bnd)
neck = bverts[B[bverts, 2] > 1.30]
c0 = B[neck, :2].mean(0)
ang = np.arctan2(B[neck, 1] - c0[1], B[neck, 0] - c0[0])
neck = neck[np.argsort(ang)]
NN = len(neck)

UP = np.array([0, 0, 1.0])
HOOD_R = 14            # eilių skaičius nuo kaklo iki kapišono krašto
PU, PV = 14, 9         # kišenės tinklelis
STR_N, STR_SIDES = 9, 6

def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)

def build(P):
    """Visų detalių taškai vienai kūno formai (P – tos formos džemperio taškai)."""
    bvh = BVHTree.FromPolygons([Vector(p) for p in P], tri.tolist())
    c = P[neck, :2].mean(0)
    cz = P[:, :2].mean(0)
    def onto(g, nrm, fallback, off):
        """Taškas g „nuleidžiamas“ ant džemperio paviršiaus iš išorės (kryptis nrm) ir atitraukiamas per off."""
        loc, *_ = bvh.ray_cast(Vector(g + nrm * 0.15), Vector(-nrm), 0.3)
        return fallback if loc is None else np.array(loc) + nrm * off
    pts = []
    # Kapišonas: iš kaklo pakyla apykakle ir guli ant nugaros ir pečių
    rows = []
    for i in neck:
        p = P[i]
        u = unit(np.array([p[0] - c[0], p[1] - c[1], 0]))
        side = abs(u[0])                                          # 1 – šonas (ant peties), 0 – priekis/nugara
        wb = float(np.clip(0.45 + 0.55 * u[1], 0, 1)) ** 1.4     # 1 – nugara, 0 – priekis
        L = 0.05 + 0.26 * wb
        nrm = unit(u * (1 - 0.7 * side) + UP * 0.9 * side)       # kur „išorė“: nugarai – atgal, pečiui – aukštyn
        row = []
        prev = p
        for j in range(HOOD_R + 1):
            t = j / HOOD_R
            if t <= 0.2:
                q = p + UP * (0.045 * t / 0.2) + u * (0.004 + 0.03 * t / 0.2)
            else:
                s = (t - 0.2) / 0.8
                g = p + UP * (0.045 - s * L * (1 - 0.6 * side)) + u * (0.03 + s * L * side * 0.8)
                g[0] = c[0] + (g[0] - c[0]) * (1 + 0.6 * wb * math.sin(math.pi / 2 * s))   # nugaroje platėja
                puff = 0.012 + 0.03 * wb * math.sin(math.pi * min(1.0, s * 1.15)) ** 0.8
                q = onto(g, nrm, prev - UP * 0.015, puff)
            row.append(q); prev = q
        rows.append(row)
    pts += [q for row in rows for q in row]
    # Kengūros kišenė priekyje
    hem = P[:, 2].min()
    for jv in range(PV):
        v = jv / (PV - 1)
        zz = hem + 0.065 + v * 0.18
        hw = 0.15 - 0.05 * v
        for ju in range(PU):
            x = cz[0] + (ju / (PU - 1) - 0.5) * 2 * hw
            o = Vector((x, cz[1] - 0.6, zz)); loc, *_ = bvh.ray_cast(o, Vector((0, 1, 0)), 1.5)
            q = np.array(loc) + np.array([0, -0.005, 0]) if loc is not None else np.array([x, cz[1] - 0.12, zz])
            pts.append(q)
    # Raišteliai: du ploni vamzdeliai žemyn nuo kaklo priekio
    front = neck[P[neck, 1] < c[1]]
    for sx in (-1, 1):
        i0 = front[np.argmin(np.abs(P[front, 0] - (c[0] + sx * 0.045)))]
        top = P[i0]
        for k in range(STR_N):
            zz = top[2] - 0.035 - k * 0.02
            x = top[0] + sx * 0.004 * k
            o = Vector((x, c[1] - 0.6, zz)); loc, *_ = bvh.ray_cast(o, Vector((0, 1, 0)), 1.5)
            centre = np.array(loc) + np.array([0, -0.017, 0]) if loc is not None else np.array([x, top[1] - 0.03, zz])
            for a in range(STR_SIDES):
                th = 2 * math.pi * a / STR_SIDES
                pts.append(centre + np.array([math.cos(th) * 0.0035, math.sin(th) * 0.0035, 0]))
    return np.array(pts, np.float32)

# --- Topologija (sukuriama vieną kartą pagal bazinę formą) ---
basis_pts = build(B)
bm = bmesh.new(); bm.from_mesh(me)
new = [bm.verts.new(Vector(p)) for p in basis_pts]
bm.verts.ensure_lookup_table()
off = 0
def V(i): return new[i]
faces = []
W = HOOD_R + 1
for a in range(NN):                                         # kapišonas – žiedas aplink kaklą
    b = (a + 1) % NN
    for j in range(HOOD_R):
        faces.append((V(a * W + j), V(b * W + j), V(b * W + j + 1), V(a * W + j + 1)))
o = NN * W
for jv in range(PV - 1):                                    # kišenė
    for ju in range(PU - 1):
        i = o + jv * PU + ju
        faces.append((V(i), V(i + 1), V(i + PU + 1), V(i + PU)))
o += PU * PV
for s in range(2):                                          # raišteliai
    base = o + s * STR_N * STR_SIDES
    for k in range(STR_N - 1):
        for a in range(STR_SIDES):
            b = (a + 1) % STR_SIDES
            i0, i1 = base + k * STR_SIDES + a, base + k * STR_SIDES + b
            faces.append((V(i0), V(i1), V(i1 + STR_SIDES), V(i0 + STR_SIDES)))
    faces.append(tuple(V(base + (STR_N - 1) * STR_SIDES + a) for a in range(STR_SIDES)))   # galas
for f in faces:
    bm.faces.new(f)
# kapišono pradžią „prisiuvam“ prie kaklo (tie patys taškai)
weld = {new[a * W]: bm.verts[int(neck[a])] for a in range(NN)}
bmesh.ops.weld_verts(bm, targetmap=weld)
bm.to_mesh(me); bm.free()
me.update()

# --- Kiekvienai kūno formai – detalių padėtis ---
n_all = len(me.vertices)
new_idx_start = nv                   # nauji taškai po senųjų (be suvirintų pirmų eilės taškų)
keep = np.ones(len(basis_pts), bool)
keep[[a * W for a in range(NN)]] = False
for k in kb:
    pts = build(K[k.name])[keep]
    full = np.empty(n_all * 3, np.float32); k.data.foreach_get("co", full); full = full.reshape(-1, 3)
    full[nv:] = pts
    k.data.foreach_set("co", full.ravel())
    if k.name == "Basis":
        me.vertices.foreach_set("co", full.ravel())
me.update()
result = {"neck_verts": NN, "added_verts": int(n_all - nv), "faces": len(me.polygons)}
