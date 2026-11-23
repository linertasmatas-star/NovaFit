"""NovaFit džinsai Blenderyje, 3 žingsnis: užpakalinės kišenės, priekinių kišenių angos, monetų kišenėlė,
užtrauktuko juosta, kilpos diržui ir saga.

Kiekviena detalė – plona „lopinėlio“ forma, prisegta prie džinsų paviršiaus (spinduliu randama vieta bazinėje
formoje, tada ta pati vieta skaičiuojama kiekvienai iš 14 kūno formų). Leisti iškart po 2 žingsnio.
"""
import bpy, bmesh, numpy as np, math
from mathutils import Vector
from mathutils.bvhtree import BVHTree

jn = bpy.data.objects["jeans"]
me = jn.data
kb = me.shape_keys.key_blocks
nv = len(me.vertices)
me.calc_loop_triangles()
tri = np.empty(len(me.loop_triangles) * 3, np.int32)
me.loop_triangles.foreach_get("vertices", tri); tri = tri.reshape(-1, 3)

def getco(k):
    a = np.empty(nv * 3, np.float32); k.data.foreach_get("co", a); return a.reshape(-1, 3)
def vnormals(P):
    fn = np.cross(P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]])
    n = np.zeros_like(P)
    for j in range(3): np.add.at(n, tri[:, j], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)

K = {k.name: getco(k) for k in kb}
B = K["Basis"]
TOP = float(B[:, 2].max())
CROTCH = float(jn.get("crotch", TOP - 0.2))
hip = (B[:, 2] > TOP - 0.16) & (B[:, 2] < TOP - 0.05)
HW = float(np.abs(B[hip, 0]).max())                # pusė klubų pločio
CY = float(np.median(B[hip, 1]))                   # klubų centras (priekis–nugara)
bvh = BVHTree.FromPolygons([Vector(p) for p in B], tri.tolist())

A_tri, A_bar, A_off = [], [], []
faces = []

def bary(p, a, b, c):
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    d20, d21 = v2 @ v0, v2 @ v1
    den = d00 * d11 - d01 * d01 + 1e-12
    v = (d11 * d20 - d01 * d21) / den; w = (d00 * d21 - d01 * d20) / den
    return np.array([1 - v - w, v, w])

def anchor(origin, direction, off):
    loc, nrm, idx, dist = bvh.ray_cast(Vector(origin), Vector(direction))
    if loc is None:
        o = np.array(origin); loc, nrm, idx, dist = bvh.find_nearest(Vector((o[0] * 0.5, CY, o[2])))
    t = tri[idx]
    A_tri.append(t); A_bar.append(bary(np.array(loc), B[t[0]], B[t[1]], B[t[2]])); A_off.append(off)
    return len(A_tri) - 1

def panel(pts, side, off, wrap=False, edge=0.0008):
    """pts[v][u] = (a, b) – 2D taškai plokštumoje, statmenoje žiūrėjimo krypčiai.
    side: 'front' (a=x), 'back' (a=x), 'left'/'right' (a=y). Kraštas prie paviršiaus, vidus pakeltas."""
    NV, NU = len(pts), len(pts[0])
    ids = []
    for v in range(NV):
        row = []
        for u in range(NU):
            a, b = pts[v][u]
            if side == "front": o, d = (a, -1.0, b), (0, 1, 0)
            elif side == "back": o, d = (a, 1.0, b), (0, -1, 0)
            elif side == "left": o, d = (-1.0, a, b), (1, 0, 0)
            else: o, d = (1.0, a, b), (-1, 0, 0)
            border = v == 0 or v == NV - 1 or (not wrap and (u == 0 or u == NU - 1))
            row.append(anchor(o, d, edge if border else off))
        ids.append(row)
    for v in range(NV - 1):
        for u in range(NU if wrap else NU - 1):
            u1 = (u + 1) % NU
            faces.append((ids[v][u], ids[v][u1], ids[v + 1][u1], ids[v + 1][u]))

def strip(curve, side, off, half=0.0045):
    """Siūlės / krašto iškilimas palei kreivę (3 eilės: kraštas – iškilęs vidurys – kraštas)."""
    c = np.array(curve)
    tg = np.gradient(c, axis=0); tg /= np.linalg.norm(tg, axis=1, keepdims=True) + 1e-9
    nn = np.stack([-tg[:, 1], tg[:, 0]], 1)
    rows = [[tuple(c[i] + nn[i] * h) for i in range(len(c))] for h in (-half, 0.0, half)]
    # galai irgi turi būti prie paviršiaus: pridedam po tašką abiejuose galuose
    rows = [[tuple(c[0] - tg[0] * 0.002 + nn[0] * h)] + r + [tuple(c[-1] + tg[-1] * 0.002 + nn[-1] * h)]
            for r, h in zip(rows, (-half, 0.0, half))]
    panel(rows, side, off)

# --- Užpakalinės kišenės (penkiakampės, šiek tiek pasuktos į šonus) ---
for s in (-1, 1):
    cx, top, w, h = s * 0.42 * HW, TOP - 0.055, 0.36 * HW, 0.13
    NU, NV = 9, 9
    pts = []
    for j in range(NV):
        v = j / (NV - 1)
        row = []
        for i in range(NU):
            u = i / (NU - 1)
            x = (u - 0.5) * w * (1 - 0.12 * v)
            zz = -v * (h + 0.018 * (1 - abs(2 * u - 1)) * (v ** 2))
            zz -= s * x * 0.12                            # išorinis kraštas šiek tiek žemiau
            row.append((cx + x, top + zz))
        pts.append(row)
    panel(pts, "back", 0.0042)

# --- Priekinių kišenių angos (lenkta „J“ linija) ir monetų kišenėlė ---
for s in (-1, 1):
    curve = []
    for i in range(12):
        t = i / 11
        x = s * (0.46 * HW + 0.42 * HW * (1 - math.cos(t * math.pi / 2)))
        zz = TOP - 0.042 - 0.085 * math.sin(t * math.pi / 2)
        curve.append((x, zz))
    strip(curve, "front", 0.0035)
cx, top, w, h = 0.66 * HW, TOP - 0.048, 0.045, 0.038
panel([[(cx + (i / 4 - 0.5) * w, top - j / 4 * h) for i in range(5)] for j in range(5)], "front", 0.003)

# --- Užtrauktuko juosta (J forma) ir vidurio siūlė ---
fx, fz0, fz1 = -0.032, TOP - 0.042, CROTCH + 0.07
curve = [(fx, fz0 - (fz0 - fz1) * i / 7) for i in range(8)]
for i in range(1, 7):
    a = i / 6 * math.pi / 2
    curve.append((fx * math.cos(a), fz1 - 0.028 * math.sin(a)))
strip(curve, "front", 0.0032)

# --- Kilpos diržui (7) ---
LOOP_W, LOOP_H, LOOP_OFF = 0.012, 0.045, 0.006
def loop(a, side):
    panel([[(a + (i / 2 - 0.5) * LOOP_W, TOP - 0.003 - j / 3 * LOOP_H) for i in range(3)] for j in range(4)], side, LOOP_OFF)
for s in (-1, 1):
    loop(s * 0.52 * HW, "front")
    loop(s * 0.45 * HW, "back")
loop(CY, "left"); loop(CY, "right")
loop(0.0, "back")

# --- Saga juosmens priekyje ---
R, A = 0.0085, 12
rings = [[(fx * 0.2 + r * math.cos(2 * math.pi * a / A), TOP - 0.021 + r * math.sin(2 * math.pi * a / A)) for a in range(A)]
         for r in (0.0005, R * 0.55, R)]
panel(rings[::-1], "front", 0.005, wrap=True)

# --- Topologija ir padėtis kiekvienai formai ---
A_tri = np.array(A_tri); A_bar = np.array(A_bar, np.float32); A_off = np.array(A_off, np.float32)
def place(P):
    N = vnormals(P)
    pos = (P[A_tri] * A_bar[:, :, None]).sum(1)
    nr = (N[A_tri] * A_bar[:, :, None]).sum(1)
    nr /= np.linalg.norm(nr, axis=1, keepdims=True) + 1e-9
    return pos + nr * A_off[:, None]

bm = bmesh.new(); bm.from_mesh(me)
new = [bm.verts.new(Vector(p)) for p in place(B)]
for f in faces:
    try: bm.faces.new([new[i] for i in f]).smooth = True
    except ValueError: pass
bm.to_mesh(me); bm.free()
me.update()
n_all = len(me.vertices)
for k in kb:
    full = np.empty(n_all * 3, np.float32); k.data.foreach_get("co", full); full = full.reshape(-1, 3)
    full[nv:] = place(K[k.name])
    k.data.foreach_set("co", full.ravel())
    if k.name == "Basis":
        me.vertices.foreach_set("co", full.ravel())
me.update()
result = {"added_verts": int(n_all - nv), "faces": len(me.polygons), "HW": round(HW, 3), "top": round(TOP, 3), "crotch": round(CROTCH, 3)}
