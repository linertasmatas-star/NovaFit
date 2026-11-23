"""NovaFit kostiuminės kelnės Blenderyje, 3 žingsnis: lygintas kantas, įstrižos priekinės kišenės,
užpakalinės kišenės su siaura anga, užtrauktuko juosta, kilpos diržui ir maža saga (be kniedžių ir lopų).

Kiekviena detalė – plona „lopinėlio“ forma, prisegta prie džinsų paviršiaus (spinduliu randama vieta bazinėje
formoje, tada ta pati vieta skaičiuojama kiekvienai iš 14 kūno formų). Leisti iškart po trousers_2_shape.py (tas pats būdas kaip džinsų).
"""
import bpy, bmesh, numpy as np, math
from mathutils import Vector
from mathutils.bvhtree import BVHTree

jn = bpy.data.objects["trousers"]
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

HEM = float(B[:, 2].min())

# --- Lygintas kantas: tiesi briauna per kiekvienos klešnės priekio vidurį ---
# Klešnės kraštai randami spinduliais iš šono (išorinis) ir iš tarpo tarp kojų (vidinis).
# Žemiau tarpkojo kantas eina klešnės viduriu (tiesė), aukščiau – tiesiai į viršų iki juosmens.
def leg_center(s, z):
    yc = float(np.median(B[(np.abs(B[:, 2] - z) < 0.02) & (B[:, 0] * s > 0.02), 1]))
    out_hit = bvh.ray_cast(Vector((s * 1.0, yc, z)), Vector((-s, 0, 0)))[0]
    in_hit = bvh.ray_cast(Vector((0.0, yc, z)), Vector((s, 0, 0)))[0]
    if out_hit is None or in_hit is None: return None
    return 0.5 * (out_hit.x + in_hit.x)
for s in (-1, 1):
    zs, cx = [], []
    for z in np.linspace(HEM + 0.03, CROTCH - 0.05, 20):
        c = leg_center(s, z)
        if c is not None: zs.append(z); cx.append(c)
    a, b = np.polyfit(zs, cx, 1)
    z_knee_top = CROTCH - 0.03
    z_top, z_bot = TOP - 0.085, HEM + 0.012
    x_up = a * z_knee_top + b
    curve = [(x_up, z) for z in np.linspace(z_top, z_knee_top, 8, endpoint=False)]
    curve += [(a * z + b, z) for z in np.linspace(z_knee_top, z_bot, 34)]
    strip(curve, "front", 0.0016, half=0.0018)

# --- Įstrižos priekinės kišenės (tiesi linija nuo juosmens į šoną) ---
for s in (-1, 1):
    x0, z0_, x1, z1 = s * 0.50 * HW, TOP - 0.04, s * 0.93 * HW, TOP - 0.16
    curve = [(x0 + (x1 - x0) * t, z0_ + (z1 - z0_) * t) for t in np.linspace(0, 1, 10)]
    strip(curve, "front", 0.0028, half=0.0035)

# --- Užpakalinės kišenės: tik siaura horizontali anga ---
for s in (-1, 1):
    cx, zz, w = s * 0.45 * HW, TOP - 0.075, 0.30 * HW
    curve = [(cx + (t - 0.5) * w, zz) for t in np.linspace(0, 1, 8)]
    strip(curve, "back", 0.003, half=0.004)

# --- Užtrauktuko juosta (J forma) ---
fx, fz0, fz1 = -0.028, TOP - 0.042, CROTCH + 0.065
curve = [(fx, fz0 - (fz0 - fz1) * i / 7) for i in range(8)]
for i in range(1, 7):
    ang = i / 6 * math.pi / 2
    curve.append((fx * math.cos(ang), fz1 - 0.025 * math.sin(ang)))
strip(curve, "front", 0.0026, half=0.0035)

# --- Kilpos diržui (siauresnės, 7) ---
LOOP_W, LOOP_H, LOOP_OFF = 0.010, 0.040, 0.005
def loop(a, side):
    panel([[(a + (i / 2 - 0.5) * LOOP_W, TOP - 0.003 - j / 3 * LOOP_H) for i in range(3)] for j in range(4)], side, LOOP_OFF)
for s in (-1, 1):
    loop(s * 0.55 * HW, "front")
    loop(s * 0.45 * HW, "back")
loop(CY, "left"); loop(CY, "right")
loop(0.0, "back")

# --- Maža plokščia saga juosmens priekyje ---
R, A = 0.0065, 12
rings = [[(fx * 0.2 + r * math.cos(2 * math.pi * a / A), TOP - 0.021 + r * math.sin(2 * math.pi * a / A)) for a in range(A)]
         for r in (0.0005, R * 0.6, R)]
panel(rings[::-1], "front", 0.0035, wrap=True)

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
