"""NovaFit džemperis Blenderyje, 2 žingsnis: forma (laisvumas, juostos, raukšlės) kiekvienai kūno formai.

Pradeda nuo nepakeistos kopijos „hoodie_src“ (1 žingsnio rezultatas), todėl galima leisti daug kartų.
Kiekviena iš 14 formų (Basis + 13) apskaičiuojama atskirai, todėl džemperis keičiasi kartu su figūra.
"""
import bpy, numpy as np, math

src = bpy.data.objects.get("hoodie_src")
cur = bpy.data.objects.get("hoodie")
if src is None:                       # pirmas kartas – išsaugom 1 žingsnio rezultatą
    src = cur.copy(); src.data = cur.data.copy(); src.name = "hoodie_src"
    bpy.context.collection.objects.link(src)
src.hide_set(True); src.hide_render = True
if cur is not None:
    bpy.data.objects.remove(cur, do_unlink=True)
hd = src.copy(); hd.data = src.data.copy(); hd.name = "hoodie"; hd.data.name = "hoodie"
bpy.context.collection.objects.link(hd)
hd.hide_set(False)
me = hd.data
kb = me.shape_keys.key_blocks
nv = len(me.vertices)

def attr(name, dtype):
    a = np.empty(nv, dtype); me.attributes[name].data.foreach_get("value", a); return a
part = attr("part", np.int32)          # 0 liemuo, 1 ranka
apos = attr("apos", np.float32)        # padėtis rankoje: 0 petys … 1 riešas

me.calc_loop_triangles()
tri = np.empty(len(me.loop_triangles) * 3, np.int32)
me.loop_triangles.foreach_get("vertices", tri); tri = tri.reshape(-1, 3)
edges = np.empty(len(me.edges) * 2, np.int32); me.edges.foreach_get("vertices", edges); edges = edges.reshape(-1, 2)
deg = np.bincount(edges.ravel(), minlength=nv).astype(np.float32); deg[deg == 0] = 1

def getco(k):
    a = np.empty(nv * 3, np.float32); k.data.foreach_get("co", a); return a.reshape(-1, 3)
def setco(k, P):
    k.data.foreach_set("co", P.astype(np.float32).ravel())
def vnormals(P):
    fn = np.cross(P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]])
    n = np.zeros_like(P)
    for j in range(3): np.add.at(n, tri[:, j], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
def neighbor_avg(P):
    acc = np.zeros_like(P)
    np.add.at(acc, edges[:, 0], P[edges[:, 1]]); np.add.at(acc, edges[:, 1], P[edges[:, 0]])
    return acc / deg[:, None]
def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)

B = getco(kb["Basis"])                 # bazinė figūra – pagal ją nustatomos sritys
z = B[:, 2]
torso = part == 0
hem_z = z[torso].min()
CHEST = torso & (z > 1.17) & (z < 1.26) & (np.abs(B[:, 0]) < 0.13)   # be pažastų – kitaip apačia per plati
DRAPE = smoothstep(1.17, 1.10, z) * torso           # žemiau krūtinės – krenta laisvai
HANG = torso & (z < 1.26) & (np.abs(B[:, 0]) < 0.16)  # iš ko skaičiuojamas „kabantis“ kontūras (krūtinė, pilvas, sėdmenys)
RIB = smoothstep(hem_z + 0.065, hem_z + 0.015, z) * torso   # apačios juosta
BUST = torso & (z > 1.08) & (z < 1.36) & (B[:, 1] < 0.0)            # priekis (Blender: -Y)
CUFF = (part == 1) & (apos > 0.925)
FOLD = (part == 1) & (apos > 0.76) & (apos <= 0.925)

thick = np.where(part == 1, 0.014, 0.012).astype(np.float32)
thick = thick * (1 - RIB) + 0.006 * RIB
thick[CUFF] = 0.006
w = (apos[FOLD] - 0.76) / 0.165
thick[FOLD] += 0.006 * (0.5 + 0.5 * np.sin(w * math.pi * 5)) * np.sin(w * math.pi)   # susikaupusios raukšlės

NB = 48
def shape(P):
    n = vnormals(P)
    S = P.copy()
    for _ in range(40):                                   # audinys neseka smulkmenų (krūtinė, mentės)
        S[torso] += 0.5 * (neighbor_avg(S)[torso] - S[torso])
    for _ in range(160):                                  # krūtinės sritis – dar lygiau (audinys tempiasi per ją)
        S[BUST] += 0.5 * (neighbor_avg(S)[BUST] - S[BUST])
    # į išorę – kiek reikia; į vidų – tik iki 3 mm nuo odos (kad nelįstų į kūną)
    lift = np.maximum(((S - P) * n).sum(1), -(thick - 0.003) * torso)
    Q = P + n * (thick + lift)[:, None]
    # Laisvas kritimas nuo krūtinės: kiekviena kryptimi spindulys ne mažesnis nei krūtinės
    c = Q[CHEST, :2].mean(0)
    d = Q[:, :2] - c
    r = np.linalg.norm(d, axis=1) + 1e-9
    b = ((np.arctan2(d[:, 1], d[:, 0]) + math.pi) / (2 * math.pi) * NB).astype(int) % NB
    R = np.zeros(NB)
    np.maximum.at(R, b[HANG], r[HANG])                    # audinys kabo tiesiai: ne siauriau už plačiausią vietą
    for _ in range(3):                                    # užpildom tuščias kryptis
        R = np.maximum(R, np.maximum(np.roll(R, 1), np.roll(R, -1)) * (R == 0))
    R = np.maximum(R, (np.roll(R, 1) + np.roll(R, -1)) / 2 * 0.98)
    target = np.maximum(r, 0.97 * R[b])
    rn = r + (target - r) * DRAPE
    rn = rn * (1 - 0.6 * RIB) + r * 0.6 * RIB             # apačioje juosta šiek tiek suima
    Q[:, :2] = c + d / r[:, None] * rn[:, None]
    # Krūtinė: iš šono žiūrint audinys tempiasi tiesiai (viršutinis gaubtas), o ne seka formą
    for x0 in np.arange(-0.16, 0.16, 0.012):
        m = BUST & (np.abs(Q[:, 0] - x0) < 0.009)
        if m.sum() < 4: continue
        zz, dd = Q[m, 2], -Q[m, 1]                         # gylis į priekį (Blender: priekis = -Y)
        o = np.argsort(zz); zz, dd = zz[o], dd[o]
        hull = []
        for pt in zip(zz, dd):
            while len(hull) >= 2 and ((hull[-1][0] - hull[-2][0]) * (pt[1] - hull[-2][1]) - (hull[-1][1] - hull[-2][1]) * (pt[0] - hull[-2][0])) >= 0:
                hull.pop()
            hull.append(pt)
        hz, hdp = np.array(hull).T
        env = np.interp(Q[m, 2], hz, hdp)
        idx = np.where(m)[0]
        Q[idx, 1] = -np.maximum(-Q[idx, 1], env)
    for _ in range(6):
        Q[BUST] += 0.5 * (neighbor_avg(Q)[BUST] - Q[BUST])
    return Q

out = {}
for k in kb:
    Q = shape(getco(k))
    setco(k, Q)
    if k.name == "Basis":
        me.vertices.foreach_set("co", Q.astype(np.float32).ravel())
    out[k.name] = round(float(np.abs(Q - getco(k)).max()), 4)
me.update()
hd.color = (0.36, 0.45, 0.62, 1)
result = {"verts": nv, "hem_z": round(float(hem_z), 3), "chest_verts": int(CHEST.sum())}
