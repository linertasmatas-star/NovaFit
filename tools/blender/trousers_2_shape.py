"""NovaFit kostiuminės kelnės Blenderyje, 2 žingsnis: forma kiekvienai kūno formai.

Pagrindas – džinsų (jeans_src, ta pati kūno sritis). Skirtumai nuo džinsų: laisvesnės, apvalesnės klešnės,
šlaunis tiesiai siaurėja į kelį (be kūno formų), nuo kelio žemyn beveik tiesiai, be apačios atlenkimo.
Lygintas kantas pridedamas 3 žingsnyje. Galima leisti daug kartų.
"""
import bpy, numpy as np, math

src = bpy.data.objects["jeans_src"]
cur = bpy.data.objects.get("trousers")
src.hide_set(True); src.hide_render = True
if cur is not None:
    bpy.data.objects.remove(cur, do_unlink=True)
jn = src.copy(); jn.data = src.data.copy(); jn.name = "trousers"; jn.data.name = "trousers"
bpy.context.collection.objects.link(jn)
jn.hide_set(False); jn.hide_render = False
me = jn.data
kb = me.shape_keys.key_blocks
nv = len(me.vertices)

def attr(name, dtype):
    a = np.empty(nv, dtype); me.attributes[name].data.foreach_get("value", a); return a
part = attr("part", np.int32)          # 0 liemuo, 2 koja
apos = attr("apos", np.float32)        # kojoje: 0 klubas … 0,5 kelis … 1 kulkšnis

me.calc_loop_triangles()
tri = np.empty(len(me.loop_triangles) * 3, np.int32)
me.loop_triangles.foreach_get("vertices", tri); tri = tri.reshape(-1, 3)
edges = np.empty(len(me.edges) * 2, np.int32); me.edges.foreach_get("vertices", edges); edges = edges.reshape(-1, 2)
deg = np.bincount(edges.ravel(), minlength=nv).astype(np.float32); deg[deg == 0] = 1
# kraštų taškai (juosmuo, apačia) – jų nelyginam, kad kraštai liktų tiesūs
cnt = {}
for t in tri:
    for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
        k = (min(a, b), max(a, b)); cnt[k] = cnt.get(k, 0) + 1
rim = np.zeros(nv, bool)
for (a, b), c in cnt.items():
    if c == 1: rim[a] = rim[b] = True

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
def convex_hull(p):
    p = p[np.lexsort((p[:, 1], p[:, 0]))]
    def half(pts):
        h = []
        for q in pts:
            while len(h) >= 2 and np.cross(h[-1] - h[-2], q - h[-2]) <= 0: h.pop()
            h.append(q)
        return h
    lo, up = half(p), half(p[::-1])
    return np.array(lo[:-1] + up[:-1])
def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)

B = getco(kb["Basis"])
z0 = B[:, 2]
TOP = z0.max()
HEM = z0.min()
CROTCH = z0[np.abs(B[:, 0]) < 0.012].min()                     # tarpkojis (viduryje žemiausias taškas)
leg = part == 2
KNEE_Z = float(np.median(z0[leg & (np.abs(apos - 0.5) < 0.03)]))
TUBE = smoothstep(CROTCH + 0.01, CROTCH - 0.05, z0)            # 1 – atskiros kojos, 0 – klubai
BAND = smoothstep(TOP - 0.045, TOP - 0.035, z0)                 # juosmens juosta
thick = 0.010 + 0.0035 * BAND                                   # be atlenkimo apačioje
NB, DZ = 32, 0.015

def shape(P):
    n = vnormals(P)
    Sm = P.copy()                                   # išlygintas kūnas (be smulkmenų, tūris lieka)
    for it in range(160):
        Sm += (0.5 if it % 2 == 0 else -0.53) * (neighbor_avg(Sm) - Sm)
    lift = np.maximum(((Sm - P) * n).sum(1), 0)
    Q = P + n * (thick + lift)[:, None]
    z = P[:, 2]
    # --- Kojos: kiekvienai kojai – lygus vamzdis aplink jos ašį ---
    for s in (-1, 1):
        L = (B[:, 0] * s > 0) & (z0 < CROTCH + 0.03)
        if not L.any(): continue
        zb = ((z0 - HEM) / DZ).astype(int)                 # juostos pagal bazinę figūrą (stabilu visoms formoms)
        NZ = zb[L].max() + 2
        C = np.zeros((NZ, 2))
        for k in range(NZ):
            m = L & (zb == k)
            C[k] = np.median(Sm[m, :2], 0) if m.sum() >= 3 else np.nan
        ok = ~np.isnan(C[:, 0])
        idx = np.arange(NZ)
        for j in range(2): C[:, j] = np.interp(idx, idx[ok], C[ok, j])
        for _ in range(6): C[1:-1] = (C[:-2] + 2 * C[1:-1] + C[2:]) / 4
        kk = int((KNEE_Z - HEM) / DZ)
        kk = min(max(kk, 1), NZ - 1)
        # nuo kelio žemyn ašis tiesi (linija nuo kelio iki kulkšnies)
        for k in range(kk):
            C[k] = C[0] + (C[kk] - C[0]) * k / kk
        cz = C[np.clip(zb, 0, NZ - 1)]
        d = Sm[:, :2] - cz
        r = np.linalg.norm(d, axis=1) + 1e-9
        th = np.arctan2(d[:, 1], d[:, 0])
        bi = ((th + math.pi) / (2 * math.pi) * NB).astype(int) % NB
        E = np.zeros((NZ, NB))
        for k in range(NZ):
            mk = L & (zb == k)
            for jb in range(NB):
                sel = mk & ((bi == jb) | (bi == (jb + 1) % NB) | (bi == (jb - 1) % NB))
                if sel.sum() >= 2: E[k, jb] = np.percentile(r[sel], 95)
        for _ in range(NB):
            e0 = E == 0
            if not e0.any(): break
            nb = np.maximum(np.roll(E, 1, 1), np.roll(E, -1, 1))
            E[e0] = nb[e0]
        # vamzdis apvalesnis: kiekviename aukštyje artinam prie vidutinio spindulio
        E = 0.3 * E + 0.7 * E.mean(1, keepdims=True)
        for _ in range(3):
            E = (np.roll(E, 1, 1) + 2 * E + np.roll(E, -1, 1)) / 4
            E[1:-1] = (E[:-2] + 2 * E[1:-1] + E[2:]) / 4
        # truputį laisviau nei džinsai
        E = E + 0.003
        # nuo kelio žemyn – tiesiai, apačioje švelniai siaurėja (iki 92 % kelio), bet ne arčiau nei 1 cm nuo kojos
        knee = E[kk].copy()
        for k in range(kk, -1, -1):
            E[k] = np.maximum(E[k] + 0.01, knee * (1.0 - 0.08 * (kk - k) / kk))
        # šlaunis: tiesi linija nuo tarpkojo iki kelio (audinys krenta, nesek kūno formos)
        kc = int((CROTCH - HEM) / DZ); kc = min(max(kc, kk + 1), NZ - 1)
        for k in range(kk + 1, kc + 1):
            t = (k - kk) / (kc - kk)
            E[k] = np.maximum(E[k], E[kk] * (1 - t) + E[kc] * t)
        for k in range(kk + 1, NZ):
            E[k] = np.maximum(E[k], E[k - 1])
        fz = np.clip((z0 - HEM) / DZ, 0, NZ - 1.001)
        fb = (th + math.pi) / (2 * math.pi) * NB - 0.5
        k0 = np.floor(fz).astype(int); tz = fz - k0
        b0 = np.floor(fb).astype(int); tb = fb - b0
        b0m, b1m = b0 % NB, (b0 + 1) % NB
        k1 = np.minimum(k0 + 1, NZ - 1)
        env = ((E[k0, b0m] * (1 - tb) + E[k0, b1m] * tb) * (1 - tz) + (E[k1, b0m] * (1 - tb) + E[k1, b1m] * tb) * tz)
        new_r = np.maximum(env + thick, r + 0.006)
        w = TUBE * L
        target = cz + d / r[:, None] * new_r[:, None]
        Q[:, :2] = Q[:, :2] * (1 - w[:, None]) + target * w[:, None]
    # --- Klubai: audinys įsitempia tiesiai per įdubas (sėdmenų tarpą, tarpkojį) – iškilas apvalkalas kiekviename aukštyje ---
    HIP = smoothstep(CROTCH - 0.005, CROTCH + 0.03, z0)
    hm = HIP > 0
    zb = np.round((z0 - CROTCH) / 0.01).astype(int)
    for k in np.unique(zb[hm]):
        m = hm & (zb == k)
        if m.sum() < 6: continue
        pts = Q[m, :2]
        c = pts.mean(0)
        hull = convex_hull(pts)
        dense = np.concatenate([hull[i] + (hull[(i + 1) % len(hull)] - hull[i]) * t[:, None]
                                for i in range(len(hull)) for t in [np.linspace(0, 1, 8, endpoint=False)]])
        da = np.arctan2(dense[:, 1] - c[1], dense[:, 0] - c[0]); dr = np.linalg.norm(dense - c, axis=1)
        o = np.argsort(da); da, dr = da[o], dr[o]
        va = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]); vr = np.linalg.norm(pts - c, axis=1) + 1e-9
        hr = np.interp(va, da, dr, period=2 * math.pi)
        nr_ = np.maximum(vr, hr)
        w = HIP[m][:, None]
        Q[m, :2] = pts + (pts - c) / vr[:, None] * ((nr_ - vr)[:, None] * w)
    # --- Audinio paviršių išlyginam (be kraštų) ---
    wq = (~rim).astype(np.float32)[:, None] * 0.9
    for it in range(60):
        Q += wq * (0.5 if it % 2 == 0 else -0.52) * (neighbor_avg(Q) - Q)
    # --- Niekur arčiau nei 4–6 mm nuo odos (stūmimas paskleidžiamas plačiai) ---
    nr = vnormals(Sm)
    dn = ((Q - Sm) * nr).sum(1)
    dreal = ((Q - P) * n).sum(1)
    push = np.maximum(np.maximum(0.006 - dn, 0.004 - dreal), 0)
    for _ in range(10):
        push = np.maximum(push, 0.85 * neighbor_avg(push[:, None])[:, 0])
    for _ in range(4):
        push = np.maximum(push, neighbor_avg(push[:, None])[:, 0])
    Q += nr * push[:, None]
    return Q

for k in kb:
    Q = shape(getco(k))
    setco(k, Q)
    if k.name == "Basis":
        me.vertices.foreach_set("co", Q.astype(np.float32).ravel())
me.update()
jn.color = (0.25, 0.26, 0.3, 1)
jn["crotch"] = float(CROTCH)                              # 3 žingsniui (užtrauktuko ilgis)
result = {"verts": nv, "top": round(float(TOP), 3), "crotch": round(float(CROTCH), 3),
          "knee": round(float(KNEE_Z), 3), "hem": round(float(HEM), 3)}
