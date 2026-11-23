"""NovaFit džemperis Blenderyje, 2 žingsnis: forma (laisvumas, juostos, raukšlės) kiekvienai kūno formai.

Pradeda nuo nepakeistos kopijos „hoodie_src“ (1 žingsnio rezultatas), todėl galima leisti daug kartų.
Kiekviena iš 14 formų (Basis + 13) apskaičiuojama atskirai, todėl džemperis keičiasi kartu su figūra.
"""
import bpy, numpy as np, math

NAME = globals().get("NAME", "hoodie")   # tas pats ir megztiniui (NAME="sweater")
src = bpy.data.objects.get(NAME + "_src")
cur = bpy.data.objects.get(NAME)
if src is None:                       # pirmas kartas – išsaugom 1 žingsnio rezultatą
    src = cur.copy(); src.data = cur.data.copy(); src.name = NAME + "_src"
    bpy.context.collection.objects.link(src)
src.hide_set(True); src.hide_render = True
if cur is not None:
    bpy.data.objects.remove(cur, do_unlink=True)
hd = src.copy(); hd.data = src.data.copy(); hd.name = NAME; hd.data.name = NAME
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

thick = np.where(part == 1, 0.02, 0.016).astype(np.float32)
thick = thick * (1 - RIB) + 0.012 * RIB
thick[CUFF] = 0.009
w = (apos[FOLD] - 0.76) / 0.165
thick[FOLD] += 0.006 * (0.5 + 0.5 * np.sin(w * math.pi * 5)) * np.sin(w * math.pi)   # susikaupusios raukšlės

NB = 64
ARMPIT_Z = 1.27
X = B[:, 0]; _cx = float(np.median(X[torso]))
# Krūtinės priekis (iki raktikaulių) irgi lygus – speneliai patenka į apvalkalą, ne į pereinamąją zoną
FRONT = smoothstep(ARMPIT_Z + 0.15, ARMPIT_Z + 0.09, z) * smoothstep(0.16, 0.11, np.abs(X - _cx))
SHELL = torso * np.maximum(smoothstep(ARMPIT_Z + 0.06, ARMPIT_Z - 0.02, z), FRONT)
arm = part == 1

DBG = {}
FRONTCHEST = torso & (B[:, 1] < -0.05) & (z > 1.05) & (z < 1.30) & (np.abs(B[:, 0]) < 0.15)
def bump(Q, tag):
    nn = vnormals(Q)
    lap = ((Q - neighbor_avg(Q)) * nn).sum(1)[FRONTCHEST]
    i = int(np.argmax(lap)); vi = np.where(FRONTCHEST)[0][i]
    DBG.setdefault(tag, [round(float(lap.max() * 1000), 2), round(float(B[vi, 0]), 3), round(float(B[vi, 2]), 3), int(vi)])
def shape(P):
    n = vnormals(P)
    bump(P, "0_body")
    # 1) Rankovės ir pečiai: išlygintas paviršius, atitrauktas nuo odos (smulkmenų nebelieka)
    S = P.copy()
    for _ in range(40):
        S += 0.5 * (neighbor_avg(S) - S)
    lift = np.maximum(((S - P) * n).sum(1), 0)
    Q = P + n * (thick + lift)[:, None]
    # 2) Liemuo: kiekviename aukštyje – lygi apimtis (apvalkalas), o ne kūno forma.
    # Apvalkalas skaičiuojamas pagal stipriai išlygintą kūną – speneliai ir kitos smulkmenos nebeturi įtakos
    Sm = P.copy()
    for it in range(160):                                # Taubin: smulkmenos dingsta, tūris lieka
        Sm += (0.5 if it % 2 == 0 else -0.53) * (neighbor_avg(Sm) - Sm)
    c = P[CHEST, :2].mean(0)
    d = P[:, :2] - c
    # Stiprus išlyginimas (visos smulkmenos dingsta), o sumažėjusi apimtis grąžinama kiekviename aukštyje
    Sb = P.copy()
    for _ in range(80):
        Sb += 0.5 * (neighbor_avg(Sb) - Sb)
    rP = np.linalg.norm(P[:, :2] - c, axis=1)
    rb = np.linalg.norm(Sb[:, :2] - c, axis=1)
    zbb = np.round((z - hem_z) / 0.02).astype(int)
    comp = np.zeros(zbb.max() + 1)
    for k in range(len(comp)):
        mk = torso & (zbb == k)
        if mk.sum() > 5: comp[k] = np.median(rP[mk] - rb[mk])
    r = rb + comp[zbb] + 1e-9
    th = np.arctan2(d[:, 1], d[:, 0])
    bi = ((th + math.pi) / (2 * math.pi) * NB).astype(int) % NB
    zb = ((z - hem_z) / 0.015).astype(int)
    NZ = zb[torso].max() + 2
    E = np.zeros((NZ, NB))
    rr = np.linalg.norm(d, axis=1)                       # tikras kūno spindulys (su speneliais)
    Ereq = np.zeros((NZ, NB))
    for k in range(NZ):                       # 80-asis procentilis: maži iškilimai (speneliai) neįskaičiuojami
        mk = torso & (zb == k) & (np.abs(P[:, 0] - c[0]) < 0.19)
        for jb in range(NB):
            sel = mk & ((bi == jb) | (bi == (jb + 1) % NB) | (bi == (jb - 1) % NB))
            if sel.sum() >= 2:
                E[k, jb] = np.percentile(r[sel], 95)
                Ereq[k, jb] = rr[sel].max() + 0.004
    for _ in range(NB):                       # tuščios kryptys – iš kaimynų
        e0 = E == 0
        if not e0.any(): break
        nb = np.maximum(np.roll(E, 1, 1), np.roll(E, -1, 1))
        E[e0] = nb[e0]
    # Kur kūnas kyšo (speneliai) – audinys ištempiamas plačiu, švelniu „tentu“, ne smulkiu guzu
    D = Ereq.copy()
    for _ in range(2):
        D = np.maximum.reduce([D, np.roll(D, 1, 1), np.roll(D, -1, 1),
                               np.vstack([D[1:], D[-1:]]), np.vstack([D[:1], D[:-1]])])
    if globals().get("TENT", 0): E = np.maximum(E, 0.5 * (E + D))
    for _ in range(4):                        # lyginam aplink ir aukštyn
        E = (np.roll(E, 1, 1) + 2 * E + np.roll(E, -1, 1)) / 4
        E[1:-1] = (E[:-2] + 2 * E[1:-1] + E[2:]) / 4
    # Žemiau krūtinės krenta tiesiai: ne siauriau už plačiausią vietą virš jos (krūtinė, pilvas, sėdmenys)
    kc = int((1.25 - hem_z) / 0.015)                     # nuo krūtinės viršūnės – be „lentynos“ po krūtine
    top = E[min(kc, NZ - 1)].copy()
    for k in range(min(kc, NZ - 1), -1, -1):
        top = np.maximum(top, E[k])
        E[k] = np.maximum(E[k], 0.98 * top)
    # Apačios juosta šiek tiek suima
    for k in range(min(5, NZ)):
        E[k] *= 0.94 + 0.012 * k
    fz = np.clip((z - hem_z) / 0.015, 0, NZ - 1.001)
    fb = (th + math.pi) / (2 * math.pi) * NB - 0.5
    k0 = np.floor(fz).astype(int); tz = fz - k0
    b0 = np.floor(fb).astype(int); tb = fb - b0
    b0m, b1m = b0 % NB, (b0 + 1) % NB
    env = ((E[k0, b0m] * (1 - tb) + E[k0, b1m] * tb) * (1 - tz) +
           (E[np.minimum(k0 + 1, NZ - 1), b0m] * (1 - tb) + E[np.minimum(k0 + 1, NZ - 1), b1m] * tb) * tz)
    shell_r = env + thick
    # Apvalkalo taškų kryptis ir aukštis – iš išlyginto kūno (ne iš odos normalių: ties speneliu jos „išsiskleidžia“)
    sd = Sb[:, :2] - c
    sr = np.linalg.norm(sd, axis=1) + 1e-9
    shellQ = np.column_stack([c + sd / sr[:, None] * np.maximum(shell_r, r + 0.006)[:, None], P[:, 2]])
    Q = Q * (1 - SHELL[:, None]) + shellQ * SHELL[:, None]
    # Audinys užsitempia virš tikro kūno lygiu kupolu: spindulys ≥ odos + 5 mm, bet išlygintas (be smaigalių)
    qd = Q[:, :2] - c
    qs = np.linalg.norm(qd, axis=1) + 1e-9
    req = rP + 0.005
    sr_ = qs.copy()
    ws = SHELL
    for _ in range(60):
        sr_ = np.maximum(sr_, req * ws + sr_ * (1 - ws))
        sr_ = sr_ + ws * 0.5 * (neighbor_avg(sr_[:, None])[:, 0] - sr_)
    sr_ = np.maximum(sr_, (req - 0.002) * ws + sr_ * (1 - ws))
    Q[:, :2] = c + qd / qs[:, None] * sr_[:, None]
    bump(Sm, "1_Sm"); bump(Q, "2_shell")
    # 3) Audinio paviršių išlyginam pačiu (tankus tinklas ties speneliais nebeduoda „žiedų“)
    w = SHELL[:, None]
    for it in range(60):
        Q += w * (0.5 if it % 2 == 0 else -0.52) * (neighbor_avg(Q) - Q)
    for _ in range(25):                                  # papildomai – paprastas lyginimas (nuapvalina krūtinės viršūnes)
        Q += w * 0.5 * (neighbor_avg(Q) - Q)
    bump(Q, "3_smooth")
    # 4) Niekur arčiau nei 6 mm nuo odos (liemenyje – nuo išlyginto kūno, kad nekiltų smulkūs guzai)
    ref = P * (1 - w) + Sm * w
    nr = vnormals(ref)
    dn = ((Q - ref) * nr).sum(1)
    dreal = ((Q - P) * n).sum(1)                         # tikra oda (speneliai) – bent 4 mm, kad nekyšotų
    # liemenyje (apvalkale) nuo tikros odos užtenka 1,5 mm – kitaip ant krūtinės vėl atsiranda guzeliai
    need_real = 0.004 * (1 - SHELL)
    # ties klubais – bent 2,4 cm nuo kūno, kad pro apačią nekyšotų džinsų juosta, kilpos ir saga
    hipb = smoothstep(1.00, 0.96, z) * torso
    chest_shell = SHELL * (1 - hipb)                      # krūtinėje atstumą jau užtikrino kupolas
    need_sm = (0.006 + 0.018 * hipb) * (1 - chest_shell)
    push = np.maximum(np.maximum(need_sm - dn, need_real - dreal), 0)
    for _ in range(10):                                  # stūmimą paskleidžiam plačiai ir švelniai
        push = np.maximum(push, 0.85 * neighbor_avg(push[:, None])[:, 0])
    for _ in range(4):
        push = np.maximum(push, neighbor_avg(push[:, None])[:, 0])
    Q += nr * push[:, None]
    bump(Q, "4_push")
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
result = {"dbg": DBG, "verts": nv, "hem_z": round(float(hem_z), 3), "chest_verts": int(CHEST.sum())}
