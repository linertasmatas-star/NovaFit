"""NovaFit: 3D avataras naršyklei – vienas .glb failas su „formų slankikliais“ (morph targets).

Paleidimas:  python3 tools/mh_export_glb.py <mpfb2 data aplankas> assets/avatar3d

Reikia MPFB2 duomenų (github.com/makehumancommunity/mpfb2, aplankas src/mpfb/data):
  3dobjs/base.obj, targets/macrodetails, targets/torso, targets/arms

Kas sukuriama (assets/avatar3d/):
  body.glb   – bazinė figūra (vidutinė moteris) + formos:
               male                  – iš moters į vyrą
               f-thin, f-heavy       – plonesnė / stambesnė moteris
               m-thin, m-heavy       – plonesnis / stambesnis vyras (naudojamos kartu su „male“)
               waist-, waist+        – juosmens apimtis (MakeHuman „measure“)
               hips-, hips+          – klubų apimtis
               upperarm-, upperarm+  – žasto ilgis
               lowerarm-, lowerarm+  – dilbio ilgis
             Kūnas turi dvi dalis: oda ir juodi apatiniai (tie patys taškai, skirtinga medžiaga).
  body.json  – papildoma informacija naršyklei (natūralus ūgis, matmenys centimetrais kiekvienai formai).

Kodėl „vidutinė moteris + skirtumai“: MakeHuman figūra = bazė + formų suma, todėl naršyklėje
tą pačią figūrą galima atkurti tiesiog nustatant kiekvienos formos stiprumą (0…1).
Viskas tiesiška, todėl rezultatas sutampa su tuo, ką skaičiuoja mh_wardrobe.py.
"""
import gzip, json, os, struct, sys
import numpy as np

DATA, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------
# 1. Bazinis kūnas (tik „body“ grupė; pagalbiniai taškai išmetami)
# ---------------------------------------------------------------------
verts, quads, group, joints = [], [], None, {}
with open(os.path.join(DATA, "3dobjs/base.obj")) as f:
    for line in f:
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("g "):
            group = line.split()[1]
        elif line.startswith("f "):
            idx = [int(p.split("/")[0]) - 1 for p in line.split()[1:]]
            if group == "body":
                quads.append(idx if len(idx) == 4 else idx + [idx[-1]])
            elif group.startswith("joint-"):
                joints.setdefault(group[6:], set()).update(idx)
base = np.array(verts)
quads = np.array(quads)
joints = {k: sorted(v) for k, v in joints.items()}
USED = np.unique(quads)

def J(v, name):
    return v[joints[name]].mean(axis=0)

_cache = {}
def target(path):
    if path not in _cache:
        off = np.zeros_like(base)
        with gzip.open(os.path.join(DATA, "targets", path + ".target.gz"), "rt") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                i, x, y, z = line.split()
                off[int(i)] = (float(x), float(y), float(z))
        _cache[path] = off
    return _cache[path]

def weight_levels(w):
    if w < 0.5:
        lo = 1 - w / 0.5
        return {"minweight": lo, "averageweight": 1 - lo}
    hi = (w - 0.5) / 0.5
    return {"averageweight": 1 - hi, "maxweight": hi}

def build(gender, weight):
    """Tas pats kaip mh_wardrobe.py: gender 0 = moteris, 1 = vyras; weight 0…1."""
    v = base.copy()
    for g, gv in (("female", 1 - gender), ("male", gender)):
        if gv == 0:
            continue
        for race in ("african", "asian", "caucasian"):
            v += target(f"macrodetails/{race}-{g}-young") * gv / 3
        for wl, wv in weight_levels(weight).items():
            v += target(f"macrodetails/universal-{g}-young-averagemuscle-{wl}") * gv * wv
    return v

F0 = build(0, 0.5)
M0 = build(1, 0.5)
MORPHS = {
    "male":      M0 - F0,
    "f-thin":    build(0, 0.0) - F0,
    "f-heavy":   build(0, 1.0) - F0,
    "m-thin":    build(1, 0.0) - M0,
    "m-heavy":   build(1, 1.0) - M0,
    "waist-":    target("torso/measure-waist-circ-decr"),
    "waist+":    target("torso/measure-waist-circ-incr"),
    "hips-":     target("torso/measure-hips-circ-decr"),
    "hips+":     target("torso/measure-hips-circ-incr"),
    "upperarm-": target("arms/measure-upperarm-length-decr"),
    "upperarm+": target("arms/measure-upperarm-length-incr"),
    "lowerarm-": target("arms/measure-lowerarm-length-decr"),
    "lowerarm+": target("arms/measure-lowerarm-length-incr"),
}

# ---------------------------------------------------------------------
# 2. Matmenys centimetrais – kad naršyklė žinotų, kokia forma atitinka kokį skaičių
# ---------------------------------------------------------------------
def circumference(v, y, part_mask):
    """Kūno apimtis aukštyje y: iškirpto pjūvio išgaubtas kontūras (kaip matuojama juostele)."""
    m = part_mask & (np.abs(v[:, 1] - y) < 0.08)
    pts = v[m][:, [0, 2]]
    if len(pts) < 3:
        return 0.0
    # išgaubtas apvalkalas (Andrew algoritmas)
    pts = sorted(map(tuple, pts))
    def cross(o, a, b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lo, hi = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0: hi.pop()
        hi.append(p)
    hull = np.array(lo[:-1] + hi[:-1])
    return float(np.linalg.norm(np.roll(hull, -1, axis=0) - hull, axis=1).sum()) * 10   # dm → cm

# Liemuo be rankų: taškai, kurie arčiau kūno vidurio nei pečių sąnariai
def torso_mask(v):
    sh = abs(J(v, "l-shoulder")[0])
    return np.isin(np.arange(len(v)), USED) & (np.abs(v[:, 0]) < sh * 0.95)

def measures(v):
    pel, neck = J(v, "pelvis")[1], J(v, "neck")[1]
    tm = torso_mask(v)
    # juosmuo – siauriausia vieta tarp dubens ir krūtinės; klubai – plačiausia žemiau dubens
    waist = min(circumference(v, y, tm) for y in np.linspace(pel + 0.4, pel + 0.45 * (neck - pel), 12))
    hips = max(circumference(v, y, tm) for y in np.linspace(pel - 0.9, pel + 0.2, 12))
    arm = sum(float(np.linalg.norm(J(v, f"l-{a}") - J(v, f"l-{b}")))
              for a, b in (("shoulder", "elbow"), ("elbow", "hand"))) * 10
    height = float(v[USED, 1].max() - v[USED, 1].min()) * 10
    return {"waist": round(waist, 1), "hips": round(hips, 1), "arm": round(arm, 1), "height": round(height, 1)}

info = {"units": "1 vienetas = 1 m", "base": "female, vidutinis svoris", "measures": {}}
for gname, g0 in (("female", F0), ("male", M0)):
    for wname, w in (("thin", 0.0), ("avg", 0.5), ("heavy", 1.0)):
        v = build(0 if gname == "female" else 1, w)
        info["measures"][f"{gname}-{wname}"] = measures(v)
        for name in ("waist-", "waist+", "hips-", "hips+", "upperarm-", "upperarm+", "lowerarm-", "lowerarm+"):
            info["measures"][f"{gname}-{wname}|{name}"] = measures(v + MORPHS[name])
print(json.dumps({k: v for k, v in info["measures"].items() if "|" not in k}, indent=1))

# ---------------------------------------------------------------------
# 3. Trikampiai ir apatinių sritis
# ---------------------------------------------------------------------
remap = -np.ones(len(base), int)
remap[USED] = np.arange(len(USED))
tris = []
for q in quads:
    a, b, c, d = q
    tris.append((a, b, c))
    if d != c:
        tris.append((a, c, d))
tris = np.array(tris)

# Kūno dalių žemėlapis (kaip mh_wardrobe.py): kiekvienam taškui – dalis ir padėtis joje
def seg_param(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
    return t, np.linalg.norm(p - (a + t[:, None] * ab), axis=1)

def body_map(v):
    bones = [("torso", J(v, "pelvis") - [0, 1.2, 0], J(v, "neck"), 1.7, 0, 1), ("head", J(v, "neck"), J(v, "head-2"), 0.9, 0, 1)]
    for s in ("l", "r"):
        sh, el, ha = J(v, f"{s}-shoulder"), J(v, f"{s}-elbow"), J(v, f"{s}-hand")
        bones += [("arm", sh, el, 0.55, 0.0, 0.5), ("arm", el, ha, 0.45, 0.5, 1.0), ("arm", ha, ha + (ha - el) * 0.55, 0.4, 1.0, 1.4)]
        up, kn, an, toe = J(v, f"{s}-upper-leg"), J(v, f"{s}-knee"), J(v, f"{s}-ankle"), J(v, f"{s}-foot-2")
        bones += [("leg", up, kn, 0.85, 0.0, 0.5), ("leg", kn, an, 0.6, 0.5, 1.0), ("leg", an, toe, 0.5, 1.0, 1.4)]
    best = np.full(len(v), np.inf); part = np.empty(len(v), dtype=object); pos = np.zeros(len(v))
    for name, a, b, r, p0, p1 in bones:
        t, d = seg_param(v, a, b)
        m = d / r < best
        best[m], part[m], pos[m] = (d / r)[m], name, (p0 + (p1 - p0) * t)[m]
    return part, pos

PART, POS = body_map(base)
Y = base[:, 1]
PELVIS_Y, NECK_Y = J(base, "pelvis")[1], J(base, "neck")[1]
OUTSIDE = -0.4

def rule(torso=None, leg=None):
    """Reikšmė kiekvienam taškui: > 0 – srities viduje (decimetrais). Naršyklė ją tolygiai
    interpoliuoja tarp taškų ir kerpa ties 0 – todėl kraštai tiesūs, ne laiptuoti."""
    f = np.full(len(base), OUTSIDE)
    if torso:
        m = PART == "torso"
        f[m] = np.minimum(Y[m] - torso[0], torso[1] - Y[m])
    if leg:
        m = PART == "leg"
        f[m] = np.maximum(f[m], np.minimum(POS[m] - leg[0], leg[1] - POS[m]) * 9.0)
    return f

L = NECK_Y - PELVIS_Y
SHORTS = rule(torso=(-99, PELVIS_Y + 0.12 * L), leg=(-1, 0.26))
BRA = rule(torso=(PELVIS_Y + 0.47 * L, PELVIS_Y + 0.9 * L))
UNDER_F = np.maximum(SHORTS, BRA).astype(np.float32)   # moteriai: šortukai + palaidinukė
UNDER_M = SHORTS.astype(np.float32)                    # vyrui: tik šortukai

# ---------------------------------------------------------------------
# 4. .glb rašymas (be papildomų bibliotekų)
# ---------------------------------------------------------------------
S = 0.1   # decimetrai → metrai
ground = F0[USED, 1].min()

def vnormals(v):
    p = v[USED]
    t = remap[tris]
    fn = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
    n = np.zeros_like(p)
    for k in range(3):
        np.add.at(n, t[:, k], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)

bin_parts, views, accessors = [], [], []
def add(arr, comp, kind, target=None, minmax=False):
    data = np.ascontiguousarray(arr)
    raw = data.tobytes()
    off = sum(len(b) for b in bin_parts)
    pad = (-len(raw)) % 4
    bin_parts.append(raw + b"\0" * pad)
    view = {"buffer": 0, "byteOffset": off, "byteLength": len(raw)}
    if target:
        view["target"] = target
    views.append(view)
    acc = {"bufferView": len(views) - 1, "componentType": comp, "count": len(data), "type": kind}
    if minmax:
        acc["min"] = data.min(axis=0).tolist(); acc["max"] = data.max(axis=0).tolist()
    accessors.append(acc)
    return len(accessors) - 1

pos = ((F0[USED] - [0, ground, 0]) * S).astype(np.float32)
POS = add(pos, 5126, "VEC3", 34962, True)
NRM = add(vnormals(F0).astype(np.float32), 5126, "VEC3", 34962)
targets, names = [], []
for name, d in MORPHS.items():
    targets.append({"POSITION": add((d[USED] * S).astype(np.float32), 5126, "VEC3", 34962, True)})
    names.append(name)

ind = remap[tris].astype(np.uint16).reshape(-1)
IDX = add(ind, 5123, "SCALAR", 34963)
UF = add(UNDER_F[USED].reshape(-1, 1)[:, 0].copy(), 5126, "SCALAR", 34962)
UM = add(UNDER_M[USED].reshape(-1, 1)[:, 0].copy(), 5126, "SCALAR", 34962)
# Vienas kūnas; apatinių sritis – papildomi taškų laukai _UNDERF (moteriai) ir _UNDERM (vyrui).
# Naršyklė juos nuspalvina juodai ten, kur laukas > 0.
prims = [{"attributes": {"POSITION": POS, "NORMAL": NRM, "_UNDERF": UF, "_UNDERM": UM},
          "indices": IDX, "material": 0, "targets": targets}]
extra = {"targetNames": names}

gltf = {
    "asset": {"version": "2.0", "generator": "NovaFit mh_export_glb.py (MakeHuman CC0)"},
    "scene": 0, "scenes": [{"nodes": [0]}],
    "nodes": [{"mesh": 0, "name": "avatar"}],
    "meshes": [{"name": "body", "primitives": prims, "weights": [0.0] * len(names), "extras": extra}],
    "materials": [
        {"name": "skin", "pbrMetallicRoughness": {"baseColorFactor": [0.886, 0.839, 0.784, 1], "metallicFactor": 0, "roughnessFactor": 0.75}},
    ],
    "accessors": accessors, "bufferViews": views,
}
binary = b"".join(bin_parts)
gltf["buffers"] = [{"byteLength": len(binary)}]
js = json.dumps(gltf, separators=(",", ":")).encode()
js += b" " * ((-len(js)) % 4)
glb = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(binary))
glb += struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(binary), 0x004E4942) + binary
open(os.path.join(OUT, "body.glb"), "wb").write(glb)

info["targetNames"] = names
info["groundOffsetM"] = 0.0
json.dump(info, open(os.path.join(OUT, "body.json"), "w"), indent=1, ensure_ascii=False)
print(f"body.glb: {len(glb) / 1e6:.2f} MB, taškų {len(USED)}, trikampių {len(tris)}, formų {len(names)}")
