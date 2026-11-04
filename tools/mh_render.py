"""Generuoja NovaFit avataro figūras iš MakeHuman kūno duomenų (CC0).

Paleidimas:  python3 mh_render.py <mpfb2 data aplankas> <išvesties aplankas>
Sukuria PNG paveikslėlius be fono: {female,male}-w{0..4}.png
(w0 = lieknas ... w4 = stambus), visi vienodu masteliu, pėdos apačioje.
"""
import gzip, sys, os, json
import numpy as np
from PIL import Image, ImageDraw

DATA, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

# ---------- 1. Bazinis kūnas ----------
verts, faces, group = [], [], None
joints = {}   # sąnarių taškai (pvz. dubuo, kaklas) – naudojam apatinių vietai rasti
with open(os.path.join(DATA, "3dobjs/base.obj")) as f:
    for line in f:
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("g "):
            group = line.split()[1]
        elif line.startswith("f "):
            idx = [int(p.split("/")[0]) - 1 for p in line.split()[1:]]
            if group == "body":
                faces.append(idx)
            elif group.startswith("joint-"):
                joints.setdefault(group[6:], set()).update(idx)
joints = {k: sorted(v) for k, v in joints.items()}
base = np.array(verts)
faces = np.array([f if len(f) == 4 else f + [f[-1]] for f in faces])

# ---------- 2. Formos (targets) ----------
MACRO = os.path.join(DATA, "targets/macrodetails")
_cache = {}
def target(name):
    if name not in _cache:
        off = np.zeros_like(base)
        with gzip.open(os.path.join(MACRO, name + ".target.gz"), "rt") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                i, x, y, z = line.split()
                off[int(i)] = (float(x), float(y), float(z))
        _cache[name] = off
    return _cache[name]

def weight_levels(w):
    if w < 0.5:
        lo = 1 - w / 0.5
        return {"minweight": lo, "averageweight": 1 - lo}
    hi = (w - 0.5) / 0.5
    return {"averageweight": 1 - hi, "maxweight": hi}

def build(gender, weight):
    """gender: 0 = moteris, 1 = vyras; weight: 0..1 (MakeHuman svorio slankiklis)."""
    v = base.copy()
    for g, gv in (("female", 1 - gender), ("male", gender)):
        if gv == 0:
            continue
        for race in ("african", "asian", "caucasian"):
            v += target(f"{race}-{g}-young") * gv / 3
        for wl, wv in weight_levels(weight).items():
            v += target(f"universal-{g}-young-averagemuscle-{wl}") * gv * wv
    return v

# ---------- 3. Piešimas (vaizdas iš priekio) ----------
SS = 3                 # glotninimas: piešiam 3x didesnį ir sumažinam
PX = 36                # taškų vienam MakeHuman vienetui (decimetrui)
W_UNITS, H_UNITS = 13.0, 18.0
LIGHT = np.array([-0.35, 0.45, 0.82]); LIGHT /= np.linalg.norm(LIGHT)
COLOR = np.array([226, 214, 200], float)   # šviesus manekeno atspalvis

UNDER = np.array([138, 150, 168], float)  # neutralūs bazinio sluoksnio apatiniai

def joint(v, name):
    return v[joints[name]].mean(axis=0)

def underwear_shapes(v, female, sx, sy):
    """Apatinių formos ekrane (lygūs kraštai): kelnaitės ir (moteriai) sportinė liemenėlė."""
    pelvis, neck = joint(v, "pelvis"), joint(v, "neck")
    leg, shoulder = joint(v, "l-upper-leg"), joint(v, "l-shoulder")
    L = neck[1] - pelvis[1]
    hw = abs(leg[0]) * 2.6
    top, outer, crotch = pelvis[1] + 0.10 * L, leg[1] - 0.04 * L, leg[1] - 0.17 * L
    shapes = [[(-hw, top), (hw, top), (hw, outer), (0.35 * hw, crotch), (-0.35 * hw, crotch), (-hw, outer)]]
    if female:
        sw = abs(shoulder[0]) * 0.92
        y1, y2 = pelvis[1] + 0.55 * L, pelvis[1] + 0.80 * L
        shapes.append([(-sw, y2), (sw, y2), (sw, y1), (-sw, y1)])
    return [[(sx(x), sy(y)) for x, y in s] for s in shapes]

def render(v, path, female):
    W, H = int(W_UNITS * PX * SS), int(H_UNITS * PX * SS)
    used = np.unique(faces)
    ground = v[used, 1].min()
    sx = lambda x: W / 2 + x * PX * SS
    sy = lambda y: H - 6 * SS - (y - ground) * PX * SS
    skin = draw(v, COLOR, W, H, sx, sy)
    cloth = draw(v, UNDER, W, H, sx, sy)
    mask = Image.new("L", (W, H), 0)
    md = ImageDraw.Draw(mask)
    for s in underwear_shapes(v, female, sx, sy):
        md.polygon(s, fill=255)
    img = Image.composite(cloth, skin, mask)
    img = img.resize((W // SS, H // SS), Image.LANCZOS)
    img.save(path)

def draw(v, color, W, H, sx, sy):
    q = v[faces]                                       # (n, 4, 3)
    fn = np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 1])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    # Taškų normalės – kad šešėliai būtų glotnūs, o ne „kampuoti“
    vn = np.zeros_like(v)
    for k in range(4):
        np.add.at(vn, faces[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    front = np.where(fn[:, 2] > 0)[0]
    order = front[np.argsort(q[front, :, 2].mean(axis=1))]   # tolimiausi – pirmi
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i in order:
        P = [np.array((sx(p[0]), sy(p[1]))) for p in q[i]]
        mid = sum(P) / 4
        for k in range(4):   # kiekvieną gabaliuką dalinam į 4, kiekvienas su savo kampo šviesa
            nrm = 0.65 * vn[faces[i, k]] + 0.35 * fn[i]
            nrm /= np.linalg.norm(nrm)
            s = 0.42 + 0.58 * max(0.0, float(nrm @ LIGHT))
            c = tuple(int(x) for x in np.clip(color * s, 0, 255)) + (255,)
            pts = [P[k], (P[k] + P[(k + 1) % 4]) / 2, mid, (P[k] + P[k - 1]) / 2]
            pts = [tuple(p) for p in pts]
            d.polygon(pts, fill=c, outline=c)
    return img

meta = {}
for gname, gval in (("female", 0.0), ("male", 1.0)):
    for k, w in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
        v = build(gval, w)
        name = f"{gname}-w{k}.png"
        render(v, os.path.join(OUT, name), female=(gname == "female"))
        used = np.unique(faces)
        meta[name] = {"gender": gname, "mh_weight": w,
                      "height_units": float(v[used, 1].max() - v[used, 1].min())}
        print(name, round(meta[name]["height_units"], 2))
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
