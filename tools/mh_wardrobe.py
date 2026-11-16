"""NovaFit: MakeHuman figūros + drabužių sluoksniai, „pasiūti“ tiesiai ant 3D kūno.

Paleidimas:  python3 mh_wardrobe.py <mpfb2 data aplankas> <išvesties aplankas>

Kiekvienai figūrai (2 lytys × 5 svorio lygiai) sukuria:
  body.png          – manekenas su neutraliais apatiniais
  <tipas>.png       – drabužio sluoksnis pilkais atspalviais (spalva uždedama naršyklėje)
Visi paveikslėliai vienodo dydžio, todėl juos galima tiesiog dėti vieną ant kito.

Kaip veikia: kiekvienas kūno taškas priskiriamas kūno daliai (liemuo, ranka, koja...)
ir gauna „padėtį“ toje dalyje (pvz. 0 = petys, 1 = riešas). Drabužis aprašomas
taisyklėmis („liemuo nuo juosmens iki kaklo + rankos iki alkūnės“), iškeliamas
truputį virš odos (storis) ir nupiešiamas su šešėliais. Kraštai kerpami tolygiai.
"""
import gzip, sys, os, json
import numpy as np
from PIL import Image, ImageDraw

DATA, OUT = sys.argv[1], sys.argv[2]

# =====================================================================
# 1. Bazinis kūnas ir sąnariai
# =====================================================================
verts, faces, group, joints = [], [], None, {}
with open(os.path.join(DATA, "3dobjs/base.obj")) as f:
    for line in f:
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("g "):
            group = line.split()[1]
        elif line.startswith("f "):
            idx = [int(p.split("/")[0]) - 1 for p in line.split()[1:]]
            if group == "body":
                faces.append(idx if len(idx) == 4 else idx + [idx[-1]])
            elif group.startswith("joint-"):
                joints.setdefault(group[6:], set()).update(idx)
base = np.array(verts)
faces = np.array(faces)
joints = {k: sorted(v) for k, v in joints.items()}
USED = np.unique(faces)

def J(v, name):
    return v[joints[name]].mean(axis=0)

# =====================================================================
# 2. Formos (MakeHuman targets): lytis ir svoris
# =====================================================================
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
    v = base.copy()
    for g, gv in (("female", 1 - gender), ("male", gender)):
        if gv == 0:
            continue
        for race in ("african", "asian", "caucasian"):
            v += target(f"{race}-{g}-young") * gv / 3
        for wl, wv in weight_levels(weight).items():
            v += target(f"universal-{g}-young-averagemuscle-{wl}") * gv * wv
    return v

# =====================================================================
# 3. Kūno dalių žemėlapis (skaičiuojamas vieną kartą baziniam kūnui)
# =====================================================================
def seg_param(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
    d = np.linalg.norm(p - (a + t[:, None] * ab), axis=1)
    return t, d

def body_map(v):
    """Kiekvienam taškui: dalis ('torso','arm','leg','head') ir padėtis toje dalyje."""
    P = v
    bones = []   # (dalis, a, b, spindulys, padėtis nuo, padėtis iki)
    bones.append(("torso", J(v, "pelvis") - [0, 1.2, 0], J(v, "neck"), 1.7, 0, 1))
    bones.append(("head", J(v, "neck"), J(v, "head-2"), 0.9, 0, 1))
    for s in ("l", "r"):
        sh, el, ha = J(v, f"{s}-shoulder"), J(v, f"{s}-elbow"), J(v, f"{s}-hand")
        bones += [("arm", sh, el, 0.55, 0.0, 0.5), ("arm", el, ha, 0.45, 0.5, 1.0),
                  ("arm", ha, ha + (ha - el) * 0.55, 0.4, 1.0, 1.4)]
        up, kn, an = J(v, f"{s}-upper-leg"), J(v, f"{s}-knee"), J(v, f"{s}-ankle")
        toe = J(v, f"{s}-foot-2")
        bones += [("leg", up, kn, 0.85, 0.0, 0.5), ("leg", kn, an, 0.6, 0.5, 1.0),
                  ("leg", an, toe, 0.5, 1.0, 1.4)]
    best = np.full(len(P), np.inf)
    part = np.empty(len(P), dtype=object)
    pos = np.zeros(len(P))
    for name, a, b, r, p0, p1 in bones:
        t, d = seg_param(P, a, b)
        score = d / r
        m = score < best
        best[m], part[m], pos[m] = score[m], name, (p0 + (p1 - p0) * t)[m]
    return part, pos

PART, POS = body_map(base)
Y = base[:, 1]
X = base[:, 0]
Z = base[:, 2]
NECK_Y = J(base, "neck")[1]
PELVIS_Y = J(base, "pelvis")[1]
EYE_Y = J(base, "l-eye")[1]
WAIST_Y = PELVIS_Y + 0.62          # kelnių juosmuo
FRONT = Z > 0.2

# =====================================================================
# 4. Drabužių aprašymai.
# Kiekviena taisyklė grąžina skaičių kiekvienam taškui: >0 – drabužio viduje.
# Naudojami decimetrai (1 = 10 cm), kad kraštai būtų kerpami tolygiai.
# =====================================================================
OUTSIDE = -0.4

def rule(torso=None, arm=None, leg=None, head=None, extra=None):
    """torso=(y_nuo, y_iki), arm/leg=(padėtis_nuo, padėtis_iki), head=y_nuo."""
    f = np.full(len(base), OUTSIDE)
    if torso:
        y0, y1 = torso
        m = PART == "torso"
        f[m] = np.minimum(Y[m] - y0, y1[m] - Y[m] if isinstance(y1, np.ndarray) else y1 - Y[m])
    for name, rng, L in (("arm", arm, 7.0), ("leg", leg, 9.0)):
        if rng:
            m = PART == name
            f[m] = np.minimum(POS[m] - rng[0], rng[1] - POS[m]) * L
    if head is not None:
        m = PART == "head"
        f[m] = Y[m] - head
    if extra is not None:
        f = np.minimum(f, extra)
    return f

# Apvali kaklo iškirptė priekyje
CREW = NECK_Y - 0.08 - 0.55 * np.exp(-(X / 0.5) ** 2) * FRONT
COLLAR = NECK_Y - 0.15 - 0.15 * np.exp(-(X / 0.4) ** 2) * FRONT
JACKET_OPEN = np.where(FRONT & (PART == "torso"), np.abs(X) - 0.18, 9)
COAT_OPEN = np.where(FRONT & (PART == "torso") & (Y > PELVIS_Y + 0.6), np.abs(X) - 0.18, 9)
# Viršutinė riba visam drabužiui (ir pečiams), kad kaklo linija būtų tolygi
CAP_CREW, CAP_COLLAR, CAP_NECK = CREW - Y, COLLAR - Y, NECK_Y - 0.12 - Y

GARMENTS = {
    #  pavadinimas: (taisyklė, storis dm)
    "tshirt":   (rule(torso=(PELVIS_Y + 0.25, CREW), arm=(-1, 0.2), extra=CAP_CREW), 0.05),
    "sweater":  (rule(torso=(PELVIS_Y + 0.15, CREW), arm=(-1, 0.96), extra=CAP_CREW), 0.08),
    "shirt":    (rule(torso=(PELVIS_Y + 0.2, COLLAR), arm=(-1, 0.95), extra=CAP_COLLAR), 0.05),
    "pants":    (rule(torso=(-99, WAIST_Y), leg=(-1, 0.97)), 0.06),
    "shorts":   (rule(torso=(-99, WAIST_Y), leg=(-1, 0.28)), 0.06),
    "skirt":    (rule(torso=(-99, WAIST_Y)), 0.05),
    "jacket":   (rule(torso=(PELVIS_Y - 0.25, NECK_Y - 0.15), arm=(-1, 0.98), extra=np.minimum(JACKET_OPEN, CAP_NECK)), 0.15),
    "coat":     (rule(torso=(-99, NECK_Y - 0.1), arm=(-1, 0.98), extra=np.minimum(COAT_OPEN, CAP_NECK)), 0.17),
    "sneakers": (rule(leg=(0.965, 9)), 0.13),
    "boots":    (rule(leg=(0.74, 9)), 0.10),
    "cap":      (rule(head=EYE_Y + 0.55), 0.07),
    "beanie":   (rule(head=EYE_Y + 0.35), 0.11),
    # Laisvi (oversize) marškinėliai: ilgesni, rankovės iki alkūnių; forma keičiama funkcijoje baggy()
    "oversize": (rule(torso=(PELVIS_Y - 0.15, CREW), arm=(-1, 0.40), extra=CAP_CREW), 0.08),
}

def baggy(vs, wide=1.10, sleeve=0.22):
    """Laisvas kirpimas: audinys nebekartoja kūno, o kabo kaip lygus „vamzdis“ nuo plačiausios vietos
    (pečių į šonus, krūtinės į priekį, menčių į nugarą). Prie kaklo ir pečių lieka kūno forma.
    Rankovės – platesnės nei ranka."""
    v = vs.copy()
    used = np.zeros(len(v), bool); used[USED] = True
    t = (PART == "torso") & used
    ys = np.arange(v[t, 1].min(), v[t, 1].max() + 0.12, 0.12)
    k = np.clip(np.digitize(v[:, 1], ys) - 1, 0, len(ys) - 1)
    at = lambda prof: np.interp(v[:, 1], ys + 0.06, prof)
    blur = lambda p: np.convolve(np.pad(p, 3, mode="edge"), np.ones(7) / 7, mode="valid")
    def profile(vals):      # didžiausia reikšmė kiekviename aukštyje
        return np.array([vals[t & (k == i)].max() if (t & (k == i)).any() else 1e-3 for i in range(len(ys))])
    hang = lambda prof: np.maximum.accumulate(prof[::-1])[::-1]   # nuo viršaus žemyn nebemažėja – audinys kabo
    zc = blur(np.array([v[t & (k == i), 2].mean() if (t & (k == i)).any() else 0.0 for i in range(len(ys))]))
    dz = v[:, 2] - at(zc)
    top = v[t, 1].max()
    ramp = np.clip((top - ys - 0.7) / 0.9, 0, 1)            # prie kaklo ir pečių nieko nekeičiam
    px, pf, pb = profile(np.abs(v[:, 0])), profile(np.maximum(dz, 0)), profile(np.maximum(-dz, 0))
    a_ = blur(hang(px) * (1 + (wide - 1) * ramp))            # pusė pločio
    bf, bb = blur(hang(pf) * 1.05), blur(hang(pb) * 1.05)    # gylis į priekį / į nugarą
    m = (PART == "torso") | ((PART == "leg") & (v[:, 1] > PELVIS_Y - 2.5))
    r = at(ramp)
    x_new = v[:, 0] * (1 + (np.clip(at(a_) / np.maximum(at(px), 1e-3), 1, 2.2) - 1) * r)
    shape = (1 - np.clip(np.abs(x_new) / np.maximum(at(a_), 1e-3), 0, 1) ** 3) ** (1 / 3)   # suapvalintas stačiakampis
    z_tube = at(zc) + np.where(dz > 0, at(bf), -at(bb)) * shape
    v[m, 0] = x_new[m]
    v[m, 2] = (v[:, 2] * (1 - r) + z_tube * r)[m]
    v = smooth(v, 40, mask=m)
    a = PART == "arm"
    _, vn = normals(v)
    v[a] += vn[a] * (sleeve * np.clip(POS[a] / 0.12, 0, 1))[:, None]   # nuo peties platėja palaipsniui
    v = smooth(v, 6, mask=m | a)
    return v

# =====================================================================
# 5. Piešimas
# =====================================================================
SS, PX = 3, 36
W_UNITS, H_UNITS = 13.0, 18.0
W, H = int(W_UNITS * PX * SS), int(H_UNITS * PX * SS)
LIGHT = np.array([-0.35, 0.45, 0.82]); LIGHT /= np.linalg.norm(LIGHT)
SKIN = np.array([226, 214, 200], float)
UNDER = np.array([138, 150, 168], float)
CLOTH = np.array([236, 236, 236], float)   # pilka bazė – naršyklė padaugina iš spalvos

_edges = np.concatenate([faces[:, [k, (k + 1) % 4]] for k in range(4)])
_edges = _edges[_edges[:, 0] != _edges[:, 1]]
_edges = np.unique(np.sort(_edges, axis=1), axis=0)

def smooth(v, iters, mask=None, lam=0.5, taubin=True):
    """Išlygina paviršių (audinys nesilanksto pagal kiekvieną kūno smulkmeną)."""
    v = v.copy()
    deg = np.zeros(len(v))
    np.add.at(deg, _edges[:, 0], 1); np.add.at(deg, _edges[:, 1], 1)
    deg[deg == 0] = 1
    for it in range(iters * 2):
        # Taubin metodas: pakaitomis lyginam ir „išpučiam“ atgal, kad kūnas nesusitrauktų
        step = lam if (it % 2 == 0 or not taubin) else -(lam + 0.03)
        acc = np.zeros_like(v)
        np.add.at(acc, _edges[:, 0], v[_edges[:, 1]])
        np.add.at(acc, _edges[:, 1], v[_edges[:, 0]])
        delta = acc / deg[:, None] - v
        if mask is not None:
            delta[~mask] = 0
        v += step * delta
    return v

def normals(v):
    q = v[faces]
    fn = np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 1])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    vn = np.zeros_like(v)
    for k in range(4):
        np.add.at(vn, faces[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    return fn, vn

def clip(poly, vals):
    """Nukerpa daugiakampį ten, kur reikšmė < 0 (tolygus kraštas)."""
    out = []
    n = len(poly)
    for i in range(n):
        p, q, a, b = poly[i], poly[(i + 1) % n], vals[i], vals[(i + 1) % n]
        if a >= 0:
            out.append(p)
        if (a >= 0) != (b >= 0):
            t = a / (a - b)
            out.append(p + (q - p) * t)
    return out

def draw_layer(v, color, fsel=None, inflate=0.0, ctx=None):
    fn, vn = normals(v)
    vv = v + vn * inflate
    ground = ctx["ground"]
    sx = lambda x: W / 2 + x * PX * SS
    sy = lambda y: H - 6 * SS - (y - ground) * PX * SS
    q = vv[faces]
    if fsel is not None:
        fv = fsel[faces]                       # (n,4)
        keep = (fv > 0).any(axis=1)
    else:
        keep = np.ones(len(faces), bool)
    cand = np.where(keep & (fn[:, 2] > 0))[0]
    order = cand[np.argsort(q[cand, :, 2].mean(axis=1))]
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i in order:
        P = [np.array((sx(p[0]), sy(p[1]))) for p in q[i]]
        mid = sum(P) / 4
        if fsel is not None:
            F = fv[i]
            fm = F.mean()
        for k in range(4):
            nrm = 0.65 * vn[faces[i, k]] + 0.35 * fn[i]
            nrm /= np.linalg.norm(nrm)
            s = 0.42 + 0.58 * max(0.0, float(nrm @ LIGHT))
            c = tuple(int(x) for x in np.clip(color * s, 0, 255)) + (255,)
            pts = [P[k], (P[k] + P[(k + 1) % 4]) / 2, mid, (P[k] + P[k - 1]) / 2]
            if fsel is not None:
                vals = [F[k], (F[k] + F[(k + 1) % 4]) / 2, fm, (F[k] + F[k - 1]) / 2]
                if min(vals) < 0:
                    pts = clip(pts, vals)
                    if len(pts) < 3:
                        continue
            d.polygon([tuple(p) for p in pts], fill=c, outline=c)
    return img, sx, sy

def silhouette_x(v, part, y, band=0.15):
    """Kūno pločio pusė (|x|) duotame aukštyje – sijonui ir paltui."""
    sel = (PART != "arm") if part == "body" else (PART == part)
    m = sel & (np.abs(v[:, 1] - y) < band)
    return np.abs(v[m, 0]).max()

def shaded_poly(d, pts, color, top, bottom, light=True):
    """Plokščias audinio gabalas su vertikaliu šešėliu (sijono/palto klostė)."""
    d.polygon(pts, fill=tuple(int(c) for c in color) + (255,))

def add_flare(img, v, sx, sy, y_top, y_bottom, extra, color, center_gap=0.0):
    """Platėjantis audinys žemiau klubų (sijonas, palto skvernai)."""
    wt = silhouette_x(v, "body", y_top, 0.2) + 0.1
    wb = max(wt, silhouette_x(v, "leg", y_bottom, 0.3) * 1.0) + extra
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    steps = 24
    for side in (-1, 1):
        for k in range(steps):
            # vertikalios juostos su šviesos kaita – kad atrodytų kaip klostės
            a, b = k / steps, (k + 1) / steps
            xa_t, xb_t = center_gap + (wt - center_gap) * a, center_gap + (wt - center_gap) * b
            xa_b, xb_b = center_gap + (wb - center_gap) * a, center_gap + (wb - center_gap) * b
            s = 0.84 + 0.06 * np.cos((a + 0.5 / steps) * np.pi * 3) - 0.14 * a ** 3
            if side < 0:
                s += 0.04
            c = tuple(int(x) for x in np.clip(color * s, 0, 255)) + (255,)
            pts = [(sx(side * xa_t), sy(y_top)), (sx(side * xb_t), sy(y_top)),
                   (sx(side * xb_b), sy(y_bottom)), (sx(side * xa_b), sy(y_bottom))]
            d.polygon(pts, fill=c, outline=c)
    return Image.alpha_composite(img, layer)   # skvernai – virš kūno

def details(img, kind, v, sx, sy):
    d = ImageDraw.Draw(img)
    dark = (120, 120, 120, 255)
    neck = J(v, "neck")
    if kind == "shirt":
        top, bot = neck[1] - 0.35, J(v, "pelvis")[1] + 0.25
        d.line([(sx(0), sy(top)), (sx(0), sy(bot))], fill=dark, width=2 * SS)
        for k in range(6):
            y = top - 0.15 - k * (top - bot - 0.3) / 5
            r = 0.05 * PX * SS
            d.ellipse([sx(0) + 2 * r - r, sy(y) - r, sx(0) + 2 * r + r, sy(y) + r], fill=dark)
        cw = 0.55
        for s in (-1, 1):
            d.polygon([(sx(s * 0.12), sy(neck[1] - 0.05)), (sx(s * cw), sy(neck[1] - 0.35)),
                       (sx(s * 0.08), sy(neck[1] - 0.6))], fill=(215, 215, 215, 255), outline=dark)
    if kind in ("jacket", "coat"):
        for s in (-1, 1):
            d.polygon([(sx(s * 0.18), sy(neck[1] - 0.15)), (sx(s * 0.75), sy(neck[1] - 0.4)),
                       (sx(s * 0.2), sy(neck[1] - 1.6))], fill=(200, 200, 200, 255), outline=(150, 150, 150, 255))
    eye = J(v, "l-eye")[1]
    if kind == "cap":
        hw = silhouette_x(v, "head", eye + 0.55, 0.12)
        yb = eye + 0.5
        d.ellipse([sx(-hw * 1.05), sy(yb + 0.12), sx(hw * 1.05), sy(yb - 0.22)], fill=(190, 190, 190, 255))
    if kind == "beanie":
        hw = silhouette_x(v, "head", eye + 0.45, 0.12)
        yb = eye + 0.35
        d.rounded_rectangle([sx(-hw - 0.08), sy(yb + 0.45), sx(hw + 0.08), sy(yb - 0.05)],
                            radius=0.15 * PX * SS, fill=(205, 205, 205, 255))
        top = J(v, "head-2")[1]
        r = 0.32
        d.ellipse([sx(-r), sy(top + 0.15 + r), sx(r), sy(top + 0.15 - r)], fill=(225, 225, 225, 255))
    if kind == "sneakers":
        for s in (-1, 1):
            an = J(v, f"{'l' if s > 0 else 'r'}-ankle")
            ground = v[USED, 1].min()
            fw = 0.55
            d.rectangle([sx(an[0] - fw), sy(ground + 0.12), sx(an[0] + fw), sy(ground - 0.02)],
                        fill=(250, 250, 250, 255))
    return img

def underwear(img_skin, img_cloth, v, female, sx, sy):
    """Neutralus bazinis sluoksnis: šortukai visiems, moteriai – sportinė palaidinukė."""
    pelvis, neck = J(v, "pelvis"), J(v, "neck")
    leg, shoulder = J(v, "l-upper-leg"), J(v, "l-shoulder")
    L = neck[1] - pelvis[1]
    hw = abs(leg[0]) * 2.6
    top, outer, crotch = pelvis[1] + 0.12 * L, leg[1] - 0.22 * L, leg[1] - 0.30 * L
    shapes = [[(-hw, top), (hw, top), (hw, outer), (0.3 * hw, crotch), (-0.3 * hw, crotch), (-hw, outer)]]
    if female:
        sw = abs(shoulder[0]) * 0.9
        y1, y_side, y2 = pelvis[1] + 0.45 * L, pelvis[1] + 0.78 * L, pelvis[1] + 0.97 * L
        shapes.append([(-sw, y1), (sw, y1), (sw, y_side), (0.55 * sw, y2), (0.35 * sw, y2),
                       (0.2 * sw, y_side + 0.05 * L), (-0.2 * sw, y_side + 0.05 * L),
                       (-0.35 * sw, y2), (-0.55 * sw, y2), (-sw, y_side)])
    mask = Image.new("L", (W, H), 0)
    md = ImageDraw.Draw(mask)
    for s in shapes:
        md.polygon([(sx(x), sy(y)) for x, y in s], fill=255)
    return Image.composite(img_cloth, img_skin, mask)

def save(img, path):
    img.resize((W // SS, H // SS), Image.LANCZOS).save(path, optimize=True)

# Krūtinės „smaigaliai“: mažas plotelis, kurį išlyginam (manekeno ir audinio išvaizdai)
def _tip_zone():
    vf = build(0.0, 0.5)
    used = np.zeros(len(vf), bool); used[USED] = True
    zone = np.zeros(len(vf), bool)
    for side in (-1, 1):
        cand = used & (PART == "torso") & (np.sign(vf[:, 0]) == side)
        tip = np.argmax(np.where(cand, vf[:, 2], -9))
        zone |= used & (np.linalg.norm(vf - vf[tip], axis=1) < 0.24)
    return zone
TIP_ZONE = _tip_zone()

# =====================================================================
# 6. Generavimas
# =====================================================================
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
meta = {"canvas": [W // SS, H // SS], "px_per_dm": PX, "figures": {}}
for gname, gval in (("female", 0.0), ("male", 1.0)):
    for k, w in enumerate(WEIGHTS):
        fig = f"{gname}-w{k}"
        if only and fig not in only:
            continue
        os.makedirs(os.path.join(OUT, fig), exist_ok=True)
        v = build(gval, w)
        ctx = {"ground": v[USED, 1].min()}
        # Manekenui švelniai išlyginam krūtinės/liemens smulkmenas
        vb = smooth(v, 3, mask=(PART == "torso") & (Y > PELVIS_Y + 1.5))
        vb = smooth(vb, 40, mask=TIP_ZONE, taubin=False)
        if gname == "female":   # po palaidinuke – lygi krūtinės forma be smulkmenų
            vb = smooth(vb, 30, mask=(PART == "torso") & (Y > PELVIS_Y + 1.6), taubin=True)
        skin, sx, sy = draw_layer(vb, SKIN, ctx=ctx)
        cloth, _, _ = draw_layer(vb, UNDER, ctx=ctx)
        save(underwear(skin, cloth, v, gname == "female", sx, sy), os.path.join(OUT, fig, "body.png"))
        vs = smooth(v, 30, mask=PART != "head")     # audiniui – lygesnis paviršius
        # krūtinės srityje audinys „ištempiamas“ – be smulkių kūno detalių
        vs = smooth(vs, 40, mask=TIP_ZONE, taubin=False)
        for kind, (fsel, thick) in GARMENTS.items():
            src = v if kind in ("cap", "beanie") else baggy(vs) if kind == "oversize" else vs
            img, sx, sy = draw_layer(src, CLOTH, fsel=fsel, inflate=thick, ctx=ctx)
            pel = J(v, "pelvis")[1]
            knee = J(v, "l-knee")[1]
            if kind == "skirt":
                img = add_flare(img, v, sx, sy, pel + 0.55, knee + 0.7, 0.6, CLOTH)
            if kind == "coat":
                img = add_flare(img, v, sx, sy, pel + 0.2, knee - 0.6, 0.5, CLOTH)
            img = details(img, kind, v, sx, sy)
            save(img, os.path.join(OUT, fig, f"{kind}.png"))
        meta["figures"][fig] = {"gender": gname, "mh_weight": w,
                                "height_dm": float(v[USED, 1].max() - v[USED, 1].min())}
        print(fig, "ok")
if not only:
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
