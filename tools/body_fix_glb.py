"""NovaFit: kūno modelio (assets/avatar3d/body.glb) pataisymas. Leisti po tools/mh_export_glb.py.

1) Krūtinė – kaip manekeno: speneliai išlyginami (erdvinis ~1,2 cm išlyginimas krūtinės priekyje),
   krūtinės forma lieka. Tai daroma bazinei figūrai ir visoms 13 formų.
2) Šešėliavimas: kiekvienai formai (vyras, stambesnis, ...) pridedamos jos normalės. Be jų vyro figūra
   šešėliuojama moters normalėmis – atrodo, lyg turėtų krūtis.

Naudojimas: python3 tools/body_fix_glb.py assets/avatar3d/body.glb
"""
import json, struct, sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "assets/avatar3d/body.glb"
raw = open(path, "rb").read()
jl = struct.unpack("<I", raw[12:16])[0]
J = json.loads(raw[20:20 + jl])
bstart = 20 + jl
bl = struct.unpack("<I", raw[bstart:bstart + 4])[0]
BIN = bytearray(raw[bstart + 8:bstart + 8 + bl])

CT = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
def read(ai):
    a = J["accessors"][ai]; bv = J["bufferViews"][a["bufferView"]]
    off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    n = a["count"] * NC[a["type"]]
    return np.frombuffer(BIN, CT[a["componentType"]], n, off).reshape(a["count"], -1).copy()
def write_inplace(ai, arr):
    a = J["accessors"][ai]; bv = J["bufferViews"][a["bufferView"]]
    off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    b = arr.astype(np.float32).tobytes()
    BIN[off:off + len(b)] = b
    if a["type"] == "VEC3" and "min" in a:
        a["min"] = arr.min(0).tolist(); a["max"] = arr.max(0).tolist()
def append(arr, target=34962):
    global BIN
    while len(BIN) % 4: BIN.append(0)
    off = len(BIN); b = arr.astype(np.float32).tobytes(); BIN += b
    J["bufferViews"].append({"buffer": 0, "byteOffset": off, "byteLength": len(b), "target": target})
    J["accessors"].append({"bufferView": len(J["bufferViews"]) - 1, "componentType": 5126, "count": len(arr), "type": "VEC3",
                           "min": arr.min(0).tolist(), "max": arr.max(0).tolist()})
    return len(J["accessors"]) - 1

prim = next(m for m in J["meshes"] if m.get("name") == "body")["primitives"][0]
if prim.get("extras", {}).get("novafitFixed"):
    print("jau pataisyta – nieko nedarau"); sys.exit(0)
P0 = read(prim["attributes"]["POSITION"]).astype(np.float64)
N0 = read(prim["attributes"]["NORMAL"]).astype(np.float64)
idx = read(prim["indices"]).ravel().astype(np.int64).reshape(-1, 3)
targets = prim["targets"]
D = [read(t["POSITION"]).astype(np.float64) for t in targets]
nv = len(P0)

def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)
def vnormals(P):
    fn = np.cross(P[idx[:, 1]] - P[idx[:, 0]], P[idx[:, 2]] - P[idx[:, 0]])
    n = np.zeros_like(P)
    for j in range(3): np.add.at(n, idx[:, j], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
used = np.zeros(nv, bool); used[idx.ravel()] = True

# Krūtinės priekio sritis (glTF: y – aukštis, z – į priekį). Apskaičiuojama pagal bazinę figūrą.
H = P0[:, 1].max()
y, x, z = P0[:, 1] / H, P0[:, 0], P0[:, 2]
W = (smoothstep(0.62, 0.66, y) * smoothstep(0.80, 0.76, y) * smoothstep(0.19, 0.15, np.abs(x))
     * smoothstep(0.0, 0.04, z) * used)
reg = np.where(W > 0)[0]
fa = np.linalg.norm(np.cross(P0[idx[:, 1]] - P0[idx[:, 0]], P0[idx[:, 2]] - P0[idx[:, 0]]), axis=1)
va = np.zeros(nv)
for j in range(3): np.add.at(va, idx[:, j], fa)
# kaimynystė erdvėje: visi naudojami taškai aplink sritį
near = np.where(used & (np.abs(P0[:, 1] / H - 0.71) < 0.14) & (z > -0.05))[0]
def clean(P):
    T = P[near]; R = P[reg]
    d2 = ((R[:, None, :] - T[None, :, :]) ** 2).sum(2)
    w = np.exp(-d2 / (2 * 0.012 ** 2)) * va[near][None, :]
    S = (w @ T) / w.sum(1, keepdims=True)
    P = P.copy()
    m = W[reg][:, None]
    P[reg] = R * (1 - m) + S * m
    return P

Pb = clean(P0)
Nb = vnormals(Pb)
# bazinė normalė: originali, o pataisytoje srityje – perskaičiuota
Nbase = N0.copy()
m = (smoothstep(0, 0.2, W))[:, None]
Nbase = Nbase * (1 - m) + Nb * m
Nbase /= np.linalg.norm(Nbase, axis=1, keepdims=True) + 1e-12
Nbase[~used] = N0[~used]
write_inplace(prim["attributes"]["POSITION"], Pb)
write_inplace(prim["attributes"]["NORMAL"], Nbase)
names = next(m for m in J["meshes"] if m.get("name") == "body").get("extras", {}).get("targetNames", [])
OWN = {"male", "f-thin", "f-heavy", "m-thin", "m-heavy"}     # tik šioms formoms šešėliai ryškiai keičiasi
zero_acc = None
for i, (t, d) in enumerate(zip(targets, D)):
    Pk = clean(P0 + d)
    write_inplace(t["POSITION"], Pk - Pb)
    if i < len(names) and names[i] in OWN:
        dn = vnormals(Pk) - Nb
        dn[~used] = 0
        t["NORMAL"] = append(dn)
    else:                                                  # kitos formos – bendras nulinis (failas mažesnis)
        if zero_acc is None: zero_acc = append(np.zeros((nv, 3)))
        t["NORMAL"] = zero_acc
prim.setdefault("extras", {})["novafitFixed"] = True

J["buffers"][0]["byteLength"] = len(BIN)
js = json.dumps(J, separators=(",", ":")).encode()
while len(js) % 4: js += b" "
while len(BIN) % 4: BIN.append(0)
out = (struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(BIN)) +
       struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(BIN), 0x004E4942) + bytes(BIN))
open(path, "wb").write(out)
print("ok: sritis", len(reg), "taškų; dydis", len(out))
