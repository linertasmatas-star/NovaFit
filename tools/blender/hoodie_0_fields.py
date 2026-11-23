import sys, base64, zlib, numpy as np, io, contextlib
sys.argv=["x","/home/claude/mpfb2/src/mpfb/data","/tmp/av3d_tmp"]
src=open("tools/mh_export_glb.py").read()
g={"__name__":"exp"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src,"exp","exec"),g)
USED=g["USED"]; PART, POS = g["body_map"](g["base"]); g["PART"]=PART; g["POS"]=POS; rule=g["rule"]; P=g["PELVIS_Y"]; N=g["NECK_Y"]; X=g["X"]; Y=g["Y"]; Z=g["Z"]; FRONT=g["FRONT"]
# Džemperis: liemuo nuo ~klubų iki kaklo, rankovės iki riešo
HOOD = rule(arm=(-1, 0.965))
# Liemuo ir viršutinė kojų dalis (sėdmenys) – kerpama tiesiai pagal aukštį, kad apačia būtų lygi
body = (PART == "torso") | (PART == "leg")
HOOD[body] = np.maximum(HOOD[body], np.minimum(Y[body] - (P - 0.15), (N + 0.05) - Y[body]))
code = {"torso":0,"arm":1,"leg":0,"head":3}          # džemperiui sėdmenų sritis – liemens dalis
part = np.array([code[p] for p in PART[USED]], np.uint8)
pos = POS[USED].astype(np.float16)
f = np.clip(HOOD[USED], -1, 1).astype(np.float16)
blob = np.concatenate([np.frombuffer(f.tobytes(),np.uint8), part, np.frombuffer(pos.tobytes(),np.uint8)])
s = base64.b64encode(zlib.compress(blob.tobytes(), 9)).decode()
open("/tmp/hoodie_blob.txt","w").write(s)
print(len(s), "pelvis/neck (m, nuo žemės):", round((P - g["ground"])*0.1,4), round((N - g["ground"])*0.1,4))
