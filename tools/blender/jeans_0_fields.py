# Džinsų sritis: nuo juosmens (šiek tiek žemiau bambos) iki kulkšnies. Paleisti iš repozitorijos šaknies.
import sys, base64, zlib, numpy as np, io, contextlib
sys.argv=["x","/home/claude/mpfb2/src/mpfb/data","/tmp/av3d_tmp"]
src=open("tools/mh_export_glb.py").read()
g={"__name__":"exp"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src,"exp","exec"),g)
USED=g["USED"]; PART, POS = g["body_map"](g["base"]); g["PART"]=PART; g["POS"]=POS; rule=g["rule"]; P=g["PELVIS_Y"]; Y=g["Y"]
F = rule(torso=(-99, P + 0.75), leg=(-1, 1.3))
# apačia kerpama tiesiai pagal aukštį (šiek tiek virš kulkšnies), ne pagal kojos kryptį
J = g["J"]; A = (J(g["base"], "l-ankle")[1] + J(g["base"], "r-ankle")[1]) / 2
legm = PART == "leg"
F[legm] = np.minimum(F[legm], Y[legm] - (A + 0.12))
code = {"torso":0,"arm":1,"leg":2,"head":3}
part = np.array([code[p] for p in PART[USED]], np.uint8)
pos = POS[USED].astype(np.float16)
f = np.clip(F[USED], -1, 1).astype(np.float16)
blob = np.concatenate([np.frombuffer(f.tobytes(),np.uint8), part, np.frombuffer(pos.tobytes(),np.uint8)])
open("tools/blender/jeans_data.b64","w").write(base64.b64encode(zlib.compress(blob.tobytes(), 9)).decode())
print("ok", (f > 0).sum())
