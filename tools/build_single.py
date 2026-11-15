"""Sukuria vieno failo NovaFit versiją: visi paveikslėliai įdedami į patį HTML.
Tokį failą galima atidaryti bet kur – jam nereikia aplanko `assets`.

Naudojimas:  python3 tools/build_single.py  [kur_išsaugoti.html]
"""
import base64, json, os, sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "NovaFit-vienas-failas.html")
html = open(os.path.join(root, "index.html"), encoding="utf-8").read()

def data_uri(rel):
    with open(os.path.join(root, rel), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

assets = {}
for folder, _, names in os.walk(os.path.join(root, "assets", "avatar")):
    for n in sorted(names):
        if n.endswith(".png"):
            rel = os.path.relpath(os.path.join(folder, n), root).replace(os.sep, "/")
            assets[rel] = data_uri(rel)

def swap(old, new):
    global html
    assert html.count(old) == 1, "index.html pasikeitė, nerandu: " + old
    html = html.replace(old, new)

assert 'src="assets/logo.png"' in html
html = html.replace('src="assets/logo.png"', 'src="' + data_uri("assets/logo.png") + '"')   # logotipas – keliose vietose
swap("im.src = src;", "im.src = NF_ASSETS[src] || src;")
swap("<script>\n// ====", "<script>\nconst NF_ASSETS = " + json.dumps(assets) + ";\n// ====")
open(out, "w", encoding="utf-8").write(html)
print(out, round(os.path.getsize(out) / 1e6, 1), "MB,", len(assets), "paveikslėliai")
