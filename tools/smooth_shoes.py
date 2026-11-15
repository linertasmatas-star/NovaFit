"""Išlygina batų sluoksnius: panaikina atskirus kojų pirštus.

Batų sluoksniai (sneakers.png, boots.png) sugeneruoti „aptraukiant“ pėdą, todėl
matėsi pirštai. Šis skriptas apatinę bato dalį (nosį) padaro lygią ir apvalią.

Naudojimas:  python3 tools/smooth_shoes.py
Paleisti vieną kartą po `mh_wardrobe.py` (jei sluoksniai generuojami iš naujo).
"""
import glob, os
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZONE, RAMP = 27, 8          # kiek taškų nuo apačios lyginam; perėjimo aukštis


def smooth_shoe(lum, alpha):
    """lum, alpha – vieno bato (pusės paveikslėlio) masyvai 0..1. Grąžina pataisytus."""
    ys = np.nonzero(alpha.max(axis=1) > 0.08)[0]
    if not len(ys):
        return lum, alpha
    bottom = ys.max()
    top = max(ys.min(), bottom - ZONE)
    rows = np.arange(lum.shape[0])
    # svoris: 0 virš zonos, tolygiai iki 1 zonoje
    wgt = np.clip((rows - top) / RAMP, 0, 1)[:, None] * (rows <= bottom + 2)[:, None]

    # Balta pado juostelė (sportbačiams) – išsaugom tokią, kokia buvo
    sole = (lum > 0.93) & (alpha > 0.5) & (rows[:, None] > bottom - 12)
    sole_rows = np.nonzero(sole.sum(axis=1) > 12)[0]

    # 1) Forma: užpildom įdubas tarp pirštų ir suapvalinam kontūrą
    closed = ndi.grey_closing(alpha, size=(7, 9))
    closed = ndi.gaussian_filter(closed, 0.6)
    new_alpha = np.maximum(alpha, closed * wgt)

    # 2) Šešėliai: vietoj pirštų – lygus paviršius (išlyginam tik bato viduje)
    src = alpha * ~sole
    blur = ndi.gaussian_filter(lum * src, (3.0, 6.0)) / np.maximum(ndi.gaussian_filter(src, (3.0, 6.0)), 1e-4)
    # švelnus apvalumas: kraštai šiek tiek tamsesni nei vidurys
    cols = np.nonzero(new_alpha.max(axis=0) > 0.5)[0]
    cx, half = (cols.min() + cols.max()) / 2, max(1.0, (cols.max() - cols.min()) / 2)
    round_shade = 1 - 0.16 * ((np.arange(lum.shape[1]) - cx) / half) ** 2
    new_lum = lum * (1 - wgt) + np.clip(blur * round_shade[None, :], 0, 1) * wgt

    if len(sole_rows):      # pado juostelę nupiešiam iš naujo – lygią, per visą bato plotį
        y0, y1 = sole_rows.min(), sole_rows.max()
        band = (rows[:, None] >= y0) & (rows[:, None] <= y1) & (new_alpha > 0.5)
        new_lum[band] = 0.98
    return new_lum, new_alpha


def process(path):
    im = np.asarray(Image.open(path).convert("RGBA")).astype(float) / 255
    lum, alpha = im[..., 0].copy(), im[..., 3].copy()
    mid = lum.shape[1] // 2
    for sl in (slice(0, mid), slice(mid, None)):
        lum[:, sl], alpha[:, sl] = smooth_shoe(lum[:, sl], alpha[:, sl])
    g = np.round(lum * 255).astype(np.uint8)
    out = np.dstack([g, g, g, np.round(alpha * 255).astype(np.uint8)])
    Image.fromarray(out, "RGBA").save(path, optimize=True)


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(ROOT, "assets", "avatar", "*", "sneakers.png")) +
                   glob.glob(os.path.join(ROOT, "assets", "avatar", "*", "boots.png")))
    for f in files:
        process(f)
    print("Išlyginta:", len(files), "failų")
