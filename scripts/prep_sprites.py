#!/usr/bin/env python3
"""Decoupe les planches de vegetation (photos CC0 Poly Haven, fond transparent) en sprites individuels
-> assets/veg/<nom>.png + assets/veg/meta.json (groupe, hauteur reelle en m, ratio largeur/hauteur).
Usage : prep_sprites.py <dossier des .webp> (a lancer une fois ; le resultat est commite)."""
import json, os, sys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

SRC = sys.argv[1]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "veg")
os.makedirs(OUT, exist_ok=True)
# (fichier, groupe, hauteur reelle min-max en m, hauteur max en px, nom)
SPEC = [
    ("fir_tree_01", "fir", (11, 15), 512, "fir"),
    ("pine_tree_01", "pine", (12, 17), 512, "pine"),
    ("pine_sapling_medium", "pine", (9, 13), 512, "pines"),
    ("orth_front (8)", "pine", (9, 13), 512, "pinef"),
    ("island_tree_01", "round", (7, 10), 448, "island"),
    ("jacaranda_tree", "round", (8, 11), 448, "jacaranda"),
    ("tree_small_02", "round", (6, 9), 448, "acacia"),
    ("orth_front (7)", "round", (7, 10), 448, "arbre7"),
    ("orth_side (5)", "round", (7, 10), 448, "arbre5"),
    ("shrub_04", "bush", (1.4, 2.4), 256, "arbuste"),
    ("orth_side (1)", "grass", (0.55, 0.9), 128, "touffe"),
    ("orth_front (1)", "grass", (0.6, 1.0), 128, "herbe"),
    ("orth_front", "grass", (0.7, 1.2), 128, "herbe_haute"),
    ("orth_side", "grass", (0.5, 0.8), 128, "herbe_f"),
    ("orth_front (2)", "bush", (0.7, 1.1), 128, "feuille"),
    ("weed_plant_02", "weed", (0.3, 0.55), 192, "plante"),
    ("orth_front (3)", "weed", (0.5, 0.9), 128, "tige"),
]


def components(im, grow):
    a = np.array(im)[..., 3] > 30
    lab, n = ndi.label(ndi.binary_dilation(a, iterations=grow))
    boxes = ndi.find_objects(lab)
    out = []
    for i, sl in enumerate(boxes):
        m = (lab[sl] == i + 1) & ndi.binary_dilation(a[sl], iterations=grow)
        if m.sum() < 400:
            continue
        out.append(sl)
    out.sort(key=lambda s: s[1].start)
    return out


def bleed(rgba):
    """Etend la couleur dans les pixels transparents (evite les halos sombres/magenta au filtrage)."""
    a = rgba[..., 3] > 8
    rgb = rgba[..., :3].astype(float)
    idx = ndi.distance_transform_edt(~a, return_distances=False, return_indices=True)
    rgb = rgb[idx[0], idx[1]]
    return np.dstack([rgb, rgba[..., 3]]).astype(np.uint8)


meta = {}
for fn, grp, (h0, h1), maxh, base in SPEC:
    p = os.path.join(SRC, fn + ".webp")
    if not os.path.exists(p):
        print("absent :", p)
        continue
    im = Image.open(p).convert("RGBA")
    comps = components(im, 3 if im.height > 300 else 1)
    for k, sl in enumerate(comps):
        c = im.crop((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop))
        bb = c.getchannel("A").point(lambda v: 255 if v > 30 else 0).getbbox()
        if not bb:
            continue
        c = c.crop(bb)
        if c.height < 40 or c.width < 16:
            continue
        s = min(1.0, maxh / c.height)
        c = c.resize((max(8, int(c.width * s)), max(8, int(c.height * s))), Image.LANCZOS)
        arr = bleed(np.array(c))
        name = f"{base}{k}" if len(comps) > 1 else base
        Image.fromarray(arr).save(os.path.join(OUT, f"{name}.png"), optimize=True)
        meta[name] = {"group": grp, "h": [h0, h1], "aspect": round(c.width / c.height, 3)}
        print(name, c.size)
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
