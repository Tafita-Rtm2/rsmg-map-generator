#!/usr/bin/env python3
"""Assemble le dossier final (out/), le manifeste, puis cree le .zip (deflate 9) et le .7z (lzma2 ultra)."""
import json, os, shutil, subprocess, sys, zipfile

WORK, OUT, DIST = "work", "out", "dist"
man = json.load(open(f"{WORK}/manifest.json"))
os.makedirs(f"{OUT}/trees", exist_ok=True)
for t in man["tiles"]:
    shutil.copy(f"{WORK}/tiles/tile_{t['index']:02d}_trees.csv", f"{OUT}/trees/tile_{t['index']:02d}_trees.csv")
    t["file"] = f"tiles/tile_{t['index']:02d}.gltf"
    t["trees_file"] = f"trees/tile_{t['index']:02d}_trees.csv"
shutil.copy(f"{WORK}/route.csv", f"{OUT}/route.csv")
man["format"] = {
    "geometry": "glTF 2.0 (.gltf + .bin), textures partagees dans tiles/textures et models/textures",
    "coordinates": "x = est, y = nord (metres, projection transverse Mercator centree sur l'origine du jeu), z = altitude. "
                   "Dans le glTF (Y vers le haut) : x_gltf = x, y_gltf = z, z_gltf = -y. "
                   "Les sommets d'un troncon sont relatifs a tiles[i].origin_xy ; z est absolu.",
    "object_prefixes": {"GND_": "sol roulable (collision)", "SOLID_": "murs et toits (collision)", "DECOR_": "sans collision"},
    "trees": "trees/tile_NN_trees.csv : espece,x,y,z,echelle,rotation(rad) ; x,y relatifs a l'origine du troncon. "
             "Especes : 0 eucalyptus, 1 arbre rond, 2 pin, 3 ravinala, 4 buisson. "
             "Modeles : models/tree_<euc|round|pine|ravi|bush>.gltf (pres) et tree_<...>_far.gltf (loin).",
    "route": "route.csv : axe de la route tous les 5 m (x,y,z,km)",
    "roads": "toutes les routes OSM voisines (pistes, rues) sont dans les troncons, materiaux road_minor / track",
}
man["sources"] = "OpenStreetMap (c) contributeurs OSM, licence ODbL ; Copernicus DEM GLO-30 ; textures Poly Haven (CC0) quand disponibles"
json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=1)
open(f"{OUT}/CREDITS.txt", "w").write(
    "Donnees cartographiques : (c) OpenStreetMap contributors (ODbL)\n"
    "Relief : Copernicus DEM GLO-30 (Copernicus / ESA)\n"
    "Textures photo : Poly Haven (CC0), le reste est procedural\n")

os.makedirs(DIST, exist_ok=True)
name = f"{man['road']}_km{man['km_from']:g}-{man['km_to']:g}"
zp = f"{DIST}/{name}.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for root, _, files in os.walk(OUT):
        for f in files:
            full = os.path.join(root, f)
            z.write(full, os.path.relpath(full, OUT))
try:
    subprocess.run(["7z", "a", "-t7z", "-mx=9", "-m0=lzma2", "-md=64m", "-ms=on", os.path.abspath(f"{DIST}/{name}.7z"), "."],
                   cwd=OUT, check=True, stdout=subprocess.DEVNULL)
except Exception as e:  # noqa
    print("7z indisponible :", e)
for f in sorted(os.listdir(DIST)):
    print(f, round(os.path.getsize(f"{DIST}/{f}") / 1e6, 2), "Mo")
