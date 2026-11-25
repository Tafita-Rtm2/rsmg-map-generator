#!/usr/bin/env python3
"""Retouche les trees.txt d'une carte RSMG deja generee, SANS regenerer la carte.
 - retire une part des arbres (classes T et P)         --cut 0.10  (10 %)
 - ajoute de l'herbe (classe G) a cote de l'herbe existante  --grass 1  (1 copie par touffe = x2)
Usage (Termux) :
  python patch_trees.py /sdcard/RSMG/maps/rn1_km0-4 --cut 0.10 --grass 0
Une sauvegarde trees.txt.bak est faite une seule fois a cote de chaque fichier.
"""
import argparse, glob, os, random, shutil

ap = argparse.ArgumentParser()
ap.add_argument("map_dir", help="dossier de la carte, ex: /sdcard/RSMG/maps/rn1_km0-4")
ap.add_argument("--cut", type=float, default=0.10, help="part d'arbres T/P a retirer (0.10 = 10 %%)")
ap.add_argument("--grass", type=int, default=0, help="copies d'herbe ajoutees par touffe G existante (0 = aucune)")
ap.add_argument("--spread", type=float, default=1.2, help="decalage max des copies d'herbe (m)")
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
rng = random.Random(a.seed)

files = glob.glob(os.path.join(a.map_dir, "**", "trees.txt"), recursive=True)
if not files:
    raise SystemExit("Aucun trees.txt trouve dans " + a.map_dir)
tot_cut = tot_add = 0
for f in sorted(files):
    bak = f + ".bak"
    if not os.path.exists(bak):
        shutil.copyfile(f, bak)                 # on repart toujours de l'original
    with open(bak) as fh:
        lines = fh.read().splitlines()
    out = []
    for ln in lines:
        p = ln.split()
        if ln.startswith("#") or len(p) < 6:
            out.append(ln)
            continue
        cls = p[3]
        if cls in ("T", "P") and rng.random() < a.cut:
            tot_cut += 1
            continue
        out.append(ln)
        if cls == "G" and a.grass > 0:
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            for _ in range(a.grass):
                dx, dz = rng.uniform(-a.spread, a.spread), rng.uniform(-a.spread, a.spread)
                size = max(0.3, float(p[4]) * rng.uniform(0.85, 1.15))
                out.append(f"{x + dx:.2f} {y:.2f} {z + dz:.2f} G {size:.2f} {p[5]}")   # meme pick (riz reste riz)
                tot_add += 1
    with open(f, "w") as fh:
        fh.write("\n".join(out) + "\n")
print(f"{len(files)} fichier(s) : {tot_cut} arbres retires, {tot_add} touffes d'herbe ajoutees")
