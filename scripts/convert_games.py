#!/usr/bin/env python3
"""Transforme la carte generee (zip RSMG) en cartes compatibles avec les autres jeux :

  <nom>_ProtonBus.zip   dossier de carte Proton Bus complet (mapModVersion=3) : .3ds, collisions _gencol_, points de depart,
                        arrets de bus avec declencheur et passagers, trafic (av1.000...). Sans arbres ni herbe.
  <nom>_BUSSID.zip      .obj + .mtl + textures par troncon, maillages de collision, points : a donner au BUSSID Mod Editor
                        (le .bussidmod lui-meme ne peut etre ecrit que par l'outil de Maleo).
  rapport.txt           controle de la carte RSMG (fichiers .3ds lisibles, textures presentes, comptes).
La carte RSMG d'origine reste la carte principale pour Simulator MG."""
import argparse, glob, math, os, re, shutil, struct, subprocess, sys, tempfile, unicodedata, zipfile
import numpy as np
from PIL import Image
from export_rsmg import write3ds


# ----------------------------------------------------------------------------- lecture 3DS
def read3ds(path):
    b = open(path, "rb").read()
    mats, objs = [], []

    def walk(o, end, ctx):
        while o < end:
            cid, ln = struct.unpack_from("<HI", b, o)
            body = o + 6
            if cid in (0x4D4D, 0x3D3D):
                walk(body, o + ln, ctx)
            elif cid == 0xAFFF:
                z = body + 6
                mats.append(b[z:b.index(b"\0", z)].decode("latin-1"))
            elif cid == 0x4000:
                z = b.index(b"\0", body)
                ctx = {"name": b[body:z].decode("latin-1"), "V": None, "UV": None, "F": None, "grp": []}
                walk(z + 1, o + ln, ctx)
                objs.append(ctx)
            elif cid == 0x4100:
                walk(body, o + ln, ctx)
            elif cid == 0x4110:
                n = struct.unpack_from("<H", b, body)[0]
                ctx["V"] = np.frombuffer(b, "<f4", n * 3, body + 2).reshape(-1, 3).astype(float)
            elif cid == 0x4140:
                n = struct.unpack_from("<H", b, body)[0]
                ctx["UV"] = np.frombuffer(b, "<f4", n * 2, body + 2).reshape(-1, 2).astype(float)
            elif cid == 0x4120:
                m = struct.unpack_from("<H", b, body)[0]
                ctx["F"] = np.frombuffer(b, "<u2", m * 4, body + 2).reshape(-1, 4)[:, :3].astype(int)
                p = body + 2 + m * 8
                while p < o + ln:
                    c2, l2 = struct.unpack_from("<HI", b, p)
                    if c2 == 0x4130:
                        z2 = b.index(b"\0", p + 6)
                        k = struct.unpack_from("<H", b, z2 + 1)[0]
                        ctx["grp"].append((b[p + 6:z2].decode("latin-1"), np.frombuffer(b, "<u2", k, z2 + 3).astype(int)))
                    p += l2
            o += ln
    walk(0, len(b), None)
    return mats, objs


def obj_tuple(o):
    """-> tuple pour write3ds : (nom, V, UV, F, indice de matiere par face, noms de matieres)"""
    names = [g[0] for g in o["grp"]] or ["route"]
    fm = np.zeros(len(o["F"]), int)
    for i, (_, idx) in enumerate(o["grp"]):
        fm[idx] = i
    return (o["name"], o["V"], o["UV"], o["F"], fm, names)


def ascii_name(s, fallback):
    t = unicodedata.normalize("NFD", s)
    t = re.sub(r"[^A-Za-z0-9 _-]", "", t).strip()
    t = re.sub(r"\s+", " ", t)
    return t[:32] if t else fallback


# ----------------------------------------------------------------------------- formes Proton Bus
def tri_obj(name, x, y, z, mat):
    s = 0.05
    return (name, np.array([[x - s, -z, y], [x + s, -z, y], [x, -z, y + s]]), None, np.array([[0, 1, 2]]), np.zeros(1, int), [mat])


def quad_obj(name, x, y, z, mat):
    s = 0.15
    V = np.array([[x - s, -(z - s), y], [x + s, -(z - s), y], [x + s, -(z + s), y], [x - s, -(z + s), y]])
    return (name, V, None, np.array([[0, 1, 2], [0, 2, 3]]), np.zeros(2, int), [mat])


def box_obj(name, cx, cy, cz, hf, hr, h, heading, mat):
    rad = math.radians(heading)
    fx, fz = math.sin(rad), math.cos(rad)
    rx, rz = fz, -fx
    vs = []
    for yy in (0, 1):
        for sf in (-1, 1):
            for sr in (-1, 1):
                vs.append([cx + fx * hf * sf + rx * hr * sr, -(cz + fz * hf * sf + rz * hr * sr), cy + yy * h])
    F = np.array([0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 1, 5, 7, 1, 7, 3, 3, 7, 6, 3, 6, 2, 2, 6, 4, 2, 4, 0]).reshape(-1, 3)
    return (name, np.array(vs), None, F, np.zeros(12, int), [mat])


def resample(pts, step):
    d = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0], np.cumsum(d)])
    if s[-1] < step:
        return pts
    t = np.arange(0, s[-1], step)
    return np.column_stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", default="games")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(a.zip) as z:
        names = z.namelist()
        dest = tmp if any(n.startswith("RSMG/maps/") for n in names) else f"{tmp}/RSMG/maps"
        z.extractall(dest)
    maps = glob.glob(f"{tmp}/RSMG/maps/*.map.txt")
    if not maps:
        sys.exit("Archive RSMG invalide : RSMG/maps/<nom>.map.txt introuvable")
    name = os.path.basename(maps[0])[:-len(".map.txt")]
    base = f"{tmp}/RSMG/maps/{name}"
    tiles = f"{base}/tiles/{name}"
    os.makedirs(a.out, exist_ok=True)
    report = []

    # ---- lecture + controle de toutes les parties .3ds
    parts = {}
    tex_missing = set()
    nobj = nfaces = 0
    for f in sorted(glob.glob(f"{tiles}/part_*.3ds")):
        mats, objs = read3ds(f)
        parts[f] = (mats, objs)
        for m in mats:
            if not os.path.exists(f"{base}/textures/{m}.png"):
                tex_missing.add(m)
        nobj += len(objs)
        nfaces += sum(len(o["F"]) for o in objs if o["F"] is not None)
    report.append(f"RSMG {name} : {len(parts)} fichier(s) .3ds, {nobj} objets, {nfaces} faces")
    report.append("textures manquantes : " + (", ".join(sorted(tex_missing)) if tex_missing else "aucune"))
    trees = sum(1 for _ in glob.glob(f"{tiles}/c*/trees.txt"))
    report.append(f"fichiers de vegetation : {trees}")

    # ---- chemins de trafic (objets Car1_NNN) + arrets + depart
    paths = []
    for f, (mats, objs) in parts.items():
        for o in objs:
            if o["name"].startswith("Car1_") and o["V"] is not None and len(o["V"]) >= 3:
                paths.append(o["V"])
    allpts = np.vstack(paths) if paths else np.zeros((0, 3))
    from scipy.spatial import cKDTree
    ptree = cKDTree(allpts[:, :2]) if len(allpts) else None

    def drive_h(x, z):                 # x, z du repere jeu ; la hauteur est lue sur le chemin de trafic le plus proche
        if ptree is None:
            return 0.0
        _, i = ptree.query([x, -z])
        return float(allpts[i, 2]) - 0.1

    stops = []
    sf = f"{base}/rsmg_stops.txt"
    if os.path.exists(sf):
        for line in open(sf, encoding="utf-8"):
            if line.startswith("#") or not line.strip():
                continue
            p = [t.strip() for t in line.split(",")]
            if len(p) >= 5:
                stops.append({"name": p[0], "x": float(p[1]), "z": float(p[2]), "cap": float(p[3]), "right": p[4].upper() == "D",
                              "depart": "depart" in [t.lower() for t in p[5:]]})
    spawn = None
    spf = f"{base}/spawnpoints.txt"
    if os.path.exists(spf):
        for line in open(spf, encoding="utf-8"):
            q = line.split()
            if len(q) >= 5 and q[0].lower() == "depart":
                spawn = (float(q[1]), float(q[2]), float(q[3]), float(q[4]))

    # =============================================================================== Proton Bus
    pn = ascii_name(name, "carte")
    proot = f"{tmp}/_proton"
    pbase = f"{proot}/{pn}"
    ptiles = f"{pbase}/tiles/{pn}"
    os.makedirs(ptiles, exist_ok=True)
    shutil.copytree(f"{base}/textures", f"{pbase}/textures")
    if os.path.exists(f"{base}/preview.png"):
        shutil.copy(f"{base}/preview.png", f"{pbase}/preview.png")
    for f, (mats, objs) in parts.items():
        keep = [obj_tuple(o) for o in objs if not o["name"].startswith("Car1_") and o["F"] is not None]
        if not keep:
            continue
        used = []
        for o in keep:
            for m in o[5]:
                if m not in used:
                    used.append(m)
        open(f"{ptiles}/{os.path.basename(f)}", "wb").write(write3ds(used, keep))
    first_mat = next((m for mats, _ in parts.values() for m in mats), "route")
    os.makedirs(f"{ptiles}/aivehicles", exist_ok=True)
    os.makedirs(f"{ptiles}/busstops", exist_ok=True)
    extra = []
    npaths = npts = 0
    for V in sorted(paths, key=lambda v: -np.hypot(*np.diff(v[:, :2], axis=0).T).sum()):
        if npaths >= 60 or npts >= 1500:
            break
        xy = resample(V[:, :2], 30.0)
        if len(xy) < 3 or npts + len(xy) > 1500:
            continue
        pref = f"av{npaths + 1}"
        zs = np.interp(np.linspace(0, 1, len(xy)), np.linspace(0, 1, len(V)), V[:, 2])
        for k, (q, zz) in enumerate(zip(xy, zs)):
            extra.append(tri_obj(f"{pref}.{k:03d}", q[0], zz, -q[1], first_mat))
        npts += len(xy)
        npaths += 1
        open(f"{ptiles}/aivehicles/{pref}.txt", "w").write(
            f"[automatic_setup]\nenabled=1\nloop=0\nreverse=0\n\n[from_3d]\nreadFrom3D=1\nprefix={pref}\nmaxPathsToCheck={len(xy) + 5}\n\n"
            "[defaults]\nisSpawner=1\nisBusSpawner=0\nspawnInterval=9\nmaxSpeed=40\nspeedMultiplier=1\n")
    for i, st in enumerate(stops):
        sid = f"zzstop{i + 1}"
        y = drive_h(st["x"], st["z"])
        rad = math.radians(st["cap"])
        fx, fz = math.sin(rad), math.cos(rad)
        rx, rz = fz, -fx
        side = 1.0 if st["right"] else -1.0
        extra.append(box_obj(f"{sid}_trigger", st["x"], y - 0.3, st["z"], 7.0, 3.2, 3.6, st["cap"], first_mat))
        for k in range(6):
            al = (k - 2.5) * 1.3
            extra.append(quad_obj(f"{sid}.{k:03d}", st["x"] + fx * al + rx * side * 3.6, y + 0.05, st["z"] + fz * al + rz * side * 3.6, first_mat))
        open(f"{ptiles}/busstops/{sid}.txt", "w").write(
            f"[busstop]\nname={ascii_name(st['name'], 'Arret ' + str(i + 1))}\nisLeft={0 if st['right'] else 1}\npaxAmount=12\n\n"
            f"[from_3d]\nreadFrom3D=1\nprefix={sid}\nmaxPathsToCheck=30\ndefPaxRotY=90\n")
    if extra:
        open(f"{ptiles}/part_zz_game.3ds", "wb").write(write3ds([first_mat], extra))
    entries, names = [], set()
    if spawn:
        entries.append(("Depart", spawn[0], spawn[1] + 0.3, spawn[2], spawn[3]))
        names.add("depart")
    for i, st in enumerate(stops):
        if not st["depart"]:
            continue
        nm = ascii_name(st["name"], f"Arret {i + 1}")
        if nm.lower() in names:
            nm = f"{nm} {i + 1}"
        names.add(nm.lower())
        entries.append((nm, st["x"], drive_h(st["x"], st["z"]) + 0.3, st["z"], st["cap"]))
    open(f"{ptiles}/entrypoints_list.txt", "w").write("\n".join(e[0] for e in entries) + "\n")
    ep = []
    for n_, e in enumerate(entries):
        rot = ((180.0 - e[4]) % 360.0 + 360.0) % 360.0
        ep.append(f"[entrypoint_{n_ + 1}]\nname={e[0]}\nposX={e[1]:.3f}\nposY={e[2]:.3f}\nposZ={-e[3]:.3f}\nrotX=0\nrotY={rot:.3f}\nrotZ=0\n\n")
    open(f"{ptiles}/entrypoints.txt", "w").write("".join(ep))
    for e in entries:
        os.makedirs(f"{pbase}/dest/{e[0]}", exist_ok=True)
        Image.new("RGB", (1024, 128), (24, 24, 24)).save(f"{pbase}/dest/{e[0]}/1.png")
        Image.new("RGB", (256, 128), (24, 24, 24)).save(f"{pbase}/dest/{e[0]}/placa1.png")
    os.makedirs(f"{pbase}/skins/0/pbc", exist_ok=True)
    open(f"{pbase}/skins/0/pbc/LISEZMOI.txt", "w").write("Mets ici les skins PNG des bus du trafic (facultatif).\n")
    open(f"{proot}/{pn}.map.txt", "w").write(f"[map]\nbaseDir={pn}\nmodelsDir={pn}\ntextures=textures\nmapModVersion=3\npreview=preview.png\n")
    open(f"{pbase}/LISEZMOI_ProtonBus.txt", "w", encoding="utf-8").write(
        f"CARTE {pn} POUR PROTON BUS SIMULATOR (generee automatiquement par rsmg-map-generator)\n\n"
        "Installation 1 : dans Proton Bus, ecran de choix de la carte, bouton +, choisis ce ZIP sans le decompresser.\n"
        "Installation 2 : decompresse le ZIP dans Android/data/com.viamep.protonbussimulator/files/maps\n"
        f"  (il doit y avoir {pn}.map.txt et le dossier {pn} directement dans maps).\n\n"
        "Inclus : geometrie .3ds, collisions automatiques (objets _gencol_), points de depart, arrets de bus, trafic.\n"
        "Non inclus : arbres et herbe, pietons, feux, eclairage, GPS. Les panneaux de destination sont des images vides.\n"
        "Proton Bus charge la carte en entier : une carte de plus de 5 km peut etre lourde pour un telephone.\n")
    pz = f"{a.out}/{name}_ProtonBus.zip"
    with zipfile.ZipFile(pz, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for rt, _, fs in os.walk(proot):
            for fn in fs:
                full = os.path.join(rt, fn)
                z.write(full, os.path.relpath(full, proot))
    report.append(f"Proton Bus : {npaths} chemin(s) de trafic ({npts} points), {len(stops)} arret(s), {len(entries)} point(s) de depart")

    # =============================================================================== BUSSID (modele pour le BUSSID Mod Editor)
    bdir = f"{tmp}/_bussid/{name}_bussid"
    os.makedirs(f"{bdir}/textures", exist_ok=True)
    mtl_used = set()
    bymap = {}
    for f, (mats, objs) in parts.items():
        m = re.search(r"part_c(\d+)_", os.path.basename(f))
        bymap.setdefault(int(m.group(1)) if m else 0, []).append((mats, objs))
    ncol = 0
    for ck, lst in sorted(bymap.items()):
        vis, col = [], []
        vbase = 0
        vlines, flines, tlines = [], [], []
        cvl, cfl, cbase = [], [], 0
        tbase = 0
        for mats, objs in lst:
            for o in objs:
                if o["F"] is None or o["name"].startswith("Car1_"):
                    continue
                V = o["V"]
                G = np.column_stack([V[:, 0], V[:, 2], -V[:, 1]])        # 3DS (x, y, z haut) -> modele (x, hauteur, z)
                vlines.append("\n".join(f"v {p[0]:.3f} {p[1]:.3f} {p[2]:.3f}" for p in G))
                UV = o["UV"] if o["UV"] is not None else np.zeros((len(V), 2))
                tlines.append("\n".join(f"vt {t[0]:.4f} {t[1]:.4f}" for t in UV))
                for mname, idx in o["grp"]:
                    mtl_used.add(mname)
                    flines.append(f"usemtl {mname}")
                    flines.append("\n".join(f"f {F[0] + 1 + vbase}/{F[0] + 1 + tbase} {F[1] + 1 + vbase}/{F[1] + 1 + tbase} {F[2] + 1 + vbase}/{F[2] + 1 + tbase}"
                                            for F in o["F"][idx]))
                if o["name"].startswith(("_gencol_", "solid")):
                    cvl.append("\n".join(f"v {p[0]:.3f} {p[1]:.3f} {p[2]:.3f}" for p in G))
                    cfl.append("\n".join(f"f {F[0] + 1 + cbase} {F[1] + 1 + cbase} {F[2] + 1 + cbase}" for F in o["F"]))
                    cbase += len(V)
                    ncol += len(o["F"])
                vbase += len(V)
                tbase += len(V)
        open(f"{bdir}/{name}_c{ck:02d}.obj", "w").write(f"# {name} troncon {ck} (Y = hauteur)\nmtllib {name}.mtl\no {name}_c{ck:02d}\n" +
                                                      "\n".join(vlines) + "\n" + "\n".join(tlines) + "\n" + "\n".join(flines) + "\n")
        open(f"{bdir}/{name}_c{ck:02d}_collision.obj", "w").write(f"# collision troncon {ck}\no {name}_c{ck:02d}_collision\n" + "\n".join(cvl) + "\n" + "\n".join(cfl) + "\n")
    with open(f"{bdir}/{name}.mtl", "w") as f:
        for m in sorted(mtl_used):
            f.write(f"newmtl {m}\nKd 1 1 1\nmap_Kd textures/{m}.png\n")
            if os.path.exists(f"{base}/textures/{m}.png"):
                shutil.copy(f"{base}/textures/{m}.png", f"{bdir}/textures/{m}.png")
    with open(f"{bdir}/{name}_points.txt", "w") as f:
        f.write("# x, y (hauteur), z, cap en degres (0 = vers +Z)\n")
        if spawn:
            f.write(f"Depart, {spawn[0]:.3f}, {spawn[1]:.3f}, {spawn[2]:.3f}, {spawn[3]:.1f}\n")
        for st in stops:
            f.write(f"Arret {ascii_name(st['name'], 'arret')}, {st['x']:.3f}, {drive_h(st['x'], st['z']):.3f}, {st['z']:.3f}, {st['cap']:.1f}\n")
    open(f"{bdir}/LISEZMOI_BUSSID.txt", "w", encoding="utf-8").write(
        f"CARTE {name} POUR BUSSID (generee automatiquement par rsmg-map-generator)\n\n"
        "Le fichier .bussidmod ne peut etre fabrique que par le BUSSID Mod Editor de Maleo (format non public).\n"
        "Ce ZIP contient ce qu'il faut pour le lui donner :\n"
        f"  {name}_cNN.obj / {name}.mtl / textures/   modele visible, un fichier par troncon de 2 km\n"
        f"  {name}_cNN_collision.obj                  maillage de collision (sols, routes, murs, batiments)\n"
        f"  {name}_points.txt                         depart et arrets (x, y, z, cap)\n\n"
        "Etapes : BUSSID Mod Editor > Map Mod > importer les .obj (un troncon a la fois) puis declarer les *_collision.obj.\n"
        "Si la carte apparait en miroir, inverse l'axe X a l'import.\n")
    bz = f"{a.out}/{name}_BUSSID.zip"
    with zipfile.ZipFile(bz, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for rt, _, fs in os.walk(f"{tmp}/_bussid"):
            for fn in fs:
                full = os.path.join(rt, fn)
                z.write(full, os.path.relpath(full, f"{tmp}/_bussid"))
    report.append(f"BUSSID : {len(bymap)} troncon(s) .obj, {len(mtl_used)} materiaux, collision {ncol} faces")

    shutil.copy(a.zip, f"{a.out}/{os.path.basename(a.zip)}")
    open(f"{a.out}/rapport.txt", "w", encoding="utf-8").write("\n".join(report) + "\n")
    print("\n".join(report))
    for fn in sorted(os.listdir(a.out)):
        print(fn, round(os.path.getsize(f"{a.out}/{fn}") / 1e6, 2), "Mo")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
