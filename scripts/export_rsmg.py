#!/usr/bin/env python3
"""Exporte la carte au format RSMG du jeu (celui que charge deja Simulator MG, comme les cartes automatiques) :

  RSMG/maps/<nom>.map.txt                         descripteur
  RSMG/maps/<nom>/preview.png, spawnpoints.txt, rsmg_stops.txt
  RSMG/maps/<nom>/textures/<tag>.png
  RSMG/maps/<nom>/tiles/<nom>/part_cNN_KK.3ds     geometrie (sols = objets _gencol_, murs/toits = objets pleins)
  RSMG/maps/<nom>/tiles/<nom>/cNN/trees.txt       arbres (T/P), buissons (B), herbe (G)
  RSMG/maps/<nom>/tiles/<nom>/aivehicles/Car1.txt + objets Car1_NNN (chemins de trafic, une voie par sens)

Repere : le jeu (x, y haut, z) devient en 3DS (x, -z, y) ; ici x = est, y = nord, z = altitude (= repere 3DS)."""
import json, math, os, re, struct, subprocess, zipfile
import numpy as np
from PIL import Image, ImageDraw

W = "work"
ROOT = "rsmg_out"
DIST = "dist"
MAX_VERTS = 60000        # par objet 3DS (limite 65535)
MAX_FACES = 20000        # par morceau avant regroupement
MAX_TRIS_FILE = 120000   # par fichier .3ds
MAX_TREES = 10**9        # arbres (T/P) sur toute la carte
MAX_UNDER = 10**9       # buissons + herbe sur toute la carte (herbe dense au bord des routes)

KEY = {"grass": "herbe", "laterite": "laterite", "rock": "roche", "forest": "sol_foret", "paddy": "riziere",
       "mud": "boue", "water": "eau", "shoulder": "bord", "road_main": "route", "road_minor": "route_sec",
       "track": "piste", "wall_white": "mur_blanc", "wall_ochre": "mur_ocre", "wall_pink": "mur_rose",
       "wall_brick": "mur_brique", "wall_green": "mur_vert", "roof_tile": "toit_tuile", "roof_tin": "toit_tole"}
TEXSIZE = {"road_main": (512, 2048), "wall_white": (256, 256), "wall_ochre": (256, 256), "wall_pink": (256, 256),
           "wall_brick": (256, 256), "wall_green": (256, 256), "roof_tile": (256, 256), "roof_tin": (256, 256),
           "road_minor": (256, 256), "track": (256, 256), "cable": (16, 16), "ligne": (16, 16), "water_surface": (256, 256), "pole": (256, 256), "beton": (256, 256), "bois": (256, 256), "trottoir": (256, 256)}
GROUND = {"grass", "laterite", "rock", "forest", "paddy", "mud", "shoulder", "falaise", "water"}
PREFIX = {"GND": "_gencol_ground", "SOLID": "solid", "DECOR": "_fake_decor"}


# ----------------------------------------------------------------------------- ecriture 3DS
def chunk(cid, body):
    return struct.pack("<HI", cid, 6 + len(body)) + body


def cstr(s):
    return s.encode("latin-1", "replace") + b"\0"


def color_chunk(cid, rgb):
    return chunk(cid, chunk(0x0011, bytes(rgb)))


def material_chunk(name, tex):
    b = chunk(0xA000, cstr(name)) + color_chunk(0xA010, (127, 127, 127)) + color_chunk(0xA020, (255, 255, 255)) \
        + color_chunk(0xA030, (0, 0, 0))
    if tex:
        b += chunk(0xA200, chunk(0x0030, struct.pack("<H", 100)) + chunk(0xA300, cstr(tex)))
    return chunk(0xAFFF, b)


def object_chunk(name, V, UV, F, fmat, mat_names):
    n, m = len(V), len(F)
    assert n <= 65535 and m <= 65535, name
    mesh = chunk(0x4110, struct.pack("<H", n) + V.astype("<f4").tobytes())
    fb = struct.pack("<H", m) + np.column_stack([F, np.zeros(m, int)]).astype("<u2").tobytes()
    for mi, mn in enumerate(mat_names):
        idx = np.where(fmat == mi)[0]
        if len(idx):
            fb += chunk(0x4130, cstr(mn) + struct.pack("<H", len(idx)) + idx.astype("<u2").tobytes())
    mesh += chunk(0x4120, fb)
    if UV is not None:
        mesh += chunk(0x4140, struct.pack("<H", n) + UV.astype("<f4").tobytes())
    return chunk(0x4000, cstr(name) + chunk(0x4100, mesh))


def write3ds(mats, objects):
    edit = chunk(0x3D3E, struct.pack("<I", 3))
    for mname in mats:
        edit += material_chunk(mname, mname + ".png")
    for o in objects:
        edit += object_chunk(*o)
    return chunk(0x4D4D, chunk(0x0002, struct.pack("<I", 3)) + chunk(0x3D3D, edit))


# ----------------------------------------------------------------------------- outils
def pieces(V, UV, F):
    for s in range(0, len(F), MAX_FACES):
        fs = F[s:s + MAX_FACES]
        used, inv = np.unique(fs.ravel(), return_inverse=True)
        yield V[used], UV[used], inv.reshape(-1, 3)


def main():
    man = json.load(open(f"{W}/manifest.json"))
    osm = json.load(open(f"{W}/osm.json"))
    R = np.loadtxt(f"{W}/route.csv", delimiter=",", skiprows=1)      # x, y, z, km
    ox, oy = R[0, 0], R[0, 1]
    name = re.sub(r"[^a-z0-9_-]+", "_", f"{man['road']}_km{man['km_from']:g}-{man['km_to']:g}".lower()).strip("_")
    base = f"{ROOT}/RSMG/maps"
    tiles_dir = f"{base}/{name}/tiles/{name}"
    os.makedirs(f"{base}/{name}/textures", exist_ok=True)
    os.makedirs(tiles_dir, exist_ok=True)
    rng = np.random.default_rng(5)

    n = len(R)
    T = np.gradient(R[:, :2], axis=0)
    T /= np.maximum(np.hypot(T[:, 0], T[:, 1]), 1e-9)[:, None]
    RIGHT = np.column_stack([T[:, 1], -T[:, 0]])
    used_mats = set()
    traffic_no = 0
    trees_all = []
    for t in man["tiles"]:
        ti = t["index"]
        tx, ty = t["origin_xy"]
        data = json.load(open(f"{W}/tiles/tile_{ti:02d}.json"))
        objects, vcount = [], 0          # objets en cours
        cur_parts, cur_v = [], 0
        out_objs = []
        counters = {"GND": 0, "SOLID": 0, "DECOR": 0}

        def flush_obj(kind):
            nonlocal cur_parts, cur_v
            if not cur_parts:
                return
            mats = sorted({p[0] for p in cur_parts})
            V = np.vstack([p[1] for p in cur_parts])
            UV = np.vstack([p[2] for p in cur_parts])
            offs = np.cumsum([0] + [len(p[1]) for p in cur_parts])
            F = np.vstack([p[3] + offs[i] for i, p in enumerate(cur_parts)])
            fm = np.concatenate([np.full(len(p[3]), mats.index(p[0])) for p in cur_parts])
            nm = f"{PREFIX[kind]}{counters[kind]}"
            counters[kind] += 1
            out_objs.append((nm, V, UV, F, fm, mats))
            cur_parts, cur_v = [], 0

        for kind in ("GND", "SOLID", "DECOR"):
            for mat, d in data.items():
                if d["kind"] != kind:
                    continue
                V = np.array(d["v"], float).reshape(-1, 3)
                V[:, 0] += tx - ox
                V[:, 1] += ty - oy
                UV = np.array(d["uv"], float).reshape(-1, 2)
                # V en 'haut = 1' tel quel : le jeu relit 1 - V (sinon textes et logos sont a l'envers)
                F = np.array(d["f"], int).reshape(-1, 3)
                key = KEY.get(mat, mat)
                used_mats.add((mat, key))
                for pv, puv, pf in pieces(V, UV, F):
                    if cur_v + len(pv) > MAX_VERTS:
                        flush_obj(kind)
                    cur_parts.append((key, pv, puv, pf))
                    cur_v += len(pv)
            flush_obj(kind)

        # trafic : une voie par sens (conduite a droite), points tous les 10 m, objets Car1_NNN
        s0, s1 = t["station_from"], t["station_to"]
        idx = np.arange(s0, min(s1 + 1, n), 2)
        if len(idx) >= 4:
            for lane, order in ((1.75, 1), (-1.75, -1)):
                ii = idx if order == 1 else idx[::-1]
                pts = R[ii, :2] + RIGHT[ii] * lane
                z = R[ii, 2] + 0.1
                VV = np.column_stack([pts[:, 0] - ox, pts[:, 1] - oy, z])
                k = np.arange(len(VV) - 1)
                FF = np.column_stack([k, k + 1, k + 1])
                traffic_no += 1
                first = out_objs[0][5][0] if out_objs else "route"
                out_objs.append((f"Car1_{traffic_no:03d}", VV, None, FF, np.zeros(len(FF), int), [first]))

        # regroupement en fichiers part_cNN_KK.3ds
        fileno, cur, tris = 0, [], 0
        def write_file():
            nonlocal fileno, cur, tris
            if not cur:
                return
            mats = []
            for o in cur:
                for mn in o[5]:
                    if mn not in mats:
                        mats.append(mn)
            fixed = []
            for (nm, V, UV, F, fm, mn) in cur:
                fixed.append((nm, V, UV, F, fm, mn))
            with open(f"{tiles_dir}/part_c{ti:02d}_{fileno:02d}.3ds", "wb") as f:
                f.write(write3ds(mats, fixed))
            fileno += 1
            cur, tris = [], 0
        for o in out_objs:
            if tris + len(o[3]) > MAX_TRIS_FILE:
                write_file()
            cur.append(o)
            tris += len(o[3])
        write_file()

        # vegetation du troncon (x y z classe taille pick) ; repere jeu : x, y haut, z = -nord
        rows = []
        p = f"{W}/tiles/tile_{ti:02d}_trees.csv"
        if os.path.exists(p):
            arr = np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2)
            for sp, x, y, z, sc, rot in arr:
                rows.append((int(sp), x + tx - ox, z, -(y + ty - oy), sc))
        trees_all.append((ti, rows))
        print(f"troncon {ti}: {fileno} fichier(s) 3DS, {len(rows)} vegetaux", flush=True)

    # plafonds de vegetation sur toute la carte, puis ecriture par troncon
    flat = [(ti, r) for ti, rows in trees_all for r in rows]
    big = [i for i, (_, r) in enumerate(flat) if r[0] in (0, 1, 2, 3)]
    small = [i for i, (_, r) in enumerate(flat) if r[0] in (4, 5)]
    keep = set(rng.permutation(big)[:MAX_TREES].tolist()) | set(rng.permutation(small)[:MAX_UNDER].tolist())
    per = {ti: [] for ti, _ in trees_all}
    for i, (ti, r) in enumerate(flat):
        if i not in keep:
            continue
        sp, x, y, z, sc = r
        size = 0.35 + 0.65 * float(np.clip((sc - 0.6) / 0.9, 0, 1))   # arbres un peu plus grands
        if sp == 0:
            cls, pick = "T", rng.uniform(0.0, 0.3)
        elif sp == 1:
            cls, pick = "T", rng.uniform(0.4, 0.7)
        elif sp == 2:
            cls, pick = "P", rng.uniform(0, 1)
        elif sp == 3:
            cls, pick = "T", rng.uniform(0.8, 1.0)
        elif sp == 4:
            cls, pick = "B", rng.uniform(0, 1)
        else:
            cls, pick = "G", rng.uniform(0, 1)
        per[ti].append(f"{x:.2f} {y:.2f} {z:.2f} {cls} {size:.2f} {pick:.2f}")
    for ti, lines in per.items():
        os.makedirs(f"{tiles_dir}/c{ti:02d}", exist_ok=True)
        with open(f"{tiles_dir}/c{ti:02d}/trees.txt", "w") as f:
            f.write("# x y z classe taille pick  (P pin 3D, T arbre, B buisson, G herbe) - genere par rsmg-map-generator\n")
            f.write("\n".join(lines) + "\n")

    # trafic
    os.makedirs(f"{tiles_dir}/aivehicles", exist_ok=True)
    open(f"{tiles_dir}/aivehicles/Car1.txt", "w").write(
        "[automatic_setup]\nenabled=1\nreverse=0\nloop=0\n\n[from_3d]\nreadFrom3D=1\nprefix=Car1\nmaxPathsToCheck=999\n\n"
        "[defaults]\nisSpawner=1\nisBusSpawner=0\nspawnInterval=9\nspeedMultiplier=1\n")

    # depart et arrets (repere jeu : x, z = -nord ; cap en degres, 0 = vers +Z)
    def game_pos(i, lane=1.75):
        p = R[i, :2] + RIGHT[i] * lane
        return p[0] - ox, -(p[1] - oy), R[i, 2]

    def heading(i):
        j = min(i + 1, n - 1)
        i0 = max(j - 1, 0)
        dx, dz = R[j, 0] - R[i0, 0], -(R[j, 1] - R[i0, 1])
        return math.degrees(math.atan2(dx, dz))
    i0 = min(10, n - 1)
    x, z, y = game_pos(i0)
    open(f"{base}/{name}/spawnpoints.txt", "w").write(
        f"# Point de depart de la carte {name}\nDepart {x:.2f} {y + 0.5:.2f} {z:.2f} {heading(i0):.1f}\n")
    stops = [(f"Debut {man['road']}", i0)]
    tree = None
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(R[:, :2])
    except Exception:  # noqa
        pass
    seen = set()
    if tree is not None:
        for nd in osm["nodes"]:
            nm = nd["tags"].get("name")
            if not nm or nd["tags"].get("place") not in ("city", "town", "village", "hamlet", "suburb", "neighbourhood"):
                continue
            d, i = tree.query(nd["pt"])
            if d < 350 and nm not in seen and 5 < i < n - 5:
                seen.add(nm)
                stops.append((nm, int(i)))
    stops.append((f"Fin km {man['km_to']:g}", n - 6))
    stops.sort(key=lambda s: s[1])
    with open(f"{base}/{name}/rsmg_stops.txt", "w") as f:
        f.write("# Arrets de bus de la carte : nom, x, z, cap, cote (D/G), options\n")
        for nm, i in stops:
            x, z, _ = game_pos(i)
            f.write(f"{nm.replace(',', ' ')}, {x:.2f}, {z:.2f}, {heading(i):.1f}, D, depart\n")

    # textures (PNG, tailles raisonnables pour un telephone)
    for mat, key in sorted(used_mats):
        src = f"{W}/tex/{mat}.jpg"
        if not os.path.exists(src):
            continue
        im = Image.open(src).convert("RGB")
        if mat.startswith(("borne_", "panneau_", "jovena_", "cashpoint", "sac_charbon", "marchandise", "bache")):
            if max(im.size) > 1024:
                im.thumbnail((1024, 1024), Image.LANCZOS)
        else:
            im = im.resize(TEXSIZE.get(mat, (1024, 1024) if mat in GROUND else (512, 512)), Image.LANCZOS)
        im.save(f"{base}/{name}/textures/{key}.png", optimize=True)

    # apercu : vue du dessus de la route et des arrets
    pv = Image.new("RGB", (256, 256), (15, 23, 41))
    dr = ImageDraw.Draw(pv)
    mn, mx = R[:, :2].min(0), R[:, :2].max(0)
    sc = 240 / max(mx[0] - mn[0], mx[1] - mn[1], 1)
    pts = [(8 + (p[0] - mn[0]) * sc, 248 - (p[1] - mn[1]) * sc) for p in R[::4, :2]]
    dr.line(pts, fill=(236, 190, 110), width=3)
    for _, i in stops:
        q = (8 + (R[i, 0] - mn[0]) * sc, 248 - (R[i, 1] - mn[1]) * sc)
        dr.ellipse([q[0] - 3, q[1] - 3, q[0] + 3, q[1] + 3], fill=(80, 200, 255))
    pv.save(f"{base}/{name}/preview.png")
    open(f"{base}/{name}.map.txt", "w").write(f"[map]\nbaseDir={name}\nmodelsDir={name}\ntextures=textures\npreview=preview.png\n")

    # archives : a extraire dans /sdcard/ (elles contiennent deja RSMG/maps/...)
    os.makedirs(DIST, exist_ok=True)
    zp = f"{DIST}/{name}_RSMG.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for rt, _, files in os.walk(ROOT):
            for fn in files:
                full = os.path.join(rt, fn)
                z.write(full, os.path.relpath(full, ROOT))
    for fn in sorted(os.listdir(DIST)):
        print(fn, round(os.path.getsize(f"{DIST}/{fn}") / 1e6, 2), "Mo")


if __name__ == "__main__":
    main()
