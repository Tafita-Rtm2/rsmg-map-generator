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

KEY = {"grass": "herbe", "grass2": "herbe2", "laterite": "laterite", "rock": "roche", "forest": "sol_foret", "paddy": "riziere",
       "mud": "boue", "water": "eau", "shoulder": "bord", "road_main": "route", "road_minor": "route_sec",
       "track": "piste", "wall_white": "mur_blanc", "wall_ochre": "mur_ocre", "wall_pink": "mur_rose",
       "wall_brick": "mur_brique", "wall_green": "mur_vert", "roof_tile": "toit_tuile", "roof_tin": "toit_tole"}
TEXSIZE = {"road_main": (512, 2048), "wall_white": (256, 256), "wall_ochre": (256, 256), "wall_pink": (256, 256),
           "wall_brick": (256, 256), "wall_green": (256, 256), "roof_tile": (256, 256), "roof_tin": (256, 256),
           "road_minor": (256, 256), "track": (256, 256), "cable": (16, 16), "ligne": (16, 16), "water_surface": (256, 256), "pole": (256, 256), "beton": (256, 256), "bois": (256, 256), "trottoir": (256, 256)}
GROUND = {"grass", "grass2", "laterite", "rock", "forest", "paddy", "mud", "shoulder", "falaise", "water"}
CARDS_MODE = os.environ.get("RSMG_CARDS", "auto")          # auto = cartes photo seulement si la carte fait <= 25 km ; 1 = toujours ; 0 = jamais
CARDS_MAX_KM = 25     # arbres / buissons / herbes = plans croises photo (RSMG_CARDS=0 : modeles du jeu)
CAP_TILE = {"T": 1200, "B": 800, "G": 1800}         # cartes max par troncon de 2 km
GROUP_OF = {0: ("pine",), 1: ("round",), 2: ("fir",), 3: ("round",), 4: ("bush", "bush", "weed"), 5: ("grass", "grass", "grass", "weed")}
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
        if name.startswith("veg_"):       # sprite avec transparence : carte d'opacite = le meme PNG (canal alpha)
            b += chunk(0xA210, chunk(0x0030, struct.pack("<H", 100)) + chunk(0xA300, cstr(tex)))
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


# ----------------------------------------------------------------------------- vegetation en cartes croisees (photos a fond transparent)
def load_veg(enabled):
    p = f"{W}/tex/veg_meta.json"
    if not (enabled and os.path.exists(p)):
        return None
    meta = json.load(open(p))
    groups = {}
    for k, m in sorted(meta.items()):
        groups.setdefault(m["group"], []).append(k)
    return meta, groups


def build_cards(arr, dx, dy, veg, rng):
    """arr : lignes (sp, x, y, z, sc, rot) du troncon (y = nord). Retourne {materiau: (V, UV, F)} en repere 3DS."""
    meta, groups = veg
    out = {}
    if not len(arr):
        return out
    kind = np.where(arr[:, 0] <= 3, "T", np.where(arr[:, 0] == 4, "B", "G"))
    sel = []
    for kd, cap in CAP_TILE.items():
        ii = np.where(kind == kd)[0]
        if len(ii) > cap:
            ii = rng.choice(ii, cap, replace=False)
        sel.append(ii)
    arr = arr[np.concatenate(sel)]
    r1, r2, r3 = rng.random(len(arr)), rng.random(len(arr)), rng.random(len(arr))
    var = {}
    for i, row in enumerate(arr):
        grp = GROUP_OF[int(row[0])]
        g = grp[int(r1[i] * len(grp)) % len(grp)]
        names = groups.get(g)
        if not names:
            continue
        var.setdefault(names[int(r2[i] * len(names)) % len(names)], []).append(i)
    for nm, idx in var.items():
        idx = np.array(idx)
        m = meta[nm]
        a = arr[idx]
        h0, h1 = m["h"]
        h = (h0 + (h1 - h0) * r3[idx]) * np.clip(a[:, 4], 0.6, 1.4)
        hw = h * m["aspect"] / 2
        X, Y, Z = a[:, 1] + dx, a[:, 2] + dy, a[:, 3] - 0.12
        Vs, Fs = [], []
        for pl, off in enumerate((0.0, math.pi / 2)):
            ang = a[:, 5] + off
            ux, uy = np.cos(ang) * hw, np.sin(ang) * hw
            n_ = len(idx)
            V = np.zeros((n_, 4, 3))
            V[:, 0] = np.column_stack([X - ux, Y - uy, Z])
            V[:, 1] = np.column_stack([X + ux, Y + uy, Z])
            V[:, 2] = np.column_stack([X + ux, Y + uy, Z + h])
            V[:, 3] = np.column_stack([X - ux, Y - uy, Z + h])
            Vs.append(V)
        V = np.concatenate(Vs, 1).reshape(-1, 3)                  # (n*8, 3)
        UV = np.tile(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float), (len(idx) * 2, 1))
        base = np.arange(len(idx) * 2)[:, None] * 4
        quad = np.array([[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]])      # recto + verso
        F = (base[:, None, :1] + quad[None]).reshape(-1, 3)
        out[f"veg_{nm}"] = (V, UV, F)
    return out


# ----------------------------------------------------------------------------- outils
def pieces(V, UV, F):
    for s in range(0, len(F), MAX_FACES):
        fs = F[s:s + MAX_FACES]
        used, inv = np.unique(fs.ravel(), return_inverse=True)
        yield V[used], UV[used], inv.reshape(-1, 3)


def export_part(man, osm, R, tiles, name, use_cards):
    """Ecrit UNE carte RSMG (dossier RSMG/maps/<name>) pour les troncons `tiles`, origine recentree sur son debut."""
    s_first, s_last = tiles[0]["station_from"], tiles[-1]["station_to"]
    ox, oy = R[s_first, 0], R[s_first, 1]
    base = f"{ROOT}/RSMG/maps"
    tiles_dir = f"{base}/{name}/tiles/{name}"
    os.makedirs(f"{base}/{name}/textures", exist_ok=True)
    os.makedirs(tiles_dir, exist_ok=True)
    rng = np.random.default_rng(5)
    km_total = (s_last - s_first) * 5 / 1000.0
    VEG = load_veg(use_cards)
    used_cards = set()
    print(f"vegetation : cartes photo dans les .3ds ({km_total:.0f} km)" if VEG else f"vegetation : trees.txt, streamee par le jeu autour du bus ({km_total:.0f} km)", flush=True)

    n = len(R)
    T = np.gradient(R[:, :2], axis=0)
    T /= np.maximum(np.hypot(T[:, 0], T[:, 1]), 1e-9)[:, None]
    RIGHT = np.column_stack([T[:, 1], -T[:, 0]])
    used_mats = set()
    traffic_no = 0
    trees_all = []
    for t in tiles:
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

        p_tr = f"{W}/tiles/tile_{ti:02d}_trees.csv"
        arr_tr = np.loadtxt(p_tr, delimiter=",", skiprows=1, ndmin=2) if os.path.exists(p_tr) else np.zeros((0, 6))
        if arr_tr.size == 0:
            arr_tr = np.zeros((0, 6))
        for kind in ("GND", "SOLID", "DECOR"):
            if kind == "DECOR" and VEG is not None:
                for key, (cv, cuv, cf) in build_cards(arr_tr, tx - ox, ty - oy, VEG, rng).items():
                    used_cards.add(key)
                    for pv, puv, pf in pieces(cv, cuv, cf):
                        if cur_v + len(pv) > MAX_VERTS:
                            flush_obj(kind)
                        cur_parts.append((key, pv, puv, pf))
                        cur_v += len(pv)
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
        if VEG is None and arr_tr.size:
            for sp, x, y, z, sc, rot in arr_tr:
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
    i0 = min(s_first + 10, n - 1)
    x, z, y = game_pos(i0)
    open(f"{base}/{name}/spawnpoints.txt", "w").write(
        f"# Point de depart de la carte {name}\nDepart {x:.2f} {y + 0.5:.2f} {z:.2f} {heading(i0):.1f}\n")
    stops = [(f"Km {man['km_from'] + R[i0, 3]:.0f} {man['road']}", i0)]
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
            if d < 350 and nm not in seen and s_first + 5 < i < s_last - 5:
                seen.add(nm)
                stops.append((nm, int(i)))
    stops.append((f"Km {man['km_from'] + R[s_last - 6, 3]:.0f} {man['road']}", s_last - 6))
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

    for key in sorted(used_cards):
        src = f"{W}/tex/{key}.png"
        if os.path.exists(src):
            Image.open(src).convert("RGBA").save(f"{base}/{name}/textures/{key}.png", optimize=True)

    # apercu : vue du dessus de la route et des arrets
    pv = Image.new("RGB", (256, 256), (15, 23, 41))
    dr = ImageDraw.Draw(pv)
    Rp = R[s_first:s_last + 1]
    mn, mx = Rp[:, :2].min(0), Rp[:, :2].max(0)
    sc = 240 / max(mx[0] - mn[0], mx[1] - mn[1], 1)
    pts = [(8 + (p[0] - mn[0]) * sc, 248 - (p[1] - mn[1]) * sc) for p in Rp[::4, :2]]
    dr.line(pts, fill=(236, 190, 110), width=3)
    for _, i in stops:
        q = (8 + (R[i, 0] - mn[0]) * sc, 248 - (R[i, 1] - mn[1]) * sc)
        dr.ellipse([q[0] - 3, q[1] - 3, q[0] + 3, q[1] + 3], fill=(80, 200, 255))
    pv.save(f"{base}/{name}/preview.png")
    open(f"{base}/{name}.map.txt", "w").write(f"[map]\nbaseDir={name}\nmodelsDir={name}\ntextures=textures\npreview=preview.png\n")



def main():
    man = json.load(open(f"{W}/manifest.json"))
    osm = json.load(open(f"{W}/osm.json"))
    R = np.loadtxt(f"{W}/route.csv", delimiter=",", skiprows=1)      # x, y, z, km
    km_total = len(R) * 5 / 1000.0
    use_cards = CARDS_MODE == "1" or (CARDS_MODE == "auto" and km_total <= CARDS_MAX_KM)
    print(f"vegetation : cartes photo dans les .3ds ({km_total:.0f} km)" if use_cards else
          f"vegetation : trees.txt, streamee par le jeu autour du bus ({km_total:.0f} km)", flush=True)
    tiles = man["tiles"]
    forced = os.environ.get("RSMG_PART_TILES")
    P = max(int(forced), 1) if forced else 15            # 15 troncons = 30 km par carte (la memoire du telephone est limitee)
    if len(tiles) <= (P if forced else int(P * 1.34)):
        parts = [tiles]
    else:
        parts = [tiles[i:i + P] for i in range(0, len(tiles), P)]
        if len(parts) > 1 and len(parts[-1]) < max(P // 3, 1):         # un dernier morceau trop court est rattache au precedent
            parts[-2] = parts[-2] + parts.pop()
    base_name = re.sub(r"[^a-z0-9_-]+", "_", f"{man['road']}".lower()).strip("_")
    names = []
    for k, ts in enumerate(parts):
        a = man["km_from"] + R[ts[0]["station_from"], 3]
        b = man["km_from"] + R[ts[-1]["station_to"], 3]
        nm = f"{base_name}_km{a:.0f}-{b:.0f}" if len(parts) == 1 else f"{base_name}_partie{k + 1}_km{a:.0f}-{b:.0f}"
        names.append(nm)
        export_part(man, osm, R, ts, nm, use_cards)
        print(f"carte {k + 1}/{len(parts)} : {nm} ({len(ts)} troncons)", flush=True)

    # une seule archive, a extraire dans /sdcard/ (elle contient deja RSMG/maps/...)
    os.makedirs(DIST, exist_ok=True)
    zp = f"{DIST}/{base_name}_km{man['km_from']:.0f}-{man['km_to']:.0f}_RSMG.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for rt, _, files in os.walk(ROOT):
            for fn in files:
                full = os.path.join(rt, fn)
                z.write(full, os.path.relpath(full, ROOT))
    for fn in sorted(os.listdir(DIST)):
        print(fn, round(os.path.getsize(f"{DIST}/{fn}") / 1e6, 2), "Mo")


if __name__ == "__main__":
    main()
