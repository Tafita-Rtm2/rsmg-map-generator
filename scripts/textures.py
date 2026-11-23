#!/usr/bin/env python3
"""Textures : photos CC0 Poly Haven quand elles sont disponibles, sinon textures procedurales.
Usage : textures.py <dossier> [taille]"""
import io, os, sys
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageEnhance

OUT = sys.argv[1] if len(sys.argv) > 1 else "work/tex"
SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(11)
UA = {"User-Agent": "rsmg-map-generator/1.0"}


def tnoise(size, cycles):
    """Bruit continu qui se repete sans raccord (filtre passe-bas par FFT)."""
    f = np.fft.fft2(rng.normal(size=(size, size)))
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2) * size
    out = np.real(np.fft.ifft2(f * np.exp(-(r / cycles) ** 2)))
    return (out - out.min()) / (out.max() - out.min() + 1e-9)


def col(c):
    return np.array(c, float)


def blend(c1, c2, t):
    return col(c1)[None, None, :] * (1 - t[..., None]) + col(c2)[None, None, :] * t[..., None]


def save_rgb(name, arr, q=82):
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(f"{OUT}/{name}.jpg", quality=q, optimize=True)


_CAT = None


def ph_pick(keywords, avoid=()):
    if NO_NET:
        return []
    """Cherche dans le catalogue Poly Haven les textures dont l'id/les tags contiennent les mots-cles."""
    global _CAT
    if _CAT is None:
        try:
            _CAT = requests.get("https://api.polyhaven.com/assets?t=textures", headers=UA, timeout=60).json()
        except Exception as e:  # noqa
            print("[texture] catalogue Poly Haven indisponible :", e, flush=True)
            _CAT = {}
    scored = []
    for pid, a in _CAT.items():
        text = (pid + " " + str(a.get("name", "")) + " " + " ".join(a.get("tags", [])) + " " +
                " ".join(a.get("categories", []))).lower()
        if any(b in text for b in avoid):
            continue
        sc = sum((3 if k in pid.lower() else 1) for k in keywords if k in text)
        if sc:
            scored.append((sc, pid))
    scored.sort(key=lambda t: -t[0])
    return [pid for _, pid in scored[:5]]


NO_NET = False   # True quand les photos locales (assets/polyhaven) sont presentes : plus d'appel reseau


def ph_fetch(ids):
    if NO_NET:
        return None, None
    for pid in ids:
        try:
            r = requests.get(f"https://api.polyhaven.com/files/{pid}", headers=UA, timeout=30)
            if r.status_code != 200:
                continue
            j = r.json()
            d = j.get("Diffuse") or j.get("diff") or j.get("Color")
            url = d["1k"]["jpg"]["url"]
            im = Image.open(io.BytesIO(requests.get(url, headers=UA, timeout=120).content)).convert("RGB")
            return pid, im
        except Exception as e:  # noqa
            print(f"[texture] {pid} indisponible ({e})", flush=True)
    return None, None


def polyhaven(name, ids, desat=1.0, keywords=(), avoid=()):
    pid, im = ph_fetch(list(ids) + (ph_pick(list(keywords), avoid) if keywords else []))
    if im is None:
        print(f"[texture] {name} : procedurale (aucune photo)", flush=True)
        return False
    im = im.resize((SIZE, SIZE))
    if desat != 1.0:
        im = ImageEnhance.Color(im).enhance(desat)
    im.save(f"{OUT}/{name}.jpg", quality=82, optimize=True)
    print(f"[texture] {name} <- Poly Haven {pid}", flush=True)
    return True


def ground(name, c1, c2, size=512, speck=0.18, pebbles=None):
    t = 0.55 * tnoise(size, 5) + 0.3 * tnoise(size, 18) + 0.15 * tnoise(size, 70)
    t = (t - t.min()) / (t.max() - t.min())
    img = blend(c1, c2, t)
    f = tnoise(size, size // 3)
    img *= (1 - speck + 2 * speck * f)[..., None]
    if pebbles:
        p = tnoise(size, 45) > 0.80
        img = np.where(p[..., None], col(pebbles) * (0.85 + 0.3 * f[..., None]), img)
    save_rgb(name, img)


def wall(name, base, brick=False, shutter=(70, 112, 76)):
    W, H = 256, 192
    n = (0.5 * tnoise(256, 6) + 0.5 * tnoise(256, 40))[:H]
    img = col(base)[None, None, :] * (0.88 + 0.24 * n[..., None])
    g = np.clip((np.arange(H)[:, None] - (H - 34)) / 34, 0, 1) * 0.28
    img = img * (1 - g)[..., None]
    if brick:
        for y in range(0, H, 8):
            img[y:y + 1, :, :] = col((205, 192, 170))
            off = 0 if (y // 8) % 2 == 0 else 12
            for x in range(off, W, 24):
                img[y:y + 8, x:x + 1, :] = col((205, 192, 170))
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.rectangle([70, 62, 94, 142], fill=shutter)
    d.rectangle([162, 62, 186, 142], fill=shutter)
    for y in range(66, 142, 8):
        d.line([70, y, 94, y], fill=tuple(int(c * 0.7) for c in shutter))
        d.line([162, y, 186, y], fill=tuple(int(c * 0.7) for c in shutter))
    d.rectangle([94, 62, 162, 142], fill=(236, 231, 221))
    d.rectangle([98, 66, 158, 138], fill=(46, 54, 62))
    d.line([128, 66, 128, 138], fill=(236, 231, 221), width=3)
    d.line([98, 102, 158, 102], fill=(236, 231, 221), width=3)
    im.save(f"{OUT}/{name}.jpg", quality=84, optimize=True)


def roof_tile(name="roof_tile"):
    s = 256
    n = tnoise(s, 30)
    img = col((156, 74, 48))[None, None, :] * (0.8 + 0.4 * n[..., None])
    for y in range(0, s, 20):
        img[y:y + 3, :, :] *= 0.62
    x = np.arange(s)[None, :]
    img *= (0.9 + 0.1 * np.sin(x * 2 * np.pi / 16))[..., None]
    moss = tnoise(s, 6) > 0.72
    img = np.where(moss[..., None], img * np.array([0.8, 0.95, 0.75]), img)
    save_rgb(name, img)


def roof_tin(name="roof_tin"):
    s = 256
    x = np.arange(s)[None, :]
    base = np.repeat(0.82 + 0.18 * np.sin(x * 2 * np.pi / 16), s, axis=0)
    n = tnoise(s, 8)
    img = col((140, 146, 150))[None, None, :] * base[..., None] * (0.88 + 0.2 * n[..., None])
    rust = (tnoise(s, 5) > 0.62).astype(float) * 0.7
    img = img * (1 - rust[..., None]) + col((150, 92, 62))[None, None, :] * rust[..., None] * base[..., None]
    save_rgb(name, img)


def road_main():
    W, H = 512, 2048          # 7 m de large, 28 m de long
    pid, ph = ph_fetch(["asphalt_02", "asphalt_01", "asphalt_04"] + ph_pick(["asphalt"], ("wet",)))
    if ph is not None:
        base = np.tile(np.array(ph.resize((512, 512)), float), (4, 1, 1)) * 0.9
        print(f"[texture] road_main <- Poly Haven {pid}", flush=True)
    else:
        n = tnoise(2048, 300)[:, :W]
        base = col((66, 66, 70))[None, None, :] * (0.82 + 0.3 * n[..., None])
        print("[texture] road_main : procedurale", flush=True)
    x = np.arange(W)[None, :]
    y = np.arange(H)[:, None]
    wear = (0.75 + 0.25 * tnoise(2048, 400)[:, :W])[..., None]
    img = base * wear   # asphalte seul, sans lignes blanches
    save_rgb("road_main", img, 85)


def road_minor():
    s = 256
    n = tnoise(s, 40)
    img = col((78, 76, 78))[None, None, :] * (0.8 + 0.35 * n[..., None])
    save_rgb("road_minor", img)


def track():
    s = 256
    n = tnoise(s, 30)
    img = col((176, 122, 86))[None, None, :] * (0.85 + 0.3 * n[..., None])
    x = np.arange(s)[None, :]
    rut = ((x > 56) & (x < 92)) | ((x > 164) & (x < 200))
    img = np.where(rut[..., None], img * 0.82, img)
    save_rgb("track", img)


def leaf(name, c1, c2):
    s = 256
    t = 0.6 * tnoise(s, 10) + 0.4 * tnoise(s, 50)
    img = blend(c1, c2, (t - t.min()) / (t.max() - t.min()))
    dark = tnoise(s, 60) < 0.2
    img = np.where(dark[..., None], img * 0.62, img)
    save_rgb(name, img)


def bark(name, c1, c2):
    s = 128
    t = tnoise(s, 24)
    x = np.arange(s)[None, :]
    streak = 0.85 + 0.15 * np.sin(x * 2 * np.pi / 9 + t * 4)
    img = blend(c1, c2, t) * streak[..., None]
    save_rgb(name, img)


def far(name, kind):
    """Plan de feuillage lointain (RGBA 128x256)."""
    im = Image.new("RGBA", (128, 256), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    r = np.random.default_rng(5)
    greens = {"euc": [(88, 110, 86), (110, 130, 96)], "round": [(58, 92, 44), (84, 112, 52)],
              "pine": [(34, 66, 44), (50, 84, 54)], "ravi": [(70, 122, 52), (100, 146, 66)],
              "bush": [(66, 100, 50), (90, 118, 58)]}[kind]
    trunk = (110, 90, 70)
    if kind == "euc":
        d.rectangle([60, 110, 68, 256], fill=trunk)
        for _ in range(60):
            x, y = 64 + r.normal(0, 16), 70 + r.normal(0, 38)
            rr = r.uniform(8, 16)
            d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=greens[r.integers(2)] + (255,))
    elif kind == "round":
        d.rectangle([58, 170, 70, 256], fill=trunk)
        for _ in range(70):
            x, y = 64 + r.normal(0, 30), 120 + r.normal(0, 22)
            rr = r.uniform(10, 20)
            d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=greens[r.integers(2)] + (255,))
    elif kind == "pine":
        d.rectangle([60, 200, 68, 256], fill=trunk)
        for i in range(6):
            y = 30 + i * 32
            w = 14 + i * 9
            d.polygon([(64, y - 24), (64 - w, y + 16), (64 + w, y + 16)], fill=greens[i % 2] + (255,))
    elif kind == "ravi":
        d.rectangle([60, 140, 66, 256], fill=trunk)
        for a in np.linspace(-1.2, 1.2, 11):
            d.line([(64, 150), (64 + 90 * np.sin(a), 150 - 110 * np.cos(a))], fill=greens[0] + (255,), width=9)
    else:
        for _ in range(40):
            x, y = 64 + r.normal(0, 26), 200 + r.normal(0, 14)
            rr = r.uniform(8, 16)
            d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=greens[r.integers(2)] + (255,))
    im.save(f"{OUT}/far_{kind}.png", optimize=True)


ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")


def solid(name, rgb, size=16):
    Image.new("RGB", (size, size), rgb).save(f"{OUT}/{name}.jpg", quality=92)


def atlas2(name, crop, size, top, bottom=None):
    """Atlas : moitie gauche = image, moitie droite = couleur unie (haut) et seconde couleur (bas)."""
    im = Image.new("RGB", size, top)
    if bottom is not None:
        ImageDraw.Draw(im).rectangle([size[0] // 2, size[1] // 2, size[0], size[1]], fill=bottom)
    im.paste(crop.resize((size[0] // 2, size[1]), Image.LANCZOS), (0, 0))
    im.save(f"{OUT}/{name}.jpg", quality=90, optimize=True)


def user_assets():
    """Textures fournies par l'utilisateur (assets/) + textures simples des equipements de route."""
    try:
        jo = Image.open(f"{ASSETS}/jovena.jpg").convert("RGB")
        atlas2("jovena_pompe", jo.crop((205, 20, 415, 320)), (512, 512), (214, 214, 218))
        atlas2("jovena_logo", jo.crop((60, 655, 395, 755)), (1024, 256), (245, 245, 245))
        atlas2("jovena_boutique", jo.crop((0, 385, 418, 590)), (1024, 512), (228, 228, 222))
        atlas2("jovena_entretien", jo.crop((435, 305, 715, 585)), (512, 512), (228, 228, 222))
        atlas2("jovena_prix", jo.crop((35, 200, 158, 345)), (512, 512), (60, 64, 70))
        atlas2("jovena_panneau", jo.crop((28, 5, 165, 195)), (512, 512), (245, 245, 245))
        solid("jovena_rouge", (196, 58, 38))
        print("[texture] Jovena <- assets/jovena.jpg", flush=True)
    except Exception as e:  # noqa
        print("[texture] Jovena indisponible :", e, flush=True)
    try:
        sc = Image.open(f"{ASSETS}/sac_charbon.jpg").convert("RGB")
        atlas2("sac_charbon", sc.crop((0, 38, 165, 258)), (512, 512), (60, 44, 36))
        print("[texture] sac de charbon <- assets/sac_charbon.jpg", flush=True)
    except Exception as e:  # noqa
        print("[texture] sac indisponible :", e, flush=True)
    try:
        cp = Image.open(f"{ASSETS}/cashpoint.jpg").convert("RGB")
        w, h = cp.size
        crop = cp.crop((int(0.282 * w), int(0.194 * h), int(0.648 * w), int(0.898 * h)))
        atlas2("cashpoint", crop, (512, 1024), (205, 40, 36), (232, 226, 210))
        print("[texture] cash point <- assets/cashpoint.jpg", flush=True)
    except Exception as e:  # noqa
        print("[texture] cashpoint indisponible :", e, flush=True)
    # marchandises : 8 couleurs (tomates, carottes, feuilles, maïs, riz, manioc, fruits, haricots)
    cols = [(196, 52, 40), (226, 128, 36), (70, 128, 58), (226, 190, 60), (232, 228, 214), (160, 120, 80), (214, 96, 120), (110, 70, 50)]
    im = Image.new("RGB", (256, 64))
    for i, c in enumerate(cols):
        ImageDraw.Draw(im).rectangle([i * 32, 0, i * 32 + 31, 63], fill=c)
    im.save(f"{OUT}/marchandise.jpg", quality=92)
    # toile d'etal rayee
    t = Image.new("RGB", (128, 128), (240, 240, 240))
    dt = ImageDraw.Draw(t)
    for x in range(0, 128, 32):
        dt.rectangle([x, 0, x + 15, 127], fill=(30, 90, 170))
    t.save(f"{OUT}/bache.jpg", quality=90)
    ground("bois", (110, 80, 52), (142, 106, 70), speck=0.25)
    polyhaven("brique", ["red_brick", "brick_wall_005", "brick_wall_001"], desat=0.85, keywords=["brick"], avoid=["floor", "roof"])
    if not os.path.exists(f"{OUT}/brique.jpg"):
        ground("brique", (150, 78, 56), (178, 98, 70), speck=0.3)
    ground("beton", (150, 148, 142), (178, 176, 170), speck=0.2)
    polyhaven("trottoir", ["concrete_pavement", "concrete_floor_worn_001"], keywords=["concrete", "pavement"], avoid=["wall"])
    if not os.path.exists(f"{OUT}/trottoir.jpg"):
        ground("trottoir", (160, 158, 152), (186, 184, 178), speck=0.2)
    ground("pole", (132, 130, 124), (160, 158, 150), speck=0.2)
    solid("cable", (28, 28, 30), 8)
    solid("ligne", (232, 232, 226), 8)
    ground("water_surface", (58, 84, 90), (84, 108, 110), speck=0.06)
    try:
        lat = Image.open(f"{OUT}/laterite.jpg").convert("RGB")
        ImageEnhance.Brightness(ImageEnhance.Color(lat).enhance(0.8)).enhance(0.95).save(f"{OUT}/falaise.jpg", quality=84)
    except Exception as e:  # noqa
        print("falaise :", e)


# ----------------------------------------------------------------------------- photos reelles fournies (assets/polyhaven, assets/veg)
PHDIR = os.path.join(ASSETS, "polyhaven")
VEGDIR = os.path.join(ASSETS, "veg")


def ph_local(name):
    """Photo locale -> carre RGB (moitie gauche des vignettes 2:1)."""
    im = Image.open(f"{PHDIR}/{name}.webp").convert("RGB")
    s = min(im.size)
    return im.crop((0, 0, s, s))


def make_tileable(im, size):
    """Rend l'image sans raccord : fondu avec sa copie decalee d'une demi-image (masque nul sur les bords)."""
    a = np.array(im.resize((size, size), Image.LANCZOS), float)
    r = np.roll(np.roll(a, size // 2, 0), size // 2, 1)
    t = np.linspace(0, 1, size)
    w1 = np.sin(np.pi * t) ** 2
    m = (w1[:, None] * w1[None, :])[..., None]
    out = a * m + r * (1 - m)
    # le fondu aplatit le contraste au centre : on le restitue
    mean, std = a.mean((0, 1)), a.std((0, 1))
    o_std = out.std((0, 1))
    out = (out - out.mean((0, 1))) * (std / np.maximum(o_std, 1e-6)) + mean
    return np.clip(out, 0, 255)


def grade(arr, hue=None, mix=1.0, sat=1.0, val=1.0, gamma=1.0):
    """Etalonnage : pousse la teinte vers `hue` (degres), puis sature / eclaire."""
    im = Image.fromarray(arr.astype(np.uint8)).convert("HSV")
    h, s, v = [np.array(c, float) for c in im.split()]
    if hue is not None:
        tgt = hue / 360 * 255
        h = h + mix * (tgt - h)
    s = np.clip(s * sat, 0, 255)
    v = np.clip(255 * (v / 255) ** gamma * val, 0, 255)
    out = Image.merge("HSV", [Image.fromarray(np.clip(c, 0, 255).astype(np.uint8)) for c in (h, s, v)]).convert("RGB")
    return np.array(out, float)


def pro_ground(name, src, size=None, mix_with=None, w=0.0, **kw):
    size = size or SIZE
    a = make_tileable(ph_local(src), size)
    if mix_with:
        b = make_tileable(ph_local(mix_with), size)
        a = a * (1 - w) + b * w
    a = grade(a, **kw) if kw else a
    save_rgb(name, a, 88)
    print(f"[texture] {name} <- photo {src}", flush=True)
    return a


def pro_textures():
    """Remplace les sols par les vraies photos : tout est couvert d'herbe verte (herbe, foret, rizieres), roches et falaises reelles."""
    if not os.path.isdir(PHDIR):
        print("[texture] assets/polyhaven absent : textures precedentes conservees", flush=True)
        return
    # herbe verte (sol entierement couvert) : deux variantes pour casser la repetition
    pro_ground("grass", "rocky_terrain_02", hue=96, mix=0.55, sat=1.25, val=1.05)
    pro_ground("grass2", "grass_ground_2", hue=84, mix=0.9, sat=1.9, val=1.12)
    pro_ground("paddy", "rocky_terrain_02", hue=100, mix=0.7, sat=1.3, val=1.08)
    pro_ground("forest", "rocky_terrain_02", mix_with="leaf_scattered_gravel_1", w=0.22, hue=100, mix=0.7, sat=1.35, val=0.8)
    # terre rouge (laterite), talus, roches
    pro_ground("laterite", "rocky_trail_02", hue=14, mix=0.75, sat=1.9, val=0.98)
    pro_ground("falaise", "rocky_terrain", hue=16, mix=0.5, sat=1.3, val=0.95)
    pro_ground("rock", "rocks_ground_03", sat=1.0, val=1.0)
    pro_ground("mud", "pebbles", hue=24, mix=0.6, sat=1.5, val=1.2)
    pro_ground("shoulder", "rocky_trail", sat=0.9, val=1.0)
    # chaussees et ouvrages
    pro_ground("road_minor", "gravel_floor_03_1", sat=0.7, val=1.15)
    pro_ground("trottoir", "rock_ground_02", sat=0.6, val=1.25)
    pro_ground("beton", "rock_ground_02", sat=0.4, val=1.3)
    pro_ground("pole", "rock_ground_02", sat=0.4, val=1.2)
    t = pro_ground("track", "rocky_trail_02", hue=18, mix=0.6, sat=1.5, val=1.0)
    x = np.arange(t.shape[1])[None, :] / t.shape[1]
    rut = np.exp(-((x - 0.3) / 0.07) ** 2) + np.exp(-((x - 0.7) / 0.07) ** 2)
    save_rgb("track", t * (1 - 0.22 * rut[..., None]), 88)
    # asphalte : 7 m x 28 m ; photo de gravier fin sombre repetee 4 fois, usure + traces de roues
    W, H = 512, 2048
    a = make_tileable(ph_local("gravel_stones"), 256)       # grain fin : 256 px repetes 2 x 8 fois
    a = grade(a, sat=0.3, val=1.1)
    base = np.tile(a, (8, 2, 1))
    xs = np.arange(W)[None, :] / W
    ys = np.arange(H)[:, None]
    wheel = np.exp(-((xs - 0.3) / 0.09) ** 2) + np.exp(-((xs - 0.7) / 0.09) ** 2)
    wear = 1.0 - 0.10 * wheel + 0.08 * (tnoise(2048, 160)[:, :W] - 0.5)
    save_rgb("road_main", base * wear[..., None], 90)
    print("[texture] road_main <- photo gravel_stones", flush=True)
    # sprites de vegetation (arbres, buissons, herbes avec transparence)
    if os.path.isdir(VEGDIR):
        import json, shutil
        meta = json.load(open(f"{VEGDIR}/meta.json"))
        for k in meta:
            shutil.copy(f"{VEGDIR}/{k}.png", f"{OUT}/veg_{k}.png")
        json.dump(meta, open(f"{OUT}/veg_meta.json", "w"))
        print(f"[texture] {len(meta)} sprites de vegetation", flush=True)


def main():
    global NO_NET
    NO_NET = os.path.isdir(PHDIR)
    ground("grass", (86, 98, 52), (128, 120, 70))
    ground("laterite", (150, 82, 52), (118, 64, 44), pebbles=(186, 150, 120))
    ground("rock", (118, 108, 98), (160, 150, 135), speck=0.3)
    ground("forest", (52, 58, 34), (86, 70, 44), speck=0.25)
    ground("paddy", (112, 138, 62), (150, 150, 80))
    ground("mud", (92, 70, 50), (118, 92, 66))
    ground("water", (74, 92, 86), (96, 110, 100), speck=0.05)
    ground("shoulder", (150, 128, 100), (176, 160, 138), speck=0.3, pebbles=(200, 190, 170))

    # photos reelles CC0 (Poly Haven) quand c'est possible : remplacent les textures procedurales
    polyhaven("grass", ["leafy_grass", "aerial_grass_rock", "grass_path_2"], desat=0.8, keywords=["grass"], avoid=["snow", "dry_grass", "dead"])
    polyhaven("laterite", ["red_laterite_soil_stones", "aerial_mud_1", "brown_mud_dry"], desat=0.75, keywords=["laterite", "red", "clay"], avoid=["brick", "wall"])
    polyhaven("rock", ["rock_boulder_cracked", "aerial_rocks_02", "rocky_terrain_02"], keywords=["rock", "cliff"], avoid=["wall", "tile"])
    polyhaven("forest", ["forrest_ground_01", "brown_mud_leaves_01"], desat=0.85, keywords=["forest", "leaves", "ground"], avoid=["snow"])
    polyhaven("mud", ["brown_mud", "aerial_mud_1"], keywords=["mud"])
    polyhaven("paddy", ["leafy_grass"], desat=0.9, keywords=["grass", "field"], avoid=["snow", "dry"])
    polyhaven("shoulder", ["gravel_road", "gravelly_sand", "rocky_trail"], keywords=["gravel", "sand"], avoid=["wet"])

    wall("wall_white", (214, 208, 196))
    wall("wall_ochre", (200, 168, 110))
    wall("wall_pink", (200, 150, 130))
    wall("wall_brick", (160, 84, 60), brick=True, shutter=(110, 70, 40))
    wall("wall_green", (150, 176, 150))
    roof_tile()
    roof_tin()
    polyhaven("roof_tile", ["roof_tiles_14", "roof_tiles"], desat=0.9, keywords=["roof", "tiles"])
    polyhaven("roof_tin", ["corrugated_iron", "corrugated_iron_02"], keywords=["corrugated", "metal"], avoid=["wall"])
    road_main()
    road_minor()
    track()

    leaf("leaf_round", (58, 92, 44), (104, 128, 60))
    leaf("leaf_euc", (88, 110, 86), (120, 138, 100))
    leaf("leaf_pine", (34, 66, 44), (58, 92, 56))
    leaf("leaf_ravi", (70, 122, 52), (110, 150, 70))
    bark("bark", (88, 70, 56), (120, 100, 80))
    bark("bark_light", (190, 180, 165), (150, 140, 125))
    for k in ("euc", "round", "pine", "ravi", "bush"):
        far(f"{k}", k)
    user_assets()
    pro_textures()
    print("textures pretes :", len(os.listdir(OUT)), "fichiers", flush=True)


if __name__ == "__main__":
    main()
