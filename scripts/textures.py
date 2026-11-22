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


def ph_fetch(ids):
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
    line = np.zeros((H, W), bool)
    line[:, 18:27] = True
    line[:, 485:494] = True
    line |= ((x >= 251) & (x < 261)) & ((y % 683) < 219)
    img = np.where(line[..., None], col((224, 224, 216)) * wear, base)
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


def main():
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
    polyhaven("laterite", ["red_laterite_soil_stones", "aerial_mud_1", "brown_mud_dry"], keywords=["laterite", "red", "clay"], avoid=["brick", "wall"])
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
    print("textures pretes :", len(os.listdir(OUT)), "fichiers", flush=True)


if __name__ == "__main__":
    main()
