#!/usr/bin/env python3
"""Textures procedurales detaillees, style ETS2 / Madagascar (sans telechargement, sans reseau).

- herbe : savane seche (vert olive + paille + terre nue), brins d'herbe individuels
- bord  : accotement de sable/gravier clair avec cailloux, brins secs
- route : asphalte granuleux, traces de roues polies, fissures, rustines, nids de poule,
          bord effrite avec sable, lignes de rive delavees (la ligne centrale est desactivee)
- laterite / falaise : sol rouge avec cailloux, falaise de deblai erodee (ravines verticales)
- murs, toits (tole rouillee, tuiles), brique, beton, bois ...
Toutes les textures se raccordent sans couture (tileable)."""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from scipy.ndimage import gaussian_filter

RNG = np.random.default_rng(2026)


# ----------------------------------------------------------------------------- bruit
def noise(h, w, cycles, rng=None):
    """Bruit gaussien filtre, sans couture, centre-reduit (z-score)."""
    rng = rng or RNG
    f = np.fft.fft2(rng.normal(size=(h, w)))
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r = np.hypot(fx, fy) * w
    o = np.real(np.fft.ifft2(f * np.exp(-(r / cycles) ** 2)))
    return (o - o.mean()) / (o.std() + 1e-9)


def n01(z, k=0.26):
    return np.clip(0.5 + k * z, 0.0, 1.0)


def fbm(h, w, base, octs=5, gain=0.55):
    tot = 0
    amp = 1.0
    for i in range(octs):
        tot = tot + amp * noise(h, w, base * 2 ** i)
        amp *= gain
    return n01(tot / (tot.std() + 1e-9))


def ramp(t, stops):
    pos = [s[0] for s in stops]
    return np.stack([np.interp(t, pos, [s[1][c] for s in stops]) for c in range(3)], -1)


def lerp(a, b, t):
    t = np.asarray(t, float)
    if t.ndim == 2:
        t = t[..., None]
    return a * (1 - t) + b * t


def smooth(x, a, b):
    t = np.clip((x - a) / (b - a + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def to_img(arr):
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def save(out, name, arr, q=88):
    im = arr if isinstance(arr, Image.Image) else to_img(arr)
    im.convert("RGB").save(f"{out}/{name}.jpg", quality=q, optimize=True)


def grain(h, w, amt=0.06):
    return 1 + amt * noise(h, w, w / 2.0)


# ----------------------------------------------------------------------------- dessin (avec raccord)
def _offsets(x0, y0, x1, y1, w, h):
    xs = [0]
    ys = [0]
    if min(x0, x1) < 0:
        xs.append(w)
    if max(x0, x1) >= w:
        xs.append(-w)
    if min(y0, y1) < 0:
        ys.append(h)
    if max(y0, y1) >= h:
        ys.append(-h)
    return [(a, b) for a in xs for b in ys]


def strokes(arr, n, length, width, colfn, angle=0.0, spread=0.35, rng=None, hscale=1.0):
    """Brins : lignes courtes orientees (angle depuis la verticale), couleur = colfn(x, y) -> (n,3)."""
    rng = rng or RNG
    h, w = arr.shape[:2]
    im = to_img(arr).convert("RGBA")
    lay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    xs = rng.uniform(0, w, n)
    ys = rng.uniform(0, h, n)
    L = rng.uniform(length[0], length[1], n) * hscale
    A = angle + rng.normal(0, spread, n)
    cols = np.clip(colfn(xs.astype(int) % w, ys.astype(int) % h), 0, 255).astype(int)
    wd = rng.integers(width[0], width[1] + 1, n)
    for i in range(n):
        x0, y0 = xs[i], ys[i]
        x1, y1 = x0 + np.sin(A[i]) * L[i], y0 - np.cos(A[i]) * L[i]
        c = (int(cols[i, 0]), int(cols[i, 1]), int(cols[i, 2]), 255)
        for ox, oy in _offsets(x0, y0, x1, y1, w, h):
            d.line([x0 + ox, y0 + oy, x1 + ox, y1 + oy], fill=c, width=int(wd[i]))
    return np.array(Image.alpha_composite(im, lay).convert("RGB"), float)


def pebbles(arr, n, rad, palette, rng=None, shadow=0.55, hscale=1.0):
    """Cailloux ombres (ombre en bas a droite, reflet en haut a gauche)."""
    rng = rng or RNG
    h, w = arr.shape[:2]
    im = to_img(arr).convert("RGBA")
    lay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    xs, ys = rng.uniform(0, w, n), rng.uniform(0, h, n)
    rs = rng.uniform(rad[0], rad[1], n) * hscale
    asp = rng.uniform(0.65, 1.0, n)
    for i in range(n):
        x, y, r = xs[i], ys[i], rs[i]
        col = palette[rng.integers(len(palette))]
        col = tuple(int(c * rng.uniform(0.85, 1.12)) for c in col)
        dark = tuple(int(c * shadow) for c in col)
        lite = tuple(min(255, int(c * 1.22)) for c in col)
        for ox, oy in _offsets(x - r, y - r, x + r, y + r, w, h):
            cx, cy = x + ox, y + oy
            d.ellipse([cx - r * 0.95 + r * 0.35, cy - r * asp[i] + r * 0.35, cx + r * 1.05 + r * 0.35, cy + r * asp[i] + r * 0.4], fill=dark + (190,))
            d.ellipse([cx - r, cy - r * asp[i], cx + r, cy + r * asp[i]], fill=col + (255,))
            d.ellipse([cx - r * 0.7, cy - r * asp[i] * 0.75, cx + r * 0.1, cy - r * asp[i] * 0.05], fill=lite + (120,))
    return np.array(Image.alpha_composite(im, lay).convert("RGB"), float)


def polylines(arr, paths, color, width=1, alpha=1.0):
    """paths: liste de listes de points (x, y); dessine avec raccord."""
    h, w = arr.shape[:2]
    im = to_img(arr).convert("RGBA")
    lay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    for pts in paths:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        for ox, oy in _offsets(min(xs), min(ys), max(xs), max(ys), w, h):
            d.line([(x + ox, y + oy) for x, y in pts], fill=tuple(int(c) for c in color) + (int(255 * alpha),), width=width)
    return np.array(Image.alpha_composite(im, lay).convert("RGB"), float)


def crack_paths(n, w, h, length, rng, vertical=False, wander=0.25, branch=0.25):
    paths = []
    for _ in range(n):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        ang = (np.pi / 2 if vertical else 0.0) + rng.normal(0, 0.5)
        pts = [(x, y)]
        for _k in range(int(rng.integers(6, 22))):
            ang += rng.normal(0, wander)
            step = length / 12 * rng.uniform(0.6, 1.4)
            x, y = x + np.cos(ang) * step, y + np.sin(ang) * step
            pts.append((x, y))
            if rng.random() < branch * 0.15:
                a2 = ang + rng.choice([-1, 1]) * rng.uniform(0.5, 1.1)
                pts2 = [(x, y)]
                xx, yy = x, y
                for _j in range(int(rng.integers(2, 6))):
                    a2 += rng.normal(0, wander)
                    xx, yy = xx + np.cos(a2) * step * 0.8, yy + np.sin(a2) * step * 0.8
                    pts2.append((xx, yy))
                paths.append(pts2)
        paths.append(pts)
    return paths


# ----------------------------------------------------------------------------- sols
def herbe(S, rng=RNG):
    """Savane malgache : olive + paille doree + plaques de terre rouge, brins individuels."""
    m = fbm(S, S, 2.5, 4)
    m2 = fbm(S, S, 7, 3)
    base = ramp(m, [(0.0, (84, 92, 44)), (0.35, (112, 112, 56)), (0.62, (156, 140, 78)), (1.0, (190, 168, 102))])
    soil = smooth(m2, 0.66, 0.80)
    base = lerp(base, ramp(fbm(S, S, 12, 3), [(0, (138, 82, 52)), (1, (176, 116, 74))]), soil * 0.85)
    base *= (0.82 + 0.34 * fbm(S, S, 40, 3))[..., None]

    def col(x, y):
        c = base[y, x]
        r = rng.random(len(x))[:, None]
        c = c * rng.uniform(0.72, 1.28, (len(x), 1))
        c = np.where(r < 0.14, np.array([214, 192, 124]) * rng.uniform(0.8, 1.05, (len(x), 1)), c)
        c = np.where(r > 0.90, np.array([66, 88, 40]) * rng.uniform(0.8, 1.2, (len(x), 1)), c)
        return c

    k = S / 1024.0
    base = strokes(base, int(S * S / 30), (7, 26), (1, 1), col, angle=0.0, spread=0.55, rng=rng, hscale=k)
    base = strokes(base, int(S * S / 90), (12, 34), (1, 2), col, angle=0.1, spread=0.45, rng=rng, hscale=k)
    base *= (0.9 + 0.2 * fbm(S, S, 90, 2))[..., None]
    return base


def bord(S, rng=RNG):
    """Accotement : sable clair + gravillons + quelques brins secs."""
    t = fbm(S, S, 4, 4)
    base = ramp(t, [(0, (166, 146, 112)), (0.5, (192, 172, 134)), (1, (214, 198, 160))])
    base *= (0.9 + 0.2 * fbm(S, S, 60, 3))[..., None]
    k = S / 1024.0
    pal = [(128, 118, 104), (162, 148, 126), (98, 88, 78), (186, 168, 138), (146, 110, 84), (210, 198, 176)]
    base = pebbles(base, int(S * S / 900), (1.6, 6.5), pal, rng=rng, hscale=k)
    base = pebbles(base, int(S * S / 250), (0.8, 2.2), pal, rng=rng, hscale=k)
    # traces de passage (pneus, pieds)
    tr = fbm(S, S, 3, 2)
    base *= (0.94 + 0.1 * smooth(tr, 0.4, 0.7))[..., None]

    def col(x, y):
        return np.array([150, 138, 80]) * rng.uniform(0.7, 1.3, (len(x), 1))
    base = strokes(base, int(S * S / 1800), (8, 26), (1, 1), col, spread=0.6, rng=rng, hscale=k)
    return base


def laterite(S, rng=RNG):
    t = fbm(S, S, 3, 5)
    base = ramp(t, [(0, (132, 58, 36)), (0.35, (166, 80, 46)), (0.7, (192, 102, 58)), (1, (214, 130, 78))])
    # ravinement : sillons de pluie
    rid = 1 - np.abs(noise(S, S, 9) * 0.5)
    base *= (0.86 + 0.14 * np.clip(rid, 0, 1))[..., None]
    base *= (0.85 + 0.3 * fbm(S, S, 50, 3))[..., None]
    k = S / 1024.0
    pal = [(206, 172, 138), (172, 134, 104), (92, 48, 38), (120, 66, 44), (226, 198, 168)]
    base = pebbles(base, int(S * S / 500), (1.0, 4.5), pal, rng=rng, hscale=k)
    base = pebbles(base, int(S * S / 6000), (4, 9), pal, rng=rng, hscale=k)

    def col(x, y):
        return np.array([140, 126, 66]) * rng.uniform(0.7, 1.25, (len(x), 1))
    base = strokes(base, int(S * S / 1500), (8, 22), (1, 1), col, spread=0.6, rng=rng, hscale=k)
    return base


def falaise(S, rng=RNG):
    """Talus de deblai : ravines verticales rouges/orange, strates, eboulis."""
    wn = rng.normal(size=(S, S))
    vert = gaussian_filter(wn, sigma=(S / 10, S / 220), mode="wrap")
    vert = (vert - vert.mean()) / vert.std()
    strat = gaussian_filter(rng.normal(size=(S, S)), sigma=(S / 300, S / 25), mode="wrap")
    strat = (strat - strat.mean()) / strat.std()
    t = n01(0.8 * vert + 0.07 * strat + 0.5 * noise(S, S, 5) + 0.12 * noise(S, S, 40), 0.22)
    base = ramp(t, [(0, (112, 44, 30)), (0.28, (162, 66, 38)), (0.55, (204, 92, 48)), (0.8, (226, 128, 70)), (1, (238, 176, 120))])
    base *= (0.86 + 0.28 * fbm(S, S, 60, 3))[..., None]
    k = S / 1024.0
    pal = [(214, 150, 104), (150, 70, 44), (232, 186, 140), (96, 44, 32)]
    base = pebbles(base, int(S * S / 2500), (1.2, 4.0), pal, rng=rng, hscale=k)
    base = polylines(base, crack_paths(int(14 * S / 1024) + 4, S, S, S / 2.0, rng, vertical=True, wander=0.08, branch=0.0), (84, 32, 22), 1, 0.4)
    return base


def sol_foret(S, rng=RNG):
    t = fbm(S, S, 4, 4)
    base = ramp(t, [(0, (44, 38, 22)), (0.5, (74, 58, 34)), (1, (104, 80, 46))])
    base *= (0.8 + 0.4 * fbm(S, S, 40, 3))[..., None]
    k = S / 1024.0
    pal = [(118, 88, 46), (150, 104, 52), (84, 76, 36), (170, 128, 70), (100, 70, 40)]

    def col(x, y):
        return np.array(pal)[rng.integers(len(pal), size=len(x))] * rng.uniform(0.75, 1.2, (len(x), 1))
    base = strokes(base, int(S * S / 70), (5, 16), (2, 3), col, angle=0.0, spread=3.0, rng=rng, hscale=k)       # feuilles mortes
    base = strokes(base, int(S * S / 40), (8, 22), (1, 1), col, angle=0.0, spread=3.0, rng=rng, hscale=k)       # aiguilles / brindilles
    base *= (0.8 + 0.3 * fbm(S, S, 14, 3))[..., None]
    return base


def riziere(S, rng=RNG):
    """Rizieres : rangs de plants verts sur eau boueuse qui reflete le ciel."""
    mud = ramp(fbm(S, S, 4, 4), [(0, (86, 78, 50)), (1, (130, 124, 80))])
    sky = smooth(fbm(S, S, 5, 3), 0.62, 0.8)
    base = lerp(mud, np.array([156, 166, 150.0]), sky * 0.45)
    xs = np.arange(S)[None, :]
    nrow = 16
    rows = 0.5 + 0.5 * np.sin(2 * np.pi * (xs / S * nrow + 0.35 * noise(S, S, 4)))
    base = lerp(base, np.array([96, 150, 52.0]), smooth(rows, 0.55, 1.0) * 0.4)
    k = S / 1024.0
    n = int(S * S / 16)
    cen = (rng.integers(0, nrow, n) + 0.5) / nrow * S + rng.normal(0, S / nrow * 0.14, n)
    ys = rng.uniform(0, S, n)
    im = to_img(base).convert("RGBA")
    lay = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    greens = np.array([(88, 150, 46), (118, 176, 62), (150, 196, 84), (70, 122, 40), (176, 200, 100)])
    for i in range(n):
        x, y = cen[i] % S, ys[i]
        L = rng.uniform(6, 20) * k
        a = rng.normal(0, 0.5)
        c = greens[rng.integers(len(greens))] * rng.uniform(0.85, 1.15)
        d.line([x, y, x + np.sin(a) * L, y - np.cos(a) * L], fill=tuple(int(v) for v in np.clip(c, 0, 255)) + (255,), width=1)
    base = np.array(Image.alpha_composite(im, lay).convert("RGB"), float)
    return base * (0.88 + 0.24 * fbm(S, S, 30, 3))[..., None]


def boue(S, rng=RNG):
    t = fbm(S, S, 4, 5)
    base = ramp(t, [(0, (62, 44, 30)), (0.5, (96, 70, 46)), (1, (134, 104, 72))])
    cr = np.abs(noise(S, S, 24))
    cracks = smooth(0.07 - cr, 0, 0.07) * smooth(fbm(S, S, 5, 2), 0.45, 0.65)
    base *= (1 - 0.4 * cracks)[..., None]
    wet = smooth(fbm(S, S, 5, 3), 0.7, 0.85)
    base = lerp(base, base * 0.6 + np.array([40, 46, 50.0]) * 0.5, wet * 0.6)
    k = S / 1024.0
    base = pebbles(base, int(S * S / 3000), (1.5, 4), [(110, 96, 80), (150, 130, 106)], rng=rng, hscale=k)
    return base


def roche(S, rng=RNG):
    t = 0.6 * fbm(S, S, 3, 5) + 0.4 * n01(noise(S, S, 8) * 0.8 + 0.6 * gaussian_filter(noise(S, S, 20), (S / 150, S / 20), mode="wrap"))
    base = ramp(np.clip(t, 0, 1), [(0, (84, 76, 70)), (0.45, (128, 118, 106)), (1, (176, 164, 148))])
    cr = np.abs(noise(S, S, 18))
    base *= (1 - 0.45 * smooth(0.06 - cr, 0, 0.06) * smooth(fbm(S, S, 5, 2), 0.45, 0.65))[..., None]
    lich = smooth(fbm(S, S, 18, 3), 0.76, 0.9)
    base = lerp(base, np.array([150, 150, 90.0]), lich * 0.35)
    base *= (0.85 + 0.3 * fbm(S, S, 70, 3))[..., None]
    return base


def eau(S, rng=RNG):
    t = fbm(S, S, 3, 4)
    base = ramp(t, [(0, (54, 78, 66)), (0.5, (72, 96, 82)), (1, (98, 120, 102))])
    rip = np.abs(noise(S, S, 22))
    base = lerp(base, np.array([170, 186, 180.0]), smooth(0.1 - rip, 0, 0.1) * 0.25)
    return base * (0.94 + 0.12 * fbm(S, S, 50, 2))[..., None]


# ----------------------------------------------------------------------------- route
def route(W=768, H=3072, rng=RNG, edge_lines=True, center_line=False):
    """Asphalte 7 m x 28 m. u (0..1) en travers : 0/1 = bords de chaussee, v = longueur."""
    u = ((np.arange(W) + 0.5) / W)[None, :]
    v = ((np.arange(H) + 0.5) / H)[:, None]
    d = np.minimum(u, 1 - u) * 7.0                      # distance au bord (m)
    mpx = W / 7.0                                       # pixels par metre

    macro = fbm(H, W, 5, 4)
    L = 66 + 24 * (macro - 0.5)
    fine = noise(H, W, W / 2.2)
    mid = noise(H, W, W / 6)
    coarse = noise(H, W, W / 14)

    # voies : 2 traces de roues par voie (polies, plus sombres, grain adouci)
    wheel = np.zeros((H, 1 * W))
    for c in (0.136, 0.364, 0.636, 0.864):
        wheel = np.maximum(wheel, np.exp(-((u - c) / 0.034) ** 2))
    wmod = (0.65 + 0.35 * n01(noise(H, 1, 6)))
    wheel = wheel * wmod
    oil = np.maximum(np.exp(-((u - 0.25) / 0.014) ** 2), np.exp(-((u - 0.75) / 0.014) ** 2)) * n01(noise(H, 1, 12), 0.35)

    g = 1.0 - 0.45 * wheel
    L = L * (1 - 0.12 * wheel) * (1 - 0.18 * oil)
    L = L + g * (7.0 * fine + 4.0 * mid + 3.0 * coarse)
    # gravillons clairs / sombres
    L = np.where(noise(H, W, W / 1.4) > 1.7, L + 26 * g, L)
    L = np.where(noise(H, W, W / 1.4) < -1.9, L - 14, L)
    img = np.stack([L, L, L + 3.5], -1)
    img[..., 0] += 2.0

    # fissures longues + fines
    k = W / 768.0
    cr = crack_paths(34, W, H, 360 * k, rng, vertical=True, wander=0.18, branch=1.0)
    img = polylines(img, cr, (24, 24, 26), 1, 0.85)
    cr2 = crack_paths(18, W, H, 140 * k, rng, vertical=False, wander=0.35, branch=1.2)
    img = polylines(img, cr2, (26, 26, 28), 1, 0.75)

    # rustines (reparations) : rectangles plus noirs avec joint de bitume
    for _ in range(7):
        bw, bh = rng.uniform(0.8, 3.2) * mpx, rng.uniform(1.2, 5.0) * mpx
        x0, y0 = rng.uniform(0.2, 6.0) * mpx, rng.uniform(0, H)
        ys = (np.arange(H) - y0) % H
        xs = np.arange(W) - x0
        inside = ((ys[:, None] < bh) & (xs[None, :] >= 0) & (xs[None, :] < bw)).astype(float)
        edge = inside - (np.roll(np.roll(inside, 3, 0), 3, 1) * np.roll(np.roll(inside, -3, 0), -3, 1))
        edge = np.clip(edge, 0, 1) + np.clip(inside - np.roll(inside, 3, 1), 0, 1)
        inside = gaussian_filter(inside, 2.0)
        img = img * (1 - 0.13 * inside)[..., None]
        img = img * (1 - 0.18 * np.clip(gaussian_filter(edge, 1.2), 0, 1))[..., None]

    # nids de poule bien visibles : ovales de 0,3 a 0,9 m, fond sombre avec ombre en haut, gravats clairs en bas, rebord casse
    for _ in range(10):
        r = rng.uniform(0.15, 0.45) * mpx
        cx, cy = rng.choice([0.7, 1.3, 2.2, 3.1, 3.9, 4.8, 5.6, 6.3]) * mpx, rng.uniform(0, H)
        yy = ((np.arange(H) - cy + H / 2) % H) - H / 2
        xx = np.arange(W) - cx
        wob = 1.0 + 0.18 * np.sin(np.arctan2(yy[:, None], xx[None, :] + 1e-6) * 3 + rng.uniform(0, 6))
        q = ((xx[None, :] / (r * 1.3)) ** 2 + (yy[:, None] / r) ** 2) * wob
        hole = smooth(1.0 - q, 0, 0.18)
        rim = np.clip(smooth(1.5 - q, 0, 0.25) - smooth(1.0 - q, 0, 0.1), 0, 1)
        shade = np.clip(-yy[:, None] / (r + 1e-6), -1, 1)          # haut du trou plus sombre, bas plus clair (profondeur)
        img = img * (1 - hole * (0.62 - 0.18 * shade))[..., None]
        img = img + (np.array([138, 124, 98.0]) * (0.22 * rim))[..., None]                       # rebord de bitume casse / sable
        img = img + (np.array([120, 110, 92.0]) * (0.20 * hole * np.clip(shade, 0, 1)))[..., None]  # gravats au fond
    # zones d'asphalte arrachees (plaques plus grandes, bord irregulier)
    for _ in range(4):
        r = rng.uniform(0.5, 1.1) * mpx
        cx, cy = rng.uniform(0.5, 6.5) * mpx, rng.uniform(0, H)
        yy = ((np.arange(H) - cy + H / 2) % H) - H / 2
        xx = np.arange(W) - cx
        q = (xx[None, :] / (r * 0.8)) ** 2 + (yy[:, None] / (r * 1.6)) ** 2 + 0.25 * noise(H, W, 18)
        zone = smooth(1.0 - q, 0, 0.35)
        gravel = ramp(fbm(H, W, 14, 3), [(0, (104, 92, 74)), (1, (150, 136, 108))])
        img = lerp(img, gravel, 0.55 * zone[..., None])

    # lignes de rive delavees (la ligne centrale n'existe pas sur les routes nationales rurales)
    wear = n01(noise(H, W, 11) + 0.8 * noise(H, W, W / 5))
    if edge_lines:
        line = np.clip(1 - np.abs(d - 0.62) / 0.085, 0, 1)
        line = smooth(line, 0.0, 0.5) * np.clip((wear - 0.30) * 2.2, 0, 1) * (noise(H, W, W / 2.5) > -1.1)
        img = lerp(img, np.array([226, 224, 212.0]) * (0.88 + 0.08 * fine[..., None]), line * 0.82)
    if center_line:
        dash = ((v * 28.0) % 9.0 < 3.0).astype(float)
        cl = np.clip(1 - np.abs(u - 0.5) * 7.0 / 0.07, 0, 1) * dash * np.clip((wear - 0.3) * 2, 0, 1)
        img = lerp(img, np.array([230, 226, 200.0]), cl * 0.8)

    # bord effrite : bitume casse irregulier, puis sable pousse par le vent / les roues
    e_b = 0.16 + 0.12 * noise(H, W, 16) + 0.07 * noise(H, W, 60)
    broken = smooth(e_b - d, -0.03, 0.05)
    sand = ramp(fbm(H, W, 10, 3), [(0, (160, 140, 108)), (1, (206, 190, 154))]) * (0.9 + 0.12 * fine[..., None])
    img = lerp(img, sand, broken)
    drift = np.clip(1 - d / (0.55 + 0.55 * n01(noise(H, W, 9))), 0, 1) * n01(noise(H, W, 7), 0.4)
    img = lerp(img, sand, drift * 0.55)
    # petits cailloux de gravier sur le bord
    edge_mask = (d < 0.55)
    img = np.where(edge_mask[..., None] & (noise(H, W, W / 1.6) > 1.4)[..., None], img * 1.25 + 8, img)
    return img


def route_secondaire(S, rng=RNG):
    base = ramp(fbm(S, S, 4, 4), [(0, (66, 62, 60)), (1, (96, 90, 84))])
    base *= (0.88 + 0.22 * fbm(S, S, S / 3, 2))[..., None]
    base = lerp(base, np.array([170, 130, 98.0]), smooth(fbm(S, S, 6, 3), 0.62, 0.82) * 0.55)       # latérite qui remonte
    base = polylines(base, crack_paths(22, S, S, S / 3, rng, wander=0.3), (30, 28, 28), 1, 0.7)
    k = S / 256.0
    base = pebbles(base, int(S * S / 900), (0.6, 1.8), [(132, 120, 108), (166, 150, 130), (100, 90, 82)], rng=rng, hscale=k)
    return base


def piste(S, rng=RNG):
    """Piste de terre : deux ornieres, bande d'herbe au milieu, bords herbeux."""
    xs = (np.arange(S)[None, :] + 0.5) / S
    t = fbm(S, S, 4, 4)
    base = ramp(t, [(0, (140, 70, 44)), (0.5, (176, 94, 56)), (1, (204, 124, 76))])
    rut = np.exp(-((xs - 0.28) / 0.07) ** 2) + np.exp(-((xs - 0.72) / 0.07) ** 2)
    base *= (1 - 0.22 * rut)[..., None]
    rut_sheen = rut * smooth(fbm(S, S, 6, 3), 0.65, 0.85)
    base = lerp(base, np.array([96, 70, 60.0]), rut_sheen * 0.5)
    center = np.exp(-((xs - 0.5) / 0.07) ** 2)
    edges = np.clip(np.abs(xs - 0.5) * 2 - 0.84, 0, 1) / 0.16
    gmask = np.clip(center * 0.8 + edges, 0, 1) * (0.55 + 0.45 * fbm(S, S, 14, 3))
    k = S / 256.0
    base = lerp(base, ramp(fbm(S, S, 20, 2), [(0, (92, 98, 48)), (1, (150, 134, 72))]), gmask * 0.85)

    def col(x, y):
        return np.array([130, 128, 66]) * rng.uniform(0.7, 1.3, (len(x), 1))
    # brins uniquement la ou il y a de l'herbe (rejet simple)
    base = strokes(base, int(S * S / 28), (5, 14), (1, 1), col, spread=0.5, rng=rng, hscale=k)
    base = pebbles(base, int(S * S / 700), (0.7, 2.4), [(206, 172, 138), (140, 90, 64), (110, 70, 54)], rng=rng, hscale=k)
    return base


# ----------------------------------------------------------------------------- batiments
def _plaster(h, w, base, rng, stain=True):
    t = fbm(h, w, 4, 4)
    img = np.array(base, float)[None, None, :] * (0.88 + 0.22 * t[..., None])
    img *= (0.93 + 0.14 * fbm(h, w, 40, 3))[..., None]
    if stain:
        # coulees de pluie
        streak = gaussian_filter(rng.normal(size=(h, w)), sigma=(h / 8, 1.2), mode="wrap")
        streak = (streak - streak.mean()) / streak.std()
        img *= (1 - 0.07 * np.clip(streak, 0, 3))[..., None]
        # eclaboussures de terre rouge + humidite en bas du mur
        ys = np.arange(h)[:, None] / h
        splash = np.clip((ys - (0.78 - 0.07 * noise(h, w, 14))) / 0.22, 0, 1)
        img = lerp(img, np.array([150, 80, 52.0]) * (0.8 + 0.3 * fbm(h, w, 30, 2))[..., None], splash * 0.6)
        mold = smooth(fbm(h, w, 8, 3), 0.7, 0.86)
        img = lerp(img, np.array([86, 98, 70.0]), mold * 0.25 * (ys < 0.9))
    return img


def mur(S_w, S_h, base, brick=False, shutter=(70, 112, 76), rng=RNG):
    w, h = S_w, S_h
    k = w / 256.0
    img = _plaster(h, w, base, rng)
    if brick:
        bh, bw = int(8 * k), int(24 * k)
        for r_, y in enumerate(range(0, h, bh)):
            img[y:y + max(1, int(k)), :, :] = (206, 190, 166)
            off = 0 if r_ % 2 == 0 else bw // 2
            for x in range(off, w, bw):
                img[y:y + bh, x:x + max(1, int(k)), :] = (206, 190, 166)
        tone = np.kron(rng.uniform(0.82, 1.12, (h // bh + 1, w // bw + 2)), np.ones((bh, bw)))[:h, :w]
        img = img * tone[..., None]
    img = polylines(img, crack_paths(5, w, h, w / 3, rng, wander=0.4), (60, 54, 50), max(1, int(k)), 0.45)
    im = to_img(img)
    dr = ImageDraw.Draw(im)
    s = lambda *a: [int(v * k) for v in a]
    sc = tuple(int(c * 0.65) for c in shutter)
    # volets de bois (lames)
    for x0 in (70, 162):
        dr.rectangle(s(x0, 62, x0 + 24, 142), fill=shutter)
        for y in range(66, 142, 6):
            dr.line(s(x0, y, x0 + 24, y), fill=sc, width=max(1, int(k)))
        dr.line(s(x0, 62, x0, 142), fill=sc, width=max(1, int(k)))
    # cadre + vitre avec reflet de ciel
    dr.rectangle(s(94, 62, 162, 142), fill=(226, 220, 208))
    dr.rectangle(s(98, 66, 158, 138), fill=(40, 50, 60))
    for i in range(0, 72, 3):
        c = 70 + int(i * 0.9)
        dr.line(s(98, 66 + i, 158, 66 + i), fill=(c, c + 14, c + 26), width=max(1, int(k * 3)))
    dr.line(s(128, 66, 128, 138), fill=(226, 220, 208), width=max(2, int(3 * k)))
    dr.line(s(98, 102, 158, 102), fill=(226, 220, 208), width=max(2, int(3 * k)))
    # linteau / appui
    dr.rectangle(s(90, 140, 166, 145), fill=(190, 184, 172))
    im = ImageEnhance.Contrast(im).enhance(1.06)
    return np.array(im, float)


def toit_tole(S, rng=RNG):
    xs = np.arange(S)[None, :]
    prof = np.repeat(0.78 + 0.22 * np.sin(xs * 2 * np.pi / (S / 12)), S, axis=0)
    t = fbm(S, S, 5, 4)
    rust = smooth(fbm(S, S, 6, 4), 0.52, 0.72)
    base = ramp(t, [(0, (120, 124, 126)), (1, (168, 172, 172))])
    base = lerp(base, ramp(fbm(S, S, 18, 3), [(0, (112, 58, 36)), (1, (176, 96, 54))]), rust * 0.85)
    base *= prof[..., None] * (0.9 + 0.2 * fbm(S, S, 50, 2))[..., None]
    streak = gaussian_filter(rng.normal(size=(S, S)), sigma=(S / 6, 1.0), mode="wrap")
    streak = (streak - streak.mean()) / streak.std()
    base *= (1 - 0.08 * np.clip(streak, 0, 3))[..., None]
    for y in (S // 4, 3 * S // 4):     # lignes de clous
        for x in range(0, S, S // 12):
            base[y - 1:y + 2, x + 2:x + 5, :] *= 0.55
    return base


def toit_tuile(S, rng=RNG):
    tw, th = S // 10, S // 11
    gh, gw = S // th + 1, S // tw + 2
    tone = rng.uniform(0.75, 1.15, (gh, gw))
    tint = rng.uniform(-1, 1, (gh, gw))
    T = np.kron(tone, np.ones((th, tw)))[:S, :S]
    Ti = np.kron(tint, np.ones((th, tw)))[:S, :S]
    base = np.stack([170 * T + 10 * Ti, 78 * T - 6 * Ti, 48 * T - 8 * Ti], -1)
    xs = np.arange(S)[None, :]
    ys = np.arange(S)[:, None]
    shade = (0.82 + 0.18 * np.sin(xs * 2 * np.pi / tw)) * (0.62 + 0.38 * ((ys % th) / th) ** 0.6)
    base *= shade[..., None]
    base *= (0.88 + 0.24 * fbm(S, S, 30, 3))[..., None]
    moss = smooth(fbm(S, S, 6, 3), 0.7, 0.86)
    base = lerp(base, np.array([76, 96, 56.0]), moss * 0.5)
    return base


def brique(S, rng=RNG):
    bh, bw = max(4, S // 16), max(8, S // 8)
    gh, gw = S // bh + 1, S // bw + 2
    tone = np.kron(rng.uniform(0.78, 1.18, (gh, gw)), np.ones((bh, bw)))[:S, :S]
    tint = np.kron(rng.uniform(-1, 1, (gh, gw)), np.ones((bh, bw)))[:S, :S]
    base = np.stack([158 * tone + 14 * tint, 78 * tone, 56 * tone - 10 * tint], -1)
    base *= (0.85 + 0.3 * fbm(S, S, 40, 3))[..., None]
    mortar = np.zeros((S, S))
    for r_, y in enumerate(range(0, S, bh)):
        mortar[y:y + max(1, bh // 8), :] = 1
        off = 0 if r_ % 2 == 0 else bw // 2
        for x in range(off, S, bw):
            mortar[y:y + bh, x:x + max(1, bw // 16)] = 1
    return lerp(base, np.array([196, 184, 164.0]) * (0.9 + 0.15 * fbm(S, S, 50, 2))[..., None], mortar * 0.9)


def beton(S, base=(158, 156, 150), rng=RNG, rough=1.0):
    t = fbm(S, S, 4, 4)
    img = np.array(base, float)[None, None, :] * (0.84 + 0.3 * t[..., None])
    img *= (0.9 + 0.18 * fbm(S, S, S / 4, 2))[..., None]
    pores = noise(S, S, S / 2.5) < -1.7
    img = np.where(pores[..., None], img * 0.62, img)
    img = polylines(img, crack_paths(6, S, S, S / 2.5, rng, wander=0.4), (70, 68, 64), 1, 0.5)
    stain = smooth(fbm(S, S, 6, 3), 0.7, 0.88)
    return lerp(img, img * np.array([0.78, 0.82, 0.74]), stain * 0.6 * rough)


def bois(S, rng=RNG):
    ys = np.arange(S)[:, None]
    base = ramp(fbm(S, S, 3, 3), [(0, (96, 66, 40)), (1, (146, 108, 68))])
    grain_ = gaussian_filter(rng.normal(size=(S, S)), sigma=(1.0, S / 8), mode="wrap")
    grain_ = (grain_ - grain_.mean()) / grain_.std()
    base *= (0.86 + 0.12 * grain_)[..., None]
    pw = S // 5
    for y in range(0, S, pw):
        base[y:y + 2, :, :] *= 0.45
        base[y:y + pw, :, :] *= rng.uniform(0.85, 1.12)
    return base * (0.9 + 0.2 * fbm(S, S, 25, 2))[..., None]


def trottoir(S, rng=RNG):
    img = beton(S, (166, 163, 156), rng)
    d = S // 2
    img[::d, :, :] *= 0.55
    img[:, ::d, :] *= 0.55
    return img


# ----------------------------------------------------------------------------- orchestration
def build(OUT, SIZE=1024, edge_lines=None, rng=None):
    rng = rng or RNG
    if edge_lines is None:
        edge_lines = os.environ.get("ROAD_EDGE_LINES", "1") != "0"
    center = os.environ.get("ROAD_CENTER_LINE", "0") == "1"
    os.makedirs(OUT, exist_ok=True)
    jobs = [
        ("grass", lambda: herbe(SIZE, rng)),
        ("shoulder", lambda: bord(SIZE, rng)),
        ("laterite", lambda: laterite(SIZE, rng)),
        ("falaise", lambda: falaise(SIZE, rng)),
        ("forest", lambda: sol_foret(SIZE, rng)),
        ("paddy", lambda: riziere(SIZE, rng)),
        ("mud", lambda: boue(SIZE, rng)),
        ("rock", lambda: roche(SIZE, rng)),
        ("water", lambda: eau(512, rng)),
        ("water_surface", lambda: eau(256, rng)),
    ]
    for name, fn in jobs:
        save(OUT, name, fn(), 90)
        print(f"[texlib] {name}", flush=True)
    save(OUT, "road_main", route(768, 3072, rng, edge_lines, center), 90)
    print("[texlib] road_main", flush=True)
    save(OUT, "road_minor", route_secondaire(512, rng), 88)
    save(OUT, "track", piste(512, rng), 88)
    for nm, base, br, sh in (("wall_white", (222, 216, 204), False, (70, 112, 76)), ("wall_ochre", (206, 170, 108), False, (60, 100, 120)),
                             ("wall_pink", (206, 150, 132), False, (76, 108, 70)), ("wall_brick", (164, 86, 60), True, (110, 70, 40)),
                             ("wall_green", (150, 178, 150), False, (130, 70, 50))):
        save(OUT, nm, mur(512, 384, base, br, sh, rng), 88)
    save(OUT, "roof_tin", toit_tole(512, rng), 88)
    save(OUT, "roof_tile", toit_tuile(512, rng), 88)
    save(OUT, "brique", brique(512, rng), 88)
    save(OUT, "beton", beton(512, rng=rng), 88)
    save(OUT, "pole", beton(512, (140, 138, 130), rng), 88)
    save(OUT, "trottoir", trottoir(512, rng), 88)
    save(OUT, "bois", bois(512, rng), 88)
    print("[texlib] termine", flush=True)
