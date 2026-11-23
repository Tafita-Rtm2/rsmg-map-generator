"""Atlas de textures generees avec du texte (bornes kilometriques, panneaux de villes)."""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"]


def font(size):
    for p in FONTS:
        try:
            return ImageFont.truetype(p, size)
        except Exception:  # noqa
            pass
    try:
        return ImageFont.load_default(size)
    except Exception:  # noqa
        return ImageFont.load_default()


def fit(draw, text, width, max_size, min_size=10):
    for s in range(max_size, min_size - 1, -2):
        f = font(s)
        if draw.textlength(text, font=f) <= width:
            return f
    return font(min_size)


class Atlas:
    def __init__(self, prefix, cw, ch, page=1024):
        self.prefix, self.cw, self.ch, self.page = prefix, cw, ch, page
        self.cols, self.rows = page // cw, page // ch
        self.items = []

    def add(self, fn):
        idx = len(self.items)
        per = self.cols * self.rows
        pg, k = idx // per, idx % per
        col, row = k % self.cols, k // self.cols
        self.items.append((pg, col, row, fn))
        u0, u1 = col * self.cw / self.page, (col + 1) * self.cw / self.page
        v1, v0 = 1 - row * self.ch / self.page, 1 - (row + 1) * self.ch / self.page
        return f"{self.prefix}_{pg}", (u0, v0, u1, v1)

    def save(self, outdir):
        pages = {}
        for pg, col, row, fn in self.items:
            im = pages.setdefault(pg, Image.new("RGB", (self.page, self.page), (255, 255, 255)))
            im.paste(fn(self.cw, self.ch), (col * self.cw, row * self.ch))
        for pg, im in pages.items():
            im.save(os.path.join(outdir, f"{self.prefix}_{pg}.jpg"), quality=90, optimize=True)
        return len(pages)


def plaster(w, h, base=(232, 232, 226), amp=14, seed=3):
    r = np.random.default_rng(seed)
    n = r.normal(0, amp, (h, w, 1))
    return Image.fromarray(np.clip(np.array(base, float)[None, None, :] + n, 0, 255).astype(np.uint8))


def draw_borne(w, h, code, town, km, seed=0):
    """Borne kilometrique malgache : sommet rouge avec le code de la route, corps blanc avec la ville et le km."""
    im = plaster(w, h, seed=seed)
    d = ImageDraw.Draw(im)
    top = int(h * 0.32)
    d.rectangle([0, 0, w, top], fill=(200, 32, 38))
    d.rectangle([0, top, w, top + 6], fill=(150, 20, 25))
    f = fit(d, code, w * 0.7, int(h * 0.2))
    d.text((w / 2, top / 2), code, font=f, fill=(255, 255, 255), anchor="mm")
    if town:
        f = fit(d, town, w * 0.92, int(h * 0.1))
        d.text((w / 2, top + (h - top) * 0.28), town, font=f, fill=(15, 15, 15), anchor="mm")
    f = fit(d, f"{km}", w * 0.7, int(h * 0.2))
    d.text((w / 2, top + (h - top) * 0.62), f"{km}", font=f, fill=(15, 15, 15), anchor="mm")
    f = fit(d, "km", w * 0.5, int(h * 0.08))
    d.text((w / 2, top + (h - top) * 0.80), "km", font=f, fill=(15, 15, 15), anchor="mm")
    d.rectangle([0, int(h * 0.93), w, h], fill=(190, 190, 184))
    return im


def draw_panneau(w, h, text, style="ville", sub=""):
    if style == "shop":
        im = Image.new("RGB", (w, h), (250, 214, 60))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, w - 1, h - 1], outline=(190, 30, 36), width=max(6, h // 12))
        fg = (190, 30, 36)
    elif style == "ville":
        im = Image.new("RGB", (w, h), (246, 246, 242))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, w - 1, h - 1], outline=(200, 30, 36), width=max(8, h // 14))
        fg = (10, 10, 10)
    else:
        im = Image.new("RGB", (w, h), (0, 84, 160))
        d = ImageDraw.Draw(im)
        d.rectangle([6, 6, w - 7, h - 7], outline=(255, 255, 255), width=max(4, h // 28))
        fg = (255, 255, 255)
    f = fit(d, text, w * 0.86, int(h * 0.46))
    d.text((w / 2, h * (0.42 if sub else 0.5)), text, font=f, fill=fg, anchor="mm")
    if sub:
        f2 = fit(d, sub, w * 0.8, int(h * 0.2))
        d.text((w / 2, h * 0.76), sub, font=f2, fill=fg, anchor="mm")
    return im
