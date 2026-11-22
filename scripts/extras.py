"""Petites formes 3D (boites, poteaux, cables, abribus, station-service) renvoyees comme (V, UV, F)."""
import math
import numpy as np


def _fix_out(V, F, center):
    tri = V[F]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = (nrm * (tri.mean(1) - center)).sum(1) < 0
    F = F.copy()
    F[flip] = F[flip][:, [0, 2, 1]]
    return F


def box(cx, cy, z0, ang, ha, hb, h, tile=2.0, bottom=False):
    """Boite orientee : (cx, cy) centre, ang = direction de l'axe long (rad), ha/hb demi-longueur/demi-largeur, h hauteur."""
    ax = np.array([math.cos(ang), math.sin(ang)])
    bx = np.array([-ax[1], ax[0]])
    c = np.array([cx, cy])
    cor = [c - ax * ha - bx * hb, c + ax * ha - bx * hb, c + ax * ha + bx * hb, c - ax * ha + bx * hb]
    V, UV, F = [], [], []
    for k in range(4):
        p0, p1 = cor[k], cor[(k + 1) % 4]
        w = float(np.linalg.norm(p1 - p0))
        b = len(V)
        V += [[p0[0], p0[1], z0], [p1[0], p1[1], z0], [p1[0], p1[1], z0 + h], [p0[0], p0[1], z0 + h]]
        UV += [[0, 0], [w / tile, 0], [w / tile, h / tile], [0, h / tile]]
        F += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
    b = len(V)
    for p in cor:
        V.append([p[0], p[1], z0 + h])
        UV.append([p[0] / tile, p[1] / tile])
    F += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
    if bottom:
        b = len(V)
        for p in cor:
            V.append([p[0], p[1], z0])
            UV.append([p[0] / tile, p[1] / tile])
        F += [[b, b + 2, b + 1], [b, b + 3, b + 2]]
    return np.array(V, float), np.array(UV, float), np.array(F, int)


def cylinder(cx, cy, z0, h, r0, r1, seg=8, tile=3.0):
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    bot = np.column_stack([cx + r0 * np.cos(ang), cy + r0 * np.sin(ang), np.full(seg, z0)])
    top = np.column_stack([cx + r1 * np.cos(ang), cy + r1 * np.sin(ang), np.full(seg, z0 + h)])
    V = np.vstack([bot, top])
    UV = np.column_stack([np.tile(ang / (2 * np.pi) * 2, 2), np.concatenate([np.zeros(seg), np.full(seg, h / tile)])])
    F = []
    for i in range(seg):
        j = (i + 1) % seg
        F += [[i, j, seg + j], [i, seg + j, seg + i]]
    return V, UV, np.array(F, int)


def ribbon(a, b, w=0.04):
    """Cable fin entre deux points 3D : deux rubans croises, chacun visible des deux cotes."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    d2 = np.array([d[0], d[1], 0.0])
    nd = np.linalg.norm(d2)
    side = np.array([-d2[1], d2[0], 0.0]) / nd * w if nd > 1e-6 else np.array([w, 0, 0])
    up = np.array([0, 0, w])
    V, F = [], []
    for off in (side, up):
        i = len(V)
        V += [a - off, b - off, b + off, a + off]
        F += [[i, i + 1, i + 2], [i, i + 2, i + 3], [i, i + 2, i + 1], [i, i + 3, i + 2]]
    return np.array(V, float), np.zeros((len(V), 2)), np.array(F, int)


def cable(a, b, sag, seg=5, w=0.04):
    """Cable avec fleche (chainette simplifiee) entre a et b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pts = []
    for k in range(seg + 1):
        t = k / seg
        p = a + (b - a) * t
        p[2] -= 4 * sag * t * (1 - t)
        pts.append(p)
    Vs, Fs, n = [], [], 0
    for k in range(seg):
        V, _, F = ribbon(pts[k], pts[k + 1], w)
        Vs.append(V)
        Fs.append(F + n)
        n += len(V)
    V = np.vstack(Vs)
    return V, np.zeros((len(V), 2)), np.vstack(Fs)


def merge(parts):
    """Assemble plusieurs (V, UV, F)."""
    Vs, Us, Fs, n = [], [], [], 0
    for V, UV, F in parts:
        Vs.append(V)
        Us.append(UV)
        Fs.append(F + n)
        n += len(V)
    return np.vstack(Vs), np.vstack(Us), np.vstack(Fs)


def pole(x, y, z, ang, h=9.0, arm=1.8):
    """Poteau electrique (fut conique + traverse). ang = direction de la ligne. Renvoie (parties pole, tetes de cable)."""
    parts = [cylinder(x, y, z, h, 0.14, 0.09, 8)]
    parts.append(box(x, y, z + h - 0.6, ang + math.pi / 2, arm, 0.05, 0.1, 1.0))
    ax = np.array([math.cos(ang + math.pi / 2), math.sin(ang + math.pi / 2)])
    heads = [np.array([x, y, z + h - 0.45]) + np.array([ax[0] * o, ax[1] * o, 0.0]) for o in (-arm * 0.8, 0.0, arm * 0.8)]
    return merge(parts), heads


def lamp(x, y, z, ang_to_road, h=6.5):
    """Lampadaire : mat + bras vers la route + tete."""
    parts = [cylinder(x, y, z, h, 0.11, 0.07, 8)]
    ax = np.array([math.cos(ang_to_road), math.sin(ang_to_road)])
    parts.append(box(x + ax[0] * 0.6, y + ax[1] * 0.6, z + h - 0.15, ang_to_road, 0.7, 0.04, 0.08, 1.0))
    parts.append(box(x + ax[0] * 1.3, y + ax[1] * 1.3, z + h - 0.3, ang_to_road, 0.35, 0.12, 0.18, 1.0))
    return merge(parts)


def bus_shelter(x, y, z, ang):
    """Abribus : toit de tole sur 4 poteaux, banc, panneau arriere."""
    ax = np.array([math.cos(ang), math.sin(ang)])
    bx = np.array([-ax[1], ax[0]])
    roof = [box(x, y, z + 2.45, ang, 1.7, 0.95, 0.12, 3.0)]
    posts = []
    for sa in (-1.5, 1.5):
        for sb in (-0.8, 0.8):
            posts.append(box(x + ax[0] * sa + bx[0] * sb, y + ax[1] * sa + bx[1] * sb, z, ang, 0.06, 0.06, 2.45, 1.0))
    back = [box(x - bx[0] * 0.85, y - bx[1] * 0.85, z + 0.2, ang, 1.5, 0.03, 1.4, 2.0)]
    bench = [box(x - bx[0] * 0.55, y - bx[1] * 0.55, z + 0.4, ang, 1.2, 0.18, 0.06, 1.0)]
    return merge(roof), merge(posts + back + bench)


def skin_box(cx, cy, z0, ang, ha, hb, h, big=None, plain=(0.75, 0.5), tile=2.0):
    """Boite dont les deux grandes faces (+b / -b) portent une image (big = (u0, v0, u1, v1) dans l'atlas) et dont les
    autres faces prennent une couleur unie lue a `plain` dans la texture. Le texte reste lisible des deux cotes."""
    ax = np.array([math.cos(ang), math.sin(ang)])
    bx = np.array([-ax[1], ax[0]])
    c = np.array([cx, cy])
    cor = [c - ax * ha - bx * hb, c + ax * ha - bx * hb, c + ax * ha + bx * hb, c - ax * ha + bx * hb]
    V, UV, F = [], [], []
    for k in range(4):
        p0, p1 = cor[k], cor[(k + 1) % 4]
        b = len(V)
        V += [[p0[0], p0[1], z0], [p1[0], p1[1], z0], [p1[0], p1[1], z0 + h], [p0[0], p0[1], z0 + h]]
        if big is not None and k in (0, 2):
            u0, v0, u1, v1 = big
            UV += [[u0, v0], [u1, v0], [u1, v1], [u0, v1]]
        else:
            UV += [list(plain)] * 4
        F += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
    b = len(V)
    for q in cor:
        V.append([q[0], q[1], z0 + h])
        UV.append(list(plain))
    F += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
    return np.array(V, float), np.array(UV, float), np.array(F, int)


def quad2(p0, p1, p2, p3, uv=((0, 0), (1, 0), (1, 1), (0, 1))):
    """Quadrilatere visible des deux cotes (toile de marche, panneau fin)."""
    V = np.array([p0, p1, p2, p3], float)
    UV = np.array(uv, float)
    F = np.array([[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]])
    return V, UV, F


def fuel_station(x, y, z, ang):
    """Station-service Jovena : auvent rouge avec logos, 2 pompes (face Jovena), boutique, atelier d'entretien, pylone prix.
    Renvoie {materiau: [(V, UV, F), ...]}."""
    ax = np.array([math.cos(ang), math.sin(ang)])
    bx = np.array([-ax[1], ax[0]])
    parts = {}

    def add(m, p):
        parts.setdefault(m, []).append(p)

    def at(a, b):
        return x + ax[0] * a + bx[0] * b, y + ax[1] * a + bx[1] * b
    add("jovena_rouge", box(x, y, z + 5.0, ang, 7.0, 4.5, 0.6, 4.0))
    for sb in (-4.62, 4.62):                      # logos sur les deux grandes faces de l'auvent
        px, py = at(0.0, sb)
        add("jovena_logo", skin_box(px, py, z + 5.02, ang, 1.5, 0.02, 0.56, big=(0, 0, 0.5, 1)))
    for a in (-5.5, 5.5):
        for b in (-3.0, 3.0):
            px, py = at(a, b)
            add("beton", box(px, py, z, ang, 0.25, 0.25, 5.0, 2.0))
    for a in (-2.2, 2.2):
        px, py = at(a, 0.0)
        add("beton", box(px, py, z, ang, 1.4, 0.7, 0.18, 2.0))
        add("jovena_pompe", skin_box(px, py, z + 0.18, ang, 0.38, 0.3, 1.6, big=(0, 0, 0.5, 1)))
    ox, oy = at(-1.0, -7.5)
    add("jovena_boutique", skin_box(ox, oy, z, ang, 3.5, 2.5, 3.0, big=(0, 0, 0.5, 1)))
    add("roof_tin", box(ox, oy, z + 3.0, ang, 3.9, 2.9, 0.2, 3.0))
    gx, gy = at(6.6, -7.5)
    add("jovena_entretien", skin_box(gx, gy, z, ang, 2.0, 2.5, 3.2, big=(0, 0, 0.5, 1)))
    add("roof_tin", box(gx, gy, z + 3.2, ang, 2.4, 2.9, 0.2, 3.0))
    px, py = at(-9.0, 6.5)                         # pylone : panneau Jovena + tableau des prix (visibles le long de la route)
    add("pole", cylinder(px, py, z, 5.2, 0.12, 0.1, 8))
    add("jovena_panneau", skin_box(px, py, z + 3.0, ang + math.pi / 2, 0.72, 0.03, 1.95, big=(0, 0, 0.5, 1)))
    add("jovena_prix", skin_box(px, py, z + 1.45, ang + math.pi / 2, 0.55, 0.03, 1.45, big=(0, 0, 0.5, 1)))
    return parts


def kiosk(x, y, z, ang):
    """Cash point (Airtel Money / MVola / Orange Money) : cabine plus grande que le vrai, toit rouge. ang: face avant = +b."""
    parts = {"cashpoint": [skin_box(x, y, z, ang, 0.65, 0.6, 2.6, big=(0, 0, 0.5, 1), plain=(0.75, 0.25)),
                           skin_box(x, y, z + 2.6, ang, 0.78, 0.72, 0.14, big=None, plain=(0.75, 0.75))]}
    return parts


def stall(x, y, z, ang, rng, charcoal=False):
    """Etal de marche local : 4 poteaux, table, toile inclinee, marchandises colorees (ou sacs de charbon). Face avant = +b."""
    ax = np.array([math.cos(ang), math.sin(ang)])
    bx = np.array([-ax[1], ax[0]])
    parts = {}

    def add(m, p):
        parts.setdefault(m, []).append(p)

    def at(a, b, zz):
        return [x + ax[0] * a + bx[0] * b, y + ax[1] * a + bx[1] * b, z + zz]
    for a in (-1.25, 1.25):
        add("bois", box(x + ax[0] * a + bx[0] * 1.0, y + ax[1] * a + bx[1] * 1.0, z, ang, 0.04, 0.04, 2.3, 1.0))
        add("bois", box(x + ax[0] * a - bx[0] * 1.0, y + ax[1] * a - bx[1] * 1.0, z, ang, 0.04, 0.04, 1.9, 1.0))
    add("bache", quad2(at(-1.5, 1.2, 2.3), at(1.5, 1.2, 2.3), at(1.5, -1.2, 1.9), at(-1.5, -1.2, 1.9),
                       uv=((0, 0), (1, 0), (1, 1), (0, 1))))
    add("bois", box(x + bx[0] * 0.55, y + bx[1] * 0.55, z + 0.75, ang, 1.15, 0.4, 0.06, 1.0))
    for a in (-1.0, 1.0):
        add("bois", box(x + ax[0] * a + bx[0] * 0.55, y + ax[1] * a + bx[1] * 0.55, z, ang, 0.04, 0.04, 0.75, 1.0))
    if charcoal:
        for i in range(int(rng.integers(3, 6))):
            a = -0.9 + i * 0.5
            add("sac_charbon", skin_box(x + ax[0] * a + bx[0] * 0.55, y + ax[1] * a + bx[1] * 0.55, z + 0.81, ang + 0.1,
                                        0.26, 0.17, 0.8, big=(0, 0, 0.5, 1), plain=(0.75, 0.5)))
    else:
        for i in range(int(rng.integers(3, 6))):
            a = -1.0 + i * 0.5
            cell = float(rng.integers(0, 8)) / 8.0 + 1 / 16.0
            add("marchandise", skin_box(x + ax[0] * a + bx[0] * 0.55, y + ax[1] * a + bx[1] * 0.55, z + 0.81, ang,
                                        0.2, 0.2, float(rng.uniform(0.12, 0.3)), big=None, plain=(cell, 0.5)))
    return parts


def sack_pile(x, y, z, ang, rng, n=6):
    """Pile de sacs de charbon au bord de la route (2 rangees)."""
    ax = np.array([math.cos(ang), math.sin(ang)])
    out = []
    for i in range(n):
        row, col = (0, i) if i < 4 else (1, i - 3.5)
        px, py = x + ax[0] * (col * 0.58), y + ax[1] * (col * 0.58)
        out.append(skin_box(px, py, z + row * 0.8, ang + float(rng.normal(0, 0.12)), 0.26, 0.17, 0.8,
                            big=(0, 0, 0.5, 1), plain=(0.75, 0.5)))
    return {"sac_charbon": out}
