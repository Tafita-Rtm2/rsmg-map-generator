#!/usr/bin/env python3
"""Construit la geometrie (relief, route, maisons, villages, vegetation) par troncons de 2 km.
Sortie : work/tiles/tile_NN.json (maillages), tile_NN_trees.csv, work/manifest.json, work/route.csv.
Les maillages sont ensuite assembles et exportes par Blender (scripts/blender_build.py)."""
import argparse, math, os
import numpy as np
import shapely
import mapbox_earcut as earcut
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from common import transformers, load_json, save_json

STEP = 5.0          # m entre deux stations le long de la route
TILE_ST = 400       # stations par troncon (2 km)
ROAD_HALF = 3.5     # demi-largeur de chaussee
SHOULDER = 5.5      # fin de l'accotement
W_MAX = 300.0       # demi-largeur du relief genere
OFFS_POS = [3.5, 5.5, 8, 11, 15, 20, 27, 36, 48, 62, 80, 102, 130, 165, 210, 260, 300]
TEXM = {"grass": 6, "laterite": 5, "rock": 8, "forest": 6, "paddy": 5, "mud": 5, "water": 10, "shoulder": 3}
KIND = {"road_main": "GND", "road_minor": "GND", "track": "GND"}
for _m in TEXM:
    KIND[_m] = "GND"
SH, GR, LA, RO, FO, PA, MU, WA = range(8)
CLS = {SH: "shoulder", GR: "grass", LA: "laterite", RO: "rock", FO: "forest", PA: "paddy", MU: "mud", WA: "water"}


class Acc:
    def __init__(self):
        self.v, self.uv, self.f, self.n = [], [], [], 0

    def add(self, V, UV, F):
        V = np.asarray(V, float).reshape(-1, 3)
        UV = np.asarray(UV, float).reshape(-1, 2)
        F = np.asarray(F, np.int64).reshape(-1, 3)
        if len(F) == 0:
            return
        self.v.append(V)
        self.uv.append(UV)
        self.f.append(F + self.n)
        self.n += len(V)

    def finish(self):
        return np.vstack(self.v), np.vstack(self.uv), np.vstack(self.f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--max-trees", type=int, default=110000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    W = a.work
    rng = np.random.default_rng(a.seed)
    road = load_json(f"{W}/road.json")
    osm = load_json(f"{W}/osm.json")
    D = np.load(f"{W}/dem.npz")
    fwd, inv = transformers()
    dem = D["dem"]
    west, north, ddx, ddy = float(D["west"]), float(D["north"]), float(D["dx"]), float(D["dy"])

    def dem_h(x, y):
        x = np.asarray(x, float)
        shp = x.shape
        lon, lat = inv.transform(x.ravel(), np.asarray(y, float).ravel())
        col = (np.asarray(lon) - west) / ddx - 0.5
        row = (north - np.asarray(lat)) / ddy - 0.5
        return map_coordinates(dem, [row, col], order=1, mode="nearest").reshape(shp)

    # ------------------------------------------------------------------ axe de la route
    P = np.array(road["pts"], float)
    P = np.column_stack([gaussian_filter1d(P[:, 0], 2.0, mode="nearest"),
                         gaussian_filter1d(P[:, 1], 2.0, mode="nearest")])
    n = len(P)
    S = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    T = np.gradient(P, axis=0)
    T /= np.maximum(np.hypot(T[:, 0], T[:, 1]), 1e-9)[:, None]
    N = np.column_stack([-T[:, 1], T[:, 0]])
    heading = np.arctan2(T[:, 1], T[:, 0])
    curv = gaussian_filter1d(np.gradient(np.unwrap(heading), S), 3.0, mode="nearest")

    h0 = dem_h(P[:, 0], P[:, 1])
    ROADH = gaussian_filter1d(h0, 8, mode="nearest")
    mx = 0.09 * STEP
    for i in range(1, n):
        ROADH[i] = min(max(ROADH[i], ROADH[i - 1] - mx), ROADH[i - 1] + mx)
    for i in range(n - 2, -1, -1):
        ROADH[i] = min(max(ROADH[i], ROADH[i + 1] - mx), ROADH[i + 1] + mx)
    tree = cKDTree(P)
    ntile = (n - 1) // TILE_ST + 1

    def terrain_h(x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        d, i = tree.query(np.column_stack([x.ravel(), y.ravel()]))
        dh = dem_h(x, y).ravel()
        rh = ROADH[i]
        blend = np.clip(np.abs(dh - rh) * 1.6, 6, 45)
        t = np.clip((d - ROAD_HALF - 2.75) / blend, 0, 1)
        t = t * t * (3 - 2 * t)
        return (rh * (1 - t) + dh * t).reshape(x.shape)

    def tile_of_xy(xy):
        _, i = tree.query(np.asarray(xy, float).reshape(-1, 2))
        return np.minimum(i // TILE_ST, ntile - 1)

    corridor = LineString(P[::3]).buffer(W_MAX - 15, cap_style=2)
    shapely.prepare(corridor)

    # ------------------------------------------------------------------ occupation du sol OSM
    def poly_of(w):
        pts = w["pts"]
        if len(pts) < 4 or not np.allclose(pts[0], pts[-1]):
            return None
        try:
            p = Polygon(pts)
            if not p.is_valid:
                p = p.buffer(0)
            if p.is_empty:
                return None
            if p.geom_type == "MultiPolygon":
                p = max(p.geoms, key=lambda g: g.area)
            return p
        except Exception:  # noqa
            return None

    wood, farm, resid, water, rockp, bways, minor = [], [], [], [], [], [], []
    for w in osm["ways"]:
        t = w["tags"]
        if "building" in t:
            p = poly_of(w)
            if p is not None:
                bways.append((p, t))
            continue
        if "waterway" in t and len(w["pts"]) >= 2:
            wd = {"river": 7, "canal": 4, "stream": 2.5}.get(t["waterway"], 1.5)
            water.append(LineString(w["pts"]).buffer(wd))
            continue
        if "highway" in t:
            minor.append(w)
            continue
        p = poly_of(w)
        if p is None:
            continue
        nat, lu = t.get("natural"), t.get("landuse")
        if nat in ("wood", "scrub", "heath") or lu == "forest":
            wood.append(p)
        elif lu in ("farmland", "paddy", "meadow", "orchard", "vineyard") or t.get("crop") == "rice" or nat == "wetland":
            farm.append(p)
        elif lu in ("residential", "commercial", "retail", "industrial"):
            resid.append(p)
        elif nat == "water" or lu in ("reservoir", "basin"):
            water.append(p)
        elif nat in ("bare_rock", "scree"):
            rockp.append(p)

    def union(lst):
        if not lst:
            return None
        u = unary_union(lst)
        shapely.prepare(u)
        return u

    U_wood, U_farm, U_resid, U_water, U_rock = map(union, (wood, farm, resid, water, rockp))

    def inside(U, x, y):
        x = np.asarray(x, float)
        if U is None:
            return np.zeros(x.shape, bool)
        return shapely.contains_xy(U, x.ravel(), np.asarray(y, float).ravel()).reshape(x.shape)

    acc = {}

    def A(mat):
        if mat not in acc:
            acc[mat] = Acc()
        return acc[mat]

    # ------------------------------------------------------------------ relief (bande le long de la route)
    offs = np.array([-o for o in OFFS_POS[::-1]] + OFFS_POS)
    m = len(offs)
    jskip = len(OFFS_POS) - 1
    OE = np.tile(offs, (n, 1))
    lim = np.where(np.abs(curv) > 1e-4, 0.8 / np.maximum(np.abs(curv), 1e-4), 1e9)
    lim = np.maximum(lim, 4.0)[:, None]
    left_inner = (curv > 0)[:, None]
    OE = np.where((OE > 0) & left_inner, np.minimum(OE, lim), OE)
    OE = np.where((OE < 0) & ~left_inner, np.maximum(OE, -lim), OE)
    X = P[:, 0:1] + N[:, 0:1] * OE
    Y = P[:, 1:2] + N[:, 1:2] * OE
    Z = terrain_h(X, Y)
    Z = np.where(np.abs(OE) <= SHOULDER + 0.01, ROADH[:, None], Z)
    Vall = np.stack([X, Y, Z], -1).reshape(-1, 3)

    ii, jj = np.meshgrid(np.arange(n - 1), np.arange(m - 1), indexing="ij")
    v00 = ii * m + jj
    v10 = (ii + 1) * m + jj
    v11 = (ii + 1) * m + jj + 1
    v01 = ii * m + jj + 1
    keep = jj != jskip
    cx = (X[:-1, :-1] + X[1:, :-1] + X[1:, 1:] + X[:-1, 1:]) / 4
    cy = (Y[:-1, :-1] + Y[1:, :-1] + Y[1:, 1:] + Y[:-1, 1:]) / 4
    offc = np.abs((OE[:-1, :-1] + OE[:-1, 1:]) / 2)
    V0 = np.stack([X[:-1, :-1], Y[:-1, :-1], Z[:-1, :-1]], -1)
    V1 = np.stack([X[1:, :-1], Y[1:, :-1], Z[1:, :-1]], -1)
    V2 = np.stack([X[1:, 1:], Y[1:, 1:], Z[1:, 1:]], -1)
    V3 = np.stack([X[:-1, 1:], Y[:-1, 1:], Z[:-1, 1:]], -1)
    nv = np.cross(V2 - V0, V3 - V1)
    nz = np.abs(nv[..., 2]) / np.maximum(np.linalg.norm(nv, axis=-1), 1e-9)
    slope = np.degrees(np.arccos(np.clip(nz, 0, 1)))
    qarea = 0.5 * np.linalg.norm(nv, axis=-1)

    noise = np.sin(cx * 0.011 + 1.3) * np.sin(cy * 0.0093 + 0.4) + 0.5 * np.sin(cx * 0.031 + cy * 0.027)
    cls = np.where(noise > -0.25, GR, LA)
    cls = np.where(inside(U_resid, cx, cy), LA, cls)
    cls = np.where((slope > 24) & (slope <= 36), FO, cls)
    cls = np.where(inside(U_wood, cx, cy), FO, cls)
    cls = np.where(inside(U_farm, cx, cy) & (slope < 14), PA, cls)
    cls = np.where(inside(U_rock, cx, cy) | (slope > 36), RO, cls)
    cls = np.where(inside(U_water, cx, cy), WA, cls)
    cls = np.where(offc <= SHOULDER, SH, cls)

    for code, name in CLS.items():
        sel = keep & (cls == code)
        if not sel.any():
            continue
        a_, b_, c_, d_ = v00[sel], v10[sel], v11[sel], v01[sel]
        F = np.concatenate([np.stack([a_, b_, c_], 1), np.stack([a_, c_, d_], 1)], 0)
        used, invx = np.unique(F.ravel(), return_inverse=True)
        Vm = Vall[used]
        A(name).add(Vm, Vm[:, :2] / TEXM[name], invx.reshape(-1, 3))

    # ------------------------------------------------------------------ chaussee principale
    cols = np.array([-ROAD_HALF, 0.0, ROAD_HALF])
    RX = P[:, 0:1] + N[:, 0:1] * cols
    RY = P[:, 1:2] + N[:, 1:2] * cols
    RZ = np.repeat(ROADH[:, None], 3, axis=1)
    RV = np.stack([RX, RY, RZ], -1).reshape(-1, 3)
    RUV = np.stack([np.tile([0.0, 0.5, 1.0], (n, 1)), np.repeat((S / 27.0)[:, None], 3, axis=1)], -1).reshape(-1, 2)
    ri, rj = np.meshgrid(np.arange(n - 1), np.arange(2), indexing="ij")
    q00, q10, q11, q01 = ri * 3 + rj, (ri + 1) * 3 + rj, (ri + 1) * 3 + rj + 1, ri * 3 + rj + 1
    RF = np.concatenate([np.stack([q00.ravel(), q10.ravel(), q11.ravel()], 1),
                         np.stack([q00.ravel(), q11.ravel(), q01.ravel()], 1)], 0)
    A("road_main").add(RV, RUV, RF)

    # ------------------------------------------------------------------ routes secondaires et pistes (OSM)
    WID = {"primary": 7, "secondary": 6, "tertiary": 5.5, "unclassified": 4.5, "residential": 4.5,
           "service": 3.5, "track": 3.5, "path": 1.6, "trunk": 7, "living_street": 4}
    UNPAVED = {"unpaved", "dirt", "ground", "gravel", "sand", "earth", "mud", "grass", "compacted", "fine_gravel"}
    minor_lines = []
    for w in minor:
        t = w["tags"]
        hw = t.get("highway", "")
        if hw not in WID or len(w["pts"]) < 2:
            continue
        ln = LineString(w["pts"])
        if ln.length < 8:
            continue
        samp = np.array([ln.interpolate(s).coords[0] for s in np.arange(0, ln.length, 4.0)] + [ln.coords[-1]])
        dmain, _ = tree.query(samp)
        if (dmain < 10).mean() > 0.9:
            continue
        ok = shapely.contains_xy(corridor, samp[:, 0], samp[:, 1]) & (dmain > ROAD_HALF + 0.7)
        mat = "track" if (hw in ("track", "path") or t.get("surface", "") in UNPAVED) else "road_minor"
        wd = WID[hw]
        run = []
        for pt, good in list(zip(samp, ok)) + [(None, False)]:
            if good:
                run.append(pt)
                continue
            if len(run) >= 2:
                r = np.array(run)
                tg = np.gradient(r, axis=0)
                tg /= np.maximum(np.hypot(tg[:, 0], tg[:, 1]), 1e-9)[:, None]
                nn = np.column_stack([-tg[:, 1], tg[:, 0]])
                L = r[:, None, :] + nn[:, None, :] * np.array([-wd / 2, wd / 2])[None, :, None]
                zz = terrain_h(L[..., 0], L[..., 1]) + 0.08
                sd = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(r, axis=0).T))])
                Vv = np.stack([L[..., 0], L[..., 1], zz], -1).reshape(-1, 3)
                UVv = np.stack([np.tile([0.0, 1.0], (len(r), 1)), np.repeat((sd / 5.0)[:, None], 2, axis=1)], -1).reshape(-1, 2)
                k = np.arange(len(r) - 1)
                Fv = np.concatenate([np.stack([2 * k, 2 * k + 2, 2 * k + 3], 1), np.stack([2 * k, 2 * k + 3, 2 * k + 1], 1)], 0)
                A(mat).add(Vv, UVv, Fv)
                minor_lines.append(LineString(r).buffer(wd / 2 + 1.5))
            run = []
    U_minor = union(minor_lines)

    # ------------------------------------------------------------------ maisons
    highland = float(np.mean(ROADH)) > 900
    WALLS_H = (["wall_brick"] * 4 + ["wall_ochre"] * 2 + ["wall_white"] * 2 + ["wall_pink"] + ["wall_green"])
    WALLS_L = (["wall_white"] * 3 + ["wall_ochre"] * 2 + ["wall_pink"] * 2 + ["wall_green"] * 2 + ["wall_brick"] * 2)
    for wm in set(WALLS_H + WALLS_L):
        KIND[wm] = "SOLID"
    KIND["roof_tile"] = KIND["roof_tin"] = "SOLID"

    def pick_mats(z):
        hi = z > 900
        wmat = str(rng.choice(WALLS_H if hi else WALLS_L))
        tile_p = 0.55 if hi else 0.3
        return wmat, ("roof_tile" if rng.random() < tile_p else "roof_tin")

    def fix_out(V, F, center):
        tri = V[F]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        cen = tri.mean(1) - center
        flip = (nrm * cen).sum(1) < 0
        F = F.copy()
        F[flip] = F[flip][:, [0, 2, 1]]
        return F

    def add_house(ring, levels, wmat, rmat):
        ring = np.asarray(ring, float)
        k = len(ring)
        hz = terrain_h(ring[:, 0], ring[:, 1])
        base = float(hz.min())
        bottom = base - 0.5
        top = base + 3.0 * levels
        p0, p1 = ring, np.roll(ring, -1, axis=0)
        el = np.linalg.norm(p1 - p0, axis=1)
        cum = np.concatenate([[0], np.cumsum(el)[:-1]])
        V = np.zeros((k, 4, 3))
        V[:, 0, :2], V[:, 1, :2], V[:, 2, :2], V[:, 3, :2] = p0, p1, p1, p0
        V[:, 0, 2] = V[:, 1, 2] = bottom
        V[:, 2, 2] = V[:, 3, 2] = top
        UV = np.zeros((k, 4, 2))
        UV[:, 0, 0] = UV[:, 3, 0] = cum / 4
        UV[:, 1, 0] = UV[:, 2, 0] = (cum + el) / 4
        UV[:, 0, 1] = UV[:, 1, 1] = (bottom - base) / 3
        UV[:, 2, 1] = UV[:, 3, 1] = (top - base) / 3
        e = np.arange(k) * 4
        F = np.concatenate([np.stack([e, e + 1, e + 2], 1), np.stack([e, e + 2, e + 3], 1)], 0)
        A(wmat).add(V.reshape(-1, 3), UV.reshape(-1, 2), F)
        poly = Polygon(ring)
        cen = np.array([ring[:, 0].mean(), ring[:, 1].mean(), (bottom + top) / 2])
        rr = poly.minimum_rotated_rectangle
        gable = False
        if rr.geom_type == "Polygon" and poly.area / max(rr.area, 1e-6) > 0.6:
            c4 = np.array(rr.exterior.coords)[:4]
            e1, e2 = c4[1] - c4[0], c4[2] - c4[1]
            l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
            if min(l1, l2) > 3.0:
                gable = True
                ctr = c4.mean(0)
                if l1 >= l2:
                    ax, bx, la, lb = e1 / l1, e2 / l2, l1 / 2, l2 / 2
                else:
                    ax, bx, la, lb = e2 / l2, e1 / l1, l2 / 2, l1 / 2
        if gable:
            ov = 0.35
            Aa, Bb = ax * (la + ov), bx * (lb + ov)
            ze = top - 0.1
            ridge = float(np.clip(lb * 0.5, 0.9, 2.6))

            def pt(c, z):
                return [c[0], c[1], z]
            pPP, pMP = pt(ctr + Aa + Bb, ze), pt(ctr - Aa + Bb, ze)
            pMM, pPM = pt(ctr - Aa - Bb, ze), pt(ctr + Aa - Bb, ze)
            rP, rM = pt(ctr + Aa, ze + ridge), pt(ctr - Aa, ze + ridge)
            RVv = np.array([pPP, pMP, rM, rP, pMM, pPM, rP, rM])
            RFv = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
            RFv = fix_out(RVv, RFv, cen)
            RUV = np.column_stack([((RVv[:, :2] - ctr) @ ax) / 3, ((RVv[:, :2] - ctr) @ bx) / 3])
            A(rmat).add(RVv, RUV, RFv)
            GV = np.array([pPM, pPP, rP, pMP, pMM, rM])
            GF = fix_out(GV, np.array([[0, 1, 2], [3, 4, 5]]), cen)
            GUV = np.column_stack([((GV[:, :2] - ctr) @ bx) / 4, (GV[:, 2] - base) / 3])
            A(wmat).add(GV, GUV, GF)
        else:
            idx = np.array(earcut.triangulate_float64(ring.astype(np.float64), np.array([k], dtype=np.uint32))).reshape(-1, 3)
            RVv = np.column_stack([ring, np.full(k, top)])
            RFv = fix_out(RVv, idx, cen)
            A(rmat).add(RVv, ring / 3, RFv)
        return poly

    house_centers = []
    occupied = []
    n_osm = 0
    for p, t in bways:
        if p.area < 8 or p.area > 3000 or len(p.exterior.coords) > 60:
            continue
        c = p.centroid
        if not shapely.contains_xy(corridor, c.x, c.y):
            continue
        ring = np.array(orient(p, 1.0).exterior.coords)[:-1]
        try:
            lv = int(float(t.get("building:levels", "")))
        except ValueError:
            lv = int(rng.choice([1, 2, 3], p=[0.55, 0.35, 0.10]))
        lv = int(np.clip(lv, 1, 6))
        zc = float(terrain_h(c.x, c.y))
        wmat, rmat = pick_mats(zc)
        if t.get("building") in ("church", "school", "warehouse", "industrial", "commercial", "retail", "hospital"):
            wmat, rmat = "wall_white", "roof_tin"
        add_house(ring, lv, wmat, rmat)
        occupied.append(p.buffer(2.0))
        house_centers.append((c.x, c.y))
        n_osm += 1
    print(f"maisons OSM : {n_osm}", flush=True)
    occ_tree = shapely.STRtree(occupied) if occupied else None
    exist_xy = np.array(house_centers) if house_centers else np.zeros((0, 2))

    # --- villages : maisons manquantes ajoutees le long de la route et en grappes
    zones = []
    R_PL = {"city": 700, "town": 500, "suburb": 350, "village": 230, "hamlet": 120, "neighbourhood": 150}
    D_PL = {"city": 14, "town": 11, "suburb": 12, "village": 7, "hamlet": 4.5, "neighbourhood": 8}
    for nd in osm["nodes"]:
        pl = nd["tags"].get("place")
        if pl in R_PL:
            c = np.array(nd["pt"])
            d, _ = tree.query(c)
            if d < R_PL[pl] + 200:
                zones.append((Point(c).buffer(R_PL[pl]), D_PL[pl], c, R_PL[pl]))
    for p in resid:
        zones.append((p, 8, np.array([p.centroid.x, p.centroid.y]), math.sqrt(p.area / math.pi)))
    new_polys = []
    n_new = 0
    for geom, dens, ctr, rad in zones:
        g = geom.intersection(corridor)
        if g.is_empty:
            continue
        existing = int(shapely.contains_xy(geom, exist_xy[:, 0], exist_xy[:, 1]).sum()) if len(exist_xy) else 0
        need = int(max(0, dens * g.area / 1e4 * 0.9 - existing))
        need = min(need, 1500)
        if need == 0:
            continue
        zi = np.where(shapely.contains_xy(g, P[:, 0], P[:, 1]))[0]
        placed = 0
        for _ in range(need * 14):
            if placed >= need:
                break
            if len(zi) and rng.random() < 0.6:
                i = int(rng.choice(zi))
                side = rng.choice([-1, 1])
                c = P[i] + N[i] * side * rng.uniform(9.5, 24)
                ang = heading[i] + rng.normal(0, 0.1)
            else:
                c = ctr + rng.normal(0, rad * 0.45, 2)
                ang = rng.uniform(0, math.pi)
                if tree.query(c)[0] < 9.5:
                    continue
            if not shapely.contains_xy(g, c[0], c[1]):
                continue
            la, lb = rng.uniform(2.8, 5.0), rng.uniform(2.2, 3.4)
            ax = np.array([math.cos(ang), math.sin(ang)])
            bx = np.array([-ax[1], ax[0]])
            ring = np.array([c - ax * la - bx * lb, c + ax * la - bx * lb, c + ax * la + bx * lb, c - ax * la + bx * lb])
            poly = Polygon(ring).buffer(1.5, join_style=2)
            if occ_tree is not None and len(occ_tree.query(poly, predicate="intersects")):
                continue
            if new_polys and shapely.intersects(poly, np.array(new_polys, dtype=object)).any():
                continue
            if inside(U_water, c[0], c[1]) or (U_minor is not None and shapely.intersects(U_minor, poly)):
                continue
            hh = terrain_h(ring[:, 0], ring[:, 1])
            if hh.max() - hh.min() > 2.2:
                continue
            wmat, rmat = pick_mats(float(hh.mean()))
            lv = int(rng.choice([1, 2, 3], p=[0.55, 0.38, 0.07]) if highland else rng.choice([1, 2], p=[0.8, 0.2]))
            add_house(ring, lv, wmat, rmat)
            new_polys.append(poly)
            house_centers.append((c[0], c[1]))
            placed += 1
            n_new += 1
    print(f"maisons ajoutees (villages) : {n_new}", flush=True)
    U_build = union(occupied + new_polys)

    # ------------------------------------------------------------------ vegetation
    # especes : 0 eucalyptus, 1 arbre rond (manguier...), 2 pin, 3 ravinala, 4 buisson
    def species_for(z, size):
        r = rng.random(size)
        zz = np.asarray(z)
        out = np.empty(size, int)
        hi, mid = zz > 1100, (zz <= 1100) & (zz >= 400)
        for mask, probs in ((hi, [0.45, 0.2, 0.3, 0.05]), (mid, [0.3, 0.4, 0.1, 0.2]), (~hi & ~mid, [0.15, 0.45, 0.1, 0.3])):
            if mask.any():
                out[mask] = rng.choice(4, size=int(mask.sum()), p=probs)
        return out

    trees = []

    def push(x, y, sp, sc):
        x = np.asarray(x)
        if len(x) == 0:
            return
        keepm = shapely.contains_xy(corridor, x, y) & ~inside(U_build, x, y) & ~inside(U_water, x, y)
        keepm &= tree.query(np.column_stack([x, y]))[0] > SHOULDER + 1.0
        if U_minor is not None:
            keepm &= ~shapely.intersects_xy(U_minor, x, y)
        x, y, sp, sc = x[keepm], y[keepm], np.asarray(sp)[keepm], np.asarray(sc)[keepm]
        z = terrain_h(x, y) if len(x) else np.zeros(0)
        trees.append(np.column_stack([sp, x, y, z, sc, rng.uniform(0, 6.283, len(x))]))

    dens_tree = {FO: 1 / 40, GR: 1 / 1100, LA: 1 / 2500}
    dens_bush = {FO: 1 / 80, GR: 1 / 300, LA: 1 / 600}
    for dd, is_bush in ((dens_tree, False), (dens_bush, True)):
        for code, dv in dd.items():
            sel = keep & (cls == code)
            idx = np.where(sel.ravel())[0]
            if not len(idx):
                continue
            cnt = rng.poisson(qarea.ravel()[idx] * dv)
            rep = np.repeat(idx, cnt)
            if not len(rep):
                continue
            u, v = rng.random(len(rep)), rng.random(len(rep))
            q0, q1, q2, q3 = (arr.reshape(-1, 3)[rep] for arr in (V0, V1, V2, V3))
            pxy = ((1 - u)[:, None] * (1 - v)[:, None] * q0 + u[:, None] * (1 - v)[:, None] * q1
                   + u[:, None] * v[:, None] * q2 + (1 - u)[:, None] * v[:, None] * q3)
            if is_bush:
                push(pxy[:, 0], pxy[:, 1], np.full(len(rep), 4), rng.uniform(0.7, 1.5, len(rep)))
            else:
                sp = species_for(pxy[:, 2], len(rep))
                push(pxy[:, 0], pxy[:, 1], sp, rng.uniform(0.75, 1.4, len(rep)))
    # alignements d'arbres le long de la route
    for side in (-1, 1):
        ii_ = np.arange(int(rng.integers(0, 5)), n, 3)
        ok = np.sin(S[ii_] * 0.004 + side * 2.0) > 0.0
        ii_ = ii_[ok & (rng.random(len(ii_)) < 0.5)]
        off = rng.uniform(7.5, 10.5, len(ii_))
        xy = P[ii_] + N[ii_] * (side * off)[:, None]
        sp = rng.choice([0, 1, 3], size=len(ii_), p=[0.55, 0.35, 0.10])
        push(xy[:, 0], xy[:, 1], sp, rng.uniform(0.9, 1.5, len(ii_)))
    # arbres fruitiers autour des maisons, arbres isoles OSM
    if house_centers:
        hc = np.array(house_centers)
        pick = rng.random(len(hc)) < 0.4
        ang = rng.uniform(0, 6.283, pick.sum())
        rr = rng.uniform(5, 9, pick.sum())
        push(hc[pick, 0] + rr * np.cos(ang), hc[pick, 1] + rr * np.sin(ang),
             np.ones(pick.sum(), int), rng.uniform(0.8, 1.2, pick.sum()))
    tn = [nd["pt"] for nd in osm["nodes"] if nd["tags"].get("natural") == "tree"]
    if tn:
        tn = np.array(tn)
        push(tn[:, 0], tn[:, 1], np.ones(len(tn), int), rng.uniform(0.9, 1.3, len(tn)))
    TR = np.vstack(trees) if trees else np.zeros((0, 6))
    if len(TR) > a.max_trees:
        TR = TR[rng.choice(len(TR), a.max_trees, replace=False)]
    print(f"arbres : {len(TR)}", flush=True)

    # ------------------------------------------------------------------ decoupage en troncons et ecriture
    os.makedirs(f"{W}/tiles", exist_ok=True)
    tiles_meta = []
    fin = {mat: ac.finish() for mat, ac in acc.items()}
    ftile = {}
    for mat, (V, UV, F) in fin.items():
        cen = V[F].mean(1)[:, :2]
        ftile[mat] = tile_of_xy(cen)
    ttile = tile_of_xy(TR[:, 1:3]) if len(TR) else np.zeros(0, int)
    for t in range(ntile):
        s0 = t * TILE_ST
        origin = P[s0]
        meshes = {}
        for mat, (V, UV, F) in fin.items():
            fs = F[ftile[mat] == t]
            if not len(fs):
                continue
            used, invx = np.unique(fs.ravel(), return_inverse=True)
            Vt = V[used].copy()
            Vt[:, 0] -= origin[0]
            Vt[:, 1] -= origin[1]
            meshes[mat] = {"kind": KIND.get(mat, "DECOR"),
                           "v": np.round(Vt, 3).ravel().tolist(),
                           "uv": np.round(UV[used], 4).ravel().tolist(),
                           "f": invx.astype(int).tolist()}
        save_json(f"{W}/tiles/tile_{t:02d}.json", meshes)
        tr = TR[ttile == t] if len(TR) else TR
        with open(f"{W}/tiles/tile_{t:02d}_trees.csv", "w") as f:
            f.write("species,x,y,z,scale,rot\n")
            for r in tr:
                f.write(f"{int(r[0])},{r[1] - origin[0]:.1f},{r[2] - origin[1]:.1f},{r[3]:.1f},{r[4]:.2f},{r[5]:.2f}\n")
        tiles_meta.append({"index": t, "origin_xy": [round(float(origin[0]), 2), round(float(origin[1]), 2)],
                           "station_from": s0, "station_to": min(s0 + TILE_ST, n - 1),
                           "trees": int(len(tr)), "materials": sorted(meshes.keys())})
    with open(f"{W}/route.csv", "w") as f:
        f.write("x,y,z,km\n")
        for i in range(n):
            f.write(f"{P[i, 0]:.2f},{P[i, 1]:.2f},{ROADH[i] + 0.0:.2f},{S[i] / 1000:.3f}\n")
    save_json(f"{W}/manifest.json", {
        "road": road["road"], "km_from": road["km_from"], "km_to": road["km_to"],
        "origin_latlon": road["origin"], "step_m": STEP, "stations": n,
        "tile_length_m": TILE_ST * STEP, "tiles": tiles_meta,
        "kinds": KIND})
    print(f"termine : {ntile} troncons", flush=True)


if __name__ == "__main__":
    main()
