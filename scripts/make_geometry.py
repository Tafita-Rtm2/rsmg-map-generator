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
import atlas as atl
import extras as ex3d
from common import transformers, load_json, save_json

STEP = 5.0          # m entre deux stations le long de la route
TILE_ST = 400       # stations par troncon (2 km)
ROAD_HALF = 3.5     # demi-largeur de chaussee
SHOULDER = 5.5      # fin de l'accotement
W_MAX = 300.0       # demi-largeur du relief genere
OFFS_POS = [3.5, 5.5, 8, 11, 15, 20, 27, 36, 48, 62, 80, 102, 130, 165, 210, 260, 300]
TEXM = {"grass": 4.5, "grass2": 4.5, "laterite": 4, "rock": 5, "forest": 4.5, "paddy": 4.5, "mud": 3.5, "water": 10, "shoulder": 2.5, "falaise": 5}
KIND = {"road_main": "GND", "road_minor": "GND", "track": "GND"}
for _m in TEXM:
    KIND[_m] = "GND"
KIND.update({"trottoir": "GND", "beton": "SOLID", "pole": "SOLID", "borne": "SOLID", "fuel": "SOLID", "brique": "SOLID"})
SH, GR, LA, RO, FO, PA, MU, WA, CL, GR2 = range(10)
CLS = {SH: "shoulder", GR: "grass", LA: "laterite", RO: "rock", FO: "forest", PA: "paddy", MU: "mud", WA: "water", CL: "falaise", GR2: "grass2"}
STEEP = {"rock", "falaise"}     # UV projetes selon la normale (pas d'etirement sur les parois)


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
    ap.add_argument("--max-trees", type=int, default=600000)
    ap.add_argument("--grass", type=float, default=1.0)
    ap.add_argument("--grass-cap", type=int, default=60000)
    ap.add_argument("--tree-cap", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    W = a.work
    rng = np.random.default_rng(a.seed)
    road = load_json(f"{W}/road.json")
    osm = load_json(f"{W}/osm.json")
    if os.path.exists(f"{W}/osm_extra.json"):
        ex_ = load_json(f"{W}/osm_extra.json")
        osm["ways"] += ex_["ways"]
        osm["nodes"] += ex_["nodes"]
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

    CARVE = {"fn": None}

    def terrain_h(x, y, carve=True):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        xr, yr = x.ravel(), y.ravel()
        d, i = tree.query(np.column_stack([xr, yr]))
        dh = dem_h(x, y).ravel()
        rh = ROADH[i]
        blend = np.clip(np.abs(dh - rh) * 1.6, 6, 45)
        t = np.clip((d - ROAD_HALF - 2.75) / blend, 0, 1)
        t = t * t * (3 - 2 * t)
        h = rh * (1 - t) + dh * t
        micro = 0.30 * np.sin(xr * 0.19 + 1.0) * np.sin(yr * 0.23 - 0.5) + 0.20 * np.sin(xr * 0.07 + yr * 0.05)
        h = h + micro * np.clip((d - 12) / 25, 0, 1)      # petites bosses naturelles loin de la route
        if carve and CARVE["fn"] is not None:
            h = h - CARVE["fn"](xr, yr)
        return h.reshape(x.shape)

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

    # ------------------------------------------------------------------ digues, rivieres, ponts
    def regrade(h, g=0.09):
        mx_ = g * STEP
        for i_ in range(1, len(h)):
            h[i_] = min(max(h[i_], h[i_ - 1] - mx_), h[i_ - 1] + mx_)
        for i_ in range(len(h) - 2, -1, -1):
            h[i_] = min(max(h[i_], h[i_ + 1] - mx_), h[i_ + 1] + mx_)

    wet = np.zeros(n, bool)
    for U_ in (U_farm, U_water):
        if U_ is not None:
            for off_ in (-14.0, 0.0, 14.0):
                q_ = P + N * off_
                wet |= shapely.contains_xy(U_, q_[:, 0], q_[:, 1])
    ROADH += gaussian_filter1d(wet.astype(float) * 1.7, 12, mode="nearest")   # digue : route surelevee dans les rizieres
    regrade(ROADH)
    print(f"digues : {int(wet.sum()) * STEP:.0f} m de route en rizieres / zones d'eau", flush=True)

    RIV = {"river": (7.0, 2.4), "canal": (4.0, 1.6), "stream": (2.5, 1.0), "drain": (1.5, 0.6)}
    rivers = []
    for w in osm["ways"]:
        wt = w["tags"].get("waterway")
        if wt in RIV and len(w["pts"]) >= 2:
            ln_ = LineString(w["pts"])
            if ln_.length > 4:
                rivers.append((ln_, RIV[wt][0], RIV[wt][1], wt))
    cp_, chw_, cdp_ = [], [], []
    for ln_, hw_, dp_, wt_ in rivers:
        for s_ in np.arange(0, ln_.length, 3.0):
            q_ = ln_.interpolate(s_)
            cp_.append((q_.x, q_.y))
            chw_.append(hw_)
            cdp_.append(dp_)
    bridges = []
    if cp_:
        cp_, chw_, cdp_ = np.array(cp_), np.array(chw_), np.array(cdp_)
        ctree_ = cKDTree(cp_)

        def carve_fn(xr, yr):
            d_, i_ = ctree_.query(np.column_stack([xr, yr]))
            u_ = np.clip((d_ - chw_[i_]) / 3.0, 0, 1)
            return cdp_[i_] * np.where(d_ <= chw_[i_], 1.0, 0.5 * (1 + np.cos(np.pi * u_)))
        CARVE["fn"] = carve_fn
        route_ls = LineString(P[::2])
        for ln_, hw_, dp_, wt_ in rivers:
            if hw_ < 4.0:
                continue
            it_ = route_ls.intersection(ln_)
            pts_ = [it_] if it_.geom_type == "Point" else list(getattr(it_, "geoms", []))
            for pt_ in pts_:
                if pt_.geom_type != "Point":
                    continue
                ic_ = int(tree.query([pt_.x, pt_.y])[1])
                half_ = int((hw_ + 6) / STEP) + 1
                i0_, i1_ = max(ic_ - half_, 0), min(ic_ + half_, n - 1)
                if i1_ - i0_ >= 4:
                    ROADH[i0_:i1_ + 1] = np.linspace(ROADH[i0_], ROADH[i1_], i1_ - i0_ + 1)
                    bridges.append((i0_, i1_))
    print(f"cours d'eau : {len(rivers)}, ponts : {len(bridges)}", flush=True)

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
    cls = np.where(noise > -0.9, GR, LA)
    n2_ = np.sin(cx * 0.047 + 2.1) * np.sin(cy * 0.039 + 0.7) + 0.6 * np.sin(cx * 0.013 - cy * 0.021 + 1.0)
    cls = np.where((cls == GR) & (n2_ > 0.15), GR2, cls)       # 2e herbe (plus claire) : casse la repetition
    cls = np.where(inside(U_resid, cx, cy), LA, cls)
    cls = np.where((slope > 24) & (slope <= 36), FO, cls)
    cls = np.where(inside(U_wood, cx, cy), FO, cls)
    cls = np.where(inside(U_farm, cx, cy) & (slope < 14), PA, cls)
    cls = np.where(inside(U_rock, cx, cy) | (slope > 36), RO, cls)
    zc_ = (Z[:-1, :-1] + Z[1:, :-1] + Z[1:, 1:] + Z[:-1, 1:]) / 4
    _, qi_ = tree.query(np.column_stack([cx.ravel(), cy.ravel()]))
    rel_ = zc_ - ROADH[qi_].reshape(cx.shape)
    cls = np.where((rel_ > 1.5) & (slope > 24) & (offc <= 80), CL, cls)     # talus de deblai : falaise de laterite rouge
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
        if name in STEEP:
            # sommets non partages + UV par face selon l'axe dominant de la normale : roche sans etirement
            Fi = invx.reshape(-1, 3)
            Vf = Vm[Fi]                                  # (nf, 3, 3)
            nrm = np.cross(Vf[:, 1] - Vf[:, 0], Vf[:, 2] - Vf[:, 0])
            use_y = np.abs(nrm[:, 0]) > np.abs(nrm[:, 1])
            horiz = np.abs(nrm[:, 2]) > 0.7 * np.linalg.norm(nrm, axis=1)
            u_ = np.where(horiz[:, None], Vf[..., 0], np.where(use_y[:, None], Vf[..., 1], Vf[..., 0]))
            v_ = np.where(horiz[:, None], Vf[..., 1], Vf[..., 2])
            A(name).add(Vf.reshape(-1, 3), np.stack([u_, v_], -1).reshape(-1, 2) / TEXM[name],
                        np.arange(len(Fi) * 3).reshape(-1, 3))
        else:
            A(name).add(Vm, Vm[:, :2] / TEXM[name], invx.reshape(-1, 3))

    # ------------------------------------------------------------------ chaussee principale
    cols = np.array([-ROAD_HALF, 0.0, ROAD_HALF])
    RX = P[:, 0:1] + N[:, 0:1] * cols
    RY = P[:, 1:2] + N[:, 1:2] * cols
    RZ = np.repeat(ROADH[:, None], 3, axis=1)
    RV = np.stack([RX, RY, RZ], -1).reshape(-1, 3)
    RUV = np.stack([np.tile([0.0, 0.5, 1.0], (n, 1)), np.repeat((S / 28.0)[:, None], 3, axis=1)], -1).reshape(-1, 2)
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

    def add_house(ring, levels, wmat, rmat, style="gable"):
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
        if style == "villa":
            gable = False
        if gable:
            ov = 0.35
            Aa, Bb = ax * (la + ov), bx * (lb + ov)
            ze = top - 0.1
            ridge = float(np.clip(lb * (0.72 if rmat == "roof_tile" else 0.42), 1.0, 3.6))

            def pt(c, z):
                return [c[0], c[1], z]
            pPP, pMP = pt(ctr + Aa + Bb, ze), pt(ctr - Aa + Bb, ze)
            pMM, pPM = pt(ctr - Aa - Bb, ze), pt(ctr + Aa - Bb, ze)
            hip = style == "hip" and la > lb * 1.15
            if hip:                                   # toit en croupe : 4 pans, faitage raccourci
                rl = max(la - lb, 0.3)
                rP, rM = pt(ctr + ax * rl, ze + ridge), pt(ctr - ax * rl, ze + ridge)
                RVv = np.array([pPP, pMP, pMM, pPM, rP, rM])
                RFv = np.array([[0, 1, 5], [0, 5, 4], [2, 3, 4], [2, 4, 5], [3, 0, 4], [1, 2, 5]])
            else:
                rP, rM = pt(ctr + Aa, ze + ridge), pt(ctr - Aa, ze + ridge)
                RVv = np.array([pPP, pMP, rM, rP, pMM, pPM, rP, rM])
                RFv = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
            RFv = fix_out(RVv, RFv, cen)
            RUV = np.column_stack([((RVv[:, :2] - ctr) @ ax) / 3, ((RVv[:, :2] - ctr) @ bx) / 3])
            A(rmat).add(RVv, RUV, RFv)
            if not hip:
                GV = np.array([pPM, pPP, rP, pMP, pMM, rM])
                GF = fix_out(GV, np.array([[0, 1, 2], [3, 4, 5]]), cen)
                GUV = np.column_stack([((GV[:, :2] - ctr) @ bx) / 4, (GV[:, 2] - base) / 3])
                A(wmat).add(GV, GUV, GF)
            ang_ = math.atan2(ax[1], ax[0])
            if rmat == "roof_tile" and rng.random() < 0.55:      # cheminee en briques
                cxp = ctr + ax * (la * 0.45)
                A("wall_brick").add(*ex3d.box(cxp[0], cxp[1], ze + ridge * 0.55, ang_, 0.28, 0.28, 1.6, 1.5))
            if levels >= 2 and rng.random() < 0.7:             # balcon a l'etage avec garde-corps
                sd_ = 1 if rng.random() < 0.5 else -1
                bc_ = ctr + bx * sd_ * (lb + 0.6)
                A("wall_white").add(*ex3d.box(bc_[0], bc_[1], base + 2.9, ang_, la * 0.6, 0.6, 0.14, 2.0, bottom=True))
                rc_ = ctr + bx * sd_ * (lb + 1.15)
                A("wall_white").add(*ex3d.box(rc_[0], rc_[1], base + 3.04, ang_, la * 0.6, 0.04, 0.95, 2.0))
        else:
            idx = np.array(earcut.triangulate_float64(ring.astype(np.float64), np.array([k], dtype=np.uint32))).reshape(-1, 3)
            RVv = np.column_stack([ring, np.full(k, top)])
            RFv = fix_out(RVv, idx, cen)
            A(rmat).add(RVv, ring / 3, RFv)
            if style == "villa":                                       # acrotere autour du toit-terrasse
                for q0_, q1_ in zip(ring, np.roll(ring, -1, axis=0)):
                    ln_ = float(np.hypot(*(q1_ - q0_)))
                    if ln_ > 0.3:
                        mid_ = (q0_ + q1_) / 2
                        A("wall_white").add(*ex3d.box(mid_[0], mid_[1], top, math.atan2(q1_[1] - q0_[1], q1_[0] - q0_[0]), ln_ / 2, 0.1, 0.6, 2.0))
        return poly

    def add_fence(c, ax, bx, ha, hb, to_road):
        """Mur de cloture en briques autour d'une maison (3 cotes + portail du cote de la route)."""
        c = np.asarray(c, float)
        cor = [c - ax * ha - bx * hb, c + ax * ha - bx * hb, c + ax * ha + bx * hb, c - ax * ha + bx * hb]
        hz_ = terrain_h(np.array([q[0] for q in cor]), np.array([q[1] for q in cor]))
        z0_ = float(hz_.min()) - 0.3
        sides = [(0, 1, -bx), (1, 2, ax), (2, 3, bx), (3, 0, -ax)]
        for i0f, i1f, outn in sides:
            p0f, p1f = cor[i0f], cor[i1f]
            ln_ = float(np.hypot(*(p1f - p0f)))
            ang_f = math.atan2(p1f[1] - p0f[1], p1f[0] - p0f[0])
            if float(np.dot(outn, to_road)) > 0.7:                     # cote de la route : portail de 3 m au milieu
                g_ = 1.5
                for sgn in (-1, 1):
                    a_f = (ln_ / 2 - g_) / 2
                    if a_f > 0.3:
                        mid_f = (p0f + p1f) / 2 + (p1f - p0f) / ln_ * sgn * (g_ + a_f)
                        A("brique").add(*ex3d.box(mid_f[0], mid_f[1], z0_, ang_f, a_f, 0.12, 1.9, 1.2))
            elif float(np.dot(outn, to_road)) > -0.7:
                mid_f = (p0f + p1f) / 2
                A("brique").add(*ex3d.box(mid_f[0], mid_f[1], z0_, ang_f, ln_ / 2, 0.12, 1.9, 1.2))
        for q_ in cor:
            A("brique").add(*ex3d.box(q_[0], q_[1], z0_, 0.0, 0.2, 0.2, 2.2, 1.2))

    house_centers = []
    occupied = []
    n_osm = 0
    mainline = LineString(P[::2])
    for p, t in bways:
        if p.area < 8 or p.area > 3000 or len(p.exterior.coords) > 60:
            continue
        c = p.centroid
        if not shapely.contains_xy(corridor, c.x, c.y):
            continue
        if p.distance(mainline) < 7.0 or (U_minor is not None and shapely.intersects(U_minor, p)):
            continue          # pas de maison sur la route ni sur une piste
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

    def orient_faces_up(V_, F_):
        tri_ = V_[F_]
        nr_ = np.cross(tri_[:, 1] - tri_[:, 0], tri_[:, 2] - tri_[:, 0])
        F_ = F_.copy()
        F_[nr_[:, 2] < 0] = F_[nr_[:, 2] < 0][:, [0, 2, 1]]
        return F_

    # --- stations-service et abribus (OpenStreetMap)
    fuel_xy = []
    for w in osm["ways"]:
        if w["tags"].get("amenity") == "fuel" and len(w["pts"]) >= 3:
            fuel_xy.append(np.array(w["pts"]).mean(0))
    for nd in osm["nodes"]:
        if nd["tags"].get("amenity") == "fuel":
            fuel_xy.append(np.array(nd["pt"], float))
    placed_f = []
    for fx_, fy_ in fuel_xy:
        if any(np.hypot(fx_ - a_, fy_ - b_) < 40 for a_, b_ in placed_f):
            continue
        d_, i_ = tree.query([fx_, fy_])
        if d_ > 90 or not shapely.contains_xy(corridor, fx_, fy_):
            continue
        sd_ = 1.0 if (np.array([fx_, fy_]) - P[i_]) @ N[i_] >= 0 else -1.0
        pos_ = P[i_] + N[i_] * sd_ * max(float(d_), 22.0)
        ang_s = float(heading[i_]) + (math.pi if sd_ > 0 else 0.0)      # le devant (+b) regarde la route, la boutique est derriere
        ax_s = np.array([math.cos(ang_s), math.sin(ang_s)])
        bx_s = np.array([-ax_s[1], ax_s[0]])
        gx_ = np.array([pos_ + ax_s * a_ + bx_s * b_ for a_ in (-12, 0, 12) for b_ in (-14, 0, 14)])
        zz_ = float(np.clip(terrain_h(gx_[:, 0], gx_[:, 1]).mean(), ROADH[i_] - 0.5, ROADH[i_] + 1.0)) + 0.03
        # dalle (asphalte) + soubassement en beton + acces depuis la route
        cor_ = [pos_ - ax_s * 12 - bx_s * 14, pos_ + ax_s * 12 - bx_s * 14, pos_ + ax_s * 12 + bx_s * 14, pos_ - ax_s * 12 + bx_s * 14]
        Vp_ = np.array([[q_[0], q_[1], zz_] for q_ in cor_])
        A("road_minor").add(Vp_, np.array([[q_[0] / 6, q_[1] / 6] for q_ in cor_]), orient_faces_up(Vp_, np.array([[0, 1, 2], [0, 2, 3]])))
        A("beton").add(*ex3d.box(pos_[0], pos_[1], zz_ - 1.6, ang_s, 12.0, 14.0, 1.58, 3.0))
        to_road_ = -sd_ * N[i_]
        e_pt_ = pos_ + to_road_ * 14.0
        r_pt_ = P[i_] + N[i_] * sd_ * 5.4
        tg_ = np.array([math.cos(heading[i_]), math.sin(heading[i_])])
        cd_ = [r_pt_ - tg_ * 5.0, r_pt_ + tg_ * 5.0, e_pt_ + tg_ * 5.0, e_pt_ - tg_ * 5.0]
        zr_ = float(ROADH[i_]) + 0.02
        Vd_ = np.array([[cd_[0][0], cd_[0][1], zr_], [cd_[1][0], cd_[1][1], zr_], [cd_[2][0], cd_[2][1], zz_ + 0.01], [cd_[3][0], cd_[3][1], zz_ + 0.01]])
        A("road_minor").add(Vd_, np.array([[q_[0] / 6, q_[1] / 6] for q_ in Vd_[:, :2]]), orient_faces_up(Vd_, np.array([[0, 1, 2], [0, 2, 3]])))
        for fm_, fp_ in ex3d.fuel_station(pos_[0], pos_[1], zz_, ang_s).items():
            KIND[fm_] = "SOLID"
            A(fm_).add(*ex3d.merge(fp_))
        occupied.append(Polygon(cor_).buffer(1.0))
        placed_f.append((fx_, fy_))
    n_bus = 0
    for nd in osm["nodes"]:
        if nd["tags"].get("highway") == "bus_stop":
            d_, i_ = tree.query(nd["pt"])
            if d_ < 25:
                sd_ = 1.0 if (np.array(nd["pt"]) - P[i_]) @ N[i_] >= 0 else -1.0
                pos_ = P[i_] + N[i_] * sd_ * 8.0
                zz_ = float(terrain_h(pos_[0], pos_[1]))
                rf_, bd_ = ex3d.bus_shelter(pos_[0], pos_[1], zz_, float(heading[i_]))
                A("roof_tin").add(*rf_)
                A("pole").add(*bd_)
                n_bus += 1
    print(f"stations-service : {len(placed_f)}, abribus : {n_bus}", flush=True)
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
    new_grid = {}
    n_new = 0
    for geom, dens, ctr, rad in zones:
        zone_town = dens >= 11
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
                c = P[i] + N[i] * side * rng.uniform(12.5, 26)
                ang = heading[i] + rng.normal(0, 0.1)
            else:
                c = ctr + rng.normal(0, rad * 0.45, 2)
                ang = rng.uniform(0, math.pi)
                if tree.query(c)[0] < 12.5:
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
            gk_ = (int(c[0] // 40), int(c[1] // 40))
            near_new = [q_ for dx_ in (-1, 0, 1) for dy_ in (-1, 0, 1) for q_ in new_grid.get((gk_[0] + dx_, gk_[1] + dy_), ())]
            if near_new and shapely.intersects(poly, np.array(near_new, dtype=object)).any():
                continue
            if inside(U_water, c[0], c[1]) or (U_minor is not None and shapely.intersects(U_minor, poly)):
                continue
            hh = terrain_h(ring[:, 0], ring[:, 1])
            if hh.max() - hh.min() > 2.2:
                continue
            wmat, rmat = pick_mats(float(hh.mean()))
            lv = int(rng.choice([1, 2, 3], p=[0.55, 0.38, 0.07]) if highland else rng.choice([1, 2], p=[0.8, 0.2]))
            style = str(rng.choice(["gable", "hip", "villa", "L"], p=[0.5, 0.17, 0.12 if highland or zone_town else 0.05, 0.21 if highland or zone_town else 0.28]))
            if style == "villa":
                lv = max(lv, 2)
                rmat = "roof_tin"
            add_house(ring, lv, wmat, rmat, "gable" if style == "L" else style)
            if style == "L":                                           # aile perpendiculaire en L
                wl_, wb_ = rng.uniform(1.8, 2.8), rng.uniform(1.6, 2.2)
                sg_ = 1 if rng.random() < 0.5 else -1
                wc_ = c + ax * (la - wb_) * (-sg_) + bx * (lb + wl_ - 0.4)
                ring_w = np.array([wc_ - bx * wl_ - ax * wb_, wc_ + bx * wl_ - ax * wb_, wc_ + bx * wl_ + ax * wb_, wc_ - bx * wl_ + ax * wb_])
                if not (occ_tree is not None and len(occ_tree.query(Polygon(ring_w).buffer(1.0), predicate="intersects"))):
                    add_house(ring_w, 1, wmat, rmat, "gable")
            if rng.random() < 0.45:
                tr_ = P[int(tree.query(c)[1])] - c
                add_fence(c, ax, bx, la + 1.8, lb + 1.8, tr_ / max(float(np.hypot(*tr_)), 1e-6))
            new_polys.append(poly)
            new_grid.setdefault(gk_, []).append(poly)
            house_centers.append((c[0], c[1]))
            placed += 1
            n_new += 1
    print(f"maisons ajoutees (villages) : {n_new}", flush=True)
    U_build = union(occupied + new_polys)

    # ------------------------------------------------------------------ equipements de route
    bp_ = np.array(house_centers) if house_centers else np.zeros((0, 2))
    near_ = np.zeros(n, int)
    if len(bp_):
        near_ = np.array([len(v) for v in cKDTree(bp_).query_ball_point(P, 45.0)])
    settle = near_ >= 3
    town = near_ >= 8
    for nd in osm["nodes"]:
        if nd["tags"].get("place") in ("city", "town", "suburb"):
            q_ = np.array(nd["pt"])
            town |= np.hypot(P[:, 0] - q_[0], P[:, 1] - q_[1]) < 350
    for i0_, i1_ in bridges:
        town[max(i0_ - 4, 0):i1_ + 5] = False
        settle[max(i0_ - 4, 0):i1_ + 5] = False

    def runs(mask, minlen):
        out_, i_ = [], 0
        while i_ < n:
            if mask[i_]:
                j_ = i_
                while j_ + 1 < n and mask[j_ + 1]:
                    j_ += 1
                if (j_ - i_) * STEP >= minlen:
                    out_.append((i_, j_))
                i_ = j_ + 1
            else:
                i_ += 1
        return out_

    def orient_faces(V_, F_, want):
        tri_ = V_[F_]
        nr_ = np.cross(tri_[:, 1] - tri_[:, 0], tri_[:, 2] - tri_[:, 0])
        flip_ = (nr_ * want).sum(1) < 0
        F_ = F_.copy()
        F_[flip_] = F_[flip_][:, [0, 2, 1]]
        return F_

    # trottoirs : bordure de 15 cm + trottoir, des deux cotes, dans les villes
    town_runs = runs(gaussian_filter1d(town.astype(float), 6) > 0.4, 80)
    for i0_, i1_ in town_runs:
        idx_ = np.arange(i0_, i1_ + 1)
        m_ = len(idx_)
        for sd_ in (-1, 1):
            Vs_ = np.zeros((m_ * 3, 3))
            UVs_ = np.zeros((m_ * 3, 2))
            for c_, (o_, dz_) in enumerate(((3.5, 0.0), (3.5, 0.15), (5.2, 0.15))):
                Vs_[c_::3, 0] = P[idx_, 0] + N[idx_, 0] * sd_ * o_
                Vs_[c_::3, 1] = P[idx_, 1] + N[idx_, 1] * sd_ * o_
                Vs_[c_::3, 2] = ROADH[idx_] + dz_
                UVs_[c_::3, 0] = S[idx_] / 3.2
                UVs_[c_::3, 1] = o_ / 3.2 + dz_
            k_ = np.arange(m_ - 1)
            for c_, want_top in ((0, False), (1, True)):
                q0_, q1_, q2_, q3_ = k_ * 3 + c_, (k_ + 1) * 3 + c_, (k_ + 1) * 3 + c_ + 1, k_ * 3 + c_ + 1
                F_ = np.concatenate([np.stack([q0_, q1_, q2_], 1), np.stack([q0_, q2_, q3_], 1)], 0)
                if want_top:
                    F_ = orient_faces(Vs_, F_, np.array([0.0, 0.0, 1.0]))
                else:
                    st_ = np.concatenate([k_, k_])
                    F_ = orient_faces(Vs_, F_, np.column_stack([-sd_ * N[st_, 0], -sd_ * N[st_, 1], np.zeros(len(st_))]))
                A("trottoir").add(Vs_, UVs_, F_)

    # poteaux electriques : lignes OSM (power=line/minor_line) + poteaux le long des villages sans ligne connue
    osm_poles = []
    for nd in osm["nodes"]:
        if nd["tags"].get("power") in ("pole", "tower"):
            osm_poles.append(nd["pt"])

    def add_pole_line(pts_):
        prev_ = None
        for k_, (px_, py_) in enumerate(pts_):
            if not shapely.contains_xy(corridor, px_, py_):
                prev_ = None
                continue
            a_, b_ = pts_[max(k_ - 1, 0)], pts_[min(k_ + 1, len(pts_) - 1)]
            ang_ = math.atan2(b_[1] - a_[1], b_[0] - a_[0])
            z_ = float(terrain_h(px_, py_))
            (V_, UV_, F_), heads_ = ex3d.pole(px_, py_, z_, ang_)
            A("pole").add(V_, UV_, F_)
            if prev_ is not None:
                dd_ = float(np.hypot(heads_[0][0] - prev_[0][0], heads_[0][1] - prev_[0][1]))
                if dd_ < 160:
                    for h0_, h1_ in zip(prev_, heads_):
                        A("cable").add(*ex3d.cable(h0_, h1_, 0.015 * dd_))
            prev_ = heads_
    n_line = 0
    for w in osm["ways"]:
        if w["tags"].get("power") in ("line", "minor_line") and len(w["pts"]) >= 2:
            add_pole_line([tuple(q) for q in w["pts"]])
            osm_poles.extend(w["pts"])
            n_line += 1
    pole_tree = cKDTree(np.array(osm_poles)) if osm_poles else None
    n_proc = 0
    for i0_, i1_ in runs(gaussian_filter1d(settle.astype(float), 8) > 0.4, 100):
        sd_ = 1 if rng.random() < 0.5 else -1
        pts_ = []
        for i_ in range(i0_, i1_ + 1, 8):
            q_ = P[i_] + N[i_] * sd_ * 7.2
            if pole_tree is None or pole_tree.query(q_)[0] > 30:
                pts_.append((float(q_[0]), float(q_[1])))
        if len(pts_) >= 2:
            add_pole_line(pts_)
            n_proc += len(pts_)
    for q_ in osm_poles:
        pass
    print(f"lignes electriques OSM : {n_line}, poteaux ajoutes dans les villages : {n_proc}", flush=True)

    # lampadaires (OSM + villes), bornes kilometriques, passages pietons
    lamps_ = []
    for nd in osm["nodes"]:
        if nd["tags"].get("highway") == "street_lamp":
            d_, i_ = tree.query(nd["pt"])
            if d_ < 20:
                lamps_.append((nd["pt"][0], nd["pt"][1], float(heading[i_]) + (math.pi / 2 if (np.array(nd["pt"]) - P[i_]) @ N[i_] < 0 else -math.pi / 2)))
    for i0_, i1_ in town_runs:
        for j_, i_ in enumerate(range(i0_, i1_ + 1, 8)):
            sd_ = 1 if j_ % 2 == 0 else -1
            q_ = P[i_] + N[i_] * sd_ * 4.9
            lamps_.append((q_[0], q_[1], float(heading[i_]) + (-math.pi / 2 if sd_ > 0 else math.pi / 2)))
    for lx_, ly_, la_ in lamps_:
        if shapely.contains_xy(corridor, lx_, ly_):
            A("pole").add(*ex3d.lamp(lx_, ly_, float(terrain_h(lx_, ly_)), la_))
    n_zeb = 0
    for nd in osm["nodes"]:
        if nd["tags"].get("highway") == "crossing":
            d_, i_ = tree.query(nd["pt"])
            if d_ < 8 and i_ > 2:
                for j_ in range(-3, 4):
                    q_ = P[i_] + N[i_] * j_ * 1.0
                    A("ligne").add(*ex3d.box(q_[0], q_[1], ROADH[i_] + 0.0, float(heading[i_]), 1.5, 0.25, 0.03, 1.0))
                n_zeb += 1
    print(f"lampadaires : {len(lamps_)}, passages pietons : {n_zeb}", flush=True)

    # --- bornes kilometriques (sommet rouge + code de la route, corps blanc + ville + km), panneaux de villes
    DEST = {"RN1": ("ANTANANARIVO", "TSIROANOMANDIDY")}
    town_a, town_b = DEST.get(str(road["road"]).upper().replace(" ", ""), ("", ""))
    code_ = str(road["road"]).upper().replace(" ", "")
    km0_ = int(road["km_from"])
    borne_at = atl.Atlas("borne", 256, 512)
    n_bor = 0
    for km_ in range(1, int(S[-1] / 1000) + 1):
        i_ = int(np.argmin(np.abs(S - km_ * 1000)))
        q_ = P[i_] - N[i_] * 7.0
        mat_, uv_ = borne_at.add(lambda w_, h_, kk=km0_ + km_: atl.draw_borne(w_, h_, code_, town_a, kk, seed=kk))
        KIND[mat_] = "SOLID"
        A(mat_).add(*ex3d.skin_box(q_[0], q_[1], float(terrain_h(q_[0], q_[1])), float(heading[i_]) + math.pi / 2,
                                   0.27, 0.16, 1.2, big=uv_, plain=(uv_[0] + 0.01, uv_[1] + 0.01)))
        n_bor += 1
    sign_at = atl.Atlas("panneau", 512, 192)
    n_sig = 0

    def put_sign(i_, side_, text_, style_, sub_=""):
        nonlocal n_sig
        if i_ < 3 or i_ > n - 4:
            return
        q_ = P[i_] + N[i_] * side_ * 6.8
        if not shapely.contains_xy(corridor, q_[0], q_[1]):
            return
        z_ = float(terrain_h(q_[0], q_[1]))
        mat_, uv_ = sign_at.add(lambda w_, h_: atl.draw_panneau(w_, h_, text_, style_, sub_))
        KIND[mat_] = "SOLID"
        for off_ in (-0.9, 0.9):
            pp_ = q_ + (N[i_] * 0.0) + np.array([math.cos(heading[i_]), math.sin(heading[i_])]) * off_
            A("pole").add(*ex3d.cylinder(pp_[0], pp_[1], z_, 2.6, 0.05, 0.05, 6))
        A(mat_).add(*ex3d.skin_box(q_[0], q_[1], z_ + 1.75, float(heading[i_]) + math.pi / 2, 0.95, 0.02, 0.7, big=uv_,
                                   plain=(uv_[0] + 0.01, uv_[1] + 0.01)))
        n_sig += 1
    PL_R = {"city": 220, "town": 180, "suburb": 150, "village": 140, "hamlet": 100}
    named = []
    for nd in osm["nodes"]:
        nm_ = nd["tags"].get("name")
        if nm_ and nd["tags"].get("place") in PL_R:
            d_, i_ = tree.query(nd["pt"])
            if d_ < 350 and nm_ not in [x_[0] for x_ in named]:
                named.append((nm_, int(i_), PL_R[nd["tags"]["place"]]))
    for nm_, i_, rr_ in named:
        k_ = int(rr_ / STEP)
        put_sign(i_ - k_, -1, nm_.upper(), "ville")          # arrivee dans le sens du km croissant (cote droit)
        put_sign(i_ + k_, 1, nm_.upper(), "ville")           # arrivee dans l'autre sens (cote gauche)
    if town_b:
        for km_ in range(0, int(S[-1] / 1000) + 1, 5):
            put_sign(int(np.argmin(np.abs(S - (km_ * 1000 + 400)))), -1, town_b, "dest", "→")
    print(f"bornes : {n_bor}, panneaux : {n_sig}", flush=True)

    # --- marches de bord de route, cash points, sacs de charbon (villages) + marches OSM
    extra_fp = []
    ens_at = atl.Atlas("enseigne", 512, 128)
    n_stall = n_cash = n_sacks = 0

    def free_spot(x_, y_, r_=2.6):
        if not shapely.contains_xy(corridor, x_, y_):
            return False
        if tree.query([x_, y_])[0] < 7.0:
            return False
        if U_build is not None and shapely.intersects(U_build, Point(x_, y_).buffer(r_)):
            return False
        if inside(U_water, x_, y_):
            return False
        return not any(np.hypot(x_ - e_[0], y_ - e_[1]) < e_[2] + r_ for e_ in extra_fp)

    def put(parts_, x_, y_, r_):
        for m_, plist_ in parts_.items():
            if m_ not in KIND:
                KIND[m_] = "DECOR" if m_ == "bache" else "SOLID"
            A(m_).add(*ex3d.merge(plist_))
        extra_fp.append((x_, y_, r_))

    for i0_, i1_ in runs(gaussian_filter1d(settle.astype(float), 8) > 0.4, 100):
        i_ = i0_ + int(rng.integers(10, 30))
        while i_ < i1_:
            if rng.random() < 0.65:
                sd_ = 1 if rng.random() < 0.5 else -1
                k_ = int(rng.integers(2, 6))
                made = 0
                for j_ in range(k_):
                    ii_ = min(i_ + j_ * 1, n - 2)          # un etal tous les ~5 m (stations)
                    q_ = P[ii_] + N[ii_] * sd_ * 8.6
                    ang_ = float(heading[ii_]) + (math.pi if sd_ > 0 else 0.0)
                    if free_spot(q_[0], q_[1], 1.6):
                        z_ = float(terrain_h(q_[0], q_[1]))
                        put(ex3d.stall(q_[0], q_[1], z_, ang_, rng, charcoal=bool(rng.random() < 0.3)), q_[0], q_[1], 1.6)
                        n_stall += 1
                        made += 1
                if made:
                    q_ = P[min(i_ + k_ + 2, n - 2)] + N[min(i_ + k_ + 2, n - 2)] * sd_ * 8.0
                    ang_ = float(heading[min(i_ + k_ + 2, n - 2)]) + (math.pi if sd_ > 0 else 0.0)
                    if rng.random() < 0.55 and free_spot(q_[0], q_[1], 1.2):
                        put(ex3d.kiosk(q_[0], q_[1], float(terrain_h(q_[0], q_[1])), ang_), q_[0], q_[1], 1.2)
                        n_cash += 1
                    q2_ = P[max(i_ - 4, 0)] + N[max(i_ - 4, 0)] * sd_ * 8.0
                    if rng.random() < 0.4 and free_spot(q2_[0], q2_[1], 2.0):
                        put(ex3d.sack_pile(q2_[0], q2_[1], float(terrain_h(q2_[0], q2_[1])), float(heading[max(i_ - 4, 0)]), rng,
                                           int(rng.integers(4, 8))), q2_[0], q2_[1], 2.0)
                        n_sacks += 1
            if rng.random() < 0.4:                     # epicerie / hotely / boutique avec enseigne
                sd_ = 1 if rng.random() < 0.5 else -1
                ish_ = min(i_ + int(rng.integers(8, 20)), n - 2)
                q_ = P[ish_] + N[ish_] * sd_ * 10.0
                if free_spot(q_[0], q_[1], 2.6):
                    txt_ = str(rng.choice(["EPICERIE", "HOTELY", "BOUTIQUE", "MOBILE MONEY", "PHARMACIE", "QUINCAILLERIE", "BAR", "SALON DE COIFFURE", "MARCHE"]))
                    emat_, euv_ = ens_at.add(lambda w_, h_, tt=txt_: atl.draw_panneau(w_, h_, tt, "shop"))
                    KIND[emat_] = "SOLID"
                    put(ex3d.shop(q_[0], q_[1], float(terrain_h(q_[0], q_[1])), float(heading[ish_]) + (math.pi if sd_ > 0 else 0.0), rng, emat_, euv_,
                                  str(rng.choice(["wall_ochre", "wall_pink", "wall_white", "wall_green"]))), q_[0], q_[1], 2.6)
                    n_stall += 1
            i_ += int(rng.integers(30, 70))             # un groupe tous les 150-350 m
    # marches / cash points / commerces references par OpenStreetMap
    for nd in osm["nodes"]:
        t_ = nd["tags"]
        pt_ = np.array(nd["pt"], float)
        d_, i_ = tree.query(pt_)
        if d_ > 70 or not shapely.contains_xy(corridor, pt_[0], pt_[1]):
            continue
        ang_ = float(heading[i_]) + (math.pi if (pt_ - P[i_]) @ N[i_] > 0 else 0.0)
        if t_.get("amenity") in ("atm", "bank", "money_transfer") or "mobile" in str(t_.get("shop", "")):
            if free_spot(pt_[0], pt_[1], 1.2):
                put(ex3d.kiosk(pt_[0], pt_[1], float(terrain_h(pt_[0], pt_[1])), ang_), pt_[0], pt_[1], 1.2)
                n_cash += 1
        elif t_.get("amenity") == "marketplace":
            for gx_ in range(-3, 4):
                for gy_ in range(-2, 3):
                    q_ = pt_ + np.array([gx_ * 4.2, gy_ * 4.2]) + rng.normal(0, 0.3, 2)
                    if free_spot(q_[0], q_[1], 1.8):
                        put(ex3d.stall(q_[0], q_[1], float(terrain_h(q_[0], q_[1])), float(heading[i_]) + (0 if rng.random() < 0.5 else math.pi),
                                       rng, charcoal=bool(rng.random() < 0.2)), q_[0], q_[1], 1.8)
                        n_stall += 1
        elif "shop" in t_ and tree.query(pt_)[0] > 8.0:
            if free_spot(pt_[0], pt_[1], 1.6):
                put(ex3d.stall(pt_[0], pt_[1], float(terrain_h(pt_[0], pt_[1])), ang_, rng, charcoal=bool(rng.random() < 0.2)), pt_[0], pt_[1], 1.6)
                n_stall += 1
    print(f"etals : {n_stall}, cash points : {n_cash}, piles de sacs : {n_sacks}", flush=True)
    atl_pages = borne_at.save(f"{W}/tex") + sign_at.save(f"{W}/tex") + ens_at.save(f"{W}/tex")
    U_build = union(occupied + new_polys + [Point(a_, b_).buffer(c_ + 0.5) for a_, b_, c_ in extra_fp])

    # ponts : dalle de beton + garde-corps ; eau des rivieres
    for i0_, i1_ in bridges:
        for i_ in range(i0_, i1_):
            mp_ = (P[i_] + P[i_ + 1]) / 2
            ang_ = math.atan2(P[i_ + 1, 1] - P[i_, 1], P[i_ + 1, 0] - P[i_, 0])
            hm_ = float(min(ROADH[i_], ROADH[i_ + 1]))
            A("beton").add(*ex3d.box(mp_[0], mp_[1], hm_ - 0.85, ang_, 2.6, 4.1, 0.8, 3.0, bottom=True))
            for sd_ in (-1, 1):
                c_ = mp_ + N[i_] * sd_ * 3.65
                A("beton").add(*ex3d.box(c_[0], c_[1], hm_, ang_, 2.6, 0.2, 0.95, 2.0))
    n_riv = 0
    for ln_, hw_, dp_, wt_ in rivers:
        smp_ = np.array([ln_.interpolate(s_).coords[0] for s_ in np.arange(0, ln_.length, 4.0)] + [ln_.coords[-1]])
        ok_ = shapely.contains_xy(corridor, smp_[:, 0], smp_[:, 1])
        run_ = []
        for q_, g_ in list(zip(smp_, ok_)) + [(None, False)]:
            if g_:
                run_.append(q_)
                continue
            if len(run_) >= 2:
                r_ = np.array(run_)
                lev_ = terrain_h(r_[:, 0], r_[:, 1], carve=False) - 0.45 * dp_
                lev_ = gaussian_filter1d(lev_, 3, mode="nearest")
                tg_ = np.gradient(r_, axis=0)
                tg_ /= np.maximum(np.hypot(tg_[:, 0], tg_[:, 1]), 1e-9)[:, None]
                nn_ = np.column_stack([-tg_[:, 1], tg_[:, 0]])
                L_ = r_[:, None, :] + nn_[:, None, :] * np.array([-hw_ - 0.5, hw_ + 0.5])[None, :, None]
                Vv_ = np.stack([L_[..., 0], L_[..., 1], np.repeat(lev_[:, None], 2, axis=1)], -1).reshape(-1, 3)
                sd2_ = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(r_, axis=0).T))])
                UVv_ = np.stack([np.tile([0.0, 1.0], (len(r_), 1)), np.repeat((sd2_ / 8.0)[:, None], 2, axis=1)], -1).reshape(-1, 2)
                kk_ = np.arange(len(r_) - 1)
                Fv_ = np.concatenate([np.stack([2 * kk_, 2 * kk_ + 2, 2 * kk_ + 3], 1), np.stack([2 * kk_, 2 * kk_ + 3, 2 * kk_ + 1], 1)], 0)
                Fv_ = orient_faces(Vv_, Fv_, np.array([0.0, 0.0, 1.0]))
                A("water_surface").add(Vv_, UVv_, Fv_)
                n_riv += 1
            run_ = []
    print(f"surfaces d'eau : {n_riv}", flush=True)

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

    def push(x, y, sp, sc, minoff=SHOULDER + 1.0):
        x = np.asarray(x)
        if len(x) == 0:
            return
        keepm = shapely.contains_xy(corridor, x, y) & ~inside(U_build, x, y) & ~inside(U_water, x, y)
        keepm &= tree.query(np.column_stack([x, y]))[0] > minoff
        if CARVE["fn"] is not None:
            keepm &= CARVE["fn"](x, y) < 0.05
        if U_minor is not None:
            keepm &= ~shapely.intersects_xy(U_minor, x, y)
        x, y, sp, sc = x[keepm], y[keepm], np.asarray(sp)[keepm], np.asarray(sc)[keepm]
        z = terrain_h(x, y) if len(x) else np.zeros(0)
        trees.append(np.column_stack([sp, x, y, z, sc, rng.uniform(0, 6.283, len(x))]))

    TREE_D = {FO: 1 / 40, GR: 1 / 1100, GR2: 1 / 1100, LA: 1 / 2500}
    BUSH_D = {FO: 1 / 80, GR: 1 / 300, GR2: 1 / 300, LA: 1 / 600, CL: 1 / 250, RO: 1 / 900}
    # touffes d'herbe par m2 : (jusqu'a 15 m de la route, jusqu'a 60 m, au-dela)
    GRASS_D = {GR: (1.0, 0.30, 0.03), GR2: (1.0, 0.30, 0.03), PA: (1.0, 0.40, 0.06), FO: (0.8, 0.5, 0.15), LA: (0.4, 0.10, 0.015),
               CL: (0.15, 0.10, 0.03), RO: (0.03, 0.03, 0.01), MU: (0.3, 0.1, 0.01)}
    veg_rows = []
    gcap = int(a.grass_cap * a.grass)
    for t_ in range(ntile):
        r0_, r1_ = t_ * TILE_ST, min((t_ + 1) * TILE_ST, n - 1)
        if r1_ <= r0_:
            continue
        sl_ = slice(r0_, r1_)
        cls_t, off_t, area_t, keep_t = cls[sl_], offc[sl_], qarea[sl_], keep[sl_]
        Q_ = [arr[sl_].reshape(-1, 3) for arr in (V0, V1, V2, V3)]
        for kindv in ("t", "b", "g"):
            dq = np.zeros(cls_t.shape)
            if kindv == "g":
                for code, (d1_, d2_, d3_) in GRASS_D.items():
                    dq = np.where(cls_t == code, np.where(off_t <= 15, d1_, np.where(off_t <= 60, d2_, d3_)), dq)
            else:
                for code, dv in (TREE_D if kindv == "t" else BUSH_D).items():
                    dq = np.where(cls_t == code, dv, dq)
            cnt = rng.poisson(area_t * dq * keep_t).ravel()
            rep_ = np.repeat(np.arange(len(cnt)), cnt)
            if not len(rep_):
                continue
            u, v = rng.random(len(rep_)), rng.random(len(rep_))
            q0, q1, q2, q3 = (q[rep_] for q in Q_)
            pxy = ((1 - u)[:, None] * (1 - v)[:, None] * q0 + u[:, None] * (1 - v)[:, None] * q1
                   + u[:, None] * v[:, None] * q2 + (1 - u)[:, None] * v[:, None] * q3)
            mark = len(trees)
            if kindv == "b":
                push(pxy[:, 0], pxy[:, 1], np.full(len(rep_), 4), rng.uniform(0.7, 1.5, len(rep_)))
            elif kindv == "g":
                push(pxy[:, 0], pxy[:, 1], np.full(len(rep_), 5), rng.uniform(0.6, 1.3, len(rep_)), minoff=SHOULDER + 0.25)
            else:
                push(pxy[:, 0], pxy[:, 1], species_for(pxy[:, 2], len(rep_)), rng.uniform(0.75, 1.4, len(rep_)))
            got = np.vstack(trees[mark:]) if len(trees) > mark else np.zeros((0, 6))
            del trees[mark:]
            if kindv == "g" and len(got) > gcap:        # priorite aux touffes les plus proches de la route
                dist_ = tree.query(got[:, 1:3])[0] + rng.uniform(0, 25, len(got))
                got = got[np.argsort(dist_)[:gcap]]
            elif kindv == "t" and len(got) > a.tree_cap:
                got = got[rng.choice(len(got), a.tree_cap, replace=False)]
            elif kindv == "b" and len(got) > 2 * a.tree_cap:
                got = got[rng.choice(len(got), 2 * a.tree_cap, replace=False)]
            veg_rows.append(got)
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
    TR = np.vstack(veg_rows + trees) if (veg_rows or trees) else np.zeros((0, 6))
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
