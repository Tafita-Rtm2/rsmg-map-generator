#!/usr/bin/env python3
"""Telecharge la route (OSM), les objets autour (OSM) et le relief (Copernicus DEM 30 m)."""
import argparse, gzip, math, re, sys, time
import numpy as np
import requests
import networkx as nx
from common import ORIGIN_LAT, ORIGIN_LON, transformers, save_json

OVERPASS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
DEFAULT_START = {"RN1": (-18.950, 47.470)}   # Anosizato, Antananarivo -> vers Tsiroanomandidy
UA = {"User-Agent": "rsmg-map-generator/1.0"}


def overpass(q):
    last = None
    for attempt in range(15):
        url = OVERPASS[attempt % len(OVERPASS)]
        try:
            r = requests.post(url, data={"data": q}, headers=UA, timeout=400)
            if r.status_code == 200:
                return r.json()
            last = f"{url} -> HTTP {r.status_code}"
        except Exception as e:  # noqa
            last = f"{url} -> {e}"
        print("overpass: nouvel essai :", last, flush=True)
        time.sleep(min(6 + 4 * attempt, 40))
    raise RuntimeError("Overpass indisponible : " + str(last))


def road_graph(road, fwd, start_ll, km_to):
    m = re.match(r"^([A-Za-z]+)\s*0*(\d+)$", road.strip())
    if not m:
        sys.exit("Nom de route invalide : " + road)
    prefix, num = m.group(1).upper(), m.group(2)
    pre = "R?N" if prefix in ("RN", "N") else prefix
    pat = f"(^|;) ?{pre} ?0*{num}(;|$)"
    # petite zone autour du depart (requete legere) au lieu de tout Madagascar
    h = max(0.35, km_to / 70.0)
    lat0, lon0 = start_ll
    bb = f"{lat0 - h:.3f},{lon0 - h:.3f},{lat0 + h:.3f},{lon0 + h:.3f}"
    queries = [
        f'[out:json][timeout:120];way["highway"]["ref"~"{pat}"]({bb});out geom;',
        f'[out:json][timeout:120];way["highway"]["name"~"Route Nationale {num}($|[^0-9])|^RN ?{num}$"]({bb});out geom;',
    ]
    ways = []
    for q in queries:
        js = overpass(q)
        els = js.get("elements", [])
        print("requete route :", len(els), "elements", js.get("remark", ""), flush=True)
        ways = [e for e in els if e.get("type") == "way" and e.get("geometry")]
        if ways:
            break
    if not ways:
        sys.exit("Aucune route trouvee pour " + road)
    G = nx.Graph()
    pos = {}
    for w in ways:
        ids = w["nodes"]
        g = w["geometry"]
        lon = np.array([p["lon"] for p in g])
        lat = np.array([p["lat"] for p in g])
        x, y = fwd.transform(lon, lat)
        for nid, xx, yy in zip(ids, x, y):
            pos[nid] = (float(xx), float(yy))
        for a_, b_ in zip(ids[:-1], ids[1:]):
            d = math.hypot(pos[a_][0] - pos[b_][0], pos[a_][1] - pos[b_][1])
            G.add_edge(a_, b_, weight=d)
    print(f"{road}: {len(ways)} troncons OSM, {G.number_of_nodes()} noeuds", flush=True)
    return G, pos


def main_path(G, pos, start_xy):
    ids = list(G.nodes)
    arr = np.array([pos[n] for n in ids])
    s = ids[int(np.argmin(np.hypot(arr[:, 0] - start_xy[0], arr[:, 1] - start_xy[1])))]
    H = G.subgraph(nx.node_connected_component(G, s))
    dist = nx.single_source_dijkstra_path_length(H, s, weight="weight")
    f = max(dist, key=dist.get)
    path = nx.dijkstra_path(H, s, f, weight="weight")
    print(f"depart {pos[s]} -> arrivee {pos[f]} : {dist[f] / 1000:.1f} km de route continue", flush=True)
    return np.array([pos[n] for n in path])


def resample(pts, step):
    d = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0], np.cumsum(d)])
    t = np.arange(0, s[-1], step)
    return np.column_stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])])


def fetch_features(P, fwd, inv, radius):
    ways, nodes, seen = [], [], set()
    seg = 800
    for a in range(0, len(P), seg):
        sub = P[a:a + seg + 1]
        dec = sub[::20] if len(sub) > 20 else sub
        if len(dec) < 2:
            continue
        lon, lat = inv.transform(dec[:, 0], dec[:, 1])
        coords = ",".join(f"{la:.5f},{lo:.5f}" for la, lo in zip(lat, lon))
        q = f"""[out:json][timeout:300];(
way["building"](around:{radius},{coords});
way["landuse"](around:{radius},{coords});
way["natural"](around:{radius},{coords});
way["waterway"](around:{radius},{coords});
way["highway"](around:{radius},{coords});
node["natural"="tree"](around:{radius},{coords});
node["place"](around:1500,{coords});
);out geom;"""
        js = overpass(q)
        for e in js.get("elements", []):
            key = (e["type"], e["id"])
            if key in seen:
                continue
            seen.add(key)
            if e["type"] == "way" and e.get("geometry"):
                lo = np.array([p["lon"] for p in e["geometry"]])
                la = np.array([p["lat"] for p in e["geometry"]])
                x, y = fwd.transform(lo, la)
                ways.append({"id": e["id"], "tags": e.get("tags", {}),
                             "pts": np.round(np.column_stack([x, y]), 2).tolist()})
            elif e["type"] == "node" and "lat" in e:
                x, y = fwd.transform(e["lon"], e["lat"])
                nodes.append({"tags": e.get("tags", {}), "pt": [round(float(x), 2), round(float(y), 2)]})
        print(f"OSM {a // seg + 1}/{(len(P) + seg - 1) // seg} : {len(ways)} lignes, {len(nodes)} points", flush=True)
    return ways, nodes


def fill_nan(dem):
    if np.isnan(dem).all():
        raise RuntimeError("DEM vide")
    m = np.nanmean(dem)
    return np.where(np.isnan(dem), m, dem).astype("float32")


def dem_copernicus(w, s, e, n, out):
    import rasterio
    from rasterio.merge import merge
    urls = []
    for lat in range(math.floor(s), math.floor(n) + 1):
        for lon in range(math.floor(w), math.floor(e) + 1):
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            name = f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"
            urls.append(f"https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif")
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        srcs = [rasterio.open(u) for u in urls]
        arr, tr = merge(srcs, bounds=(w, s, e, n))
        for x in srcs:
            x.close()
    dem = arr[0].astype("float32")
    dem[dem < -1000] = np.nan
    np.savez_compressed(out, dem=fill_nan(dem), west=tr.c, north=tr.f, dx=tr.a, dy=-tr.e)
    print("Relief : Copernicus DEM 30 m", dem.shape, flush=True)


def dem_skadi(w, s, e, n, out):
    lats = list(range(math.floor(s), math.floor(n) + 1))
    lons = list(range(math.floor(w), math.floor(e) + 1))
    rows = []
    for lat in reversed(lats):
        row = []
        for lon in lons:
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            name = f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"
            url = f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/{ns}{abs(lat):02d}/{name}.hgt.gz"
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            a = np.frombuffer(gzip.decompress(r.content), dtype=">i2").reshape(3601, 3601).astype("float32")
            a[a < -1000] = np.nan
            row.append(a[:3600, :3600])
        rows.append(np.hstack(row))
    dem = np.vstack(rows)
    np.savez_compressed(out, dem=fill_nan(dem), west=float(min(lons)), north=float(max(lats) + 1),
                        dx=1 / 3600, dy=1 / 3600)
    print("Relief : SRTM 30 m (secours)", dem.shape, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--road", default="RN1")
    ap.add_argument("--km-from", type=float, default=0)
    ap.add_argument("--km-to", type=float, default=20)
    ap.add_argument("--corridor", type=float, default=320)
    ap.add_argument("--start-lat", type=float, default=None)
    ap.add_argument("--start-lon", type=float, default=None)
    ap.add_argument("--out", default="work")
    a = ap.parse_args()
    fwd, inv = transformers()

    if a.start_lat is not None and a.start_lon is not None:
        start_ll = (a.start_lat, a.start_lon)
    else:
        start_ll = DEFAULT_START.get(a.road.upper().replace(" ", ""), (ORIGIN_LAT, ORIGIN_LON))
    G, pos = road_graph(a.road, fwd, start_ll, a.km_to)
    sx, sy = fwd.transform(start_ll[1], start_ll[0])
    full = resample(main_path(G, pos, (sx, sy)), 5.0)
    i0, i1 = int(a.km_from * 200), int(a.km_to * 200)
    P = full[i0:i1 + 1]
    if len(P) < 20:
        sys.exit(f"Route trop courte ({len(full) * 5 / 1000:.1f} km) pour km {a.km_from}-{a.km_to}")
    print(f"Troncon retenu : {len(P) * 5 / 1000:.1f} km", flush=True)

    ways, nodes = fetch_features(P, fwd, inv, int(a.corridor))
    margin = a.corridor + 500
    w, s = inv.transform(P[:, 0].min() - margin, P[:, 1].min() - margin)
    e, n = inv.transform(P[:, 0].max() + margin, P[:, 1].max() + margin)
    try:
        dem_copernicus(w, s, e, n, f"{a.out}/dem.npz")
    except Exception as ex:  # noqa
        print("Copernicus indisponible :", ex, flush=True)
        dem_skadi(w, s, e, n, f"{a.out}/dem.npz")

    save_json(f"{a.out}/road.json", {"road": a.road, "km_from": a.km_from, "km_to": a.km_to,
                                     "origin": [ORIGIN_LAT, ORIGIN_LON],
                                     "pts": np.round(P, 2).tolist()})
    save_json(f"{a.out}/osm.json", {"ways": ways, "nodes": nodes})


if __name__ == "__main__":
    main()
