"""Lecture des donnees OpenStreetMap depuis l'extrait complet de Madagascar (Geofabrik, .osm.pbf) :
beaucoup plus rapide et fiable qu'Overpass, et sans limite de longueur (250 km et plus)."""
import math
import re
import numpy as np
import osmium
from scipy.spatial import cKDTree

LAT0, LON0 = -18.9378, 47.4999
KX, KY = 111320.0 * math.cos(math.radians(LAT0)), 110540.0
CELL = 800.0
KEEP = ("building", "building:levels", "landuse", "natural", "crop", "waterway", "highway", "surface", "ref", "name",
        "power", "amenity", "shop", "place", "bridge", "bus", "tracktype", "leisure")
WAY_KEYS = ("building", "landuse", "natural", "waterway", "highway", "power", "amenity", "leisure")
AMEN = {"fuel", "marketplace", "bus_station", "atm", "bank", "money_transfer"}
HIGH_NODE = {"bus_stop", "street_lamp", "crossing"}


class RoadHandler(osmium.SimpleHandler):
    def __init__(self, pat):
        super().__init__()
        self.pat = pat
        self.ways = []

    def way(self, w):
        t = w.tags
        if "highway" not in t:
            return
        ref = t.get("ref")
        if not ref or not self.pat.search(ref):
            return
        try:
            self.ways.append([(n.ref, n.lon, n.lat) for n in w.nodes])
        except osmium.InvalidLocationError:
            pass


def road_ways(pbf, road):
    m = re.match(r"^([A-Za-z]+)\s*0*(\d+)$", road.strip())
    prefix, num = m.group(1).upper(), m.group(2)
    pre = "R?N" if prefix in ("RN", "N") else prefix
    h = RoadHandler(re.compile(rf"(^|;) ?{pre} ?0*{num}(;|$)"))
    h.apply_file(pbf, locations=True)
    return h.ways


class FeatureHandler(osmium.SimpleHandler):
    def __init__(self, route_xy, fwd, radius):
        super().__init__()
        self.fwd = fwd
        self.radius = radius
        self.tree = cKDTree(route_xy)
        self.route = route_xy
        cells = set()
        cx = np.floor(route_xy[:, 0] / CELL).astype(int)
        cy = np.floor(route_xy[:, 1] / CELL).astype(int)
        for a, b in set(zip(cx.tolist(), cy.tolist())):
            for da in range(-2, 3):
                for db in range(-2, 3):
                    cells.add((a + da, b + db))
        self.cells = cells
        self.ways, self.nodes = [], []

    def _near_cells(self, lons, lats):
        n = len(lons)
        idx = range(n) if n <= 40 else np.unique(np.linspace(0, n - 1, 40).astype(int))
        for i in idx:
            x = (lons[i] - LON0) * KX
            y = (lats[i] - LAT0) * KY
            if (int(x // CELL), int(y // CELL)) in self.cells:
                return True
        return False

    def way(self, w):
        t = w.tags
        keep = None
        for k in WAY_KEYS:
            if k in t:
                keep = k
                break
        if keep is None:
            return
        if keep == "power" and t.get("power") not in ("line", "minor_line"):
            return
        if keep == "amenity" and t.get("amenity") not in AMEN:
            return
        try:
            lons = [n.lon for n in w.nodes]
            lats = [n.lat for n in w.nodes]
        except osmium.InvalidLocationError:
            return
        if len(lons) < 2:
            return
        poly = lons[0] == lons[-1] and lats[0] == lats[-1] and len(lons) >= 4
        near = self._near_cells(lons, lats)
        if not near and poly and keep in ("landuse", "natural"):
            # grand polygone dont aucun sommet n'est proche mais qui peut couvrir la route
            x0, x1 = (min(lons) - LON0) * KX, (max(lons) - LON0) * KX
            y0, y1 = (min(lats) - LAT0) * KY, (max(lats) - LAT0) * KY
            cc = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
            rr = math.hypot(x1 - x0, y1 - y0) / 2
            if rr < 60000:
                idx = self.tree.query_ball_point(cc, rr)
                if idx:
                    r = self.route[idx]
                    near = bool(((r[:, 0] >= x0) & (r[:, 0] <= x1) & (r[:, 1] >= y0) & (r[:, 1] <= y1)).any())
        if not near:
            return
        x, y = self.fwd.transform(np.array(lons), np.array(lats))
        pts = np.column_stack([x, y])
        d, _ = self.tree.query(pts)
        if d.min() > self.radius + 40 and not (poly and keep in ("landuse", "natural") and near):
            return
        self.ways.append({"id": w.id, "tags": {k: t[k] for k in KEEP if k in t},
                          "pts": np.round(pts, 2).tolist()})

    def node(self, n):
        t = n.tags
        if not t:
            return
        take = False
        if "place" in t:
            take = True
        elif t.get("natural") == "tree":
            take = True
        elif t.get("power") in ("pole", "tower"):
            take = True
        elif t.get("amenity") in AMEN or t.get("highway") in HIGH_NODE or "shop" in t:
            take = True
        if not take or not n.location.valid():
            return
        x = (n.location.lon - LON0) * KX
        y = (n.location.lat - LAT0) * KY
        if (int(x // CELL), int(y // CELL)) not in self.cells:
            return
        px, py = self.fwd.transform(n.location.lon, n.location.lat)
        d, _ = self.tree.query([px, py])
        lim = 1500 if "place" in t else self.radius
        if d <= lim:
            self.nodes.append({"tags": {k: t[k] for k in KEEP if k in t}, "pt": [round(float(px), 2), round(float(py), 2)]})


def features(pbf, route_xy, fwd, radius):
    h = FeatureHandler(route_xy, fwd, radius)
    h.apply_file(pbf, locations=True)
    return h.ways, h.nodes
