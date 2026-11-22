import json, os
from pyproj import Transformer

# Meme origine que le jeu (km 0 = Miadana, Antananarivo) : tous les troncons se raccordent.
ORIGIN_LAT = -18.9378
ORIGIN_LON = 47.4999


def transformers():
    proj = (f"+proj=tmerc +lat_0={ORIGIN_LAT} +lon_0={ORIGIN_LON} +k=1 +x_0=0 +y_0=0 "
            "+datum=WGS84 +units=m +no_defs")
    fwd = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
    inv = Transformer.from_crs(proj, "EPSG:4326", always_xy=True)
    return fwd, inv


def save_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
