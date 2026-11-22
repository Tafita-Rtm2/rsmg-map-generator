# rsmg-map-generator

Genere automatiquement (GitHub Actions + Blender) une carte 3D realiste d'une route de Madagascar
(relief reel Copernicus 30 m, route et batiments OpenStreetMap, villages completes, vegetation, textures),
decoupee en troncons de 2 km, avec zip et 7z au maximum de compression.

**Lancer** : onglet *Actions* > *Generate map* > *Run workflow* (route, km de debut/fin).
**Telecharger** : onglet *Releases* (lien direct) ou l'artifact du run.
Le format est decrit dans `manifest.json` (dans l'archive). Donnees (c) OpenStreetMap contributors (ODbL).
