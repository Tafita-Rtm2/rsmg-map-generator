# rsmg-map-generator

Genere automatiquement (GitHub Actions + Blender) une carte 3D realiste d'une route de Madagascar
(relief reel Copernicus 30 m, route et batiments OpenStreetMap, villages completes, vegetation, textures),
decoupee en troncons de 2 km, avec zip et 7z au maximum de compression.

**Lancer** : onglet *Actions* > *Generate map* > *Run workflow* (route, km de debut/fin).
**Telecharger** : onglet *Releases* (lien direct) ou l'artifact du run.
Le format est decrit dans `manifest.json` (dans l'archive). Donnees (c) OpenStreetMap contributors (ODbL).

## Sortie
- `RN1_km0-20_RSMG.zip` / `.7z` : **format natif de Simulator MG** (cartes .map.txt + parties .3ds, trees.txt, arrets, trafic).
  A extraire dans `/sdcard/` : le dossier `RSMG/maps/` est deja dedans. Aucune modification du jeu necessaire.
- `RN1_km0-20.zip` / `.7z` : glTF (Blender), pour d'autres usages.

## Textures photo-realistes (assets/polyhaven, assets/veg)
- **Sols** : vrais sols photo (CC0 Poly Haven) rendus sans raccord et etalonnes par `scripts/textures.py` : herbe verte x2 (sol entierement couvert d'herbe,
  y compris rizieres et sol de foret), laterite rouge, falaises/roches (UV projetes selon la normale = pas d'etirement sur les parois), boue, accotement gravier,
  asphalte, trottoir, beton, piste.
- **Vegetation** : arbres, buissons et herbes = plans croises photo a fond transparent (`assets/veg/*.png`, decoupes par `scripts/prep_sprites.py`),
  ecrits dans les .3ds (materiaux `veg_*`, carte d'opacite = canal alpha du PNG). `RSMG_CARDS=0` revient aux modeles d'arbres du jeu (trees.txt).
- Pour changer une texture : remplacer le fichier dans `assets/polyhaven/` (vignette carree a gauche) puis relancer le workflow.
