# Blender : ce qu'il faut faire (sortie glTF, optionnelle)

Le jeu Simulator MG n'utilise PAS Blender : le zip `*_RSMG.zip` est deja natif (textures PNG incluses).
Blender sert seulement a produire le glTF pour visualiser/retoucher la carte.

## Dans GitHub Actions
1. Actions > Generate map > Run workflow, cocher **blender**.
2. Le workflow installe Blender 4.2.3, puis lance :
   `blender -b --factory-startup -noaudio -P scripts/blender_build.py -- --work work --out out --compress safe`
   Le script lit `work/tiles/*.json` et les textures `work/tex/*.jpg` (les nouvelles textures realistes sont reprises automatiquement), cree un materiau par texture (rugosite par materiau), les arbres, puis exporte le glTF.
3. Telecharger `RN1_km*.zip` dans Releases / artifacts.

## Pour un rendu plus realiste dans Blender (a la main)
- Importer le glTF, mettre un **Sun** (force ~3, angle 0.5 deg, inclinaison ~35 deg) + **monde HDRI** ciel clair.
- Rendu **Cycles** ou EEVEE avec *Ambient Occlusion* et *Bloom* actifs, *Filmic/AgX* en vue couleur.
- Brume legere (Volume Scatter, densite ~0.0005) pour la profondeur comme dans ETS2.
- Ajouter une texture *Bump* depuis la couleur pour la route et la falaise si besoin.

## Arbres et herbe : pourquoi ils n'apparaissaient que par endroits (corrige)
Le jeu ne lit que le DEBUT de chaque `trees.txt` (un quota par troncon). Le generateur ecrivait les points dans l'ordre de creation :
les premiers arbres etaient tous dans les ~350 premiers metres de chaque troncon de 2 km, puis plus rien. `export_rsmg.py` les melange
maintenant : un point par case de 60 m, puis un 2e par case... Tout debut de fichier est donc reparti sur toute la longueur du troncon.
Rien a faire dans Blender pour ca : il suffit de relancer Actions > Generate map (les cartes deja generees gardent l'ancien ordre).
