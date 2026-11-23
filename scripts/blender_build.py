"""Execute DANS Blender : blender -b -P scripts/blender_build.py -- --work work --out out --compress safe|max
Lit work/tiles/*.json (maillages calcules par make_geometry.py), applique les textures, cree les modeles d'arbres
et exporte le tout en glTF (textures partagees entre troncons)."""
import json, math, os, sys
import numpy as np
import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(name, default):
    return argv[argv.index(name) + 1] if name in argv else default


WORK = arg("--work", "work")
OUT = arg("--out", "out")
COMP = arg("--compress", "safe")
TEX = os.path.join(WORK, "tex")
MAXC = COMP == "max"

bpy.ops.wm.read_factory_settings(use_empty=True)
_mats = {}


def get_mat(name, alpha=False, cull=True):
    key = (name, alpha)
    if key in _mats:
        return _mats[key]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    path = os.path.join(TEX, name + (".png" if alpha else ".jpg"))
    if os.path.exists(path):
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = bpy.data.images.load(path)
        t.extension = "REPEAT"
        nt.links.new(t.outputs["Color"], b.inputs["Base Color"])
        if alpha:
            nt.links.new(t.outputs["Alpha"], b.inputs["Alpha"])
    # rugosite par materiau : asphalte mat-poli, tole semi-brillante, eau lisse, reste mat
    rough = {"road_main": 0.82, "road_minor": 0.88, "roof_tin": 0.55, "water": 0.12, "water_surface": 0.12,
             "falaise": 1.0, "grass": 1.0, "laterite": 0.97, "shoulder": 0.98}.get(name, 0.95)
    b.inputs["Roughness"].default_value = rough
    for nm in ("Specular IOR Level", "Specular"):
        if nm in b.inputs:
            b.inputs[nm].default_value = 0.05
            break
    m.use_backface_culling = bool(cull and not alpha)
    if alpha:
        try:
            m.blend_method = "CLIP"
        except Exception:
            pass
    _mats[key] = m
    return m


def make_obj(name, V, UV, F, mat, smooth=False):
    V = np.asarray(V, np.float32).reshape(-1, 3)
    UV = np.asarray(UV, np.float32).reshape(-1, 2)
    F = np.asarray(F, np.int32).reshape(-1, 3)
    me = bpy.data.meshes.new(name)
    me.from_pydata(V.tolist(), [], F.tolist())
    me.update()
    uvl = me.uv_layers.new(name="UVMap")
    uvl.data.foreach_set("uv", UV[F.ravel()].ravel())
    if smooth:
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def export(path, objs, **extra):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    kw = dict(export_format="GLTF_SEPARATE", use_selection=True, export_apply=False,
              export_texture_dir="textures", export_image_format="WEBP" if MAXC else "AUTO",
              export_jpeg_quality=80, export_image_quality=80,
              export_draco_mesh_compression_enable=MAXC, export_draco_mesh_compression_level=7,
              export_draco_position_quantization=14, export_draco_normal_quantization=8,
              export_draco_texcoord_quantization=12, export_cameras=False, export_lights=False,
              export_yup=True, export_normals=True, export_texcoords=True, export_materials="EXPORT",
              export_image_webp_fallback=False)
    kw.update(extra)
    props = set(bpy.ops.export_scene.gltf.get_rna_type().properties.keys())
    kw = {k: v for k, v in kw.items() if k in props}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=path, **kw)


def clear_objects():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for me in list(bpy.data.meshes):
        bpy.data.meshes.remove(me)


# ---------------------------------------------------------------- troncons
def build_tiles():
    man = json.load(open(os.path.join(WORK, "manifest.json")))
    for t in man["tiles"]:
        i = t["index"]
        data = json.load(open(os.path.join(WORK, "tiles", f"tile_{i:02d}.json")))
        objs = []
        for mat, d in data.items():
            kind = d["kind"]
            culloff = mat.startswith("roof_")
            ob = make_obj(f"{kind}_{mat}", d["v"], d["uv"], d["f"], get_mat(mat, cull=not culloff), smooth=(kind == "GND"))
            objs.append(ob)
        if objs:
            export(os.path.join(OUT, "tiles", f"tile_{i:02d}.gltf"), objs)
        clear_objects()
        print(f"[blender] troncon {i} exporte ({len(objs)} objets)", flush=True)


# ---------------------------------------------------------------- modeles d'arbres
def icosphere(sub=1):
    t = (1 + 5 ** 0.5) / 2
    V = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]
    V = [np.array(v, float) / np.linalg.norm(v) for v in V]
    for _ in range(sub):
        cache, newf = {}, []

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                mm = V[a] + V[b]
                V.append(mm / np.linalg.norm(mm))
                cache[k] = len(V) - 1
            return cache[k]
        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            newf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        F = newf
    return np.array(V), np.array(F)


def blob(c, r, noise, rng):
    V, F = icosphere(1)
    V = np.array(c) + V * np.array(r) * (1 + rng.normal(0, noise, (len(V), 1)))
    return V, F


def cyl(r0, r1, h, z0=0.0, seg=7):
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    bot = np.column_stack([r0 * np.cos(ang), r0 * np.sin(ang), np.full(seg, z0)])
    top = np.column_stack([r1 * np.cos(ang), r1 * np.sin(ang), np.full(seg, z0 + h)])
    F = []
    for i in range(seg):
        j = (i + 1) % seg
        F += [(i, j, seg + j), (i, seg + j, seg + i)]
    return np.vstack([bot, top]), np.array(F)


def cone(r, h, z0, seg=8):
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    V = np.vstack([np.column_stack([r * np.cos(ang), r * np.sin(ang), np.full(seg, z0)]), [[0, 0, z0 + h]]])
    F = [(i, (i + 1) % seg, seg) for i in range(seg)]
    return V, np.array(F)


def join(parts):
    Vs, Fs, n = [], [], 0
    for V, F in parts:
        Vs.append(V)
        Fs.append(F + n)
        n += len(V)
    return np.vstack(Vs), np.vstack(Fs)


def uv_planar(V, s=0.4):
    return V[:, :2] * s


def build_models():
    rng = np.random.default_rng(3)
    specs = {}
    # eucalyptus
    trunk = cyl(0.25, 0.12, 11, 0, 7)
    crown = join([blob((rng.uniform(-1, 1), rng.uniform(-1, 1), z), (2.0, 2.0, 2.8), 0.12, rng) for z in (9.5, 11, 12.5, 10.5)])
    specs["euc"] = (trunk, crown, "bark_light", "leaf_euc")
    # arbre rond (manguier, etc.)
    trunk = cyl(0.35, 0.2, 3.2, 0, 7)
    crown = join([blob((rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(4.5, 6.2)), (3.0, 3.0, 2.4), 0.12, rng) for _ in range(6)])
    specs["round"] = (trunk, crown, "bark", "leaf_round")
    # pin
    trunk = cyl(0.25, 0.1, 12, 0, 6)
    crown = join([cone(3.2 - i * 0.5, 3.4, 2.5 + i * 2.0, 8) for i in range(5)])
    specs["pine"] = (trunk, crown, "bark", "leaf_pine")
    # ravinala (arbre du voyageur) : eventail de feuilles
    trunk = cyl(0.2, 0.15, 6, 0, 6)
    blades = []
    for a in np.linspace(-1.2, 1.2, 11):
        d = np.array([math.sin(a), 0, math.cos(a)])
        base = np.array([0, 0, 6.0])
        w = 0.3
        V = np.array([base + [-w, 0, 0], base + [w, 0, 0], base + d * 3.6 + [w * 1.4, 0, 0], base + d * 3.6 + [-w * 1.4, 0, 0]])
        blades.append((V, np.array([(0, 1, 2), (0, 2, 3)])))
    specs["ravi"] = (trunk, join(blades), "bark", "leaf_ravi")
    # buisson
    crown = join([blob((rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), 0.6), (0.9, 0.9, 0.7), 0.15, rng) for _ in range(3)])
    specs["bush"] = (None, crown, None, "leaf_round")

    sizes = {"euc": (5, 13), "round": (9, 8), "pine": (6, 12), "ravi": (7, 7), "bush": (1.8, 1.4)}
    for sp, (trunk, crown, bark_m, leaf_m) in specs.items():
        objs = []
        if trunk is not None:
            V, F = trunk
            objs.append(make_obj(f"tree_{sp}_trunk", V, uv_planar(V, 0.5), F, get_mat(bark_m), smooth=True))
        V, F = crown
        objs.append(make_obj(f"tree_{sp}_crown", V, uv_planar(V, 0.35), F, get_mat(leaf_m, cull=(sp != "ravi")), smooth=(sp != "ravi")))
        export(os.path.join(OUT, "models", f"tree_{sp}.gltf"), objs)
        clear_objects()
        # version lointaine : deux plans croises a texture transparente
        w, h = sizes[sp]
        hw = w / 2
        V = np.array([[-hw, 0, 0], [hw, 0, 0], [hw, 0, h], [-hw, 0, h], [0, -hw, 0], [0, hw, 0], [0, hw, h], [0, -hw, h]], float)
        F = np.array([(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7)])
        UV = np.array([[0, 0], [1, 0], [1, 1], [0, 1]] * 2, float)
        ob = make_obj(f"tree_{sp}_far", V, UV, F, get_mat(f"far_{sp}", alpha=True, cull=False))
        export(os.path.join(OUT, "models", f"tree_{sp}_far.gltf"), [ob])
        clear_objects()
    print("[blender] modeles d'arbres exportes", flush=True)


build_models()
build_tiles()
print("[blender] termine", flush=True)
