"""Rewind page: today's continents, each in its own colour, carried back in time by the planet's own recorded plate motions
(the way GPlates reconstructs Earth), shown on a rotatable 3D globe with a time slider. Crust shown with its shelves (land bright, shelf darker). Pieces that split from one
supercontinent visibly close back together (user 2026-10-30: the flat drift map did not show that they had been joined).
Check printed per planet: share of the rewound land that lands on continental crust at the start (should be ~1).

Earth reference (PIPELINE principle 3: a display method is first tried on Earth): the argument "earth" draws Earth's
continental crust (Müller et al. 2022 continents + present land) rewound to 200 Ma (Pangea) by the published rotations,
with exactly the same drawing, so the planets can be compared with a supercontinent that really existed.

Usage: .venv/bin/python planet-history-sim/tools/planet_rewind.py <out.html> [earth] seed [seed ...]"""
import io, base64, json, sys, numpy as np
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from plate_checks import HERE
R_PLANET, STEP, W, H = 6000.0, 5, 1024, 512
DRAW_KM = 70.0               # a pixel takes the colour of the nearest crust point within this distance
lat = np.radians(90 - (np.arange(H) + 0.5) * 180 / H); lon = np.radians(-180 + (np.arange(W) + 0.5) * 360 / W)
la, lo = np.meshgrid(lat, lon, indexing="ij")
G = np.c_[(np.cos(la) * np.cos(lo)).ravel(), (np.cos(la) * np.sin(lo)).ravel(), np.sin(la).ravel()]
PAL = np.array([[240, 190, 70], [90, 170, 240], [235, 110, 90], [120, 200, 110], [200, 130, 220], [80, 200, 190],
                [245, 150, 50], [170, 170, 120]]) / 255
SMALL, SEA, START = np.array([150, 150, 150]) / 255, np.array([18, 44, 82]) / 255, np.array([60, 80, 105]) / 255


def rotate(v, w):
    a = np.linalg.norm(w, axis=1, keepdims=True); k = w / np.maximum(a, 1e-30)
    c, s = np.cos(a), np.sin(a)
    return v * c + np.cross(k, v) * s + k * np.sum(k * v, 1, keepdims=True) * (1 - c)


def jpeg(img):
    buf = io.BytesIO(); plt.imsave(buf, img, format="jpeg", pil_kwargs={"quality": 85})
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def colours(p, is_land, plate):
    """each piece of crust is coloured by the plate that carries it today (as GPlates does): the 8 plates with the most
    continental crust (>= 2% of it) get their own colour. Tried on Earth first: colouring by crust linked within 150 km
    joined Afro-Eurasia, the Americas and Australia through their shelves into one piece."""
    lab = np.unique(plate, return_inverse=True)[1]
    size = np.bincount(lab); order = [i for i in np.argsort(-size) if size[i] >= 0.02 * len(p)][:len(PAL)]
    col = np.tile(SMALL, (len(p), 1))
    for c_, i in enumerate(order):
        col[lab == i] = PAL[c_]
    col[~is_land] = col[~is_land] * 0.55 + SEA * 0.45
    return col, len(order)


def draw(q, col, radius, under=None):
    d, j = cKDTree(q).query(G, distance_upper_bound=2 * np.sin(DRAW_KM / radius / 2))
    img = np.tile(SEA, (H * W, 1))
    if under is not None:
        img[under.ravel()] = START
    ok = np.isfinite(d); img[ok] = col[j[ok]]
    return jpeg(img.reshape(H, W, 3))


def earth(t_max=200):
    import pygplates
    from plate_checks import MODEL, ROTATION, fibonacci_points
    from engine_earth_check import partition_ids
    rot = pygplates.RotationModel(str(ROTATION))
    g = np.array([v.to_xyz() for v in fibonacci_points(60000)])
    raw = np.fromfile(HERE / "../reference/earth/topography/ETOPO5.DAT", dtype=">i2").reshape(2160, 4320)
    la_ = np.degrees(np.arcsin(g[:, 2])); lo_ = np.degrees(np.arctan2(g[:, 1], g[:, 0]))
    elev = raw[np.clip(((90 - la_) * 12).astype(int), 0, 2159), ((lo_ % 360) * 12).astype(int) % 4320]   # ETOPO5 rows from 90N, columns from 0E
    conts = pygplates.FeatureCollection(str(MODEL / "Continents/continents.gpml"))
    crust = (partition_ids(pygplates.PlatePartitioner(conts, rot, 0), g) >= 0) | (elev > 0)
    static = pygplates.FeatureCollection(str(MODEL / "Static_Polygons/static_polygons_Merdith_etal.gpml"))
    part = pygplates.PlatePartitioner(static, rot, 0)
    p = g[crust]; pid = np.full(len(p), -1); born = np.zeros(len(p))
    for i, (x, y, z) in enumerate(p):
        f = part.partition_point(pygplates.PointOnSphere(x, y, z))
        if f is not None:
            pid[i] = f.get_feature().get_reconstruction_plate_id(); born[i] = f.get_feature().get_valid_time()[0]
    from plate_checks import TOPOLOGY_FILES
    topo = [pygplates.FeatureCollection(str(MODEL / "Topologies" / f)) for f in TOPOLOGY_FILES]
    col, nl = colours(p, (elev > 0)[crust], partition_ids(pygplates.PlatePartitioner(topo, rot, 0), p))
    frames = []
    for t in range(0, t_max + 1, STEP):
        q = p.copy()
        for k in np.unique(pid[pid >= 0]):
            m_ = pid == k
            r_k = rot.get_rotation(float(t), int(k))
            q[m_] = [(r_k * pygplates.PointOnSphere(*v)).to_xyz() for v in p[m_]]
        there = (pid >= 0) & (born >= t)              # crust that did not exist yet (young arcs, Iceland) is hidden
        frames.append(dict(myr=t, src=draw(q[there], col[there], 6371.0)))
    print(f"earth: {len(frames)} frames, colours for {nl} landmasses")
    return dict(seed="地球（对照）", name="地球（对照）", T0=t_max, frames=frames)


planets = []
args = sys.argv[2:]
if "earth" in args:
    planets.append(earth()); args = [a for a in args if a != "earth"]
for s in [int(a) for a in args]:
    src = HERE / f"results/planet_m1_seed{s}"
    rec = json.loads((src / "replay_record.json").read_text()); arr = np.load(src / "replay_record.npz")
    grid, maps, conts = arr["grid"], arr["maps"], arr["conts"]; n = len(maps)
    t = np.load(src / "terrain.npz"); r_ = np.load(src / "records.npz")
    # continents are compared by their whole continental crust, shelves included: conjugate margins fit best at the
    # continental slope, not at the coastline (Bullard et al. 1965; RULES A18). Land bright, flooded shelf darker.
    crust = r_["cont"] | (t["elev"] > 0)
    p = t["grid"][crust].copy(); is_land = (t["elev"] > 0)[crust]
    tree = cKDTree(grid)
    col, nl = colours(p, is_land, maps[-1][tree.query(p)[1]]); start_mask = conts[0][cKDTree(grid).query(G)[1]].reshape(H, W)
    frames, k_list = [], list(range(n - 1, -1, -1))
    snaps = {n: (p.copy(), np.ones(len(p), bool))}
    for k in k_list:                                  # step k moves crust from T0-k to T0-k-1; undo it
        pl = maps[k][tree.query(p)[1]]
        Wk = np.array([rec["w"][k].get(str(int(q)), [0, 0, 0]) for q in pl])
        p = rotate(p, -Wk)
        if k % STEP == 0:                             # land not yet on continental crust then (later arcs, islands) is hidden
            snaps[k] = (p.copy(), conts[k][tree.query(p)[1]])
    hit = conts[0][tree.query(p)[1]].mean()
    for k in sorted(snaps, reverse=True):
        q, there = snaps[k]; ids = np.nonzero(there)[0]
        img_src = draw(q[ids], col[ids], R_PLANET, start_mask if k == 0 else None)
        frames.append(dict(myr=int(rec["T0"]) - k if k < n else 0, src=img_src))
    frames[0]["myr"] = 0
    planets.append(dict(seed=s, T0=int(rec["T0"]), frames=frames))
    print(s, f"{len(frames)} frames, rewound land on start continents {hit*100:.0f}%, colours for {nl} landmasses")
page = open(HERE / "rewind_template.html").read().replace("/*DATA*/[]", json.dumps(planets))
open(sys.argv[1], "w").write(page)
print("page", len(page) // 1024, "KB")
