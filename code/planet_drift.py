"""Plate-drift movie page: continental crust over the last 200 Myr, coloured by plate, so the user can watch the continents
split and drift (user 2026-10-29: "看不出这是板块漂移形成的"). Reads the coarse run's replay_record.npz (conts/maps per Myr).

Usage: .venv/bin/python planet-history-sim/tools/planet_drift.py <out.html> seed [seed ...]"""
import io, base64, json, sys, numpy as np
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from plate_checks import HERE
W, H, STEP = 720, 360, 5
lat = np.radians(90 - (np.arange(H) + 0.5) * 180 / H); lon = np.radians(-180 + (np.arange(W) + 0.5) * 360 / W)
la, lo = np.meshgrid(lat, lon, indexing="ij")
G = np.c_[(np.cos(la) * np.cos(lo)).ravel(), (np.cos(la) * np.sin(lo)).ravel(), np.sin(la).ravel()]
PAL = np.array([[230, 170, 80], [120, 190, 90], [210, 110, 110], [150, 140, 220], [90, 190, 190], [220, 200, 90],
                [200, 130, 200], [160, 120, 80], [110, 160, 230], [240, 140, 60], [170, 210, 140], [200, 200, 200]]) / 255
SEA, PRESENT = np.array([22, 52, 92]) / 255, np.array([255, 255, 255]) / 255
planets = []
for s in [int(a) for a in sys.argv[2:]]:
    src = HERE / f"results/planet_m1_seed{s}"
    r = np.load(src / "replay_record.npz")
    idx = cKDTree(r["grid"]).query(G)[1]
    land_now = (np.load(src / "terrain.npz")["elev"] > 0)[cKDTree(np.load(src / "terrain.npz")["grid"]).query(G)[1]].reshape(H, W)
    # present coastline, drawn faintly on every frame as a fixed reference
    coast = land_now ^ np.roll(land_now, 1, 1) | land_now ^ np.roll(land_now, 1, 0)
    n = len(r["conts"]); frames = []
    for i in list(range(0, n, STEP)) + ([n - 1] if (n - 1) % STEP else []):
        c = r["conts"][i][idx].reshape(H, W); p = r["maps"][i][idx].reshape(H, W)
        img = np.empty((H, W, 3)); img[:] = SEA
        img[c] = PAL[p[c] % len(PAL)]
        img[coast] = img[coast] * 0.5 + PRESENT * 0.5
        buf = io.BytesIO(); plt.imsave(buf, img, format="png")
        frames.append(dict(myr=n - 1 - i, src="data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()))
    planets.append(dict(seed=s, frames=frames))
    print(s, len(frames), "frames")
page = open(HERE / "drift_template.html").read().replace("/*DATA*/[]", json.dumps(planets))
open(sys.argv[1], "w").write(page)
print("page", len(page) // 1024, "KB")
