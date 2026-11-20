"""Earth check of climate_rough.py (RULES.md C2): Köppen map (Kottek 2006) vs the estimator at 1 degree, ETOPO5 relief."""
import numpy as np, time
from pathlib import Path
from climate_rough import climate_grid, biome_shares
E = Path(__file__).resolve().parent.parent / "reference/earth"
W, H = 360, 180
raw = np.fromfile(E / "topography/ETOPO5.DAT", dtype=">i2").reshape(2160, 4320).astype(float)
raw = np.roll(raw, 2160, axis=1)                               # start at 180W like the Köppen grid
elev = raw.reshape(H, 12, W, 12).mean(axis=(1, 3))
kg = np.full((H, W), -1, int); land = np.zeros((H, W), bool)
m = {"Af": 0, "Am": 0, "Aw": 1, "As": 1, "BWh": 2, "BWk": 2, "BSh": 3, "BSk": 3, "ET": 6, "EF": 7}
cnt = np.zeros((H, W)); first = {}
for line in (E / "climate/Koeppen-Geiger-ASCII.txt").read_text().split("\n")[1:]:
    p = line.split()
    if len(p) < 3: continue
    la, lo, c = float(p[0]), float(p[1]), p[2]
    y, x = int((90 - la) // 1), int((lo + 180) // 1)
    k = m.get(c)
    if k is None:
        k = 4 if c[0] == "C" else (5 if c[2] in "cd" else 4) if c[0] == "D" else None
    if k is None: continue
    cnt[y, x] += 1
    if (y, x) not in first: first[(y, x)] = k
for (y, x), k in first.items():
    if cnt[y, x] >= 2: kg[y, x] = k; land[y, x] = True
t = time.time(); r = climate_grid(elev, land, 6371.0); dt = time.time() - t
lat = 90 - (np.arange(H) + 0.5)
w = np.cos(np.radians(lat))[:, None] * np.ones((1, W))
mb = np.where(r["B"] == 8, 6, r["B"])
G = lambda k: np.where((k == 2) | (k == 3), 1, np.where((k == 6) | (k == 7), 2, 0))
sel = kg >= 0
agree = (w[sel] * (mb[sel] == kg[sel])).sum() / w[sel].sum()
agree3 = (w[sel] * (G(mb[sel]) == G(kg[sel]))).sum() / w[sel].sum()
sh = biome_shares(np.where(sel, mb, -1), lat)
real = biome_shares(np.where(sel, kg, -1), lat)
print(f"time {dt:.1f}s  8-class agreement {agree*100:.1f}%  3-group agreement {agree3*100:.1f}%")
print("habitable (0,1,3,4,5): model %.1f%%  real %.1f%%" % (100 * sum(sh[k] for k in (0, 1, 3, 4, 5)), 100 * sum(real[k] for k in (0, 1, 3, 4, 5))))
print("desert: model %.1f%% real %.1f%%; cold (6,7): model %.1f%% real %.1f%%" % (sh[2] * 100, real[2] * 100, (sh[6] + sh[7]) * 100, (real[6] + real[7]) * 100))
