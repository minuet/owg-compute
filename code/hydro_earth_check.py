"""Earth check of hydro.py: endorheic share and where the closed basins are (ETOPO5 at 10', climate_rough at 1 degree).

Reference: closed (endorheic) basins drain about 20% of the land outside Antarctica (Shiklomanov & Rodda 2003; Wang et
al. 2018, HydroSHEDS-based); the large ones are the Caspian-Aral-central Asian basins, the Sahara and Arabia, the
Tibetan plateau interior, the Great Basin, the Altiplano, Lake Eyre (Australia), the Kalahari and Lake Chad.
"""
import math, os, time
import numpy as np
from pathlib import Path
from scipy import ndimage
from climate_rough import climate_grid
from hydro import hydrology

E = Path(__file__).resolve().parent.parent / "reference/earth"
OUT = Path(__file__).resolve().parent / "results/hydro_earth"
OUT.mkdir(parents=True, exist_ok=True)
raw = np.fromfile(E / "topography/ETOPO5.DAT", dtype=">i2").reshape(2160, 4320).astype(float)
raw = np.roll(raw, 2160, axis=1)                                # start at 180W
H, W = 1080, 2160
AGG = os.environ.get("AGG", "mean")
elev = getattr(raw.reshape(H, 2, W, 2), AGG)(axis=(1, 3))       # "min" keeps valley floors (river gorges narrower than a cell)
lab, n = ndimage.label(elev < 0, structure=np.ones((3, 3)))
sizes = np.bincount(lab.ravel())
sizes[0] = 0
ocean = lab == sizes.argmax()
# the Black Sea joins the ocean through the Bosporus (~1 km wide, narrower than a cell): mark it as sea by location
cy, cx = int((90 - 43.5) * 6), int((34.0 + 180) * 6)
if lab[cy, cx] > 0 and not ocean[cy, cx]:
    ocean |= lab == lab[cy, cx]
    print("Black Sea joined to the ocean:", int(sizes[lab[cy, cx]]), "cells")
big = [(int(s), float(90 - (np.nonzero(lab == k)[0].mean() + 0.5) / 6), float(-180 + (np.nonzero(lab == k)[1].mean() + 0.5) / 6))
       for k, s in enumerate(sizes) if s > 2000 and k != sizes.argmax()]
print("below-sea-level water bodies not joined to the ocean (cells, lat, lon):", big)
# climate at 1 degree on the same land mask, repeated to 10'
e1 = elev.reshape(180, 6, 360, 6).mean(axis=(1, 3))
l1 = (~ocean).reshape(180, 6, 360, 6).mean(axis=(1, 3)) > 0.5
c = climate_grid(e1, l1, 6371.0)
up = lambda a: np.repeat(np.repeat(a, 6, axis=0), 6, axis=1)
P, T, A = up(c["P"]), up(c["T"]) - 6.5 * (np.maximum(elev, 0) - up(np.maximum(e1, 0))) / 1000, up(c["A"])
CLIM = os.environ.get("CLIMATE", "worldclim")
if CLIM == "worldclim":
    # observed climate (WorldClim 2.1, Fick & Hijmans 2017, 1970-2000, 10'): tests the hydrology method by itself
    from PIL import Image
    rd = lambda v, m: np.array(Image.open(E / f"climate/worldclim/wc2.1_10m_{v}_{m:02d}.tif")).astype(float)
    pm = np.stack([rd("prec", m) for m in range(1, 13)]); tm = np.stack([rd("tavg", m) for m in range(1, 13)])
    have = (pm[0] > -1000) & (tm[0] > -1000)
    Pw, Tw, Aw = pm.sum(0), tm.mean(0), (tm.max(0) - tm.min(0)) / 2
    land_ = ~ocean
    P = np.where(have, Pw, P); T = np.where(have, Tw, T); A = np.where(have, Aw, A)
    print(f"climate: WorldClim observed ({(have & land_).sum() / land_.sum() * 100:.0f}% of land cells covered; the rest from climate_rough)")
else:
    print("climate: climate_rough (the screening estimator)")
# seam on the meridian with the least land
shift = -int(np.argmin((~ocean).sum(axis=0)))
rl = lambda a: np.roll(a, shift, axis=1)
t0 = time.time()
SED = float(os.environ.get("SED", "0.3"))
r = hydrology(rl(elev), rl(ocean), rl(P), rl(T), rl(A), 6371.0, sed_myr=SED)
print(f"hydrology {time.time()-t0:.0f}s; depressions {r['n_depressions']} (filled by sediment {r['n_sed_filled']}, over {SED:g} Myr), closed {r['n_closed']}")
lat = 90 - (np.arange(H) + 0.5) / 6
land = ~rl(ocean)
noant = land & (lat[:, None] > -60)
a = r["area"]
share = (a * r["endorheic"] * noant).sum() / (a * noant).sum()
print(f"endorheic share of land outside Antarctica: {share*100:.1f}% (reference ~20%)")
print(f"mean runoff over land outside Antarctica: {(a*r['runoff_mm']*noant).sum()/(a*noant).sum():.0f} mm/yr (reference ~300)")
lon = (-180 + (np.arange(W) + 0.5) / 6 - shift / 6 + 180) % 360 - 180
LAT, LON = np.meshgrid(lat, lon, indexing="ij")
regions = {"Caspian-Aral-Central Asia": (35, 55, 45, 90), "Sahara-Arabia": (12, 33, -15, 60), "Tibet interior": (31, 37, 78, 95),
           "Great Basin": (36, 44, -120, -111), "Altiplano": (-24, -15, -70, -66), "Lake Eyre": (-32, -20, 132, 142),
           "Kalahari": (-25, -17, 18, 27), "Amazon (should be open)": (-15, 2, -75, -50), "Congo (open)": (-10, 5, 15, 30),
           "Europe (open)": (43, 60, -5, 30), "Eastern US (open)": (30, 45, -90, -75)}
for nm, (la0, la1, lo0, lo1) in regions.items():
    m = land & (LAT >= la0) & (LAT <= la1) & (LON >= lo0) & (LON <= lo1)
    print(f"  {nm:28s} endorheic {100*(a*r['endorheic']*m).sum()/(a*m).sum():5.1f}%")
np.savez_compressed(OUT / "hydro_earth.npz", endorheic=r["endorheic"], runoff=r["runoff_mm"].astype(np.float32), shift=shift)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    img = np.full((H, W, 3), 0.85)
    img[rl(ocean)] = (0.75, 0.85, 0.95)
    img[land & ~r["endorheic"]] = (0.55, 0.75, 0.5)
    img[r["endorheic"]] = (0.9, 0.6, 0.3)
    img[r["lakes"]] = (0.2, 0.3, 0.8)
    img = np.roll(img, -shift, axis=1)
    plt.figure(figsize=(16, 8)); plt.imshow(img, extent=(-180, 180, -90, 90)); plt.title(f"Earth endorheic (orange) {share*100:.1f}% of land outside Antarctica")
    plt.savefig(OUT / f"earth_endorheic_{CLIM}.png", dpi=80, bbox_inches="tight")
except Exception as ex:
    print("plot failed", ex)
