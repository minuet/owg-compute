"""RULES.md A12 on Earth: ocean depth predicted from seafloor age vs measured depth (ETOPO5).

Age-depth relations (depth below sea level, metres; t in Myr):
  GDH1 (Stein & Stein 1992):      2600 + 365*sqrt(t)            for t < 20
                                  5651 - 2473*exp(-0.0278*t)    for t >= 20
  Parsons & Sclater 1977:         2500 + 350*sqrt(t)            for t < 70
                                  6400 - 3200*exp(-t/62.8)      for t >= 70
Both are thermal subsidence of a cooling plate; they do not depend on gravity, so they carry over to our planet
(RULES.md E9 note). Sediment, hotspot swells and plateaus are not in the formulas, which is part of the misfit.
"""
import numpy as np
import logging
from pathlib import Path
logging.getLogger("gplately").setLevel(logging.CRITICAL)   # gplately is imported where it is used (Earth checks only)
from plate_checks import HERE

ETOPO = HERE.parent / "reference/earth/topography/ETOPO5.DAT"


def gdh1(t):
    t = np.asarray(t, float)
    return np.where(t < 20, 2600 + 365 * np.sqrt(np.maximum(t, 0)), 5651 - 2473 * np.exp(-0.0278 * t))


def parsons_sclater(t):
    t = np.asarray(t, float)
    return np.where(t < 70, 2500 + 350 * np.sqrt(np.maximum(t, 0)), 6400 - 3200 * np.exp(-t / 62.8))


def sediment_thickness(age, lat):
    """Straume et al. 2019 (GlobSed) world-ocean analytical model, eq. 2 with Table 4 coefficients:
    Z = sqrt(age) * (53.02 - 2.46*|lat| + 0.045*lat^2) metres. Fitted on open-ocean data (thickness < 1.3 km)."""
    phi = np.abs(lat)
    return np.sqrt(np.maximum(age, 0)) * (53.02 - 2.46 * phi + 0.045 * phi ** 2)


def sediment_shallowing(z, rho_m=3300.0, rho_s=2000.0, rho_w=1030.0):
    """Isostatically loaded sediment of thickness z raises the seafloor by z*(rho_m-rho_s)/(rho_m-rho_w)."""
    return z * (rho_m - rho_s) / (rho_m - rho_w)


def etopo_sample(lat, lon):
    raw = np.fromfile(ETOPO, dtype=">i2").reshape(2160, 4320)
    r = np.clip(np.rint((90 - lat) * 12).astype(int), 0, 2159)
    c = np.rint(((lon % 360) + 360) % 360 * 12).astype(int) % 4320
    return raw[r, c].astype(float)


def main():
    import gplately
    ref = gplately.Raster(str(HERE / "results/earth_seafloor_age_0Ma.nc"))
    age = np.asarray(ref.data, float)
    glon, glat = np.meshgrid(np.asarray(ref.lons), np.asarray(ref.lats))
    ok = np.isfinite(age)
    lat, lon, a = glat[ok], glon[ok], age[ok]
    depth = -etopo_sample(lat, lon)
    sea = depth > 1000                                   # deep ocean only (shelves are continental)
    w = np.cos(np.radians(lat))
    for name, f in [("GDH1 (Stein & Stein 1992)", gdh1), ("Parsons & Sclater 1977", parsons_sclater)]:
        for sed in (False, True):
            pred = f(a) - (sediment_shallowing(sediment_thickness(a, lat)) if sed else 0)
            e = (pred - depth)[sea]
            print(f"{name:28s} {'+ sediment' if sed else '          '}  median error {np.median(e):+5.0f} m, median |err| {np.median(np.abs(e)):4.0f} m")
            for lo, hi in [(0, 20), (20, 50), (50, 100), (100, 200)]:
                s_ = (a[sea] >= lo) & (a[sea] < hi)
                print(f"      age {lo:3d}-{hi:3d} Myr: measured median {np.median(depth[sea][s_]):5.0f} m, predicted median {np.median(pred[sea][s_]):5.0f} m")


if __name__ == "__main__":
    main()
