"""RULES.md A21 on Earth: can crustal thickness + density (Airy isostasy) explain continental elevation?

Data: CRUST1.0 (Laske et al. 2013), 1x1 degree, layer boundaries (km, up positive) and densities (g/cm3).
Airy isostasy: a column of thickness h_i and density rho_i floats on mantle rho_m; its surface elevation is
  e = sum_i h_i * (rho_m - rho_i) / rho_m - C      (C: a constant fixed by the reference column)
Elevation from isostasy does not depend on gravity (only density ratios), so it carries over to our planet.
"""
from pathlib import Path
import numpy as np

CRUST = Path(__file__).resolve().parent.parent / "reference/earth/crust"


def load():
    b = np.loadtxt(CRUST / "crust1.bnds")        # 64800 x 9, km
    rho = np.loadtxt(CRUST / "crust1.rho")       # 64800 x 9, g/cm3 (9th = mantle below Moho)
    lat = np.repeat(np.arange(89.5, -90, -1.0), 360)
    lon = np.tile(np.arange(-179.5, 180, 1.0), 180)
    return b, rho, lat, lon


def main():
    b, rho, lat, lon = load()
    top = b[:, 0]                                 # surface (land elevation or sea surface 0)
    h = b[:, :8] - b[:, 1:9]                       # layer thicknesses: water, ice, 3 sediments, 3 crystalline
    land = (h[:, 0] == 0) & (top > 0) & (h[:, 1] == 0)   # dry land without ice sheet
    w = np.cos(np.radians(lat))
    crust_th = (b[:, 1] - b[:, 8])                 # crust incl. sediments (below water/ice)
    rho_m = rho[:, 8]

    # 1) full Airy with each cell's own layer densities and mantle density
    solid = h[:, 2:8]
    mass_deficit = (solid * (rho_m[:, None] - rho[:, 2:8])).sum(axis=1) / rho_m
    C = np.average(mass_deficit[land] - top[land], weights=w[land])
    pred_full = mass_deficit - C
    # 2) simple Airy: one crustal density for all, thickness only (what we can generate for our planet)
    rho_c, rho_m0 = 2.85, 3.30
    simple = crust_th * (rho_m0 - rho_c) / rho_m0
    C2 = np.average(simple[land] - top[land], weights=w[land])
    pred_simple = simple - C2

    for name, pred in [("Airy, CRUST1.0 layer densities", pred_full), ("Airy, single crust density 2.85", pred_simple)]:
        e = (pred - top)[land]
        r = np.corrcoef(pred[land], top[land])[0, 1]
        print(f"{name:34s} land cells {land.sum()}  corr {r:.2f}  rms {np.sqrt(np.average(e**2, weights=w[land]))*1000:4.0f} m  "
              f"median |err| {np.median(np.abs(e))*1000:4.0f} m")
    print(f"constant C (full) = {C:.2f} km, (simple) = {C2:.2f} km")
    # thickness classes on land: how elevation follows thickness
    for lo, hi in [(20, 30), (30, 35), (35, 40), (40, 45), (45, 50), (50, 60), (60, 80)]:
        s = land & (crust_th >= lo) & (crust_th < hi)
        if s.sum():
            print(f"  crust {lo}-{hi} km: {s.sum():5d} cells, measured elevation median {np.median(top[s])*1000:5.0f} m, "
                  f"simple Airy median {np.median(pred_simple[s])*1000:5.0f} m")


if __name__ == "__main__":
    main()
