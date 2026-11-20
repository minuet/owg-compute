"""Earth reference for land on oceanic crust (PIPELINE principle 0, tier 2: numbers measured on Earth data).

Measured at the planet's cell size (planet grid 60000 cells on R = 6000 km, ~7540 km2 a cell): ETOPO5 is averaged over
equal-area Earth cells of the same area. Oceanic crust = outside the continental polygons of the Müller et al. 2022 model.
  1. island arcs: oceanic cells on the overriding side of a 0 Ma subduction zone (Müller 2022 topologies, GPlately
     trench normals), 100-240 km behind the trench (the planet's arc belt, planet_terrain.py)
  2. hotspots: oceanic cells within 500 km of an active oceanic hotspot (hotspot list: Courtillot et al. 2003 and
     Sleep 1990 catalogues; positions to ~1 deg, enough for a 500 km radius)
For each: cell count, share of cells whose mean is above sea level, land area (ETOPO5 pixels above 0) and the
quantiles of the cell-mean elevation (used by planet_terrain.py to give the planet the same distribution).
Output: results/earth_island_stats.json.  Usage: .venv/bin/python planet-history-sim/tools/earth_island_stats.py"""
import json, math
import numpy as np
from scipy.spatial import cKDTree
from plate_checks import HERE, MODEL, ROTATION, TOPOLOGY_FILES

R_E, R_P, N_P = 6371.0, 6000.0, 60000
N_E = int(round(N_P * (R_E / R_P) ** 2))
LEVELS = list(range(0, 91, 5)) + list(range(91, 100)) + [99.5, 99.8, 100]   # fine steps in the upper tail, where the land is
# oceanic hotspots (lat, lon): Courtillot et al. 2003 Table 1 and Sleep 1990, those on oceanic lithosphere
HOTSPOTS = {
    "Hawaii": (19.4, -155.3), "Iceland": (64.4, -17.3), "Galapagos": (-0.4, -91.5), "Reunion": (-21.2, 55.7),
    "Louisville": (-50.9, -138.1), "Easter": (-27.1, -109.3), "Tristan": (-37.1, -12.3), "Kerguelen": (-49.6, 69.0),
    "Marquesas": (-10.5, -139.0), "Samoa": (-14.5, -168.2), "Society": (-18.0, -148.0), "Macdonald": (-29.0, -140.3),
    "Pitcairn": (-25.4, -129.3), "Juan Fernandez": (-33.9, -81.8), "San Felix": (-26.3, -80.1), "Azores": (38.5, -28.4),
    "Canary": (28.2, -18.0), "Cape Verde": (14.9, -24.4), "Madeira": (32.6, -17.3), "Bermuda": (32.3, -64.8),
    "Ascension": (-7.9, -14.4), "St Helena": (-16.0, -5.7), "Fernando": (-3.8, -32.4), "Trindade": (-20.5, -28.8),
    "Bouvet": (-54.4, 3.4), "Meteor": (-52.0, 1.0), "Discovery": (-43.0, -2.7), "Vema": (-32.1, 4.3),
    "Great Meteor": (30.0, -28.5), "Comores": (-11.8, 43.3), "Crozet": (-46.1, 50.2), "Marion": (-46.9, 37.8),
    "Amsterdam-St Paul": (-37.8, 77.5), "Balleny": (-67.6, 164.8), "Lord Howe": (-33.0, 159.0), "Tasmantid": (-40.4, 155.5),
    "Caroline": (5.0, 164.0), "Bowie": (53.0, -135.0), "Cobb": (46.0, -130.0), "Jan Mayen": (71.0, -8.5),
    "Socorro": (18.8, -111.0), "Guadalupe": (29.0, -118.3),
}


def xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def main():
    import pygplates, gplately
    from plate_engine import fibonacci_xyz
    from engine_earth_check import partition_ids
    g = fibonacci_xyz(N_E)
    raw = np.fromfile(HERE / "../reference/earth/topography/ETOPO5.DAT", dtype=">i2").reshape(2160, 4320).astype(float)
    plat = 90 - (np.arange(2160) + 0.5) / 12; plon = (np.arange(4320) + 0.5) / 12
    LA, LO = np.meshgrid(plat, plon, indexing="ij")
    pix = xyz(LA.ravel(), LO.ravel()); wpx = np.cos(np.radians(LA.ravel()))
    cell = cKDTree(g).query(pix)[1]
    wsum = np.bincount(cell, wpx, N_E)
    elev = np.bincount(cell, wpx * raw.ravel(), N_E) / wsum                       # cell mean (m)
    land_area = np.bincount(cell, wpx * (raw.ravel() > 0), N_E) / wsum             # share of the cell above sea level
    rot = pygplates.RotationModel(str(ROTATION))
    conts = pygplates.FeatureCollection(str(MODEL / "Continents/continents.gpml"))
    oceanic = partition_ids(pygplates.PlatePartitioner(conts, rot, 0), g) < 0
    cell_km2 = 4 * math.pi * R_E ** 2 / N_E
    km = lambda chord: 2 * np.arcsin(np.clip(chord / 2, 0, 1)) * R_E

    topo = [str(MODEL / "Topologies" / f) for f in TOPOLOGY_FILES]
    model = gplately.PlateReconstruction(str(ROTATION), topo)
    sz = np.asarray(model.tessellate_subduction_zones(0, ignore_warnings=True))
    tp = xyz(sz[:, 1], sz[:, 0])
    az = np.radians(sz[:, 7])                                                      # trench normal towards the overriding plate
    la, lo = np.radians(sz[:, 1]), np.radians(sz[:, 0])
    north = np.c_[-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)]
    east = np.c_[-np.sin(lo), np.cos(lo), np.zeros_like(lo)]
    nrm = np.cos(az)[:, None] * north + np.sin(az)[:, None] * east
    d, k = cKDTree(tp).query(g)
    d_tr = km(d)
    over = np.einsum("ij,ij->i", g - tp[k], nrm[k]) > 0
    arc = oceanic & over & (d_tr > 100) & (d_tr < 240)

    hs = xyz(*np.array(list(HOTSPOTS.values())).T)
    near_hs = oceanic & (km(cKDTree(hs).query(g)[0]) < 500)

    def stats(m):
        return {"cells": int(m.sum()), "cell_mean_above_sea_frac": float((elev[m] > 0).mean()),
                "land_area_frac": float(land_area[m].mean()), "land_area_km2": float(land_area[m].sum() * cell_km2),
                "quantiles_m": [float(v) for v in np.percentile(elev[m], LEVELS)]}
    all_land = float((land_area * cell_km2).sum())
    out = {"source": "ETOPO5 averaged over equal-area cells of the planet's cell size; oceanic = outside Müller 2022 "
                     "continents; trenches Müller 2022 0 Ma via GPlately; hotspots Courtillot 2003 / Sleep 1990",
           "cell_km2": cell_km2, "levels_pct": LEVELS, "earth_land_km2": all_land,
           "island_arc_belt": stats(arc), "hotspot_500km": stats(near_hs & ~arc),
           "oceanic_all": stats(oceanic), "oceanic_other": stats(oceanic & ~arc & ~near_hs)}
    for k_ in ("island_arc_belt", "hotspot_500km", "oceanic_other"):
        out[k_]["share_of_earth_land"] = out[k_]["land_area_km2"] / all_land
    (HERE / "results/earth_island_stats.json").write_text(json.dumps(out, indent=1))
    for k_ in ("island_arc_belt", "hotspot_500km", "oceanic_other", "oceanic_all"):
        s = out[k_]
        print(f"{k_:16s} cells {s['cells']:6d}  cell mean > 0: {s['cell_mean_above_sea_frac']*100:5.2f}%  land area "
              f"{s['land_area_km2']/1e3:7.0f} thousand km2 ({s['land_area_km2']/all_land*100:.2f}% of Earth land)  "
              f"p50 {s['quantiles_m'][10]:.0f} m  p95 {s['quantiles_m'][23]:.0f}  p99 {s['quantiles_m'][27]:.0f}  max {s['quantiles_m'][-1]:.0f}")


if __name__ == "__main__":
    main()
