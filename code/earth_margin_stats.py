"""Earth reference for continental margins (PIPELINE principle 0, tier 2): how crustal thickness and the elevation residual
change with distance inland from the continent-ocean boundary, measured in CRUST1.0 (Laske et al. 2013, 1x1 deg).

Same definitions as the planet (planet_terrain.py): continental-type cell = crust (incl. sediments) >= 15 km, no ice;
distance inland = to the nearest cell with thinner crust (oceanic); active margin = within 400 km of a 0 Ma subduction
zone on its overriding side (Müller 2022 topologies via GPlately), otherwise passive. Residual = surface elevation
minus the thickness-median curve (results/crust_elevation_curve.json).
Output: results/earth_margin_stats.json.  Craton = inside the Müller 2022 craton polygons (Cratons/cratons.gpml).
Usage: .venv/bin/python planet-history-sim/tools/earth_margin_stats.py"""
import json
import numpy as np
from scipy.spatial import cKDTree
from plate_checks import HERE, MODEL, ROTATION, TOPOLOGY_FILES
from isostasy_check import load

BINS = [0, 50, 100, 150, 200, 250, 300, 400, 500, 700, 1000, 1500, 3000]
LEVELS = list(range(0, 101, 5))


def xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def main():
    import gplately
    b, rho, lat, lon = load()
    top = b[:, 1] * 1000.0                                   # solid surface: land elevation or -water depth
    h = b[:, :8] - b[:, 1:9]
    th = b[:, 1] - b[:, 8]                                   # crust incl. sediments, below water / ice
    ice = h[:, 1] > 0
    p = xyz(lat, lon); w = np.cos(np.radians(lat))
    cont = (th >= 15) & ~ice
    oce = th < 15
    km = lambda c: 2 * np.arcsin(np.clip(c / 2, 0, 1)) * 6371.0
    d_in = km(cKDTree(p[oce]).query(p)[0])
    curve = json.loads((HERE / "results/crust_elevation_curve.json").read_text())["thickness_km_vs_median_elevation_m"]
    resid = top - np.interp(th, [c[0] for c in curve], [c[1] for c in curve])

    model = gplately.PlateReconstruction(str(ROTATION), [str(MODEL / "Topologies" / f) for f in TOPOLOGY_FILES])
    sz = np.asarray(model.tessellate_subduction_zones(0, ignore_warnings=True))
    tp = xyz(sz[:, 1], sz[:, 0]); az = np.radians(sz[:, 7])
    la, lo = np.radians(sz[:, 1]), np.radians(sz[:, 0])
    nrm = np.cos(az)[:, None] * np.c_[-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)] + \
          np.sin(az)[:, None] * np.c_[-np.sin(lo), np.cos(lo), np.zeros_like(lo)]
    d, k = cKDTree(tp).query(p)
    active = (km(d) < 400) & (np.einsum("ij,ij->i", p - tp[k], nrm[k]) > 0)

    import pygplates
    from engine_earth_check import partition_ids
    crat = partition_ids(pygplates.PlatePartitioner(pygplates.FeatureCollection(str(MODEL / "Cratons/cratons.gpml")),
                                                    pygplates.RotationModel(str(ROTATION)), 0), p) >= 0
    print(f"craton share of continental-type cells (area-weighted): {np.average(crat[cont], weights=w[cont])*100:.1f}%")
    out = {"craton_share_of_continental": float(np.average(crat[cont], weights=w[cont])),
           "source": "CRUST1.0; continental-type = crust >= 15 km without ice; active = < 400 km behind a Müller 2022 "
                     "0 Ma trench", "bins_km": BINS, "levels_pct": LEVELS}
    for name, m in (("passive", cont & ~active), ("active", cont & active), ("all", cont),
                    ("passive_craton", cont & ~active & crat), ("passive_noncraton", cont & ~active & ~crat)):
        rows = []
        for a, z in zip(BINS[:-1], BINS[1:]):
            s = m & (d_in >= a) & (d_in < z)
            if s.sum() < 20:
                rows.append(None); continue
            rows.append({"n": int(s.sum()), "thickness_median_km": float(np.median(th[s])),
                         "thickness_p25_p75": [float(np.percentile(th[s], 25)), float(np.percentile(th[s], 75))],
                         "thickness_quantiles_km": [float(v) for v in np.percentile(th[s], LEVELS)],
                         "residual_median_m": float(np.median(resid[s])),
                         "residual_quantiles_m": [float(v) for v in np.percentile(resid[s], LEVELS)],
                         "elev_median_m": float(np.median(top[s]))})
        out[name] = rows
        print(name)
        for (a, z), r in zip(zip(BINS[:-1], BINS[1:]), rows):
            if r:
                print(f"  {a:4d}-{z:4d} km  n {r['n']:5d}  thickness {r['thickness_median_km']:5.1f} km "
                      f"({r['thickness_p25_p75'][0]:.0f}-{r['thickness_p25_p75'][1]:.0f})  residual median {r['residual_median_m']:+5.0f} m  "
                      f"elev median {r['elev_median_m']:+5.0f} m")
    (HERE / "results/earth_margin_stats.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
