"""Seafloor age check (RULES.md A11) with GPlately's SeafloorGrid (tracer-based seafloor age algorithm).

Builds present-day seafloor ages from the plate model, then reports the age statistics.
Usage: .venv/bin/python planet-history-sim/tools/seafloor_age_check.py [max_time] [grid_spacing_deg]
"""
import sys, logging, glob
from pathlib import Path
import numpy as np
import pygplates

logging.getLogger("gplately").setLevel(logging.CRITICAL)
import gplately

from plate_checks import MODEL, TOPOLOGY_FILES, ROTATION, HERE


def main():
    max_time = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    spacing = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    rot = pygplates.RotationModel(str(ROTATION))
    topo = pygplates.FeatureCollection()
    for f in TOPOLOGY_FILES:
        topo.add(pygplates.FeatureCollection(str(MODEL / "Topologies" / f)))
    model = gplately.PlateReconstruction(rot, topo)
    continents = pygplates.FeatureCollection(str(MODEL / "Continents/continents.gpml"))
    gplot = gplately.PlotTopologies(model, continents=continents)
    out_dir = HERE / "seafloor_age_earth"
    grid = gplately.SeafloorGrid(model, gplot, max_time=max_time, min_time=0, ridge_time_step=1,
                                 save_directory=str(out_dir), file_collection="earth", grid_spacing=spacing,
                                 refinement_levels=5)
    grid.reconstruct_by_topological_model()
    grid.save_netcdf_files("SEAFLOOR_AGE")
    files = sorted(glob.glob(str(out_dir / "**/*SEAFLOOR_AGE*0.0Ma*.nc"), recursive=True)) or \
            sorted(glob.glob(str(out_dir / "**/*SEAFLOOR_AGE*.nc"), recursive=True))
    print("files", files[:3], "...", len(files))
    g = gplately.Raster(files[0])
    age = np.asarray(g.data, dtype=float)
    lat = np.linspace(-90, 90, age.shape[0])
    w = np.cos(np.radians(lat))[:, None] * np.ones_like(age)
    ok = np.isfinite(age)
    a, ww = age[ok], w[ok]
    order = np.argsort(a)
    cum = np.cumsum(ww[order]) / ww.sum()
    q = lambda p: float(a[order][np.searchsorted(cum, p)])
    print(f"ocean grid cells {ok.sum()}  mean age {float((a*ww).sum()/ww.sum()):.1f} Ma  median {q(0.5):.1f}  "
          f"p90 {q(0.9):.1f}  p99 {q(0.99):.1f}  max {a.max():.1f}")
    for lo in range(0, 200, 20):
        s = ww[(a >= lo) & (a < lo + 20)].sum() / ww.sum()
        print(f"  age {lo:3d}-{lo+20:3d} Ma: {s*100:5.1f}% of ocean floor")


if __name__ == "__main__":
    main()
