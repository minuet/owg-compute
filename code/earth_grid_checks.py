"""The planet's own check algorithm (planet_checks.check_state) applied to the Earth plate model, 200 Ma - now, so the
planet is compared with Earth measured by the SAME algorithm (PIPELINE principle 0/3; 2026-10-31: the A8 trench band
had come from GPlately's trench tool on the model's labelled subduction zones, while the planet check counts every
converging grid boundary and sets the polarity by age - on today's Earth the two give 62% vs 77% retreating trenches).

Steps: GPlately's SeafloorGrid (the official tracer-based seafloor age algorithm, as seafloor_age_check.py) builds the
Earth model's seafloor age grids from 250 Ma; then every 10 Myr from 200 Ma to now, on a grid of the planet's cell size:
plate ids (resolved topologies), continental crust (continent polygons reconstructed), seafloor age (nearest age-grid
value), motions (stage rotations t+1 -> t, mantle frame), and check_state with Earth's radius.
Output: results/earth_grid_checks.json (per step + p10-p90 bands of the A8 and other quantities).
Usage: .venv/bin/python planet-history-sim/tools/earth_grid_checks.py [grid_spacing_deg] [max_time_Ma]   (runs on Kaggle: heavy)
"""
import sys, glob, json, logging, math, re, shutil
from pathlib import Path
import numpy as np
import pygplates

logging.getLogger("gplately").setLevel(logging.CRITICAL)
import gplately

from plate_checks import MODEL, TOPOLOGY_FILES, ROTATION, HERE
from plate_engine import fibonacci_xyz, xyz_to_latlon
from planet_checks import check_state

R = 6371.0
N = int(round(60000 * (R / 6000.0) ** 2))
KEYS = ["trench_abs_mean_cm_yr", "trench_retreat_frac", "trench_fast_advance_frac", "trench_vorth_mean_cm_yr",
        "crust_made_over_consumed", "ocean_age_mean", "ocean_age_p99", "net_rotation_deg_myr"]


def partition_ids(partitioner, xyz):
    out = np.full(len(xyz), -1, np.int64)
    for i, (x, y, z) in enumerate(xyz):
        f = partitioner.partition_point(pygplates.PointOnSphere(x, y, z))
        if f is not None:
            out[i] = f.get_feature().get_reconstruction_plate_id()
    return out


def main():
    spacing = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    max_time = int(sys.argv[2]) if len(sys.argv) > 2 else 250   # a short span = quick test of the whole chain
    rot = pygplates.RotationModel(str(ROTATION))
    topo = pygplates.FeatureCollection()
    for f in TOPOLOGY_FILES:
        topo.add(pygplates.FeatureCollection(str(MODEL / "Topologies" / f)))
    model = gplately.PlateReconstruction(rot, topo)
    conts = pygplates.FeatureCollection(str(MODEL / "Continents/continents.gpml"))
    out_dir = HERE / "seafloor_age_earth_series"
    grid = gplately.SeafloorGrid(model, gplately.PlotTopologies(model, continents=conts), max_time=max_time, min_time=0,
                                 ridge_time_step=1, save_directory=str(out_dir), file_collection="earth",
                                 grid_spacing=spacing, refinement_levels=5)
    grid.reconstruct_by_topological_model()
    grid.save_netcdf_files("SEAFLOOR_AGE")
    g = fibonacci_xyz(N)
    la, lo = xyz_to_latlon(g)
    topo_list = [pygplates.FeatureCollection(str(MODEL / "Topologies" / f)) for f in TOPOLOGY_FILES]
    rows, ages = [], {}
    # file name exactly as gplately 2.0 oceans._save_netcdf_file builds it: <dir>/<name>/<collection>_<name>_grid_<t:0.2f>_Ma.nc
    nc = {float(t): str(out_dir / "SEAFLOOR_AGE" / f"earth_SEAFLOOR_AGE_grid_{float(t):0.2f}_Ma.nc") for t in range(0, max_time + 1)}
    nc = {t: f for t, f in nc.items() if Path(f).exists()}
    if not nc:
        print("no age grid found; files present:", sorted(str(p) for p in out_dir.rglob("*.nc"))[:10], flush=True)
    keep = HERE / "results/earth_age_grids"; keep.mkdir(parents=True, exist_ok=True)
    times = [t for t in range(200, -1, -10) if t < max_time]
    for t in times:
        if float(t) in nc:
            shutil.copy(nc[float(t)], keep)   # the 10-Myr grids are kept first, so a later failure need not redo the slow step
    print(len(nc), "age grids, e.g.", sorted(nc.items())[:2], flush=True)
    for t in times:
        if float(t) not in nc:
            print(t, "no age grid", flush=True)
            continue
        ras = gplately.Raster(nc[float(t)])
        z = np.ma.filled(np.ma.asarray(ras.data, float), np.nan)
        lats = np.asarray(ras.lats, float); lons = np.asarray(ras.lons, float)
        age = z[np.clip(np.round((la - lats[0]) / (lats[1] - lats[0])).astype(int), 0, len(lats) - 1),
                np.clip(np.round((lo - lons[0]) / (lons[1] - lons[0])).astype(int), 0, len(lons) - 1)]
        plate = partition_ids(pygplates.PlatePartitioner(topo_list, rot, float(t)), g)
        cont = partition_ids(pygplates.PlatePartitioner(conts, rot, float(t)), g) >= 0
        age = np.where(cont, np.nan, age)
        ages[f"t{t}"] = age.astype(np.float16)
        w = {}
        for p in np.unique(plate):
            try:
                pole, ang = rot.get_rotation(float(t), int(p), float(t) + 1.0).get_euler_pole_and_angle()
                w[int(p)] = np.array(pole.to_xyz()) * ang
            except Exception:
                w[int(p)] = np.zeros(3)
        cs = check_state(g, {"plate": plate, "cont": cont, "age": age}, w, R)
        row = {"ma": t, **{k: cs.get(k) for k in KEYS}, "ocean_cells_without_age": float(np.isnan(age[~cont]).mean())}
        rows.append(row)
        print(t, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    vals = lambda k: [r[k] for r in rows if isinstance(r[k], (int, float)) and np.isfinite(r[k])]
    band = {k: [round(float(np.percentile(vals(k), 10)), 3), round(float(np.percentile(vals(k), 90)), 3)] for k in KEYS if vals(k)}
    (HERE / "results/earth_grid_checks.json").write_text(json.dumps(
        {"source": "Müller et al. 2022 + GPlately SeafloorGrid ages, planet_checks.check_state on the planet-size grid, 200-0 Ma every 10 Myr",
         "band_p10_p90": band, "steps": rows}, indent=1))
    print("bands:", band, flush=True)
    np.savez_compressed(HERE / "results/earth_age_series.npz", **ages)   # sampled ages, kept so the slow step need not be redone


if __name__ == "__main__":
    main()
