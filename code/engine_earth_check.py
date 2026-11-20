"""Exam for plate_engine.py on Earth (RULES.md C2): drive it with the published Earth rotations from 200 Ma
(Pangea breakup) to the present, then compare with GPlately's official seafloor ages and the reconstructed continents.

Also counts ocean pockets enclosed by continental crust (RULES C2, 2026-10-31): the engine's result vs the real Earth
(Müller 2022 continents at 0 Ma) on the same grid, to tell numerical gaps from real basins. CLOSE_POCKETS=<cells>
runs the engine's close_pockets(<cells>) every 2 Myr, as the planet generator does.

Usage: .venv/bin/python planet-history-sim/tools/engine_earth_check.py [n_points] [start_ma]
"""
import os
import sys, json, time, logging
import numpy as np
import pygplates

logging.getLogger("gplately").setLevel(logging.CRITICAL)
import gplately

from plate_checks import MODEL, TOPOLOGY_FILES, ROTATION, HERE
from plate_engine import PlateEngine, xyz_to_latlon

R_EARTH = 6371.0


def partition_ids(partitioner, xyz):
    out = np.full(len(xyz), -1, np.int64)
    for i, (x, y, z) in enumerate(xyz):
        g = partitioner.partition_point(pygplates.PointOnSphere(x, y, z))
        if g is not None:
            out[i] = g.get_feature().get_reconstruction_plate_id()
    return out


def pockets(grid, cont):
    """Ocean pieces not connected to the world ocean (enclosed by continental crust): count, sizes in cells."""
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    nb = cKDTree(grid).query(grid, k=7)[1][:, 1:]
    idx = np.nonzero(~cont)[0]; pos = -np.ones(len(grid), int); pos[idx] = np.arange(len(idx))
    ii = np.repeat(np.arange(len(idx)), 6); jj = pos[nb[idx].ravel()]; ok = jj >= 0
    lab = connected_components(coo_matrix((np.ones(ok.sum()), (ii[ok], jj[ok])), shape=(len(idx),) * 2), directed=False)[1]
    sz = np.bincount(lab); sz = np.sort(sz[sz < sz.max()])[::-1]
    return {"count": int(len(sz)), "cells": int(sz.sum()), "count_le5": int((sz <= 5).sum()), "sizes": sz[:30].tolist()}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
    t0 = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    rot = pygplates.RotationModel(str(ROTATION))
    static = pygplates.FeatureCollection(str(MODEL / "Static_Polygons/static_polygons_Merdith_etal.gpml"))
    topo = [pygplates.FeatureCollection(str(MODEL / "Topologies" / f)) for f in TOPOLOGY_FILES]
    conts = pygplates.FeatureCollection(str(MODEL / "Continents/continents.gpml"))

    eng = PlateEngine(n, R_EARTH, lambda p, a, b: rot.get_rotation(b, p, a), gap_r=0.85)
    # initial state at t0: plate ids of preserved crust from static polygons, the rest from the plate topologies
    ids = partition_ids(pygplates.PlatePartitioner(static, rot, t0), eng.grid)
    missing = ids < 0
    ids[missing] = partition_ids(pygplates.PlatePartitioner(topo, rot, t0), eng.grid[missing])
    eng.plate = ids.astype(np.int32)
    eng.cont = partition_ids(pygplates.PlatePartitioner(conts, rot, t0), eng.grid) >= 0
    eng.block[eng.cont] = eng.plate[eng.cont]
    cont0 = int(eng.cont.sum())
    print(f"start {t0} Ma: {n} points, {len(np.unique(eng.plate))} plate ids, continental {cont0/n*100:.1f}% of surface", flush=True)

    mode = sys.argv[3] if len(sys.argv) > 3 else "topology"
    assign = None
    if mode == "topology":
        # same as GPlately's algorithm: at each step, material belongs to the plate whose topology contains it
        def assign(t, pos):
            part = pygplates.PlatePartitioner(topo, rot, t)
            ids = partition_ids(part, pos)
            ids[ids < 0] = eng.plate[ids < 0]
            return ids
    print("membership mode:", mode, flush=True)
    t_start = time.time()
    close_n = int(os.environ.get("CLOSE_POCKETS", "0"))
    for t in range(t0, 0, -1):
        eng.step(t, t - 1, assign)
        if close_n and (t % 2 == 0 or t == 1):
            eng.close_pockets(close_n)
        if (t - 1) % 25 == 0:
            L = eng.log[-1]
            print(f"  {t-1:4d} Ma  new ocean cells {L['new_ocean_cells']:5d}  consumed {L['consumed_cells']:5d}  "
                  f"particles {L['particles']}  ({time.time()-t_start:.0f}s)", flush=True)

    # compare with GPlately's official present-day seafloor ages
    st = eng.grid_state()
    lat, lon = xyz_to_latlon(eng.grid)
    ref = gplately.Raster(str(HERE / "results/earth_seafloor_age_0Ma.nc"))
    grid, glon, glat = np.asarray(ref.data, dtype=float), np.asarray(ref.lons), np.asarray(ref.lats)
    iy = np.clip(np.rint((lat - glat[0]) / (glat[1] - glat[0])).astype(int), 0, len(glat) - 1)
    ix = np.clip(np.rint((lon - glon[0]) / (glon[1] - glon[0])).astype(int), 0, len(glon) - 1)
    ref_age = grid[iy, ix]  # nearest grid cell (gplately's own sampler breaks with numpy 2.5)
    eng_age = st["birth"] - 0.0
    ocean_both = (~st["cont"]) & np.isfinite(ref_age)
    known = ocean_both & np.isfinite(eng_age)
    diff = np.abs(eng_age[known] - ref_age[known])
    atl = known & (lon > -70) & (lon < 20) & (lat > -60) & (lat < 65)
    eng_age_all = np.where(np.isfinite(eng_age), eng_age, t0)  # crust older than the run counts as t0
    cont_now = partition_ids(pygplates.PlatePartitioner(conts, rot, 0), eng.grid) >= 0
    res = {
        "points": n, "start_ma": t0,
        "continental_start_frac": cont0 / n, "continental_end_frac": float(st["cont"].mean()),
        "continental_mask_agreement": float((st["cont"] == cont_now).mean()),
        "continental_particles_start_end": [cont0, int(eng.cont.sum())],
        "ocean_mean_age_engine": float(eng_age_all[ocean_both].mean()),
        "ocean_mean_age_reference": float(ref_age[ocean_both].mean()),
        "age_median_abs_diff_my": float(np.median(diff)), "age_within_10my_frac": float((diff <= 10).mean()),
        "age_corr": float(np.corrcoef(eng_age[known], ref_age[known])[0, 1]),
        "all_ocean_signed_median_diff_my": float(np.median(eng_age[known] - ref_age[known])),
        "atlantic_median_abs_diff_my": float(np.median(np.abs(eng_age[atl] - ref_age[atl]))),
        "atlantic_within_10my_frac": float((np.abs(eng_age[atl] - ref_age[atl]) <= 10).mean()),
    }
    res["pockets_engine"] = pockets(eng.grid, st["cont"])
    res["pockets_earth"] = pockets(eng.grid, cont_now)
    res["close_pockets_cells"] = close_n
    for k, v in res.items():
        print(f"{k:32s} {v:.3f}" if isinstance(v, float) else f"{k:32s} {v}")
    (HERE / f"results/engine_earth_check{'_close' + str(close_n) if close_n else ''}.json").write_text(json.dumps({"summary": res, "log": eng.log}))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))
    for a_, vals, title in [(ax[0], np.where(st["cont"], np.nan, eng_age_all), "engine ocean age (Ma)"),
                            (ax[1], np.where(np.isfinite(ref_age), ref_age, np.nan), "GPlately reference age (Ma)")]:
        sc = a_.scatter(lon, lat, c=vals, s=2, cmap="viridis_r", vmin=0, vmax=200)
        a_.set_title(title); fig.colorbar(sc, ax=a_)
    sc = ax[2].scatter(lon, lat, c=st["plate"] % 37, s=2, cmap="tab20"); ax[2].set_title("engine plate ids at 0 Ma")
    for a_ in ax: a_.set_xlim(-180, 180); a_.set_ylim(-90, 90)
    fig.tight_layout(); fig.savefig(HERE / f"results/engine_earth_check{'_close' + str(close_n) if close_n else ''}.png", dpi=70)


if __name__ == "__main__":
    main()
