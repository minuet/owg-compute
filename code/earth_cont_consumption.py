"""Earth reference (2026-11-05): how much continental crust do continent-continent overlaps consume when the particle
engine is driven by the real Earth plate model? Same engine and same collision rule as the planet (PlateEngine with
cont_collide: a continental particle overridden by another plate's continental particle is consumed, its crust handed
on), membership from the Müller et al. 2022 topologies every step (as engine_earth_check.py), 450 Ma -> 0.
The planet's Earth test (planet_m1 INIT=earth:450) consumed 6-13% of the starting continental particles per 50 Myr;
this tells whether Earth's own motions do the same (then the overlap rule / the model polygons are the cause) or far
less (then the simulated motions converge the continents too much). Continental polygons of the model grow with time
(terranes added: 26% -> 41% of the surface), so consumption is reported against the starting count.
Mode "static" (argument 3): GPlates' standard reconstruction instead - every particle keeps the plate id of the static
polygon it starts in (no re-assignment by topology, which hides overlaps by handing overridden particles to the
overriding plate), so continental particles overlap exactly where the reconstructed continents overlap.
Usage: earth_cont_consumption.py [n_points, default 20000] [start Ma, default 450] [topology | static]
"""
import sys, json, time
import numpy as np
import pygplates
from plate_checks import MODEL, TOPOLOGY_FILES, ROTATION, HERE
from plate_engine import PlateEngine
from plate_checks import partition_ids

R_EARTH = 6371.0


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    t0 = int(sys.argv[2]) if len(sys.argv) > 2 else 450
    rot = pygplates.RotationModel(str(ROTATION))
    topo = [pygplates.FeatureCollection(str(MODEL / "Topologies" / f)) for f in TOPOLOGY_FILES]
    conts = pygplates.FeatureCollection(str(MODEL / "Continents/continents.gpml"))
    eng = PlateEngine(n, R_EARTH, lambda p, a, b: rot.get_rotation(b, p, a), gap_r=0.85)
    mode = sys.argv[3] if len(sys.argv) > 3 else "topology"
    ids = partition_ids(pygplates.PlatePartitioner(topo, rot, float(t0)), eng.grid)
    ids[ids < 0] = 0
    if mode == "static":
        static = pygplates.FeatureCollection(str(MODEL / "Static_Polygons/static_polygons_Merdith_etal.gpml"))
        sid = partition_ids(pygplates.PlatePartitioner(static, rot, float(t0)), eng.grid)
        ids[sid >= 0] = sid[sid >= 0]
    eng.plate = ids.astype(np.int32)
    eng.cont = partition_ids(pygplates.PlatePartitioner(conts, rot, float(t0)), eng.grid) >= 0
    eng.cont_collide = True
    c0 = int(eng.cont.sum())

    def assign(t, pos):
        out = partition_ids(pygplates.PlatePartitioner(topo, rot, float(t)), pos)
        out[out < 0] = eng.plate[out < 0]
        return out
    used, rows, t_start = 0, {}, time.time()
    for t in range(t0, 0, -1):
        eng.step(t, t - 1, assign if mode == "topology" else None)
        lc = eng.last_cont_consumed
        used += 0 if lc is None else len(lc["receiver"]) + lc["lost"]
        if (t - 1) % 50 == 0:
            rows[t - 1] = {"consumed_share_of_start": round(used / c0, 3), "continental_share_now": round(float(eng.cont.mean()), 3)}
            print(t - 1, rows[t - 1], f"({time.time() - t_start:.0f}s)", flush=True)
    out = {"n": n, "start_Ma": t0, "mode": mode, "continental_particles_start": c0,
           "particle_area_km2": eng.cell_area_km2, "by_Ma": rows}
    (HERE / f"results/earth_cont_consumption_{mode}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
