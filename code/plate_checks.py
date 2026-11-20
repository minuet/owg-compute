"""Check a plate model against RULES.md (A2, A3, A4, A6, A7, C4).

Run on the published Earth model first: it should pass, which shows the checks are right,
and its measured values fill in the "待定" thresholds. The same checks then run on our planet.

Usage: .venv/bin/python planet-history-sim/tools/plate_checks.py [step_myr] [max_age]
"""
import json, math, sys, time
from pathlib import Path
import pygplates

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent / "reference/earth/plate-model"
TOPOLOGY_FILES = ["250-0_plate_boundaries.gpml", "410-250_plate_boundaries.gpml", "1000-410-Topologies.gpml",
                  "1000-410-Convergence.gpml", "1000-410-Divergence.gpml", "1000-410-Transforms.gpml",
                  "TopologyBuildingBlocks.gpml"]
ROTATION = MODEL / "Optimisation/1000_0_rotfile_MantleOptimised.rot"  # mantle frame: for speeds and net rotation
CONTINENTS = MODEL / "Continents/continents.gpml"
R_KM = 6371.0  # Earth; for our planet call set_radius(6000.0). pyGPlates itself defaults to Earth radius, so always pass it explicitly.


def set_radius(r_km):
    global R_KM
    R_KM = float(r_km)
KMS_PER_MY = pygplates.VelocityUnits.kms_per_my  # 1 km/Myr = 0.1 cm/yr


def fibonacci_points(n):
    pts, ga = [], math.pi * (3 - math.sqrt(5))
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = math.sqrt(1 - z * z)
        pts.append(pygplates.PointOnSphere(r * math.cos(i * ga), r * math.sin(i * ga), z))
    return pts


def plate_velocity(rot, pid, t, point):
    """Velocity (km/Myr, Vector3D) of a point on plate pid at time t, from the 1 Myr stage rotation."""
    stage = rot.get_rotation(t, pid, t + 1)
    return pygplates.calculate_velocities([point], stage, 1.0, KMS_PER_MY, earth_radius_in_kms=R_KM)[0]


def check_time(t, rot, topo_features, continents, samples):
    resolved, shared = [], []
    pygplates.resolve_topologies(topo_features, rot, resolved, t, shared)
    polys = [(r.get_resolved_geometry(), r.get_feature().get_reconstruction_plate_id()) for r in resolved]

    # A2: every sample point must lie in exactly one plate
    gaps = overlaps = 0
    owner = []
    for p in samples:
        hits = [pid for poly, pid in polys if poly.is_point_in_polygon(p)]
        if not hits:
            gaps += 1
        elif len(hits) > 1:
            overlaps += 1
        owner.append(hits[0] if hits else None)

    # continental points at this time (reconstructed continent outlines)
    rec = []
    pygplates.reconstruct(continents, rot, rec, t)
    cpolys = [r.get_reconstructed_geometry() for r in rec if isinstance(r.get_reconstructed_geometry(), pygplates.PolygonOnSphere)]
    is_cont = [any(cp.is_point_in_polygon(p) for cp in cpolys) for p in samples]

    # A6: speeds per plate; A7: net rotation from area-uniform samples
    per_plate = {}
    nx = ny = nz = 0.0
    n = 0
    for p, pid, c in zip(samples, owner, is_cont):
        if pid is None:
            continue
        v = plate_velocity(rot, pid, t, p)
        speed = v.get_magnitude()
        d = per_plate.setdefault(pid, [0.0, 0, 0])
        d[0] += speed * speed
        d[1] += 1
        d[2] += c
        x, y, z = p.to_xyz()
        vx, vy, vz = v.get_x(), v.get_y(), v.get_z()
        nx += y * vz - z * vy
        ny += z * vx - x * vz
        nz += x * vy - y * vx
        n += 1
    # omega = 3/(2 R^2) * mean(r x v), r = R * unit vector  ->  3/(2R) * mean(u x v)  [rad/Myr]
    w = 1.5 / R_KM * math.sqrt(nx * nx + ny * ny + nz * nz) / max(n, 1)
    net_rot_deg = math.degrees(w)
    plates = []
    for pid, (s2, cnt, cc) in per_plate.items():
        plates.append({"pid": pid, "rms_cm_yr": math.sqrt(s2 / cnt) / 10.0, "area_frac": cnt / n, "cont_frac": cc / cnt})
    global_rms = math.sqrt(sum(d[0] for d in per_plate.values()) / max(n, 1)) / 10.0

    # A3/A4: boundary lengths, normal rates, crust production vs consumption
    lengths = {}
    produced = consumed = 0.0  # km^2/Myr
    bad_transform_km = 0.0
    for sec in shared:
        ftype = sec.get_feature().get_feature_type().get_name()
        for sub in sec.get_shared_sub_segments():
            sharing = sub.get_sharing_resolved_topologies()
            if len(sharing) != 2:
                continue
            pa, pb = (s.get_feature().get_reconstruction_plate_id() for s in sharing)
            for arc in sub.get_resolved_geometry().get_segments():
                if arc.is_zero_length():
                    continue
                L = arc.get_arc_length() * R_KM
                lengths[ftype] = lengths.get(ftype, 0.0) + L
                a, b = arc.get_start_point().to_xyz(), arc.get_end_point().to_xyz()
                m = [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
                mn = math.sqrt(sum(c * c for c in m))
                mid = pygplates.PointOnSphere(m[0] / mn, m[1] / mn, m[2] / mn)
                va, vb = plate_velocity(rot, pa, t, mid), plate_velocity(rot, pb, t, mid)
                gn = arc.get_great_circle_normal()  # perpendicular to the boundary, tangent to the sphere
                normal_rate = abs((va - vb).get_x() * gn.get_x() + (va - vb).get_y() * gn.get_y() + (va - vb).get_z() * gn.get_z())
                if ftype == "MidOceanRidge":
                    produced += normal_rate * L
                elif ftype == "SubductionZone":
                    consumed += normal_rate * L
                elif ftype == "Transform" and normal_rate > 20.0 and L > 0:  # > 2 cm/yr opening or closing
                    bad_transform_km += L
    return {
        "time": t, "n_plates": len(polys), "gap_frac": gaps / len(samples), "overlap_frac": overlaps / len(samples),
        "global_rms_cm_yr": global_rms, "net_rotation_deg_myr": net_rot_deg, "plates": plates,
        "ridge_km": lengths.get("MidOceanRidge", 0.0), "subduction_km": lengths.get("SubductionZone", 0.0),
        "transform_km": lengths.get("Transform", 0.0), "bad_transform_km": bad_transform_km,
        "crust_produced_km2_myr": produced, "crust_consumed_km2_myr": consumed,
    }


def main():
    step = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    max_age = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    rot = pygplates.RotationModel(str(ROTATION))
    topo = [pygplates.FeatureCollection(str(MODEL / "Topologies" / f)) for f in TOPOLOGY_FILES]
    continents = pygplates.FeatureCollection(str(CONTINENTS))
    samples = fibonacci_points(1500)
    out = []
    for t in range(0, max_age + 1, step):
        t0 = time.time()
        r = check_time(t, rot, topo, continents, samples)
        out.append(r)
        print(f"{t:5d} Ma  plates {r['n_plates']:3d}  gap {r['gap_frac']*100:.2f}%  overlap {r['overlap_frac']*100:.2f}%  "
              f"rms {r['global_rms_cm_yr']:.1f} cm/yr  netrot {r['net_rotation_deg_myr']:.3f} deg/Myr  "
              f"ridge {r['ridge_km']:.0f} km  subd {r['subduction_km']:.0f} km  "
              f"prod/cons {r['crust_produced_km2_myr']/max(r['crust_consumed_km2_myr'],1):.2f}  "
              f"bad-transform {r['bad_transform_km']:.0f} km  ({time.time()-t0:.1f}s)", flush=True)
    dest = HERE / "results/earth_check_results.json"
    dest.write_text(json.dumps(out))
    print("saved", dest)


if __name__ == "__main__":
    main()


def partition_ids(partitioner, xyz):
    """Plate id of each unit vector under a pygplates.PlatePartitioner (-1 outside every polygon).
    Moved here from force_inversion_earth.py (archived with the force model, 2026-11-18)."""
    import numpy as np
    out = np.full(len(xyz), -1, np.int64)
    for i, (x, y, z) in enumerate(xyz):
        f = partitioner.partition_point(pygplates.PointOnSphere(x, y, z))
        if f is not None:
            out[i] = f.get_feature().get_reconstruction_plate_id()
    return out
