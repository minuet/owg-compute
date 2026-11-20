"""Shape of the real supercontinent (PIPELINE principle 0, tier 2): Pangea at 200 Ma in the Müller et al. 2022 model,
measured at the planet's cell size, to set and check the planet's starting supercontinent.

Continental crust = the model's continental polygons (Continents/continents.gpml) at 0 Ma, carried back to 200 Ma by
the published rotations (crust that did not exist then is left out). On a grid of the planet's cell area:
  - crust share of the surface, latitude of the crust's centre, share in the largest connected piece
  - distance of the crust to its edge (to the nearest non-crust cell): distribution, and the same shares for a round cap
    of equal area; "roughness" = crust within 200 km of the edge / that share for the round cap (size-free)
Seeds given on the command line: the same numbers for those planets' starting supercontinents (replay_record.npz).
Output: results/earth_pangea_stats.json.  Usage: .venv/bin/python planet-history-sim/tools/earth_pangea_stats.py [seed ...]"""
import json, math, sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from plate_checks import HERE, MODEL, ROTATION
from plate_engine import fibonacci_xyz

EDGES = [0, 200, 500, 1000, 1500, 99999]


def shape_stats(g, crust, radius):
    km = lambda c: 2 * np.arcsin(np.clip(c / 2, 0, 1)) * radius
    d = km(cKDTree(g[~crust]).query(g[crust])[0])
    f = crust.mean(); a = math.acos(1 - 2 * f)                          # angular radius of a cap of equal area
    cap = lambda dk: 1 - (1 - math.cos(max(a - dk / radius, 0))) / (1 - math.cos(a))
    nb = cKDTree(g).query(g, k=7)[1][:, 1:]
    idx = np.nonzero(crust)[0]; pos = -np.ones(len(g), int); pos[idx] = np.arange(len(idx))
    ii = np.repeat(np.arange(len(idx)), 6); jj = pos[nb[idx].ravel()]; ok = jj >= 0
    lab = connected_components(coo_matrix((np.ones(ok.sum()), (ii[ok], jj[ok])), shape=(len(idx),) * 2), directed=False)[1]
    c = g[crust].mean(axis=0)
    share = [float(((d >= x) & (d < y)).mean()) for x, y in zip(EDGES[:-1], EDGES[1:])]
    return {"crust_frac": float(f), "centre_lat_deg": float(math.degrees(math.asin(c[2] / np.linalg.norm(c)))),
            "largest_piece_share": float(np.bincount(lab).max() / len(idx)),
            "dist_to_edge_share": share, "cap_within_200km": cap(200.0),
            "roughness_200km": share[0] / cap(200.0), "median_dist_km": float(np.median(d)),
            "median_dist_cap_km": float(radius * a * (1 - 1 / math.sqrt(2)))}


def pangea(t=200.0):
    g, crust, crust0 = pangea_mask(t)
    return shape_stats(g, crust, 6371.0), shape_stats(g, crust0, 6371.0)


def pangea_mask(t=200.0):
    """grid, continental crust at t (rewound), continental crust at 0 Ma"""
    import pygplates
    from plate_checks import partition_ids
    n = int(round(60000 * (6371.0 / 6000.0) ** 2))
    g = fibonacci_xyz(n)
    rot = pygplates.RotationModel(str(ROTATION))
    conts = pygplates.FeatureCollection(str(MODEL / "Continents/continents.gpml"))
    part = pygplates.PlatePartitioner(conts, rot, 0)
    p = g[partition_ids(part, g) >= 0]
    pid = np.full(len(p), -1); born = np.zeros(len(p))
    for i, (x, y, z) in enumerate(p):
        f = part.partition_point(pygplates.PointOnSphere(x, y, z))
        pid[i] = f.get_feature().get_reconstruction_plate_id(); born[i] = f.get_feature().get_valid_time()[0]
    q = p.copy()
    for k in np.unique(pid):
        m = pid == k
        r = rot.get_rotation(t, int(k))
        q[m] = [(r * pygplates.PointOnSphere(*v)).to_xyz() for v in p[m]]
    q = q[born >= t]
    spacing = math.sqrt(4 * math.pi / n)
    crust = cKDTree(q).query(g)[0] < 0.75 * spacing
    return g, crust, partition_ids(part, g) >= 0


def main():
    pg, today = pangea()
    out = {"source": "Müller et al. 2022 continental polygons rewound to 200 Ma (and at 0 Ma), cells of the planet's size",
           "dist_edges_km": EDGES, "pangea_200Ma": pg, "earth_0Ma": today}
    show = lambda name, s: print(f"{name:12s} crust {s['crust_frac']*100:4.1f}%  centre lat {s['centre_lat_deg']:+5.1f}  "
                                 f"largest piece {s['largest_piece_share']*100:3.0f}%  dist to edge " +
                                 " ".join(f"{a}-{b}:{v*100:4.1f}%" for a, b, v in zip(EDGES[:-1], EDGES[1:], s["dist_to_edge_share"])) +
                                 f"  roughness {s['roughness_200km']:.2f}  median {s['median_dist_km']:.0f} km (cap {s['median_dist_cap_km']:.0f})")
    show("Pangea 200", pg); show("Earth 0 Ma", today)
    for s in sys.argv[1:]:
        r = np.load(HERE / f"results/planet_m1_seed{s}/replay_record.npz")
        st = shape_stats(r["grid"], r["cont0"], 6000.0); out[f"planet_{s}"] = st; show(f"planet {s}", st)
    (HERE / "results/earth_pangea_stats.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
