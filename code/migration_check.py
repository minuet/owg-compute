"""Migration check (RULES.md B7) on the fine replay grid.

From the story start, which share of all land can early humans reach on foot or by raft across straits up to ~150 km
(CLAUDE.md: rafts ~100-200 km), and which share by coastal boats across up to 1200 km of sea?
Land pieces are joined when their nearest points are within the crossing distance.
Checked at present sea level and at the glacial low stand (-120 m, RULES.md A22: migration happens through glacial
cycles); B7 passes if a start passes at either sea level. Both results are reported.

Usage: .venv/bin/python planet-history-sim/tools/migration_check.py <seed> [<start_lat> <start_lon>]
Without a start, every story start of the seed in results/screen_climate.json is tried and the best one is kept.
"""
import json, sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from plate_checks import HERE

R = 6000.0
SEA_LEVELS = (0.0, -120.0)   # present, last-glacial-maximum low stand (A22)
MODES = [("walk_or_raft_150km", 150.0, 0.20), ("coastal_boat_1200km", 1200.0, 0.80)]


def chord(km):
    return 2 * np.sin(km / R / 2)


def ll_to_xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def components(xyz, land, crossing_km):
    """Land points joined into pieces reachable from each other; computed once, then queried for every start."""
    pts = xyz[land]
    tree = cKDTree(pts)
    pairs = tree.query_pairs(chord(crossing_km), output_type="ndarray")
    n = len(pts)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    return tree, lab


def reach(xyz, land, start_xyz, crossing_km):
    tree, lab = components(xyz, land, crossing_km)
    s = lab[tree.query(start_xyz)[1]]
    return float((lab == s).mean()), int(lab.max() + 1)


def story_starts(seed):
    for r in json.loads((HERE / "results/screen_climate.json").read_text()):
        if r["seed"] == seed:
            return list(dict.fromkeys((float(a), float(b)) for _, a, b, _ in r["story_starts"]))
    sys.exit(f"seed {seed} is not in results/screen_climate.json: run planet_screen.py {seed} first, or give a start")


def main():
    seed = int(sys.argv[1])
    starts = [(float(sys.argv[2]), float(sys.argv[3]))] if len(sys.argv) > 3 else story_starts(seed)
    d = np.load(HERE / f"results/planet_m1_seed{seed}/fine/terrain_fine.npz")
    xyz, elev = d["grid"].astype(float), d["elev"].astype(float)
    spacing = np.sqrt(4 * np.pi / len(xyz)) * R
    res = {s: {} for s in starts}
    for sl in SEA_LEVELS:
        land = elev > sl
        print(f"sea level {sl:+.0f} m: land {land.mean()*100:.1f}% of surface")
        for name, km, _ in MODES:
            tree, lab = components(xyz, land, max(km, 1.5 * spacing))
            for s in starts:
                res[s].setdefault(sl, {})[name] = float((lab == lab[tree.query(ll_to_xyz(*s))[1]]).mean())
        for s in starts:
            r = res[s][sl]
            r["pass"] = all(r[name] >= need for name, _, need in MODES)
            print(f"  start {s[0]:6.1f},{s[1]:7.1f}: " + "  ".join(f"{name} {r[name]*100:5.1f}%" for name, _, _ in MODES)
                  + ("  PASS" if r["pass"] else "  fail"))
    # best start: passes at present sea level first, then at the glacial low stand, then the most land on foot/raft
    best = max(starts, key=lambda s: (res[s][0.0]["pass"], res[s][-120.0]["pass"], res[s][0.0][MODES[0][0]]))
    b = res[best]
    out = {"start": list(best), **{name: b[0.0][name] for name, _, _ in MODES},
           "pass": b[0.0]["pass"] or b[-120.0]["pass"],
           "sea_levels": {f"{sl:+.0f}": b[sl] for sl in SEA_LEVELS},
           "all_starts": [{"start": list(s), **{f"{sl:+.0f}": res[s][sl] for sl in SEA_LEVELS}} for s in starts]}
    print(f"best start {best}: B7", "PASS" if out["pass"] else "FAIL",
          "(needs >= 20% on foot/raft and >= 80% by coastal boat, at present or glacial sea level)")
    (HERE / f"results/planet_m1_seed{seed}/fine/migration.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
