"""Look requirements (RULES.md B9b, set by the user): checked on a 0.25 deg raster of the planet's terrain
(fine replay if present, else the coarse run). Major continents (>= 3% of land each; pieces facing each other closer than 200 km along >= 300 km of coast count as one): at least 4,
the largest <= 30% of land; land in every time zone (24 bands of 15 deg longitude, 60S-70N: a band with < 1% of that
land counts as empty; the number of empty bands does not matter, at most 3 in a row, i.e. no ocean gap wider than 45 deg (user 2026-10-28: players everywhere must find land, a gap of 1-2 zones is fine, 6-8 is not); Earth has 5 empty, 4 in a row); islands by size class compared with Earth (scaled by land area).

Usage: .venv/bin/python planet-history-sim/tools/look_check.py [earth] seed [seed ...]   (earth: the same rules on Earth's land)   (writes results/planet_m1_seed<N>/look.json)
"""
import json, os, sys
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from plate_checks import HERE

R = 6000.0
H, W = 720, 1440
LAT = np.radians(90 - (np.arange(H) + .5) / 4); LON = np.radians(-180 + (np.arange(W) + .5) / 4)
LA, LO = np.meshgrid(LAT, LON, indexing="ij")
G = np.c_[(np.cos(LA) * np.cos(LO)).ravel(), (np.cos(LA) * np.sin(LO)).ravel(), np.sin(LA).ravel()]
CELL = (np.pi * R / H) ** 2 * np.cos(LA)
# B9b largest continent: the design requirement is 40% (user 2026-10-28, RULES B9b). The 54% test-phase compromise
# was dropped with the force model (user 2026-11-18); the env override is kept only for diagnostics
LARGEST_MAX = float(os.environ.get("LOOK_LARGEST_MAX", "0.40"))
EARTH_ISLANDS = {"big": 4, "medium": 21, "small": 64}     # ETOPO5 at 0.25 deg, Earth land 146 Mkm2
EARTH_LAND = 146e6


def pieces(land, erode=0):
    """Areas (km2) of separate land pieces. erode > 0 (coarse grid only): the 87 km grid joins land across narrow straits
    and shelves, so pieces are found on the land shrunk by `erode` raster cells (2 = ~56 km) and every land cell is then
    given to the nearest piece (calibrated on 3111/3113/3120: matches the fine replay within a few percent of land)."""
    core = ndimage.binary_erosion(land, iterations=erode) if erode else land
    lab, _ = ndimage.label(core)
    for y in range(H):                                       # join across the 180 deg meridian
        a, b = lab[y, 0], lab[y, -1]
        if a and b and a != b:
            lab[lab == b] = a
    if erode:
        idx = ndimage.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
        lab = np.where(land, lab[idx[0], idx[1]], 0)
    ids = np.unique(lab[lab > 0])
    return ndimage.sum(CELL, lab, ids), lab, ids


def span(p):
    """Rough length (km) of a set of unit vectors: two passes of farthest point."""
    a = p[np.argmax(np.linalg.norm(p - p[0], axis=1))]
    return float(np.linalg.norm(p - a, axis=1).max() * R)


def merge_close(land, lab, ids, ar, gap_km=200.0, min_frac=0.03, span_km=300.0):
    """Pieces of >= min_frac of land count as one continent when a long stretch of their coasts faces each other closer
    than gap_km (the user's rule, 2026-10-28: two capes 100 km apart are still separate; two big lands with a long shore
    under 200 km apart look like one). Long = the close stretch spans >= span_km. Direct gaps only, no island chaining."""
    tot = ar.sum()
    keep = [k for k in range(len(ids)) if ar[k] >= min_frac * tot]
    trees = {}
    for k in keep:
        m = lab == ids[k]
        trees[k] = cKDTree(G[(m & ~ndimage.binary_erosion(m)).ravel()])
    parent = {k: k for k in keep}
    def root(k):
        while parent[k] != k:
            k = parent[k]
        return k
    for i, a in enumerate(keep):
        for b in keep[i + 1:]:
            d = trees[a].query(trees[b].data, distance_upper_bound=gap_km / R)[0] * R
            near = trees[b].data[d < gap_km]
            if len(near) and span(near) >= span_km:
                parent[root(b)] = root(a)
    out = {}
    for k in range(len(ids)):
        r = root(k) if k in parent else ("s", k)
        out[r] = out.get(r, 0.0) + ar[k]
    return np.array(list(out.values()))


def earth_land():
    """Earth's land on the same 0.25 deg raster (ETOPO5 block means), to run every look rule on Earth first."""
    raw = np.fromfile(HERE / "../reference/earth/topography/ETOPO5.DAT", dtype=">i2").reshape(2160, 4320).astype(float)
    e = raw.reshape(720, 3, 1440, 3).mean(axis=(1, 3))
    return np.roll(e, W // 2, axis=1) > 0                    # ETOPO5 columns start at 0E, the raster at 180W


def check(seed):
    global R, CELL
    if seed == "earth":
        R = 6371.0; CELL = (np.pi * R / H) ** 2 * np.cos(LA)
        src, land, fine = HERE / "results/earth_look", earth_land(), True
        src.mkdir(exist_ok=True)
    else:
        R = 6000.0; CELL = (np.pi * R / H) ** 2 * np.cos(LA)
        src = HERE / f"results/planet_m1_seed{seed}"
        f = src / "fine/terrain_fine.npz"; fine = f.exists()
        d = np.load(f if fine else src / "terrain.npz")
        land = (d["elev"][cKDTree(d["grid"]).query(G)[1]] > 0).reshape(H, W)
    ar, lab, ids = pieces(land, 0 if fine else 2); tot = ar.sum()
    cont = merge_close(land, lab, ids, ar)
    major = sorted([a / tot for a in cont if a >= 0.03 * tot], reverse=True)
    raw = ar if fine else pieces(land)[0]              # islands are counted on the land as it is
    rest = raw[raw < 0.03 * tot]
    scale = tot / EARTH_LAND
    isl = {"big": int(np.sum(rest >= 5e5)), "medium": int(np.sum((rest >= 5e4) & (rest < 5e5))),
           "small": int(np.sum((rest >= 5e3) & (rest < 5e4)))}
    deg = np.degrees(LA)
    m = land & (deg > -60) & (deg < 70)
    share = np.array([(CELL * m)[:, k * 60:(k + 1) * 60].sum() for k in range(24)]) / max((CELL * m).sum(), 1)
    empty = share < 0.01
    run = longest = 0
    for k in range(48):                                      # longest run of empty zones, around the globe
        run = run + 1 if empty[k % 24] else 0; longest = max(longest, min(run, 24))
    checks = {
        "B9b at least 4 sea-separated continents": (len(major) >= 4, f"{len(major)}: " + ", ".join(f"{x*100:.0f}%" for x in major)),
        f"B9b largest continent <= {LARGEST_MAX*100:.0f}% of land": (bool(major) and major[0] <= LARGEST_MAX, f"{major[0]*100:.0f}%" if major else "none"),
        "B9b time zones (info only; replaced by B9c habitable land in planet_screen.py)": (True, f"{int(empty.sum())} empty, {longest} in a row"),
        "B9b big and medium islands present (>= half of Earth's, scaled)": (
            isl["big"] >= 0.5 * EARTH_ISLANDS["big"] * scale and isl["medium"] >= 0.5 * EARTH_ISLANDS["medium"] * scale,
            f"big {isl['big']}, medium {isl['medium']}, small {isl['small']} (Earth scaled: {EARTH_ISLANDS['big']*scale:.1f}, "
            f"{EARTH_ISLANDS['medium']*scale:.1f}, {EARTH_ISLANDS['small']*scale:.1f})"),
    }
    out = {"seed": seed, "grid": "fine" if fine else "coarse", "continents": [round(x, 3) for x in major],
           "islands": isl, "empty_time_zones": int(empty.sum()),
           "checks": {k: {"pass": bool(v[0]), "value": v[1]} for k, v in checks.items()},
           "all_pass": all(v[0] for v in checks.values())}
    (src / "look.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    for s in [a if a == "earth" or "/" in a else int(a) for a in sys.argv[1:]]:   # "<seed>/scan_t<label>": a scanned moment
        r = check(s)
        print(f"seed {s} [{'PASS' if r['all_pass'] else 'fail'}] ({r['grid']}) " +
              " | ".join(f"{'ok' if v['pass'] else 'X'} {v['value']}" for v in r["checks"].values()))
