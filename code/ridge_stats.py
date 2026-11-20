"""Ridge and trench length, mean opening rate and crust production of a generated planet (RULES.md C4)."""
import json, sys
import numpy as np
from scipy.spatial import cKDTree
from plate_checks import HERE

R = 6000.0


def ridge_stats(seed):
    d = HERE / f"results/planet_m1_seed{seed}"
    st = np.load(d / "state.npz"); g, plate, cont, birth = st["grid"], st["plate"], st["cont"], st["birth"]
    W = {int(k): np.array(v) for k, v in json.load(open(d / "kinematics.json")).items()}
    sp = math_sp = np.sqrt(4 * np.pi / len(g)) * R
    nb = cKDTree(g).query(g, k=7)[1][:, 1:]
    i, k = np.nonzero(plate[nb] != plate[:, None]); j = nb[i, k]
    keep = i < j; i, j = i[keep], j[keep]
    w = lambda P: np.array([W.get(int(p), np.zeros(3)) for p in P])
    dv = np.cross(w(plate[i]), g[i]) - np.cross(w(plate[j]), g[j])
    dvec = (g[j] - g[i]) / np.linalg.norm(g[j] - g[i], axis=1)[:, None]
    rate = -(dv * dvec).sum(1) * R * 0.1                     # cm/yr, > 0 opening
    per_len = 2.5                                             # neighbour pairs per cell length of boundary
    o, c = rate > 0.5, rate < -0.5
    a = birth[~cont]
    return {"ridge_km": float(o.sum() / per_len * sp), "trench_km": float(c.sum() / per_len * sp),
            "opening_cm_yr": float(rate[o].mean()), "production_km2_yr": float((rate[o] * 1e-5).sum() / per_len * sp),
            "young_0_20_pct": float((a < 20).mean() * 100), "plates": int(len(np.unique(plate)))}


if __name__ == "__main__":
    # Earth (Müller 2022 model, 1 Gyr median): ridges ~62,000 km, trenches ~67,000 km; production ~3 km2/yr
    for s in sys.argv[1:]:
        r = ridge_stats(int(s))
        print(f"seed {s}: ridges {r['ridge_km']/1e3:.0f}k km, trenches {r['trench_km']/1e3:.0f}k km, opening {r['opening_cm_yr']:.1f} cm/yr, "
              f"production {r['production_km2_yr']:.1f} km2/yr, ocean <20 Myr {r['young_0_20_pct']:.0f}%, plates {r['plates']}")
