"""Jigsaw fit between continents (the evidence Wegener used): can one landmass be rotated on the sphere so that a long,
curved stretch of its coast closes onto another landmass's coast without the two overlapping?

For every pair of landmasses (>= 1% of land) that are now apart (>= 300 km), the coasts facing each other are fitted
with an ICP rotation about the planet centre (Kabsch, trimmed to the closest 70% of pairs, 12 starting spins).
A fit counts when, after rotation:
  - matched coast (rotated coast within MATCH_KM of the other coast) is >= 1000 km long,
  - the matched stretch is curved (bulges >= 150 km off the great circle through its ends): straight coasts fit anything,
  - the rotated landmass overlaps the other by <= 3% of its area.
Uses a 0.25 deg raster of the fine replay if present, else the coarse run (coast shapes then only at ~90 km).

Usage: .venv/bin/python planet-history-sim/tools/fit_check.py seed [seed ...]   (writes results/planet_m1_seed<N>/fit.json)
"""
import json, sys
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import look_check as L
from plate_checks import HERE

R = L.R
SPACING_KM = 50.0
MATCH_KM = 80.0       # fine grid; coarse grid uses twice this
FIT_KM = 1000.0
BULGE_KM = 150.0
FACE_KM = 1500.0      # coast counted as facing: within gap + this of the other landmass
ANCHORS = 30          # global search: anchor points on each coast
SPIN_DEG = 10
COVER = 0.5           # share of both facing coasts that closes up


def thin(p):
    """Points snapped to a ~50 km 3-D grid, one per cell: roughly even spacing at every latitude."""
    k = np.round(p * R / SPACING_KM).astype(np.int64)
    _, i = np.unique(k, axis=0, return_index=True)
    return p[np.sort(i)]


def kabsch(P, Q):
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def rot(axis, ang):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def bulge(p):
    a = p[np.argmax(np.linalg.norm(p - p[0], axis=1))]
    b = p[np.argmax(np.linalg.norm(p - a, axis=1))]
    n = np.cross(a, b)
    if np.linalg.norm(n) < 1e-9:
        return 0.0
    n /= np.linalg.norm(n)
    return float(np.abs(p @ n).max() * R)


def rotvec(v):
    a = np.linalg.norm(v)
    return np.eye(3) if a < 1e-12 else rot(v / a, a)


def fit_pair(cA, cB, landB, labA_at, match_km, rng=np.random.default_rng(0)):
    """Best rotation of B's facing coast onto A's facing coast without overlapping A.
    Global search (anchor point pairs x spins), then local refinement of a soft score."""
    treeA = cKDTree(cA)
    sB = cB[rng.choice(len(cB), min(len(cB), 120), replace=False)]
    sL = landB[rng.choice(len(landB), min(len(landB), 600), replace=False)]
    aA = cA[rng.choice(len(cA), min(len(cA), ANCHORS), replace=False)]
    aB = cB[rng.choice(len(cB), min(len(cB), ANCHORS), replace=False)]
    sig = match_km / R

    def soft(M):
        d = treeA.query(sB @ M.T)[0]
        ov = labA_at(sL @ M.T).mean()
        return float(np.exp(-(d / sig) ** 2).mean()) - 5.0 * max(0.0, ov - 0.02)

    # global search, batched: every anchor pair (b -> a) and spin about a
    Ms = []
    for b in aB:
        for a in aA:
            ax = np.cross(b, a)
            R0 = rot(ax, np.arccos(np.clip(a @ b, -1, 1))) if np.linalg.norm(ax) > 1e-9 else np.eye(3)
            for spin in np.radians(np.arange(0, 360, SPIN_DEG)):
                Ms.append(rot(a, spin) @ R0)
    Ms = np.array(Ms)
    cs = sB[::2]; cl = sL[::4]
    d = treeA.query(np.einsum("kij,nj->kni", Ms, cs).reshape(-1, 3))[0].reshape(len(Ms), -1)
    ov = labA_at(np.einsum("kij,nj->kni", Ms, cl).reshape(-1, 3)).reshape(len(Ms), -1).mean(1)
    sc = np.exp(-(d / sig) ** 2).mean(1) - 5.0 * np.maximum(0.0, ov - 0.02)
    cands = [(sc[k], Ms[k]) for k in np.argsort(-sc)[:40]]
    from scipy.optimize import minimize
    best = None
    for sc0, M0 in cands[:8]:  # refine the best few
        res = minimize(lambda v: -soft(rotvec(v) @ M0), np.zeros(3), method="Nelder-Mead",
                       options={"initial_simplex": np.vstack([np.zeros(3), np.eye(3) * 0.05]), "xatol": 1e-4, "fatol": 1e-4, "maxiter": 300})
        M = rotvec(res.x) @ M0
        P = cB @ M.T
        d = treeA.query(P)[0] * R
        m = d < match_km
        if m.sum() < 3:
            continue
        length = m.sum() * SPACING_KM
        dA = cKDTree(P).query(cA)[0] * R
        cover = min(float(m.mean()), float((dA < match_km).mean()))
        overlap = float(labA_at(landB @ M.T).mean())
        score = length if overlap <= 0.03 else 0.0
        if best is None or score > best["score"]:
            best = dict(score=score, length_km=round(length), bulge_km=round(bulge(P[m])), overlap=round(overlap, 3),
                        rms_km=round(float(np.sqrt((d[m] ** 2).mean()))), cover=round(cover, 2), M=M)
    return best


def fit_land(land, match_km, min_frac=0.01):
    """All pairs of landmasses on a 0.25 deg land raster; returns (pairs, lab, ids, areas)."""
    ar, lab, ids = L.pieces(land)
    tot = ar.sum()
    big = [k for k in range(len(ids)) if ar[k] >= min_frac * tot]
    coast, landp = {}, {}
    for k in big:
        m = lab == ids[k]
        coast[k] = thin(L.G[(m & ~ndimage.binary_erosion(m)).ravel()])
        lp = L.G[m.ravel()]
        landp[k] = lp[:: max(1, len(lp) // 4000)]

    def labA_at_factory(idA):
        def f(p):
            la = np.degrees(np.arcsin(np.clip(p[:, 2], -1, 1))); lo = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
            y = np.clip(((90 - la) * 4).astype(int), 0, L.H - 1); x = np.clip(((lo + 180) * 4).astype(int), 0, L.W - 1)
            return lab[y, x] == idA
        return f

    pairs = []
    for i, a in enumerate(big):
        for b in big[i + 1:]:
            ta, tb = cKDTree(coast[a]), cKDTree(coast[b])
            dab = ta.query(coast[b])[0] * R
            gap = dab.min()
            if gap < 300 or gap > 5000:
                continue
            dba = tb.query(coast[a])[0] * R
            fa, fb = coast[a][dba < gap + FACE_KM], coast[b][dab < gap + FACE_KM]   # coasts facing each other
            if len(fa) < 15 or len(fb) < 15:
                continue
            (A, cA, B, cB) = (a, fa, b, fb) if ar[a] >= ar[b] else (b, fb, a, fa)   # fit the smaller onto the larger
            r = fit_pair(cA, cB, landp[B], labA_at_factory(ids[A]), match_km)
            if r is None:
                continue
            r.update(idA=int(ids[A]), idB=int(ids[B]), A=round(ar[A] / tot * 100, 1), B=round(ar[B] / tot * 100, 1),
                     gap_km=round(gap), fits=bool(r["length_km"] >= FIT_KM and r["cover"] >= COVER and r["overlap"] <= 0.03 and r["bulge_km"] >= BULGE_KM))
            pairs.append(r)
    pairs.sort(key=lambda r: -r["score"])
    return pairs, lab, ids, ar


def check(seed):
    src = HERE / f"results/planet_m1_seed{seed}"
    f = src / "fine/terrain_fine.npz"
    fine = f.exists()
    d = np.load(f if fine else src / "terrain.npz")
    land = (d["elev"][cKDTree(d["grid"]).query(L.G)[1]] > 0).reshape(L.H, L.W)
    pairs = fit_land(land, MATCH_KM if fine else 2 * MATCH_KM)[0]
    for p in pairs:
        p.pop("M")
    out = {"seed": seed, "grid": "fine" if fine else "coarse", "fits": [p for p in pairs if p["fits"]], "best": pairs[:3]}
    (src / "fit.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    for s in map(int, sys.argv[1:]):
        r = check(s)
        print(f"seed {s} ({r['grid']}) fits {len(r['fits'])}: " + "; ".join(
            f"{p['A']}%<->{p['B']}% matched {p['length_km']} km bulge {p['bulge_km']} overlap {p['overlap']} gap {p['gap_km']}" for p in r["best"]))
