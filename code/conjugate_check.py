"""Conjugate margins: pairs of coasts that were split by the same continental rift and drifted apart (South America and
Africa on Earth). These are the coasts that can be seen to fit back together; finding them from coast shapes alone did
not work on Earth (fit_earth_check.py: the Bullard 1965 fit of South America and Africa scored no better than unrelated
pairs), so they are taken from the planet's own history (records.npz rift_time of the thinned rift margins).

A pair counts as a visible fit when:
  - both sides keep >= MIN_MARGIN_KM of margin from the same rift (the two sides are now on different landmasses),
  - the margins are now >= MIN_GAP_KM apart (an ocean opened between them),
  - >= 70% of each margin was not deformed afterwards (no later mountain building or folding),
  - closing the ocean (best rotation of one margin onto the other) leaves an RMS misfit <= MAX_RMS_KM.
Landmasses as in look_check (fine replay raster if present).

Usage: .venv/bin/python planet-history-sim/tools/conjugate_check.py seed [seed ...]   (writes results/planet_m1_seed<N>/conjugates.json)
"""
import json, sys
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import look_check as L
import fit_check as F
from plate_checks import HERE

MIN_MARGIN_KM, MIN_GAP_KM, MAX_RMS_KM = 1000.0, 500.0, 150.0


def close_fit(a, b):
    """Rotation of point set b onto point set a (ICP, trimmed); returns (rms km, rotation)."""
    ca, cb = a.mean(0), b.mean(0)
    ca /= np.linalg.norm(ca); cb /= np.linalg.norm(cb)
    ax = np.cross(cb, ca)
    M0 = F.rot(ax, np.arccos(np.clip(ca @ cb, -1, 1))) if np.linalg.norm(ax) > 1e-9 else np.eye(3)
    ta, best = cKDTree(a), None
    for spin in np.radians(np.arange(0, 360, 15)):
        M = F.rot(ca, spin) @ M0
        for _ in range(30):
            P = b @ M.T
            d, j = ta.query(P)
            k = d <= np.quantile(d, 0.8)
            M = F.kabsch(P[k], a[j[k]]) @ M
        d = ta.query(b @ M.T)[0] * L.R
        rms = float(np.sqrt((d ** 2).mean()))
        if best is None or rms < best[0]:
            best = (rms, M)
    return best


def check(seed):
    src = HERE / f"results/planet_m1_seed{seed}"
    rec = np.load(src / "records.npz")
    grid = np.load(src / "terrain.npz")["grid"]
    f = src / "fine/terrain_fine.npz"
    d = np.load(f if f.exists() else src / "terrain.npz")
    land = (d["elev"][cKDTree(d["grid"]).query(L.G)[1]] > 0).reshape(L.H, L.W)
    lab = L.pieces(land, 0 if f.exists() else 2)[1]
    # every particle -> nearest landmass (margins can be under the shelf now)
    idx = ndimage.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
    labn = lab[idx[0], idx[1]].ravel()
    cell = cKDTree(L.G).query(grid)[1]
    piece = labn[cell]
    tot = {i: (lab == i).sum() for i in np.unique(lab[lab > 0])}
    T_all = rec["rift_time"]
    later = lambda t: (rec["orogen_time"] > 0) & (rec["orogen_time"] < t) | (rec["fold_time"] > 0) & (rec["fold_time"] < t)
    out = []
    for T in sorted(set(np.round(T_all[T_all > 0]).tolist())):
        sel = np.nonzero(np.round(T_all) == T)[0]
        ps, cnt = np.unique(piece[sel], return_counts=True)
        sides = [p for p, c in zip(ps, cnt) if c >= 5]
        for i, a in enumerate(sides):
            for b in sides[i + 1:]:
                A, B = sel[piece[sel] == a], sel[piece[sel] == b]
                la, lb = L.span(grid[A]), L.span(grid[B])
                gap = float(cKDTree(grid[A]).query(grid[B])[0].min() * L.R)
                intact = min(1 - later(T)[A].mean(), 1 - later(T)[B].mean())
                rms, _ = close_fit(grid[A], grid[B])
                big = lambda p: round(100 * tot[p] / sum(tot.values()), 1)
                fits = bool(min(la, lb) >= MIN_MARGIN_KM and gap >= MIN_GAP_KM and intact >= 0.7 and rms <= MAX_RMS_KM)
                out.append(dict(rift_myr=T, landA=big(a), landB=big(b), marginA_km=round(la), marginB_km=round(lb),
                                gap_km=round(gap), intact=round(float(intact), 2), rms_km=round(rms), fits=fits))
    out.sort(key=lambda r: (not r["fits"], -min(r["marginA_km"], r["marginB_km"])))
    res = {"seed": seed, "grid": "fine" if f.exists() else "coarse", "fits": [r for r in out if r["fits"]], "pairs": out}
    (src / "conjugates.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    for s in map(int, sys.argv[1:]):
        r = check(s)
        print(f"seed {s} ({r['grid']}) visible fits {len(r['fits'])}: " + "; ".join(
            f"{p['rift_myr']:.0f} Myr {p['landA']}%<->{p['landB']}% margins {p['marginA_km']}/{p['marginB_km']} km gap {p['gap_km']} "
            f"intact {p['intact']} misfit {p['rms_km']} km{' FIT' if p['fits'] else ''}" for p in r["pairs"][:4]))
