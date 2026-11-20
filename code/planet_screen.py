"""Climate screening of generated planets (RULES.md B4, B5, B6, B9) with the Earth-checked rough estimator.

For every planet that passed all A/C checks: terrain -> 1 degree grid -> rough climate -> biome shares, story start,
longitude spread. Earth values come from the same estimator on Earth (habitable ~71%, desert ~13%, cold ~16%).
Migration (B7) needs straits of 100-200 km, below the coarse grid: checked after the fine replay.

Usage: .venv/bin/python planet-history-sim/tools/planet_screen.py [seed ...]   (default: all seeds of round $ROUND (v2) whose log says ALL CHECKS PASS)
Results are merged into results/screen_climate.json; each seed gets strict pass / within tolerance / fail and a score (0 = on target).
"""
import glob, json, os, re, sys
import numpy as np
from scipy.spatial import cKDTree
from climate_rough import climate_grid, biome_shares
from plate_checks import HERE
from plate_engine import xyz_to_latlon

R_PLANET = 6000.0
HAB_ZONE_KM2 = 1.0e6    # B9c: habitable land needed in every ZONES adjacent time zones (about two Frances)
ZONES = 4
TOL_FRAC = 0.5          # tolerance = this x the estimator's error on Earth (RULES.md B4c)
COLD_MAX = 0.20         # user 2026-10-29: cold details are deferred to the 3-D climate model (the rough estimator is least sure
                        # about ice and tundra); screening only rejects clearly too cold planets and planets without any ice/snow
DEFERRED = ("B5 cold", "B5 ice", "B5 tundra", "B4b tundra", "B4b ice", "B4b alpine")
W, H = 360, 180
LAT = 90 - (np.arange(H) + 0.5)
LON = -180 + np.arange(W) + 0.5


def to_grid(xyz, values):
    la, lo = np.radians(LAT)[:, None] * np.ones((1, W)), np.radians(LON)[None, :] * np.ones((H, 1))
    g = np.column_stack([(np.cos(la) * np.cos(lo)).ravel(), (np.cos(la) * np.sin(lo)).ravel(), np.sin(la).ravel()])
    return values[cKDTree(xyz).query(g)[1]].reshape(H, W)


RIFT_FLANK_KM = 150


def screen(seed, path=None):
    src = path or HERE / f"results/planet_m1_seed{seed}"
    t = np.load(src / "terrain.npz")
    st = np.load(src / "state.npz")
    elev = to_grid(t["grid"], t["elev"])
    rift_d = to_grid(t["grid"], t["rift_dist_km"]) if "rift_dist_km" in t.files else np.full((H, W), np.inf)
    land = elev > 0
    clim = climate_grid(elev, land, R_PLANET)
    B = clim["B"]
    sh = biome_shares(B, LAT)
    habit = sum(sh[k] for k in (0, 1, 3, 4, 5)); desert = sh[2]; cold = sh[6] + sh[7] + sh[8]
    w = np.cos(np.radians(LAT))[:, None] * np.ones((1, W))
    # story start (B6): tropical savanna cell with rainforest within ~4 cells and highland (>1200 m) within ~4 cells
    starts = []
    for y in np.nonzero(np.abs(LAT) <= 15)[0]:
        for x in np.nonzero(B[y] == 1)[0]:
            ys = slice(max(0, y - 4), min(H, y + 5))
            xs = [(x + d) % W for d in range(-4, 5)]
            win_b = B[ys][:, xs]; win_e = elev[ys][:, xs]
            rf = int((win_b == 0).sum()); hi = int((win_e > 1200).sum())
            # user 2026-10-29: the start must sit on the rift flank itself (<= RIFT_FLANK_KM from the rift axis);
            # "a rift somewhere within ~300 km of the window" let starts 450-850 km from the rift through
            near_rift = bool(rift_d[y, x] <= RIFT_FLANK_KM)
            if rf and (hi or near_rift):
                # a rift flank between forest and savanna is the preferred story start (rain shadow of the uplifted
                # flank, CLAUDE.md "East Side Story" reference); otherwise any highland next to the forest edge
                starts.append((min(rf, 15) + min(int((win_b == 1).sum()), 15) + 0.5 * min(hi, 10) + (20 if near_rift else 0),
                               float(LAT[y]), float(LON[x]), "rift" if near_rift else "highland"))
    starts.sort(reverse=True)
    # longitude spread (B9): largest gap in longitude without land between 60S and 70N
    band = (LAT >= -60) & (LAT <= 70)
    lon_has = land[band].any(axis=0)
    gap = run = 0
    for k in range(2 * W):
        if not lon_has[k % W]:
            run += 1; gap = max(gap, run)
        else:
            run = 0
    gap = min(gap, W)
    # B9c (user 2026-10-28): every 4 adjacent time zones (60 deg of longitude, sliding by one zone; user 2026-10-29: widest allowed) hold a large
    # habitable area where many players can be active: >= HAB_ZONE_KM2 of rainforest/savanna/steppe/temperate/boreal land
    cell_km2 = (np.pi * R_PLANET / 180) ** 2 * np.cos(np.radians(LAT))[:, None] * np.ones((1, W))
    hab_col = (np.isin(B, [0, 1, 3, 4, 5]) * cell_km2).sum(axis=0)
    hab_zone = hab_col.reshape(24, 15).sum(axis=1)
    hab_win = np.array([hab_zone[[(k + i) % 24 for i in range(ZONES)]].sum() for k in range(24)])
    # B4, B4b, B5 as ranges (lo, hi) with a tolerance: half of the same estimator's error on Earth for that quantity
    # (|model - real| in results/earth_biome_shares.json, RULES.md B4c). Strict pass = inside the range; "near" = inside
    # range +- tolerance (the rough estimator cannot tell these apart; the 3-D climate model decides later, step 11).
    E = json.loads((HERE / "results/earth_biome_shares.json").read_text())
    M, Rl = np.array(E["earth_model"]), np.array(E["earth_real"])
    HAB = [0, 1, 3, 4, 5]
    tol = TOL_FRAC * np.abs(M - Rl)
    tol_hab = TOL_FRAC * abs(M[HAB].sum() - Rl[HAB].sum())
    names = ["rainforest", "savanna", "desert", "steppe", "temperate forest", "boreal", "tundra", "ice", "alpine"]
    ranges = [("B4 habitable >= 76% (Earth same method ~71%)", habit, 0.76, 1.0, tol_hab),
              ("B4 desert <= 11% (Earth ~13%)", desert, 0.0, 0.11, tol[2]),
              ("B5 cold 8~15% (Earth ~16%)", cold, 0.08, 0.15, max(tol[6], tol[7])),
              ("B5 ice >= 3%", sh[7], 0.03, 1.0, tol[7]),
              ("B5 tundra >= 2%", sh[6], 0.02, 1.0, tol[6])]
    for k in HAB:                                   # B4b habitable types: 0.5x to 2x Earth (nothing missing, nothing dominant)
        ranges.append((f"B4b {names[k]} 0.5-2x Earth ({M[k]*100:.1f}%)", sh[k], 0.5 * M[k], 2.0 * M[k], tol[k]))
    for k, (lo, hi) in {2: (0.03, 0.11), 6: (0.02, M[6]), 7: (0.03, M[7]), 8: (0.01, 2.0 * M[8])}.items():
        ranges.append((f"B4b {names[k]} {lo*100:.0f}-{hi*100:.1f}%", sh[k], lo, hi, tol[k]))   # present, not above Earth (alpine up to 2x)
    checks, score = {}, 0.0
    for name, v, lo, hi, t in ranges:
        miss = max(lo - v, v - hi, 0.0)            # how far outside the range
        score += miss / t
        status = "pass" if miss <= 1e-9 else ("near" if miss <= t + 1e-9 else "fail")
        if name.startswith(DEFERRED) and status != "pass":
            score -= miss / t; status = "defer"        # recorded, decided later by the 3-D climate model
        checks[name] = (status, f"{v*100:.1f}%" + (f" (outside by {miss*100:.1f}, tolerance {t*100:.1f})" if miss > 1e-9 else ""))
    checks[f"B5 cold <= {COLD_MAX*100:.0f}% and some ice/snow (hard limit while details are deferred)"] = (
        "pass" if cold <= COLD_MAX and sh[7] + sh[8] > 0 else "fail", f"cold {cold*100:.1f}%, ice+alpine {(sh[7]+sh[8])*100:.1f}%")
    n_rift = sum(1 for x in starts if x[3] == "rift")
    checks["B6 story start exists"] = ("pass" if starts else "fail", f"{len(starts)} candidate cells, {n_rift} by a rift"
                                       + (f"; best ({starts[0][3]}) at {starts[0][1]:.1f}, {starts[0][2]:.1f}" if starts else ""))
    checks["B9 longitude gap <= 120 deg"] = ("pass" if gap <= 120 else "fail", f"{gap} deg")
    checks["B9c habitable land >= 1 Mkm2 in every 4 adjacent time zones"] = (
        "pass" if hab_win.min() >= HAB_ZONE_KM2 else "fail",
        f"weakest 4 zones (from {int(np.argmin(hab_win)) * 15 - 180} deg): {hab_win.min() / 1e6:.2f} Mkm2")
    st = [v[0] for v in checks.values()]
    return {"seed": seed, "shares": sh, "habitable": habit, "desert": desert, "cold": cold, "lon_gap": gap, "hab_win_min_km2": float(hab_win.min()),
            "story_starts": starts[:5], "rift_start": bool(starts) and starts[0][3] == "rift",
            "checks": {k: {"pass": v[0] == "pass", "status": v[0], "value": v[1]} for k, v in checks.items()},
            "deferred": [k for k, v in checks.items() if v[0] == "defer"],
            "all_pass": all(s in ("pass", "defer") for s in st),                 # strict
            "within_tolerance": all(s != "fail" for s in st),        # candidate (RULES.md B4c)
            "score": round(score, 2)}                                # 0 = strict pass; lower is closer to the targets


def main():
    seeds = [int(a) for a in sys.argv[1:]]
    if not seeds:                                    # every seed of the round (env ROUND, default v2) whose log says ALL CHECKS PASS
        for f in glob.glob(str(HERE / f"results/{os.environ.get('ROUND', 'v2')}/batch_*.log")):
            if "ALL CHECKS PASS" in open(f).read():
                seeds.append(int(re.search(r"batch_(\d+)\.log", f).group(1)))
    seeds = sorted(set(seeds))
    out = []
    for s in seeds:
        r = screen(s)
        out.append(r)
        flag = "PASS" if r["all_pass"] else ("near" if r["within_tolerance"] else "fail")
        print(f"seed {s:4d} [{flag}] score {r['score']:5.2f}  habitable {r['habitable']*100:4.1f}%  desert {r['desert']*100:4.1f}%  cold {r['cold']*100:4.1f}% "
              f"(ice {r['shares'][7]*100:.1f}, tundra {r['shares'][6]*100:.1f})  starts {len(r['story_starts'])}"
              f"{' (rift)' if r['rift_start'] else ''}  lon gap {r['lon_gap']}")
    print("candidates (within tolerance), best first:",
          " ".join(f"{r['seed']}{'R' if r['rift_start'] else ''}" for r in sorted(out, key=lambda r: r["score"]) if r["within_tolerance"]))
    # merge into the existing file so screening one seed does not drop the others
    path = HERE / "results/screen_climate.json"
    old = {r["seed"]: r for r in json.loads(path.read_text())} if path.exists() else {}
    old.update({r["seed"]: r for r in out})
    path.write_text(json.dumps([old[k] for k in sorted(old)], indent=1))


if __name__ == "__main__":
    main()
