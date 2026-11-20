"""What sets a plate's speed, measured in the Earth plate model (Müller et al. 2022, mantle frame), every STEP Myr from
200 Ma to now - the Earth test for driving the planet's plate speeds by their forces instead of drawing them
(PIPELINE principle 5; user 2026-11-03: speeds must not be random draws).

Per plate and time: area, continental share, RMS speed, and the lengths of its outline by the MODEL'S OWN boundary labels:
  slab   - subduction zone where this plate sinks (slab attached: pulls it; Forsyth & Uyeda 1975)
  over   - subduction zone where this plate overrides (trench suction; Conrad & Lithgow-Bertelloni 2002)
  ridge  - mid-ocean ridge (ridge push)
  perim  - whole outline
1. Check of the measurement against the published statistics of Zahirovic et al. 2015 (EPSL 418:40, Table 1, all plates
   0-200 Ma): median RMS speed of plates with no slab 2.59, > 25% of the outline slab 7.87, > 50% continental 2.80
   (max 7.85), > 25% continental and < 20% slab 2.75 (max 7.44) cm/yr.
2. Force balance of Forsyth & Uyeda 1975 in its scalar form (driving force along the boundaries = drag over the base,
   stronger under continents): v * area * (1 + k * cont) = a * slab + s * over + r * ridge  (a, s, r >= 0, least squares
   on log(speed + 0.1 cm/yr) for each k on a grid; k chosen by the smallest residual; the 0.1 only keeps log finite). Fitted on 200-100 Ma and tested on 100-0 Ma and the
   reverse (the coefficients must not depend on the period), then on all. Scores: correlation of predicted and real
   speed (log) for all plates and for continental (> 50%) plates.
Usage: .venv/bin/python planet-history-sim/tools/earth_plate_speed_controls.py [step_Myr] [t_max]  ->  results/earth_plate_speed_controls.json
"""
import sys, json, time
import numpy as np
import pygplates
from scipy.optimize import minimize
from plate_checks import MODEL, TOPOLOGY_FILES, ROTATION, HERE
from plate_engine import fibonacci_xyz
from plate_checks import partition_ids

R = 6371.0
N = int(round(60000 * (R / 6000.0) ** 2))      # the planet's cell size, as the other Earth tools
MIN_AREA = 0.001                                 # plates of >= 0.1% of the surface (Zahirovic 2015 lists all plates;
                                                 # smaller ones have < ~70 grid cells and no stable RMS speed)
K_GRID = np.r_[0.0, np.geomspace(0.1, 30.0, 40)]  # search range of the continental drag factor k (numerical)


def euler(rot, t, p):
    try:
        e = rot.get_rotation(float(t), int(p), float(t) + 1.0)
        if e.represents_identity_rotation():
            return np.zeros(3)
        pole, ang = e.get_euler_pole_and_angle()
        return np.array(pole.to_xyz()) * ang
    except Exception:
        return np.zeros(3)


def measure(step, t_max):
    rot = pygplates.RotationModel(str(ROTATION))
    topo = [pygplates.FeatureCollection(str(MODEL / "Topologies" / f)) for f in TOPOLOGY_FILES]
    conts = pygplates.FeatureCollection(str(MODEL / "Continents/continents.gpml"))
    g = fibonacci_xyz(N)
    rows = []
    for t in range(t_max, -1, -step):
        t0 = time.time()
        plate = partition_ids(pygplates.PlatePartitioner(topo, rot, float(t)), g)
        cont = partition_ids(pygplates.PlatePartitioner(conts, rot, float(t)), g) >= 0
        L = {}
        rt, ss = [], []
        pygplates.resolve_topologies(topo, rot, rt, float(t), ss)
        for s in ss:
            ft = s.get_feature().get_feature_type()
            for sub in s.get_shared_sub_segments():
                km = sub.get_resolved_geometry().get_arc_length() * R
                sh = [int(r.get_feature().get_reconstruction_plate_id()) for r in sub.get_sharing_resolved_topologies()]
                for p in set(sh):
                    L.setdefault(p, {"perim": 0.0, "slab": 0.0, "over": 0.0, "ridge": 0.0})["perim"] += km
                if ft == pygplates.FeatureType.gpml_subduction_zone:
                    os_ = sub.get_overriding_and_subducting_plates()
                    if os_ is not None:
                        ov, su = (int(x.get_feature().get_reconstruction_plate_id()) for x in os_)
                        L.setdefault(su, {"perim": 0.0, "slab": 0.0, "over": 0.0, "ridge": 0.0})["slab"] += km
                        L.setdefault(ov, {"perim": 0.0, "slab": 0.0, "over": 0.0, "ridge": 0.0})["over"] += km
                elif ft == pygplates.FeatureType.gpml_mid_ocean_ridge:
                    for p in set(sh):
                        L[p]["ridge"] += km
        ids, cnt = np.unique(plate, return_counts=True)
        for p, n in zip(ids, cnt):
            if p < 0 or n / N < MIN_AREA or int(p) not in L:
                continue
            m = plate == p
            w = euler(rot, t, p)
            v = np.linalg.norm(np.cross(w, g[m]), axis=1) * R / 10.0        # cm/yr
            rows.append({"t": t, "plate": int(p), "area": float(n / N), "cont": float(cont[m].mean()),
                         "rms": float(np.sqrt((v ** 2).mean())), **{k: round(x, 1) for k, x in L[int(p)].items()}})
        print(t, "Ma:", sum(r["t"] == t for r in rows), f"plates ({time.time() - t0:.0f}s)", flush=True)
    return rows


def med_max(v):
    return {"n": len(v), "median": round(float(np.median(v)), 2), "max": round(float(np.max(v)), 2)} if len(v) else {"n": 0}


def fit(rows):
    """Least squares on log speed (speeds spread over two orders of magnitude; a fit on relative residuals of the
    balance shrinks every prediction towards 0). Coefficients >= 0 as exp(theta); k from the grid."""
    v = np.array([r["rms"] for r in rows]); A = np.array([r["area"] for r in rows]) * 4 * np.pi * R ** 2
    c = np.array([r["cont"] for r in rows]); X = np.array([[r["slab"], r["over"], r["ridge"]] for r in rows])
    lv = np.log(v + 0.1)
    best = None
    for k in K_GRID:
        den = A * (1 + k * c)
        f = lambda th: np.mean((np.log(X @ np.exp(th) / den + 0.1) - lv) ** 2)
        th0 = np.log(np.full(3, np.median(v * den / X.sum(axis=1).clip(1.0))))
        r_ = minimize(f, th0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 4000})
        if best is None or r_.fun < best[0]:
            best = (float(r_.fun), float(k), np.exp(r_.x))
    return {"k": best[1], "a_slab": float(best[2][0]), "s_over": float(best[2][1]), "r_ridge": float(best[2][2]),
            "rms_log_residual": round(best[0] ** 0.5, 3), "k_at_grid_edge": bool(best[1] in (K_GRID[0], K_GRID[-1]))}


def predict(f, rows):
    A = np.array([r["area"] for r in rows]) * 4 * np.pi * R ** 2
    c = np.array([r["cont"] for r in rows]); X = np.array([[r["slab"], r["over"], r["ridge"]] for r in rows])
    return (X @ np.array([f["a_slab"], f["s_over"], f["r_ridge"]])) / (A * (1 + f["k"] * c))


def score(f, rows):
    v = np.array([r["rms"] for r in rows]); p = predict(f, rows); c = np.array([r["cont"] for r in rows])
    out = {}
    for name, m in (("all", np.ones(len(v), bool)), ("continental_gt50", c > 0.5), ("oceanic_lt25", c < 0.25)):
        if m.sum() > 2:
            lv, lp = np.log(v[m] + 0.1), np.log(p[m] + 0.1)
            out[name] = {"n": int(m.sum()), "corr_log": round(float(np.corrcoef(lv, lp)[0, 1]), 3),
                         "real_median_max": [round(float(np.median(v[m])), 2), round(float(v[m].max()), 2)],
                         "pred_median_max": [round(float(np.median(p[m])), 2), round(float(p[m].max()), 2)]}
    return out


def main():
    step = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    t_max = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    rows = measure(step, t_max)
    v = lambda sel: [r["rms"] for r in rows if sel(r)]
    sf = lambda r: r["slab"] / max(r["perim"], 1e-9)
    out = {"source": f"Müller et al. 2022, mantle frame, {t_max}-0 Ma every {step} Myr, plates >= {MIN_AREA:.1%}",
           "zahirovic_2015_check": {
               "no_slab (Z15 median 2.59, max 12.06)": med_max(v(lambda r: r["slab"] == 0)),
               "slab_gt25pct (Z15 7.87, max 22.72)": med_max(v(lambda r: sf(r) > 0.25)),
               "cont_gt50 (Z15 2.80, max 7.85)": med_max(v(lambda r: r["cont"] > 0.5)),
               "cont_gt25_slab_lt20 (Z15 2.75, max 7.44)": med_max(v(lambda r: r["cont"] > 0.25 and sf(r) < 0.2)),
               "cont_lt25_slab_gt20 (Z15 8.51, max 22.72)": med_max(v(lambda r: r["cont"] < 0.25 and sf(r) > 0.2)),
               "cont_gt50_no_slab": med_max(v(lambda r: r["cont"] > 0.5 and r["slab"] == 0)),
               "cont_gt50_with_slab": med_max(v(lambda r: r["cont"] > 0.5 and r["slab"] > 0))}}
    if t_max >= 100:
        old = [r for r in rows if r["t"] > 100]; young = [r for r in rows if r["t"] <= 100]
        f_old, f_young, f_all = fit(old), fit(young), fit(rows)
        out["force_balance"] = {"fit_200_100": f_old, "fit_100_0": f_young, "fit_all": f_all,
                                "test_fit_old_on_young": score(f_old, young), "test_fit_young_on_old": score(f_young, old),
                                "all_on_all": score(f_all, rows)}
    out["rows"] = rows
    (HERE / "results/earth_plate_speed_controls.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: x for k, x in out.items() if k != "rows"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
