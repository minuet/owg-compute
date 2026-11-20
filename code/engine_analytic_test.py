"""Known-answer test for plate_engine.py (RULES.md C2), on our planet's radius (6000 km).

Two plates split the sphere along the 0°/180° meridians and rotate about the north pole in opposite directions:
the ridge stays on the 0° meridian (symmetric spreading) and the plates converge along 180°.
Exact answer: ocean at longitude lon (|lon| < omega*T) has age |lon| / omega; crust made = crust consumed.
A small continent rides on plate 2 and must never be lost.

Usage: .venv/bin/python planet-history-sim/tools/engine_analytic_test.py [n_points] [half_rate_cm_yr] [duration_myr]
"""
import math, sys, json
import numpy as np
import pygplates
from plate_engine import PlateEngine, xyz_to_latlon
from plate_checks import HERE

R_PLANET = 6000.0


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
    half_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0     # cm/yr at the equator
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    omega_deg = math.degrees(half_rate * 10.0 / R_PLANET)              # deg/Myr (1 cm/yr = 10 km/Myr)
    north = pygplates.PointOnSphere(90, 0)

    def stage(p, t_from, t_to):  # times in Ma, t_from > t_to
        ang = math.radians(omega_deg * (t_from - t_to)) * (1 if p == 2 else -1)
        return pygplates.FiniteRotation(north, ang)

    eng = PlateEngine(n, R_PLANET, stage, gap_r=float(sys.argv[4]) if len(sys.argv) > 4 else 0.85)
    lat, lon = xyz_to_latlon(eng.grid)
    eng.plate = np.where(lon >= 0, 2, 1).astype(np.int32)
    eng.cont = (eng.plate == 2) & (np.abs(lat) < 20) & (lon > 60) & (lon < 100)
    cont0 = int(eng.cont.sum())
    for t in range(T, 0, -1):
        eng.step(t, t - 1)
    st = eng.grid_state()
    age = st["birth"]                           # present = 0 Ma, so age = birth
    truth = np.abs(lon) / omega_deg
    young = (np.abs(lon) < omega_deg * T * 0.95) & (np.abs(lat) < 60) & ~st["cont"]
    d = age[young] - truth[young]
    made = np.mean([L["new_ocean_cells"] for L in eng.log[10:]]) * eng.cell_area_km2
    eaten = np.mean([L["consumed_cells"] for L in eng.log[10:]]) * eng.cell_area_km2
    expected = 2 * 2 * math.radians(omega_deg) * R_PLANET ** 2          # two plates, half rate each, ridge from pole to pole
    spacing_km = eng.spacing * R_PLANET
    res = {"points": n, "spacing_km": spacing_km, "half_rate_cm_yr": half_rate, "duration_myr": T,
           "age_bias_myr": float(np.median(d)), "age_median_abs_err_myr": float(np.median(np.abs(d))),
           "age_p90_abs_err_myr": float(np.percentile(np.abs(d), 90)),
           "resolution_limit_myr": spacing_km / (half_rate * 10.0),
           "crust_made_km2_per_myr": made, "crust_consumed_km2_per_myr": eaten, "crust_expected_km2_per_myr": expected,
           "continental_cells_start_end": [cont0, int(eng.cont.sum())]}
    for k, v in res.items():
        print(f"{k:30s} {v:.2f}" if isinstance(v, float) else f"{k:30s} {v}")
    (HERE / "results/engine_analytic_test.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
