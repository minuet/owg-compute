"""Check thresholds for the planet, measured on the Earth plate model (Müller et al. 2022, the EarthByte reference
model) with the same quantities the planet checks compute (PIPELINE principle 0: numbers come from the authoritative
model, not from single papers). The planet simulates the ~200 Myr after a supercontinent's breakup, so the band is
Earth's same phase, 0-200 Ma: per time step values, p10-p90 across the steps.

Inputs: results/earth_check_results.json (plate_checks.py 10 1000); results/earth_grid_checks.json (earth_grid_checks.py:
the planet's own check algorithm run on the Earth model, so the A8 trench band is measured the same way as the planet -
2026-10-31: GPlately's trench tool on labelled trenches gave 62% retreating for today's Earth, the planet algorithm 77%). Output: results/earth_reference.json, read by planet_checks.verdict.

Usage: .venv/bin/python planet-history-sim/tools/earth_reference.py
"""
import json
import numpy as np
from plate_checks import HERE

T_MAX = 200
q = lambda a: [round(float(np.nanpercentile(a, 10)), 3), round(float(np.nanpercentile(a, 90)), 3)]


def main():
    pc = [r for r in json.loads((HERE / "results/earth_check_results.json").read_text()) if r["time"] <= T_MAX]
    eg = json.loads((HERE / "results/earth_grid_checks.json").read_text())["steps"]
    big = lambda r, f: [p["rms_cm_yr"] for p in r["plates"] if p["area_frac"] >= 0.01 and f(p["cont_frac"])]
    oce = lambda c: c < 0.25            # Zahirovic et al. 2015 classes
    con = lambda c: c > 0.5
    pooled_o = [v for r in pc for v in big(r, oce)]
    pooled_c = [v for r in pc for v in big(r, con)]
    ref = {
        "source": "Müller et al. 2022 plate model (mantle-optimised frame), 0-200 Ma every 10 Myr; band = p10-p90 over time steps",
        "net_rotation_deg_myr": q([r["net_rotation_deg_myr"] for r in pc]),
        "net_rotation_max": round(max(r["net_rotation_deg_myr"] for r in pc), 3),
        "ocean_plate_median_cm_yr": q([np.median(big(r, oce)) for r in pc if big(r, oce)]),
        "ocean_plate_max_cm_yr": round(max(pooled_o), 2),
        "cont_plate_median_cm_yr": q([np.median(big(r, con)) for r in pc if big(r, con)]),
        "cont_plate_max_cm_yr": round(max(pooled_c), 2),
        "trench_abs_mean_cm_yr": q([r["trench_abs_mean_cm_yr"] for r in eg]),
        "trench_retreat_frac": q([r["trench_retreat_frac"] for r in eg]),
        "trench_fast_advance_frac": q([r["trench_fast_advance_frac"] for r in eg]),
    }
    (HERE / "results/earth_reference.json").write_text(json.dumps(ref, indent=1, ensure_ascii=False))
    print(json.dumps(ref, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
