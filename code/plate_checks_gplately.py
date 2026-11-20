"""Checks that use GPlately's published tools (RULES.md A4 cross-check, A8 trench migration).

Usage: .venv/bin/python planet-history-sim/tools/plate_checks_gplately.py [step_myr] [max_age]
"""
import json, math, sys, logging
from pathlib import Path
import numpy as np
import pygplates

logging.getLogger("gplately").setLevel(logging.CRITICAL)
import gplately

from plate_checks import MODEL, TOPOLOGY_FILES, ROTATION, HERE


def main():
    step = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    max_age = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    rot = pygplates.RotationModel(str(ROTATION))
    topo = pygplates.FeatureCollection()
    for f in TOPOLOGY_FILES:
        topo.add(pygplates.FeatureCollection(str(MODEL / "Topologies" / f)))
    model = gplately.PlateReconstruction(rot, topo)
    out = []
    for t in range(0, max_age + 1, step):
        sz = np.asarray(model.tessellate_subduction_zones(t, ignore_warnings=True))
        if sz.size == 0:
            continue
        # col4 trench absolute speed (cm/yr), col5 obliquity to trench normal (normal points towards overriding plate)
        # col6 arc length (deg). Positive = trench advancing towards overriding plate, negative = rolling back.
        # col4 already carries a sign (negative = moving towards the overriding plate, gplately.ptt.subduction_convergence),
        # so the orthogonal component is cos(obliquity) * |speed| (as ptt's own output_trench_absolute_velocity_components).
        # Fixed 2026-10-30: col4 * cos(col5) applied the direction twice and made every trench look like it rolls back.
        v_orth = np.abs(sz[:, 4]) * np.cos(np.radians(sz[:, 5]))
        w = sz[:, 6]
        order = np.argsort(v_orth)
        cum = np.cumsum(w[order]) / w.sum()
        median = float(v_orth[order][np.searchsorted(cum, 0.5)])
        within2 = float(w[np.abs(v_orth) <= 2.0].sum() / w.sum())
        prod, dest = model.crustal_production_destruction_rate(t)
        r = {"time": t, "trench_vorth_median_cm_yr": median, "trench_within_2cm_frac": within2,
             "trench_vorth_mean_cm_yr": float((v_orth * w).sum() / w.sum()),
             "trench_abs_mean_cm_yr": float((np.abs(v_orth) * w).sum() / w.sum()),
             "trench_retreat_frac": float(w[v_orth < 0].sum() / w.sum()),
             "trench_fast_advance_frac": float(w[v_orth > 3.0].sum() / w.sum()),
             "crust_prod_km2_yr": float(prod), "crust_dest_km2_yr": float(dest)}
        out.append(r)
        print(f"{t:5d} Ma  trench orth median {median:+.2f} mean {r['trench_vorth_mean_cm_yr']:+.2f} cm/yr  "
              f"|v| {r['trench_abs_mean_cm_yr']:.2f} retreat {r['trench_retreat_frac']*100:.0f}% fast-adv {r['trench_fast_advance_frac']*100:.0f}% within ±2: {within2*100:.0f}%  crust prod/dest {prod:.2f}/{dest:.2f} km2/yr", flush=True)
    name = "earth_check_results_gplately.json" if max_age >= 1000 else f"earth_check_results_gplately_0_{max_age}.json"
    (HERE / "results" / name).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
