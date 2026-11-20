"""Measure, in the Earth plate model (Müller et al. 2022, v1.2.2), how the pieces of Pangea moved apart after breakup:
relative opening speed at each conjugate margin, distance of the relative stage pole from the margin, how much the
stage pole turns from one 10 Myr stage to the next, and the absolute (mantle-frame) speed of Africa, the piece left over
the upwelling. These measured values set the rule-based continental motions of the planet generator (PARAM_AUDIT.md).

Usage: .venv/bin/python planet-history-sim/tools/earth_breakup_stats.py   -> results/earth_breakup_stats.json
"""
import json, math
import numpy as np
import pygplates
from plate_checks import HERE, MODEL, ROTATION

R = 6371.0
STEP = 10
# (name, moving plate, fixed plate, margin point on the fixed plate (present lat, lon), breakup age Ma in the model)
PAIRS = [
    ("Central Atlantic (N America / NW Africa)", 101, 714, (25.0, -16.5), 190),
    ("E/W Gondwana (Antarctica / Africa)", 802, 701, (-22.0, 36.0), 180),
    ("India / Antarctica", 501, 802, (-67.0, 75.0), 130),
    ("South Atlantic (S America / Africa)", 201, 701, (-15.0, 12.0), 130),
    ("Madagascar / India", 501, 702, (-18.0, 49.5), 90),
    ("Australia / Antarctica", 801, 802, (-66.0, 130.0), 80),
    ("NE Atlantic (Greenland / Eurasia)", 102, 301, (65.0, 5.0), 55),
    ("Red Sea (Arabia / NE Africa)", 503, 715, (20.0, 38.5), 25),
]


def xyz(lat, lon):
    la, lo = math.radians(lat), math.radians(lon)
    return np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])


def stage_vector(rot, moving, fixed, t_old, t_young):
    """Euler vector (rad/Myr) of moving relative to fixed from t_old to t_young, in the fixed plate's present frame."""
    s = rot.get_rotation(t_young, moving, t_old, fixed)
    if s.represents_identity_rotation():
        return np.zeros(3)
    pole, ang = s.get_euler_pole_and_angle()
    return np.array(pole.to_xyz()) * ang / (t_old - t_young)


def main():
    rel = pygplates.RotationModel(str(MODEL / "Rotations/1000_0_rotfile.rot"))
    absr = pygplates.RotationModel(str(ROTATION))
    out = {"source": "Müller et al. 2022 plate model v1.2.2 (relative rotations; mantle-optimised frame for Africa)",
           "step_myr": STEP, "pairs": []}
    all_rates, early, turns, pole_d = [], [], [], []
    for name, mov, fix, (lat, lon), t0 in PAIRS:
        p = xyz(lat, lon)
        rows, prev = [], None
        for t in range(t0, 0, -STEP):
            w = stage_vector(rel, mov, fix, t, t - STEP)
            om = float(np.linalg.norm(w))
            if om == 0:
                continue
            d = math.degrees(math.acos(abs(float(w @ p)) / om))          # pole distance from the margin point
            v = om * R * math.sin(math.radians(d)) / 10.0               # cm/yr
            turn = None
            if prev is not None:
                turn = math.degrees(math.acos(np.clip(float(w @ prev) / om / np.linalg.norm(prev), -1, 1)))
            rows.append({"from": t, "to": t - STEP, "rate_cm_yr": round(v, 2), "deg_per_myr": round(math.degrees(om), 3),
                         "pole_dist_deg": round(d, 1), "pole_turn_deg": None if turn is None else round(turn, 1)})
            prev = w
            all_rates.append(v); pole_d.append(d)
            if t0 - t < 50:
                early.append(v)
            if turn is not None:
                turns.append(turn)
        r = [x["rate_cm_yr"] for x in rows]
        out["pairs"].append({"name": name, "moving": mov, "fixed": fix, "breakup_ma": t0, "stages": rows,
                             "rate_median": round(float(np.median(r)), 2), "rate_min": min(r), "rate_max": max(r)})
        print(f"{name:42s} rate median {np.median(r):4.1f} (min {min(r):.1f}, max {max(r):.1f}) cm/yr; "
              f"pole distance median {np.median([x['pole_dist_deg'] for x in rows]):.0f} deg; "
              f"turns {[x['pole_turn_deg'] for x in rows[1:]]}")
    afr = []
    for t in range(0, 200, STEP):
        w = stage_vector(absr, 701, 0, t + STEP, t)
        afr.append(round(float(np.linalg.norm(np.cross(w, xyz(0.0, 20.0)))) * R / 10.0, 2))
    q = lambda a, k: round(float(np.percentile(a, k)), 2)
    out["summary"] = {
        "opening_rate_cm_yr": {"p10": q(all_rates, 10), "median": q(all_rates, 50), "p90": q(all_rates, 90)},
        "opening_rate_first_50_myr": {"p10": q(early, 10), "median": q(early, 50), "p90": q(early, 90)},
        "pole_turn_per_10_myr_deg": {"median": q(turns, 50), "p75": q(turns, 75), "p90": q(turns, 90)},
        "pole_dist_from_margin_deg": {"p10": q(pole_d, 10), "median": q(pole_d, 50), "p90": q(pole_d, 90)},
        "africa_mantle_frame_speed_cm_yr": {"by_10_myr_from_0": afr, "median": q(afr, 50), "max": max(afr)},
    }
    print(json.dumps(out["summary"], indent=1))
    (HERE / "results/earth_breakup_stats.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
