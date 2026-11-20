"""Earth test of mantle_flow.py (Lithgow-Bertelloni & Richards 1998 plate motions from slabs), PIPELINE principle 5.

The slabs come from the Earth plate model's own subduction history (Müller et al. 2022, mantle frame): every 5 Myr, at
points along every subduction zone (the model's labels and polarity), the convergence rate of the sinking plate towards
the overriding one and the age of the sinking plate (the age grids of earth_grid_checks.py up to 200 Ma, 90 Myr before
that as LBR98 assumed for the Mesozoic). At each test time the plates' Euler vectors are predicted from these slabs
alone and compared with the model's motions, both without net rotation:
  - global correlation of the velocity fields (LBR98: 0.7-0.9 for the Cenozoic)
  - the scale that matches the speeds (gives the upper-mantle viscosity; LBR98: 4-6 x 10^20 Pa s)
  - per plate (>= 1 % of the surface): direction cosine and RMS speed, separately for continental (> 50 %) plates
Usage: .venv/bin/python planet-history-sim/tools/earth_mantle_flow_check.py [times_Ma comma-separated] [n_points]   (Kaggle: HC)
"""
import sys, json, math, time
import numpy as np
import pygplates
from fastkd import cKDTree
from plate_checks import MODEL, TOPOLOGY_FILES, ROTATION, HERE
from plate_engine import fibonacci_xyz
from plate_checks import partition_ids
import mantle_flow as mf

R = 6371.0
TESS_DEG = 0.5            # trench points every 0.5 deg (~55 km; numerical, below the 87 km grid spacing)
SIDE_KM = 100.0           # offset across the trench to tell the overriding from the sinking side (numerical)
AGE_STEP = 10             # age grids every 10 Myr (earth_age_series.npz)
RESIST_GRID = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]   # search range of the slab-resistance coefficient
                          # (dimensionless; fitted on Earth 2026-11-03: c = 3, PARAM_AUDIT). Scan only with SCAN=1.
RESIST_C = 3.0
# Pangea's breakup (2026-11-04): the global correlation is dominated by the large fast oceanic plates and says nothing
# about whether neighbouring continents separate. Conjugate continental pairs of the breakup (Müller et al. 2022 plate
# ids; the same kind of comparison as earth_breakup_stats.py, but its own 8 plate pairs, not that script's 8 conjugate margins): rate of change of the distance between their centroids, predicted
# vs the model's, at each test time (> 0: moving apart)
CONJ_PAIRS = [(101, 701, "N America - Africa"), (201, 701, "S America - Africa"), (101, 301, "N America - Eurasia"),
              (501, 802, "India - Antarctica"), (801, 802, "Australia - Antarctica"), (701, 802, "Africa - Antarctica"),
              (702, 701, "Madagascar - Africa"), (501, 701, "India - Africa")]


def separation_rate(w, plate, xyz, a, b):
    """cm/yr, d/dt of the angle between the centroids of plates a and b, times the radius (> 0 apart)"""
    ma, mb = plate == a, plate == b
    if not ma.any() or not mb.any() or a not in w or b not in w:
        return None
    ca = xyz[ma].mean(0); ca /= np.linalg.norm(ca); cb = xyz[mb].mean(0); cb /= np.linalg.norm(cb)
    d = math.acos(np.clip(ca @ cb, -1, 1))
    va, vb = np.cross(w[a], ca), np.cross(w[b], cb)
    return float(-(va @ cb + ca @ vb) / max(math.sin(d), 1e-9) * R / 10.0)


def euler(rot, t, p):
    try:
        e = rot.get_rotation(float(t), int(p), float(t) + 1.0)
        if e.represents_identity_rotation():
            return np.zeros(3)
        pole, ang = e.get_euler_pole_and_angle()
        return np.array(pole.to_xyz()) * ang
    except Exception:
        return np.zeros(3)


def offset(x, n, km):
    a = km / R
    return math.cos(a) * x + math.sin(a) * n


def slablets(rot, topo, t_lo, t_hi, ages, gtree):
    rows = []
    for tau in np.arange(t_hi, t_lo - 1e-9, -mf.STEP_MYR) if t_hi > t_lo else [float(t_lo)]:
        rt, ss = [], []
        pygplates.resolve_topologies(topo, rot, rt, float(tau), ss)
        part = pygplates.PlatePartitioner(topo, rot, float(tau))
        W = {}
        akey = f"t{int(round(tau / AGE_STEP) * AGE_STEP)}"
        age_map = ages[akey].astype(float) if akey in ages.files else None
        n0 = len(rows)
        for s in ss:
            if s.get_feature().get_feature_type() != pygplates.FeatureType.gpml_subduction_zone:
                continue
            for sub in s.get_shared_sub_segments():
                os_ = sub.get_overriding_and_subducting_plates()
                if os_ is None:
                    continue
                ov, su = (int(x.get_feature().get_reconstruction_plate_id()) for x in os_)
                for p in (ov, su):
                    if p not in W:
                        W[p] = euler(rot, tau, p)
                pts = np.array([q.to_xyz() for q in sub.get_resolved_geometry().to_tessellated(math.radians(TESS_DEG)).get_points()])
                if len(pts) < 2:
                    continue
                for a, b in zip(pts[:-1], pts[1:]):
                    x = a + b; x /= np.linalg.norm(x)
                    ds = math.acos(np.clip(a @ b, -1, 1)) * R
                    n = np.cross(x, b - a)
                    if ds <= 0 or np.linalg.norm(n) == 0:
                        continue
                    n /= np.linalg.norm(n)
                    f = part.partition_point(pygplates.PointOnSphere(*offset(x, n, SIDE_KM)))
                    pid = f.get_feature().get_reconstruction_plate_id() if f is not None else None
                    if pid == ov:
                        pass
                    elif pid == su:
                        n = -n
                    else:
                        continue
                    vc = float((np.cross(W[su], x) - np.cross(W[ov], x)) @ n) * R / 10.0      # cm/yr, > 0 closing
                    if vc <= 0:
                        continue
                    if age_map is not None:
                        ag = age_map[gtree.query(offset(x, -n, SIDE_KM))[1]]
                        if not np.isfinite(ag):
                            continue                      # continental crust on the sinking side: no oceanic slab
                    else:
                        ag = mf.H_SLAB_AGE
                    H = mf.H_SLAB_KM * math.sqrt(max(ag, 0.0) / mf.H_SLAB_AGE)
                    rows.append((tau, *x, vc, H, ds, *n, su))
        print(f"  slabs {tau:5.0f} Ma: {len(rows) - n0} trench points", flush=True)
    return np.array(rows)


def nr_vector(xyz, v):
    """rotation that best fits the velocity field (net rotation), rad/Myr; v in cm/yr."""
    u = v / (R / 10.0)
    A = len(xyz) * np.eye(3) - xyz.T @ xyz
    return np.linalg.solve(A, np.cross(xyz, u).sum(axis=0))


def main():
    # 160-200 Ma: Pangea's breakup (the test of the deep upwellings)
    times = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 180, 200]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else int(round(60000 * (R / 6000.0) ** 2))
    t0 = time.time()
    rot = pygplates.RotationModel(str(ROTATION))
    topo = [pygplates.FeatureCollection(str(MODEL / "Topologies" / f)) for f in TOPOLOGY_FILES]
    conts = pygplates.FeatureCollection(str(MODEL / "Continents/continents.gpml"))
    ages = np.load(HERE / "results/earth_age_series.npz")
    g_age = fibonacci_xyz(int(round(60000 * (R / 6000.0) ** 2)))
    S = slablets(rot, topo, min(times) + mf.STEP_MYR, max(times) + mf.HISTORY_MYR, ages, cKDTree(g_age))
    print(f"slablets: {len(S)} ({time.time() - t0:.0f}s)", flush=True)
    xyz = fibonacci_xyz(n)
    sh = mf.SH(xyz)
    hc = mf.HC(sh)
    print(f"harmonics ready, HC built ({time.time() - t0:.0f}s)", flush=True)
    depths = np.arange(mf.DENS_LAYER_KM / 2, mf.EARTH.zcmb, mf.DENS_LAYER_KM)
    llsvp = mf.tomo_llsvp_density(sh, depths, hc.bin)          # kg/m3, fixed in the mantle frame (LLSVPs stable)
    print(f"LLSVP density: min {llsvp.min():.1f} kg/m3, layers with anomalies {int((llsvp < 0).any(axis=1).sum())}", flush=True)
    rows = []
    for t in times:
        t1 = time.time()
        plate = partition_ids(pygplates.PlatePartitioner(topo, rot, float(t)), xyz)
        cont = partition_ids(pygplates.PlatePartitioner(conts, rot, float(t)), xyz) >= 0
        hc.cont_mask = cont; hc._M = None                         # KEEL_K (mantle_flow.HC.torques)
        sl, z, mass = mf.slab_state(S, float(t))
        trenches = slablets(rot, topo, float(t), float(t), ages, cKDTree(g_age))
        pull, att = mf.attached_pull(sl, z, mass, trenches)
        # ridge push variant (2026-11-06, mantle_flow.ridge_push_rows): Earth's ocean ages on this grid
        akey_ = f"t{int(round(t / AGE_STEP) * AGE_STEP)}"
        age_ = ages[akey_].astype(float)[cKDTree(g_age).query(xyz)[1]] if akey_ in ages.files else np.full(len(xyz), np.nan)
        nb_ = cKDTree(xyz).query(xyz, k=7)[1][:, 1:]
        rp_ = mf.ridge_push_rows(xyz, nb_, plate, age_, cont)
        pull_rp = np.r_[pull, rp_] if len(pull) else rp_
        Wo = np.array([euler(rot, t, p) for p in plate])
        vo = np.cross(Wo, xyz) * R / 10.0
        vo -= np.cross(nr_vector(xyz, vo), xyz) * R / 10.0
        row = {"t": t, "attached_slab_share_of_mass": round(float(mass[att].sum() / max(mass.sum(), 1e-30)), 3),
               "pull_total_N": float("%.3g" % pull[:, 7].sum()) if len(pull) else 0.0}
        import os
        grid_c = RESIST_GRID if os.environ.get("SCAN") == "1" else [RESIST_C]
        variants = [("suction_LBR98", np.ones(len(sl), bool), None, 0.0, False)] + \
                   [(f"pull_CLB02_resist{c:g}", ~att, pull, c, False) for c in grid_c] + \
                   [(f"pull_CLB02_resist{RESIST_C:g}_llsvp", ~att, pull, RESIST_C, True)] + \
                   [(f"pull_CLB02_resist{RESIST_C:g}_ridgepush", ~att, pull_rp, RESIST_C, False)]
        for name, drive_mask, pl, c, with_llsvp in variants:
            dens = mf.slab_density(xyz, sl[drive_mask], z[drive_mask], mass[drive_mask], depths)
            if with_llsvp:
                dens = dens + llsvp
            w_pred, info = hc.solve_plates(xyz, plate, depths, dens * 100.0 / mf.HC_AVG_DEN, pull=pl, slab_resist=c)
            Wp = np.array([w_pred[int(p)] for p in plate])
            w_obs = {int(p): euler(rot, t, p) for p in np.unique(plate)}
            vp = np.cross(Wp, xyz) * R / 10.0
            vp -= np.cross(nr_vector(xyz, vp), xyz) * R / 10.0
            corr = float((vp * vo).sum() / math.sqrt((vp ** 2).sum() * (vo ** 2).sum()))
            scale = float((vp * vo).sum() / (vp ** 2).sum())
            vps = vp * scale
            per = []
            for p in sorted(w_pred):
                m = plate == p
                if m.mean() < 0.01:
                    continue
                a, b = vps[m], vo[m]
                per.append({"plate": p, "area": round(float(m.mean()), 4), "cont": round(float(cont[m].mean()), 3),
                            "slab": bool(len(pull) and (pull[:, 6] == p).any()),
                            "rms_pred": round(float(np.sqrt((a ** 2).sum(1).mean())), 2),
                            "rms_obs": round(float(np.sqrt((b ** 2).sum(1).mean())), 2),
                            "cos": round(float((a * b).sum() / math.sqrt((a ** 2).sum() * (b ** 2).sum() + 1e-30)), 3)})
            pairs = []
            for a, b, lab in CONJ_PAIRS:
                so = separation_rate(w_obs, plate, xyz, a, b)
                sp_ = separation_rate({k: v * scale for k, v in w_pred.items()}, plate, xyz, a, b)
                if so is not None and sp_ is not None:
                    pairs.append((lab, round(sp_, 2), round(so, 2)))
            print(f"   {t} Ma {name} separation (pred, obs) cm/yr: " + "; ".join(f"{l}: {p:+.1f}/{o:+.1f}" for l, p, o in pairs), flush=True)
            if pl is not None:
                # split of the predicted separation into the slab-driven flow (suction) and the attached-slab pull,
                # the same linear split as planet_flow._rift_budget (x = -Minv (q_int - sgn q_pull))
                op = hc.last_op
                xs = -(op["Minv"] @ op["q_int"]) * op["om"]; xp = op["sgn"] * (op["Minv"] @ op["q_pull"]) * op["om"]
                split = []
                for comp, x_ in (("suction", xs), ("pull", xp)):
                    wc = {p: x_[3 * j:3 * j + 3] * scale for j, p in enumerate(op["ids"])}
                    for a_, b_ in op["merged"].items():
                        wc[a_] = wc[b_]
                    split.append((comp, {lab: separation_rate(wc, plate, xyz, a, b) for a, b, lab in CONJ_PAIRS}))
                print(f"   {t} Ma {name} separation split cm/yr: " + "; ".join(
                    f"{lab}: " + ", ".join(f"{c_} {d_[lab]:+.2f}" for c_, d_ in split if d_.get(lab) is not None)
                    for _, _, lab in CONJ_PAIRS), flush=True)
                row[name + "_split"] = {c_: d_ for c_, d_ in split}
            row[name] = {"global_corr": round(corr, 3), "speed_scale": round(scale, 3), "pairs": pairs,
                         "upper_mantle_viscosity_Pa_s": float("%.3g" % (1e21 / scale)) if scale > 0 else None,
                         "plates": per, "svd": info}
            print(t, "Ma", name, ": corr", round(corr, 3), "scale", round(scale, 3), info["torque_pull_Nm"], info["torque_suction_Nm"], "sign", info["hc_traction_sign"],
                  f"({time.time() - t1:.0f}s)", flush=True)
            for q in per:
                print("    ", q, flush=True)
            sp = sorted(((float(np.sqrt(((vps[plate == p]) ** 2).sum(1).mean())), float((plate == p).mean()), p) for p in w_pred), reverse=True)[:4]
            print("     fastest (any size):", [(round(a, 1), round(b * 100, 3), c) for a, b, c in sp], "merged:", info.get("merged"), flush=True)
            row[name]["fastest_any_size"] = sp
        rows.append(row)
    out = {"source": "Müller et al. 2022 slabs and plates; LBR98 / CLB02 methods via HC " + mf.HC_COMMIT[:8], "rows": rows}
    for model in [k for k in rows[0] if isinstance(rows[0][k], dict) and not k.endswith("_split")]:
        P = [q for r in rows for q in r[model]["plates"]]
        o = {"global_corr_median": float(np.median([r[model]["global_corr"] for r in rows])),
             "viscosity_median": float(np.median([r[model]["upper_mantle_viscosity_Pa_s"] or np.nan for r in rows]))}
        for name, sel in (("all_plates", lambda q: True), ("continental_gt50", lambda q: q["cont"] > 0.5),
                          ("with_slab", lambda q: q["slab"]), ("no_slab", lambda q: not q["slab"])):
            L = [(q["rms_pred"], q["rms_obs"]) for q in P if sel(q)]
            if len(L) > 2:
                a = np.log(np.array(L) + 0.1)
                o[name] = {"n": len(L), "corr_log_speed": round(float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]), 3),
                           "pred_median_max": [round(float(np.median([x for x, _ in L])), 2), round(max(x for x, _ in L), 2)],
                           "obs_median_max": [round(float(np.median([y for _, y in L])), 2), round(max(y for _, y in L), 2)]}
        out[model] = o
    (HERE / ("results/earth_mantle_flow_check" + (f"_keel{mf.KEEL_K:g}" if mf.KEEL_K != 1.0 else "") + (f"_sink{mf.SINK_LM:g}" if mf.SINK_LM else "") + ".json")).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
