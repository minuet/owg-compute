"""Measure, in the Earth plate model (Müller et al. 2022), how Earth's OCEANIC plates move and meet, every 10 Myr from
200 Ma (Panthalassa) to now - the reference for the planet's oceanic plates (PIPELINE principle 0, tier 2), as
earth_breakup_stats.py is for the continents.

A plate is oceanic when < 10% of it is continental crust (continent polygons reconstructed to that time); only plates
covering >= 1% of the surface are listed. On a grid of the planet's cell size, with the model's own motions (mantle frame):
  - oceanic plates: number, area shares, total area, RMS speed
  - boundaries between two oceanic plates, by the MODEL'S OWN LABEL of each boundary section (mid-ocean ridge,
    transform, subduction zone, other) - no judgement of ours - and by the grid opening rate (> 0.5 / < -0.5 cm/yr)
  - full spreading rate of the ridges between oceanic plates (length-weighted quantiles)
  - per oceanic plate: share of its outline that is a trench where it sinks (slab attached), and its speed
    (Forsyth & Uyeda 1975: plates with more slab move faster)
  - relative motion of oceanic plate pairs across a ridge, from one 10 Myr step to the next: turn of the relative pole
    and change of the spreading rate (like POLE_TURN_SIGMA / STAGE_RATE_SIGMA for the continents); distance of the
    relative pole from the ridge centre (like POLE_DIST)
  - motion towards the slab: for each oceanic plate with subduction zones where it sinks (the model's own subduction
    zones and polarity), cosine between its velocity at its centre and the direction from its centre to its trenches,
    and the share of its trench points where it moves towards the trench (Forsyth & Uyeda 1975: slabs pull plates)
  - births: oceanic plates reaching >= 1% that were not there (>= 1%) 10 Myr earlier
  - orthogonal convergence rate where ocean sinks (length-weighted quantiles)
  - Panthalassa at 190 Ma: where its large oceanic plates meet (triple junction cells) and the angle from there to the
    centre of the continental crust (Pangea)
Usage: .venv/bin/python planet-history-sim/tools/earth_ocean_plate_stats.py  ->  results/earth_ocean_plate_stats.json
"""
import json, math
import numpy as np
import pygplates
from scipy.spatial import cKDTree
from plate_checks import MODEL, TOPOLOGY_FILES, ROTATION, HERE
from plate_engine import fibonacci_xyz
from planet_checks import boundary_pairs

R = 6371.0
N = int(round(60000 * (R / 6000.0) ** 2))
STEP = 10
OCEANIC_MAX_CONT = 0.1      # same threshold as the archived force model (archive/force_model/code/tools/plate_forces.py)
MIN_SHARE = 0.01


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def partition_ids(partitioner, xyz):
    out = np.full(len(xyz), -1, np.int64)
    for i, (x, y, z) in enumerate(xyz):
        f = partitioner.partition_point(pygplates.PointOnSphere(x, y, z))
        if f is not None:
            out[i] = f.get_feature().get_reconstruction_plate_id()
    return out


def euler(rot, t, p, ref=0):
    try:
        e = rot.get_rotation(float(t), int(p), float(t) + 1.0, int(ref))
        if e.represents_identity_rotation():
            return np.zeros(3)
        pole, ang = e.get_euler_pole_and_angle()
        return np.array(pole.to_xyz()) * ang                 # rad/Myr
    except Exception:
        return np.zeros(3)


def wq(x, w, qs):
    o = np.argsort(x); c = np.cumsum(w[o]) / w.sum()
    return [float(np.interp(q, c, x[o])) for q in qs]


def main():
    rot = pygplates.RotationModel(str(ROTATION))
    topo = [pygplates.FeatureCollection(str(MODEL / "Topologies" / f)) for f in TOPOLOGY_FILES]
    conts = pygplates.FeatureCollection(str(MODEL / "Continents/continents.gpml"))
    g = fibonacci_xyz(N)
    km_pair = 2 * math.pi * R / len(boundary_pairs(g, (g[:, 2] > 0).astype(int))[0])
    QS = [0.1, 0.25, 0.5, 0.75, 0.9]
    rows, prev_rel = [], {}
    turns, dlog, pdist, conv_rates, births, prev_big = [], [], [], [], [], None
    speeds_all, abs_turns, prev_w = [], [], {}
    slab_cos, slab_towards = [], []
    all_rates, all_rates_w = [], []
    for t in range(200, -1, -STEP):
        plate = partition_ids(pygplates.PlatePartitioner(topo, rot, float(t)), g)
        cont = partition_ids(pygplates.PlatePartitioner(conts, rot, float(t)), g) >= 0
        ids, inv, cnt = np.unique(plate, return_inverse=True, return_counts=True); inv = inv.ravel()
        cf = np.bincount(inv, weights=cont.astype(float)) / cnt
        oceanic = {int(p) for p, c in zip(ids, cf) if c < OCEANIC_MAX_CONT}
        big = [int(p) for p, c, n_ in zip(ids, cf, cnt) if c < OCEANIC_MAX_CONT and n_ / N >= MIN_SHARE]
        w = {int(p): euler(rot, t, p) for p in ids}
        W = np.array([w[int(p)] for p in plate])
        sp = {p: float(np.sqrt(((np.linalg.norm(np.cross(w[p], g[plate == p]), axis=1) * R / 10) ** 2).mean())) for p in big}
        speeds_all += list(sp.values())
        for p in big:                                   # turn of the absolute (mantle-frame) pole of each oceanic plate
            if p in prev_w and np.linalg.norm(prev_w[p]) > 0 and np.linalg.norm(w[p]) > 0:
                abs_turns.append(math.degrees(math.acos(np.clip(prev_w[p] @ w[p] / np.linalg.norm(prev_w[p]) / np.linalg.norm(w[p]), -1, 1))))
        prev_w = {p: w[p] for p in big}
        # boundaries by the model's own labels
        rt, ss = [], []
        pygplates.resolve_topologies(topo, rot, rt, float(t), ss)
        lab = {}
        trench_pts = {}
        for s in ss:
            if s.get_feature().get_feature_type() == pygplates.FeatureType.gpml_subduction_zone:
                for sub in s.get_shared_sub_segments():
                    r_ = sub.get_overriding_and_subducting_plates()
                    if r_ is not None:
                        sp_ = int(r_[1].get_feature().get_reconstruction_plate_id())
                        trench_pts.setdefault(sp_, []).extend([pt.to_xyz() for pt in sub.get_resolved_geometry().get_points()])
        for s in ss:
            ft = s.get_feature().get_feature_type().get_name()
            kind = {"MidOceanRidge": "ridge", "SubductionZone": "subduction", "Transform": "transform"}.get(ft, "other")
            for sub in s.get_shared_sub_segments():
                sh = [int(r.get_feature().get_reconstruction_plate_id()) for r in sub.get_sharing_resolved_topologies()]
                if len(sh) == 2 and sh[0] in oceanic and sh[1] in oceanic:
                    lab[kind] = lab.get(kind, 0.0) + sub.get_resolved_geometry().get_arc_length() * R
        # grid: boundaries between two oceanic plates with ocean on both sides
        i, j = boundary_pairs(g, plate)
        oo = np.array([int(a) in oceanic and int(b) in oceanic for a, b in zip(plate[i], plate[j])], bool) & ~cont[i] & ~cont[j]
        mid = unit(g[i] + g[j]); nrm = unit(g[j] - g[i])
        op = np.einsum("ij,ij->i", (np.cross(W[j], mid) - np.cross(W[i], mid)) * R / 10, nrm)
        ridge = oo & (op > 0.5); conv = oo & (op < -0.5)
        all_rates.append(op[ridge]); all_rates_w.append(np.ones(ridge.sum()))
        # slab attached: cells of plate p on a boundary converging > 0.5 cm/yr where p is the sinking side
        # (ocean under continent, else the older side - use the model's subduction labels instead where possible:
        # here the grid version, same rule as planet_checks)
        attach = {}
        for p in big:
            on = (plate[i] == p) | (plate[j] == p)
            p_is_i = plate[i] == p
            p_ocean = np.where(p_is_i, ~cont[i], ~cont[j]); other_cont = np.where(p_is_i, cont[j], cont[i])
            c_ = on & (op < -0.5) & p_ocean
            attach[p] = (float((c_ & other_cont).sum() / max(on.sum(), 1)), float(c_.sum() / max(on.sum(), 1)))
        for p in big:
            if p not in trench_pts or np.linalg.norm(w[p]) == 0:
                continue
            T = np.array(trench_pts[p]); c = unit(g[plate == p].mean(axis=0))
            v = np.cross(w[p], c); d = unit(T.mean(axis=0)); d = d - (d @ c) * c
            if np.linalg.norm(d) > 0 and np.linalg.norm(v) > 0:
                slab_cos.append(float(v @ d / np.linalg.norm(v) / np.linalg.norm(d)))
            vt = np.cross(w[p], T); out_ = T - c * (T @ c)[:, None]
            slab_towards.append(float((np.einsum("ij,ij->i", vt, out_) > 0).mean()))
        # relative motion across oceanic ridges, compared with the same pair 10 Myr earlier
        rel = {}
        for a, b in set(zip(plate[i][ridge].tolist(), plate[j][ridge].tolist())):
            key = (min(a, b), max(a, b))
            m = ridge & (((plate[i] == key[0]) & (plate[j] == key[1])) | ((plate[i] == key[1]) & (plate[j] == key[0])))
            if m.sum() * km_pair < 500:
                continue
            rel[key] = (w[key[0]] - w[key[1]], float(op[m].mean()))
            v_ = rel[key][0]
            if np.linalg.norm(v_) > 0:
                ctr = unit(mid[m].mean(axis=0)); pole = v_ / np.linalg.norm(v_)
                pdist.append(math.degrees(math.acos(min(abs(float(pole @ ctr)), 1.0))))
            if key in prev_rel:
                v0, r0 = prev_rel[key]; v1, r1 = rel[key]
                if np.linalg.norm(v0) > 0 and np.linalg.norm(v1) > 0:
                    turns.append(math.degrees(math.acos(np.clip(v0 @ v1 / np.linalg.norm(v0) / np.linalg.norm(v1), -1, 1))))
                    if r0 > 0 and r1 > 0:
                        dlog.append(math.log(r1 / r0))
        prev_rel = rel
        sink_ocean = (op < -0.5) & ~(cont[i] & cont[j]) & (~cont[i] | ~cont[j])
        conv_rates.append(-op[sink_ocean])
        if prev_big is not None:
            births.append(len(set(big) - prev_big))
        prev_big = set(big)
        if t == 190:
            bigs = [p for p in big if (plate == p).mean() > 0.1]
            nb = cKDTree(g).query(g, k=7)[1]
            tj = np.array([all((plate[nb[c]] == p).any() for p in bigs) for c in range(N)]) if len(bigs) >= 3 else np.zeros(N, bool)
            cc = unit(g[cont].mean(axis=0))
            if tj.any():
                tjc = unit(g[tj].mean(axis=0))
                panth = {"big_plates": bigs, "tj_cells": int(tj.sum()),
                         "angle_tj_to_continent_centre_deg": round(math.degrees(math.acos(float(np.clip(tjc @ cc, -1, 1)))), 1),
                         "angle_tj_to_nearest_continent_deg": round(math.degrees(2 * math.asin(min(cKDTree(g[cont]).query(tjc)[0] / 2, 1))), 1)}
                print("Panthalassa 190 Ma:", panth, flush=True)
        row = {"ma": t, "n_oceanic_plates": len(big), "oceanic_area_shares": sorted([round(float((plate == p).mean()), 3) for p in big], reverse=True),
               "oceanic_plates_total_share": round(float(np.isin(plate, list(oceanic)).mean()), 3),
               "speed_cm_yr": {p: round(v, 1) for p, v in sp.items()},
               "oo_model_labels_km": {k: round(v) for k, v in lab.items()},
               "oo_grid_km": {"ridge": round(ridge.sum() * km_pair), "convergent": round(conv.sum() * km_pair),
                              "other": round((oo & ~ridge & ~conv).sum() * km_pair)},
               "ridge_full_rate_cm_yr_q10_25_50_75_90": [round(x, 2) for x in wq(op[ridge], np.ones(ridge.sum()), QS)] if ridge.any() else None,
               "slab_attached": {p: {"under_continent": round(a[0], 2), "all": round(a[1], 2), "speed": round(sp[p], 1)} for p, a in attach.items()}}
        rows.append(row)
        print(t, "Ma:", row["n_oceanic_plates"], "oceanic plates", row["oceanic_area_shares"], "| oo labels", row["oo_model_labels_km"],
              "| grid", row["oo_grid_km"], "| ridge rate", row["ridge_full_rate_cm_yr_q10_25_50_75_90"], flush=True)
    r = np.concatenate(all_rates); cr = np.concatenate(conv_rates)
    out = {"source": "Müller et al. 2022 (mantle frame), cells of the planet's size, every 10 Myr 200-0 Ma",
           "oceanic_plate_def": f"< {OCEANIC_MAX_CONT:.0%} continental crust, >= {MIN_SHARE:.0%} of the surface",
           "steps": rows,
           "ridge_full_rate_all_steps_q10_25_50_75_90": [round(x, 2) for x in np.quantile(r, QS)],
           "ridge_pair_pole_turn_deg_per_10Myr_q25_50_75_90": [round(float(x), 1) for x in np.quantile(turns, [0.25, 0.5, 0.75, 0.9])] if turns else None,
           "ridge_pair_rate_change_lnsd_per_10Myr": round(float(np.std(dlog)), 3) if dlog else None, "n_pair_steps": len(turns),
           "ridge_pole_dist_from_ridge_centre_deg_q10_25_50_75_90": [round(float(x), 1) for x in np.quantile(pdist, QS)],
           "convergence_orthogonal_cm_yr_q10_25_50_75_90": [round(float(x), 2) for x in np.quantile(cr, QS)],
           "births_per_10Myr": births, "births_total": int(sum(births)),
           "panthalassa_190Ma": locals().get("panth")}
    out["slab_direction_cos_q10_25_50_75_90"] = [round(float(x), 3) for x in np.quantile(slab_cos, QS)]
    out["slab_towards_share_q10_25_50_75_90"] = [round(float(x), 3) for x in np.quantile(slab_towards, QS)]
    out["n_slab_plates_steps"] = len(slab_cos)
    print("towards slab: cos", out["slab_direction_cos_q10_25_50_75_90"], "share of trench points approached",
          out["slab_towards_share_q10_25_50_75_90"], "n", len(slab_cos), flush=True)
    # full distributions (quantiles at 0.02 ... 0.98) for drawing values on the planet
    FQ = [round(x, 2) for x in np.arange(0.02, 0.981, 0.02)]
    out["dist_quantile_levels"] = FQ
    out["dist"] = {"ridge_full_rate_cm_yr": [round(float(x), 3) for x in np.quantile(r, FQ)],
                   "ridge_pole_turn_deg_per_10Myr": [round(float(x), 2) for x in np.quantile(turns, FQ)],
                   "ridge_pole_dist_deg": [round(float(x), 2) for x in np.quantile(pdist, FQ)],
                   "convergence_orthogonal_cm_yr": [round(float(x), 3) for x in np.quantile(cr, FQ)],
                   "oceanic_plate_rms_speed_cm_yr": [round(float(x), 3) for x in np.quantile(speeds_all, FQ)],
                   "oceanic_plate_abs_pole_turn_deg_per_10Myr": [round(float(x), 2) for x in np.quantile(abs_turns, FQ)],
                   "slab_direction_cos": [round(float(x), 3) for x in np.quantile(slab_cos, FQ)],
                   "slab_towards_share": [round(float(x), 3) for x in np.quantile(slab_towards, FQ)]}
    print("all steps: ridge full rate", out["ridge_full_rate_all_steps_q10_25_50_75_90"], "pole turn", out["ridge_pair_pole_turn_deg_per_10Myr_q25_50_75_90"],
          "rate ln-sd", out["ridge_pair_rate_change_lnsd_per_10Myr"], "n", len(turns), "pole dist", out["ridge_pole_dist_from_ridge_centre_deg_q10_25_50_75_90"],
          "convergence", out["convergence_orthogonal_cm_yr_q10_25_50_75_90"], "births", births, flush=True)
    (HERE / "results/earth_ocean_plate_stats.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
