"""RULES.md checks for a generated planet (grid state + Euler vectors), same quantities as the Earth checks.

Velocities use our planet's radius. Positive trench speed = trench moving towards the overriding plate (advance),
negative = rollback, the same convention as GPlately's tessellate_subduction_zones used for Earth.
"""
import json, math
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def boundary_pairs(grid, plate, k=7):
    nb = cKDTree(grid).query(grid, k=k)[1]
    i = np.repeat(np.arange(len(grid)), k - 1)
    j = nb[:, 1:].ravel()
    sel = (plate[i] != plate[j]) & (i < j)
    return i[sel], j[sel]


def check_state(grid, state, w, radius_km):
    """state: dict(plate, cont, birth) on the grid at time t (birth in Ma, t subtracted by the caller); w: plate -> Euler vector (rad/Myr)."""
    plate, cont, age = state["plate"], state["cont"], state["age"]
    W = np.array([w.get(int(p), np.zeros(3)) for p in plate])
    vel = np.cross(W, grid) * radius_km / 10.0                       # cm/yr, per grid point
    n = len(grid)
    res = {}

    # A7 net rotation (area-uniform grid)
    nr = 1.5 * np.cross(grid, vel * 10.0 / radius_km).mean(axis=0)  # rad/Myr
    res["net_rotation_deg_myr"] = float(np.degrees(np.linalg.norm(nr)))

    # A6 speeds per plate vs continental fraction; C4 plate sizes
    plates = []
    for p in np.unique(plate):
        sel = plate == p
        plates.append({"plate": int(p), "area_frac": float(sel.mean()), "cont_frac": float(cont[sel].mean()),
                       "rms_cm_yr": float(np.sqrt((np.linalg.norm(vel[sel], axis=1) ** 2).mean()))})
    res["plates"] = sorted(plates, key=lambda d: -d["area_frac"])
    res["global_rms_cm_yr"] = float(np.sqrt((np.linalg.norm(vel, axis=1) ** 2).mean()))

    # boundaries: A3 types, A4 crust made vs consumed, A8 trench motion
    i, j = boundary_pairs(grid, plate)
    mid = unit(grid[i] + grid[j])
    nrm = unit(grid[j] - grid[i])
    rel = (np.cross(W[j], mid) - np.cross(W[i], mid)) * radius_km / 10.0   # cm/yr, j relative to i
    opening = np.einsum("ij,ij->i", rel, nrm)
    sliding = np.einsum("ij,ij->i", rel, np.cross(mid, nrm))
    div = opening > 0.5
    conv = opening < -0.5
    res["boundary_share"] = {"divergent": float(div.mean()), "convergent": float(conv.mean()),
                             "transform_or_slow": float((~div & ~conv).mean())}
    made, eaten = float(opening[div].sum()), float(-opening[conv].sum())
    res["crust_made_over_consumed"] = made / eaten if eaten > 0 else float("inf")
    # trench: overriding side = continental, else the younger ocean (the same polarity rule as the model's slab pull,
    # plate_forces.solve); continent-continent contacts are collisions, not trenches, and are left out
    ci, cj = cont[i][conv], cont[j][conv]
    ai, aj = np.nan_to_num(age[i][conv], nan=1e6), np.nan_to_num(age[j][conv], nan=1e6)
    sz = ~(ci & cj)
    i_over = (ci | (~cj & (ai < aj)))[sz]
    ic, jc = i[conv][sz], j[conv][sz]
    over = np.where(i_over, ic, jc)
    sub = np.where(i_over, jc, ic)
    towards_over = unit(grid[over] - grid[sub])
    v_trench = np.einsum("ij,ij->i", vel[over], towards_over)
    has = len(v_trench) > 0
    res["trench_vorth_mean_cm_yr"] = float(v_trench.mean()) if has else float("nan")
    res["trench_abs_mean_cm_yr"] = float(np.abs(v_trench).mean()) if has else float("nan")
    res["trench_retreat_frac"] = float((v_trench < 0).mean()) if has else float("nan")
    res["trench_fast_advance_frac"] = float((v_trench > 3.0).mean()) if has else float("nan")
    res["trench_vorth_mean_cont_over_cm_yr"] = float(v_trench[cont[over]].mean()) if cont[over].any() else float("nan")
    res["trench_vorth_mean_ocean_over_cm_yr"] = float(v_trench[~cont[over]].mean()) if (~cont[over]).any() else float("nan")

    # A11 ocean ages
    oa = age[~cont & np.isfinite(age)]
    res["ocean_age_mean"] = float(oa.mean()) if len(oa) else float("nan")
    res["ocean_age_p99"] = float(np.percentile(oa, 99)) if len(oa) else float("nan")
    res["ocean_age_bins_20my"] = [float(((oa >= a) & (oa < a + 20)).mean()) for a in range(0, 200, 20)]
    res["continental_frac"] = float(cont.mean())
    # A2: every plate is one connected piece (detached parts of a plate are not allowed)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    nb = cKDTree(grid).query(grid, k=7)[1][:, 1:]
    ii = np.repeat(np.arange(n), 6); jj = nb.ravel(); same = plate[ii] == plate[jj]
    ncomp, label = connected_components(coo_matrix((np.ones(same.sum()), (ii[same], jj[same])), shape=(n, n)), directed=False)
    size = np.bincount(label, minlength=ncomp)
    comp_plate = np.zeros(ncomp, np.int64); comp_plate[label] = plate
    detached = 0
    for p in np.unique(plate):
        cs = np.nonzero(comp_plate == p)[0]
        detached += int(size[cs].sum() - size[cs].max())
    res["detached_frac"] = detached / n
    return res


def verdict(res, history):
    """Pass/fail against RULES.md thresholds (Earth-derived; see RULES.md for sources)."""
    ok = {}
    # bands measured on the Earth plate model over the same phase (0-200 Ma after Pangea): earth_reference.py
    E = json.loads((Path(__file__).parent / "results/earth_reference.json").read_text())
    inb = lambda x, b: b[0] <= x <= b[1]
    band = lambda b: f"{b[0]:g}~{b[1]:g}"
    nrs = [h["net_rotation_deg_myr"] for h in history]
    ok[f"A7 net rotation median {band(E['net_rotation_deg_myr'])}, max <= {E['net_rotation_max']:g} deg/Myr (Earth model)"] = (
        inb(np.median(nrs), E["net_rotation_deg_myr"]) and max(nrs) <= E["net_rotation_max"],
        f"median {np.median(nrs):.3f}, max {max(nrs):.3f}")
    ratios = [h["crust_made_over_consumed"] for h in history]
    ok["A4 crust made / consumed in 0.9~1.3"] = (all(0.9 <= r <= 1.3 for r in ratios),
                                                 f"min {min(ratios):.2f}, max {max(ratios):.2f}")
    # plate classes as Zahirovic et al. 2015: < 25% continental vs > 50% continental, plates >= 1% of the surface
    big = lambda h, f: [p["rms_cm_yr"] for p in h["plates"] if p["area_frac"] >= 0.01 and f(p["cont_frac"])]
    om = [np.median(big(h, lambda c: c < 0.25)) for h in history if big(h, lambda c: c < 0.25)]
    cm = [np.median(big(h, lambda c: c > 0.5)) for h in history if big(h, lambda c: c > 0.5)]
    omax = max((v for h in history for v in big(h, lambda c: c < 0.25)), default=np.nan)
    cmax = max((v for h in history for v in big(h, lambda c: c > 0.5)), default=np.nan)
    ok[f"A6 ocean plates median {band(E['ocean_plate_median_cm_yr'])}, max <= {E['ocean_plate_max_cm_yr']:g} cm/yr (Earth model)"] = (
        bool(om) and inb(np.median(om), E["ocean_plate_median_cm_yr"]) and omax <= E["ocean_plate_max_cm_yr"],
        f"median {np.median(om):.1f}, max {omax:.1f}" if om else "none")
    ok[f"A6 continental plates median {band(E['cont_plate_median_cm_yr'])}, max <= {E['cont_plate_max_cm_yr']:g} cm/yr (Earth model)"] = (
        bool(cm) and inb(np.median(cm), E["cont_plate_median_cm_yr"]) and cmax <= E["cont_plate_max_cm_yr"],
        f"median {np.median(cm):.1f}, max {cmax:.1f}" if cm else "none")
    med = lambda k: float(np.nanmedian([h[k] for h in history]))
    ok[f"A8 trenches: mean |v| {band(E['trench_abs_mean_cm_yr'])} cm/yr, retreating {band(E['trench_retreat_frac'])}, "
       f"advancing > 3 cm/yr {band(E['trench_fast_advance_frac'])} (Earth model)"] = (
        inb(med("trench_abs_mean_cm_yr"), E["trench_abs_mean_cm_yr"]) and inb(med("trench_retreat_frac"), E["trench_retreat_frac"])
        and med("trench_fast_advance_frac") <= E["trench_fast_advance_frac"][1],
        f"mean |v| {med('trench_abs_mean_cm_yr'):.1f}, retreat {med('trench_retreat_frac')*100:.0f}%, fast advance "
        f"{med('trench_fast_advance_frac')*100:.0f}% (mean v over continents {med('trench_vorth_mean_cont_over_cm_yr'):+.1f}, "
        f"over ocean {med('trench_vorth_mean_ocean_over_cm_yr'):+.1f})")
    # Earth's mean ocean-floor age since Pangea: ~38 Myr at 120-80 Ma, ~64 today (Coltice et al. 2013 and Seton et al.
    # 2012/2020, computed on the EarthByte model); +-~10 Myr between reconstructions (Williams et al. 2021)
    ok["A11 ocean age mean 35~75 Myr, p99 <= ~200"] = (35 <= res["ocean_age_mean"] <= 75 and res["ocean_age_p99"] <= 200,
                                                       f"mean {res['ocean_age_mean']:.1f}, p99 {res['ocean_age_p99']:.0f}")
    areas = [p["area_frac"] for p in res["plates"]]
    areas = [p["area_frac"] for p in res["plates"]]
    ok["C4 largest plate 20.5~38.5%, 5~8 plates >= 5% (Earth model 0-1000 Ma, p10-p90; results/earth_check_results.json, 101 times)"] = (
        0.205 <= areas[0] <= 0.385 and 5 <= sum(a >= 0.05 for a in areas) <= 8,
        f"largest {areas[0]*100:.0f}%, plates >= 5%: {sum(a >= 0.05 for a in areas)}")
    ok["A2 every plate is one connected piece (detached parts < 0.1% of surface)"] = (
        res["detached_frac"] < 0.001, f"detached {res['detached_frac']*100:.2f}% of surface")
    return ok
