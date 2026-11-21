"""E-1121-12, Earth test of the breakup criterion (PLATE_PLAN section 6 step 3): available force at a rift from the thin-shell
model against the critical force of Hopper & Buck 1993 (E-1121-11) and the dike force of Bialas, Buck & Qin 2010.

Run: SHELLS on the generated 2x grid with the planet's drag method (our flow + residual, SHELLS_RESID=1), the reasonable
parameters (tauMax 2E12; fault friction subduction 0.055, collision 0.06, others 0.085; E-1121-03), with the continental
boundaries of Somalia-Africa and Amur-Eurasia welded (shells_grid.earth_grid weld: East African rift and Baikal are intact
lithosphere, Somalia and Amur lose their velocity conditions, so no inferred traction carries their observed motion) and
the stress output patch (shells_feg_test.patch_tau).

Available force F: the vertical integral of the deviatoric stress normal to the rift, from Shells' tauMat (integrals of
Sxx-Szz, Syy-Szz, Sxy; tension positive), mean over the elements within R_AVG of each site. Critical force Fc:
lith_strength (our continental envelope, PARAM_AUDIT 11.1) at Hopper & Buck's critical strain rate EDOT_CRIT, for the
heat flow Shells used there. Dike force: lith_strength.dike_force.
Since E-1121-14 the critical force also comes from the thin-shell model's own rheology (shells_rheology, Fc_shells), so that
force and strength come from one rheology; the lith_strength values are kept for comparison.

Expected on Earth, written before the run: the East African rift (extending at 2-7 mm/yr and breaking up in Afar) must
have F above min(Fc, dike force); the cratons (stable for > 1 Gyr) far below Fc in every direction; the slow rifts of
cold or old lithosphere (Rhine, Rio Grande: < 1 mm/yr; Baikal 4 mm/yr for 30 Myr) near or below Fc. India serves as the
sign check (N-S compression from the Himalayan collision: tauMat N-S must be negative).
Usage: .venv/bin/python -I tools/rift_force_earth.py <results dir of the run>   (writes results/rift_force_earth.json)"""
import glob, json, math, os, sys, zipfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lith_strength as ls
import shells_grid as sg
from shells_rheology import shells_strength
from fastkd import cKDTree
from plate_checks import HERE

EDOT_CRIT = 1e-17               # 1/s, Hopper & Buck 1993 (their range 5e-18 - 8e-17; E-1121-11)
REFINE = 2.0                    # the run's grid refinement (E-1120-04)
R_AVG_KM = sg.H_PLATE_KM / REFINE   # one plate-interior grid spacing of the run
STEPS = HERE / "../reference/earth/plate-boundaries/PB2002_steps.dat"
# slow rifts not in PB2002: axis points (lon, lat) and the azimuth of the extension direction, degrees from north
# Rio Grande: San Luis, Espanola, Albuquerque, Socorro, Palomas basins; E-W extension (Berglund et al. 2012 GPS)
# Upper Rhine Graben: Basel to Frankfurt, strike N20E, extension normal to it (Ziegler 1992)
SLOW = {"Rio Grande": ([(-105.9, 37.5), (-106.1, 36.0), (-106.7, 35.0), (-106.9, 34.0), (-106.9, 32.5)], 90.0),
        "Rhine": ([(7.6, 47.8), (7.9, 48.6), (8.3, 49.3), (8.5, 49.9)], 110.0)}
# cratons (stable Precambrian shields): one interior point each
CRATONS = {"Canadian": (-95.0, 58.0), "Siberian": (110.0, 65.0), "West African": (-5.0, 18.0),
           "Kaapvaal": (26.0, -27.0), "Yilgarn": (119.0, -29.0), "Amazonian": (-58.0, -8.0), "Baltic": (25.0, 63.0)}
INDIA = (78.0, 20.0)


def read_run(d):
    z = zipfile.ZipFile(glob.glob(f"{d}/*_shellsF.zip")[0])
    L = z.read("INPUT/Earth5R.feg").decode().splitlines()
    n = int(L[1].split()[0]); ne = int(L[2 + n].split()[0])
    nd = np.array([[float(x) for x in l.split()[1:7]] for l in L[2:2 + n]])        # lon, lat, elev, heat (W/m2), crust, mantle lith. (m)
    tri = np.array([[int(x) - 1 for x in l.split()[1:4]] for l in L[3 + n:3 + n + ne]])
    f = [x for x in z.namelist() if "ShellsF_output/fEarth_" in x][0]
    tau = np.array([[float(v.replace("D", "E")) for v in l.split()[2:6]]
                    for l in z.read(f).decode().splitlines() if l.startswith("TAU ")])
    assert len(tau) == ne, (len(tau), ne)
    c = sg.xyz(nd[tri, 0], nd[tri, 1]).sum(1)
    return nd, tri, tau, c / np.linalg.norm(c, axis=1)[:, None]


def site(lon, lat, az_ext, nd, tri, tau, tree):
    """F normal to the rift (extension azimuth az_ext), the most tensile horizontal value (any direction), heat flow"""
    k = tree.query_ball_point(sg.xyz(np.array([lon]), np.array([lat]))[0], R_AVG_KM / sg.R_KM)
    t = tau[k].mean(0)
    a = math.radians(az_ext)
    nx, ny = -math.cos(a), math.sin(a)          # Shells' local x = south, y = east
    fn = t[0] * nx * nx + t[1] * ny * ny + 2 * t[2] * nx * ny
    fmax = (t[0] + t[1]) / 2 + math.hypot((t[0] - t[1]) / 2, t[2])
    q, zc, zm = (float(v) for v in nd[np.unique(tri[k]), 3:6].mean(0))
    phi = 0.5 * math.atan2(2 * t[2], t[0] - t[1])          # most tensile axis, from local south toward east
    return fn, fmax, t, q * 1e3, len(k), zc, zm, math.degrees(math.atan2(math.sin(phi), -math.cos(phi))), az_ext


def row(name, vals):
    fn, fmax, q, zc, zm = (float(np.median([v[i] for v in vals])) for i in (0, 1, 3, 5, 6))
    fc_s = shells_strength(q * 1e-3, zc, zm, EDOT_CRIT)
    def axial(i):                                          # mean of axial directions (degrees from north, mod 180)
        c = sum(math.cos(math.radians(2 * v[i])) for v in vals); s = sum(math.sin(math.radians(2 * v[i])) for v in vals)
        return round(math.degrees(math.atan2(s, c)) / 2 % 180.0, 0)
    fc_j, fc_c, fd = ls.strength(q, True), ls.strength(q, False), ls.dike_force(q)
    return {"site": name, "points": len(vals), "q_mW": round(q, 1), "F_normal_TN": round(fn / 1e12, 2),
            "F_max_TN": round(fmax / 1e12, 2), "Fc_jelly_TN": round(fc_j / 1e12, 2), "Fc_creme_TN": round(fc_c / 1e12, 2),
            "F_dike_TN": round(fd / 1e12, 2), "F_over_Fc_jelly": round(fn / fc_j, 2),
            "Fc_shells_TN": round(fc_s / 1e12, 2), "F_over_Fc_shells": round(fn / fc_s, 2), "Fmax_over_Fc_shells": round(fmax / fc_s, 2),
            "az_rift_normal_deg": axial(8), "az_model_tension_deg": axial(7)}


def main(d):
    ls.EDOT = EDOT_CRIT
    nd, tri, tau, cen = read_run(d)
    tree = cKDTree(cen)
    rows = []
    steps = [l.replace(":", " ").split() for l in open(STEPS) if l.strip()]
    groups = {"East Africa, Main Ethiopian": lambda p, lo, la, star: p == "SO-AF" and la >= 5.0,
              "East Africa, western branch": lambda p, lo, la, star: p == "SO-AF" and -16.0 <= la < 5.0,
              "Baikal": lambda p, lo, la, star: p in ("AM-EU", "EU-AM") and lo < 113.0 and not star}
    # Baikal: only the steps of the rift itself; PB2002 marks with * the steps inside its orogens (diffuse zones), here the
    # Mongolian part of the Amur boundary running south from the lake (E-1121-16)
    for name, sel in groups.items():
        vals = []
        for s in steps:
            lo0, la0, lo1, la1 = (float(x) for x in s[2:6])
            lo, la = (lo0 + lo1) / 2, (la0 + la1) / 2
            if s[-1].rstrip("*") in sg.WELD_CLASSES and sel(s[1], lo, la, s[-1].endswith("*")):
                strike = math.degrees(math.atan2((lo1 - lo0) * math.cos(math.radians(la)), la1 - la0))
                vals.append(site(lo, la, strike + 90.0, nd, tri, tau, tree))
        rows.append(row(name, vals))
    for name, (pts, az) in SLOW.items():
        rows.append(row(name, [site(lo, la, az, nd, tri, tau, tree) for lo, la in pts]))
    for name, (lo, la) in CRATONS.items():
        rows.append(row(f"craton {name}", [site(lo, la, 0.0, nd, tri, tau, tree)]))
    ind = site(*INDIA, 0.0, nd, tri, tau, tree)
    out = {"edot_crit": EDOT_CRIT, "r_avg_km": R_AVG_KM, "rows": rows,
           "sign_check_india_NS_TN": round(ind[0] / 1e12, 2),
           "global_tauMat_median_TN": [round(float(np.median(tau[:, i])) / 1e12, 3) for i in range(3)]}
    (HERE / "results/rift_force_earth.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
