"""Continental rift mode after Buck 1991 (JGR 96:20161, "Modes of continental lithospheric extension"), re-implemented
from the paper's text, tables 1-3 and appendix A (PARAM_AUDIT 11.12, E-1121-11).
A uniform lithosphere is stretched in pure shear over a zone of width XE at velocity difference ux until the strain is
STRAIN. One-dimensional heat transport at the rift centre (eq. 1: conduction, vertical advection v = -edot z, crustal
heat sources) gives the temperatures; lower crustal flow across the rift (eq. 7, effective diffusivity eq. 8) gives the
crustal thickness at the centre. The force needed to keep extending changes by
  dFys  integrated yield stress (brittle gB z or power-law creep, the lesser), crust rheology above the Moho, olivine below
  dFcb  crustal buoyancy, g drho* h dh (eq. 12), dh = crust at x = XL/2 minus crust at the centre
  dFtb  thermal buoyancy, -g int rho alpha dT z dz (eq. 11)
dF > 0: wide rift; dF < 0: narrow rift, or core complex when lower crustal flow cuts dFcb below CC_FRAC of its no-flow value.
Usage: .venv/bin/python tools/rift_buck.py   (reproduction checks against the paper, then its table 4 cases)"""
import json
import math
import numpy as np
from scipy.linalg import solve_banded
from plate_checks import HERE

# Buck 1991 table 2
K = 3.3                          # W/m/K
RHO_C, RHO_M = 2800.0, 3300.0    # kg/m3
H_RAD = 6.4e-7                   # W/m3, uniform in the crust
ALPHA = 3.0e-5                   # 1/K
G_BUCK = 10.0                    # m/s2
R = 8.31                         # J/mol/K
KAPPA = 1.0e-6                   # m2/s, textbook (Turcotte & Schubert); not legible in the extracted table 2
GB_EARTH = 12e6 / 1e3            # Pa/m, extensional brittle strength with hydrostatic pore pressure (eq. 2), at g = 10
# Buck 1991 table 3 (Kirby & Kronenberg 1987 and others): n, A (Pa^-n s^-1), E (J/mol)
RHEO = {"olivine": (3.0, 1.0e-15, 500e3), "pyroxene": (5.3, 2.5e-37, 380e3), "anorthosite": (3.2, 5.6e-23, 238e3),
        "dry_quartz": (2.9, 5.0e-25, 149e3), "wet_quartz": (2.4, 1.3e-20, 134e3)}
SIGMA_REF = 1e6                  # Pa, stress of the Newtonian effective viscosity C (table 3) and minimum flow stress
ETA_LAB = 1e21                   # Pa s, base of the lithosphere ZL (effective mantle viscosity, eq. 9)
XE, XL, STRAIN = 40e3, 250e3, 0.25   # m, m, -: the paper's figure settings
L_STRESS = 100e3                 # m, crustal thickness difference over which the flow stress is taken
CC_FRAC = 0.1                    # core complex: dFcb below this share of its no-flow value (fig. 11 caption)
DZ, DX = 1e3, 1e3                # m, technical grids (paper: 1 km, finer had negligible effect)
NT_MIN, DT_MAX = 400, 1e4 * 3.15576e7   # technical: at least 400 time steps, none longer than 10 kyr
SEC_YR = 3.15576e7


def drho_star():
    return RHO_C * (RHO_M - RHO_C) / RHO_M


def creep_stress(law, t_k, edot):
    n, a, e = law
    return (edot / a) ** (1 / n) * np.exp(e / (n * R * t_k))


def geotherm0(qs, h, z):
    """steady state, T(0) = 0 C, uniform H in the crust (deg C)"""
    tc = qs * z / K - H_RAD * z ** 2 / (2 * K)
    th = qs * h / K - H_RAD * h ** 2 / (2 * K)
    return np.where(z <= h, tc, th + (qs - H_RAD * h) * (z - h) / K)


def strength(t_c, z, hc, edot, crust, gb):
    t_k = t_c + 273.15
    duct = np.where(z < hc, creep_stress(RHEO[crust], t_k, edot), creep_stress(RHEO["olivine"], t_k, edot))
    return float(np.minimum(gb * z, duct).sum() * DZ)


def run(qs_mw, h_km, ux_cm_yr, crust="dry_quartz", flow=True, g=G_BUCK, xe=XE, xl=XL, strain=STRAIN):
    qs, h0 = qs_mw * 1e-3, h_km * 1e3
    gb = GB_EARTH * g / G_BUCK
    n_o, a_o, e_o = RHEO["olivine"]
    t_lab = e_o / (R * math.log(ETA_LAB * a_o * SIGMA_REF ** (n_o - 1))) - 273.15   # eta = C exp(E/RT), C = A^-1 s^(1-n)
    zz = np.arange(0, 400e3 + DZ, DZ)
    tt = geotherm0(qs, h0, zz)
    if tt.max() < t_lab:
        return None
    zl = float(zz[np.argmax(tt >= t_lab)])
    z = np.arange(0, zl + DZ / 2, DZ); nz = len(z)
    t0 = geotherm0(qs, h0, z); t0[-1] = t_lab
    ux = ux_cm_yr / 100 / SEC_YR
    edot = ux / xe
    t_end = strain / edot
    nt = max(NT_MIN, int(math.ceil(t_end / DT_MAX))); dt = t_end / nt
    # thermal operator (backward Euler): dT/dt = kappa T'' + edot z T' (upwind from below) + H/(rho c)
    rc = K / KAPPA                   # rho c (J/m3/K)
    lo = np.zeros(nz); di = np.ones(nz); up = np.zeros(nz)
    i = np.arange(1, nz - 1)
    a = KAPPA * dt / DZ ** 2; b = edot * z[i] * dt / DZ
    lo[i - 1] = -a; di[i] = 1 + 2 * a + b; up[i + 1] = -a - b
    ab = np.zeros((3, nz)); ab[0] = up; ab[1] = di; ab[2] = lo
    # crust across the rift (x from the centre to XL/2): dh/dt = kf h'' - u h' - h u'
    x = np.arange(0, xl / 2 + DX / 2, DX); nx = len(x)
    u = np.where(x <= xe / 2, edot * x, ux / 2); du = np.where(x <= xe / 2, edot, 0.0)
    hx = np.full(nx, h0)
    gdr = g * drho_star()
    n_c, a_c, e_c = RHEO[crust]
    k_stress = int(round(L_STRESS / DX))
    temp = t0.copy(); kf_max = 0.0
    for _ in range(nt):
        hc = hx[0]
        rhs = temp + dt * np.where(z < hc, H_RAD / rc, 0.0); rhs[0] = 0.0; rhs[-1] = t_lab
        temp = solve_banded((1, 1), ab, rhs)
        if flow:
            j = min(int(hc / DZ), nz - 2)
            tm = np.interp(hc, z, temp) + 273.15
            grad = max((temp[j] - temp[j - 1]) / DZ, 1e-6)
            y0 = R * tm ** 2 / (e_c * grad)
            sig = max(SIGMA_REF, 2 * y0 * gdr * (hx[min(k_stress, nx - 1)] - hx[0]) / L_STRESS)
            eta0 = sig ** (1 - n_c) / a_c * math.exp(e_c / (R * tm))
            kf = gdr * y0 ** 3 / eta0
        else:
            kf = 0.0
        kf_max = max(kf_max, kf)
        # explicit advection (upwind, u >= 0) and thinning, implicit diffusion with zero-gradient ends
        adv = np.zeros(nx); adv[1:] = u[1:] * (hx[1:] - hx[:-1]) / DX
        rhs = hx - dt * (adv + hx * du)
        c = kf * dt / DX ** 2
        ab2 = np.zeros((3, nx)); ab2[1] = 1 + 2 * c; ab2[0, 1:] = -c; ab2[2, :-1] = -c
        ab2[0, 1] = -2 * c; ab2[2, -2] = -2 * c
        hx = solve_banded((1, 1), ab2, rhs)
    dh = hx[-1] - hx[0]
    f0 = strength(t0, z, h0, edot, crust, gb)
    f1 = strength(temp, z, hx[0], edot, crust, gb)
    rho = np.where(z < hx[0], RHO_C, RHO_M)
    dftb = float(-g * (rho * ALPHA * (temp - t0) * z).sum() * DZ)
    dfcb = gdr * h0 * dh
    dfcb_noflow = gdr * h0 * h0 * (1 - math.exp(-strain))
    df = (f1 - f0) + dfcb + dftb
    mode = "wide" if df > 0 else ("core_complex" if dfcb < CC_FRAC * dfcb_noflow else "narrow")
    return {"qs": qs_mw, "h_km": h_km, "ux_cm_yr": ux_cm_yr, "crust": crust, "edot": edot,
            "T_moho0_C": round(float(geotherm0(qs, h0, np.array([h0]))[0]), 1), "ZL_km": zl / 1e3,
            "Fys0_TN": f0 / 1e12, "dFys_TN": (f1 - f0) / 1e12, "dFcb_TN": dfcb / 1e12, "dFtb_TN": dftb / 1e12,
            "dF_TN": df / 1e12, "flow_share": dfcb / dfcb_noflow, "kf_max": kf_max, "mode": mode}


def gz_analytic(tm_c, h_km, crust="dry_quartz", edot=8e-15, ts=0.0):
    """appendix A, eq. A9 with F = 1 (no lower crustal flow) and eq. A11 for the brittle-ductile temperatures"""
    h = h_km * 1e3
    b = GB_EARTH / G_BUCK

    def t_bd(law, depth):
        n, a, e = law
        return e / (n * R * math.log(GB_EARTH * depth * (a / edot) ** (1 / n))) - 273.15
    tc, tmm = t_bd(RHEO[crust], h), t_bd(RHEO["olivine"], h)
    zc = min(h, h * (tc - ts) / (tm_c - ts)); zm = h * (tmm - ts) / (tm_c - ts)
    return drho_star() * h ** 2 / (b * (zc ** 2 + max(zm ** 2 - h ** 2, 0.0)))


# Buck 1991 table 4: crustal thickness (km), heat flow (mW/m2), extension velocity (cm/yr) near the rifts, and the
# mode observed; Tibet and the Altiplano are the paper's analogs for the conditions of core complex formation
TABLE4 = [("Tibet (core complex analog)", (60, 70), 90, (0.1, 0.3), "core_complex"),
          ("Altiplano (core complex analog)", (50, 60), 90, None, "core_complex"),
          ("Northern Basin and Range (present)", (25, 35), 90, (0.8, 1.1), "wide"),
          ("Aegean (present)", (30, 35), 90, (3, 10), "wide"),
          ("Rio Grande", (30, 30), 60, (0.1, 1), "narrow"),
          ("Gulf of Suez", (30, 30), 40, (0.5, 0.7), "narrow"),
          ("Northern Red Sea", (30, 30), 40, (0.7, 1.5), "narrow"),
          ("East African Rifts", (40, 40), 50, (0.1, 0.1), "narrow"),
          ("Rhinegraben", (40, 40), 70, (0.1, 0.1), "narrow"),
          ("Baikal Rift", (40, 40), 50, None, "narrow")]
UX_UNKNOWN = 1.0                 # cm/yr where table 4 has no rate: the paper's standard figure velocity


if __name__ == "__main__":
    out = {}
    # 1. fig. 3: Fys against heat flow (h = 40, dry quartz), paper: ~30 TN/m at 50 mW/m2, < 1 TN/m at 100
    out["fig3_Fys_h40"] = {q: round(run(q, 40, 1.0)["Fys0_TN"], 2) for q in (40, 50, 60, 70, 80, 90, 100)}
    # 2. figs. 9-10: total force change against heat flow and crustal thickness, dry quartz, 1 cm/yr over 40 km
    grid = {}
    for h in (30, 40, 50, 55, 60):
        for q in range(40, 125, 5):
            r = run(q, h, 1.0)
            if r:
                grid[f"{h}/{q}"] = (r["T_moho0_C"], round(r["dF_TN"], 3), r["mode"])
    out["fig10_dry_quartz"] = grid
    # 3. fig. 12b: wet quartz never gives a wide rift (strain rates 1e-16..1e-13: 0.0126..126 cm/yr over 40 km)
    wet = [run(q, h, ux, "wet_quartz") for h in (20, 30, 40, 50, 60) for q in range(40, 125, 10)
           for ux in (0.0126, 0.126, 1.26, 12.6, 126)]
    out["fig12b_wet_quartz_wide_count"] = sum(1 for r in wet if r and r["mode"] == "wide")
    out["fig12b_wet_quartz_runs"] = sum(1 for r in wet if r)
    # 4. appendix A: analytic Goetze number vs the numerical narrow-wide boundary, no crustal heat sources
    out["gz_analytic_dry_quartz_h40"] = {tm: round(gz_analytic(tm, 40), 2) for tm in range(500, 1001, 50)}
    # 5. table 4 cases
    rows = []
    for name, hr, q, ur, obs in TABLE4:
        modes = set()
        for h in sorted(set(hr)):
            for ux in (sorted(set(ur)) if ur else [UX_UNKNOWN]):
                r = run(q, h, ux)
                modes.add(r["mode"] if r else "none")
        rows.append({"case": name, "observed": obs, "predicted": sorted(modes)})
    out["table4"] = rows
    (HERE / "results/rift_buck.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
