"""Extensional strength of the whole continental lithosphere (N/m) from its geotherm (2026-11-10; PARAM_AUDIT).

Strength envelope (Brace & Kohlstedt 1980; Kusznir & Park 1987; Ranalli 1995; Burov 2011): at each depth the
differential stress the rock can carry is the smaller of brittle failure (Byerlee normal faulting) and
power-law creep at the tectonic strain rate, capped at SIGMA_CAP; integrated down to the base of the
lithosphere (where the conductive geotherm meets the mantle adiabat). Two textbook views of the mantle are computed:
"jelly sandwich" (dry olivine, Hirth & Kohlstedt 2003) and "creme brulee" (the mantle carries no strength, Jackson 2002).
Geotherm: steady conduction with crustal heat production decaying with depth (Turcotte & Schubert 4.8; Lachenbruch
1970), reduced heat flow q_m = 0.6 q_s (Pollack & Chapman 1977).
Usage: .venv/bin/python tools/lith_strength.py   (prints the strength against surface heat flow)
"""
import math
import json
import numpy as np
import mantle_flow as mf
from plate_checks import HERE

# Rock constants formerly read from py (archived with the force model, 2026-11-18); values and sources unchanged
MU = 0.6                         # Byerlee 1978, sigma_n > 200 MPa (PARAM_AUDIT ①)
BETA = (math.sqrt(1 + MU ** 2) + MU) ** 2 - 1   # thrust-faulting factor from MU (same formula as the archived axioms.py)
RHO_CONT = 2800.0                # kg/m3, CRUST1.0
SIGMA_CAP = 300e6                # Pa, differential-stress cap of the lithosphere (Craig, Copley & Jackson 2014, ②)
R_GAS = 8.314
EDOT = 1e-15                     # 1/s, tectonic strain rate (Kusznir & Park 1987; Burov 2011)
Z_FELSIC_KM = 23.0               # upper + middle crust (Rudnick & Fountain 1995: 12 + 11 km)
Z_MOHO_KM = 40.0                 # crust (Rudnick & Fountain 1995; Christensen & Mooney 1995: mean 41 km)
QUARTZ = (1.1e-4, 4.0, 223e3)    # wet quartzite, A (MPa^-n s^-1), n, Q (J/mol) (Gleason & Tullis 1995)
DIABASE = (8.0, 4.7, 485e3)      # dry diabase (Mackwell et al. 1998)
OLIVINE = (1.1e5, 3.5, 530e3)    # dry olivine, dislocation creep (Hirth & Kohlstedt 2003)
T0_C = 0.0                       # surface temperature (Turcotte & Schubert)
H_R_KM = 10.0                    # decay depth of crustal heat production (Turcotte & Schubert 4.8; Lachenbruch 1970)
QM_SHARE = 0.6                   # reduced / surface heat flow (Pollack & Chapman 1977)
K_CRUST, K_MANTLE = 2.5, 3.3     # W/m/K (Turcotte & Schubert)
TP_C, ADIABAT_C_KM = 1315.0, 0.3 # mantle potential temperature and adiabat (McKenzie, Jackson & Priestley 2005)
RHO_MANTLE = mf.RP_RHO_M
Z_MAX_KM, N_Z = 250.0, 1000      # technical: depth samples
Q_SCAN = np.arange(40.0, 100.1, 5.0)   # mW/m2, technical scan
Q_CONT_MEAN = 65.0               # mW/m2, continental mean (Pollack, Hurter & Johnson 1993)

BETA_EXT = 1 - 1 / (math.sqrt(1 + MU ** 2) + MU) ** 2   # normal faulting, as earth_rift_tension S_EXT


def geotherm(q_s_mw, z_m):
    """T (C) at depth z (m) for surface heat flow q_s (mW/m2); conductive, then the adiabat below their crossing."""
    q_s = q_s_mw * 1e-3; q_m = QM_SHARE * q_s; hr = H_R_KM * 1e3; zm = Z_MOHO_KM * 1e3
    def crust(z):
        return T0_C + q_m * z / K_CRUST + (q_s - q_m) * hr * (1 - np.exp(-z / hr)) / K_CRUST
    q_moho = q_m + (q_s - q_m) * math.exp(-zm / hr)        # heat flow through the Moho
    t = np.where(z_m <= zm, crust(z_m), crust(zm) + q_moho * (z_m - zm) / K_MANTLE)
    return np.minimum(t, TP_C + ADIABAT_C_KM * z_m / 1e3)


def _creep(law, t_c):
    a, n, q = law
    return (EDOT / a) ** (1 / n) * np.exp(q / (n * R_GAS * (t_c + 273.15))) * 1e6   # Pa


def envelope(q_s_mw, mantle_strong=True, beta=BETA_EXT, g=mf.EARTH.g):
    """depth (m) and differential stress (Pa) of the strength envelope: extensional by default; beta=BETA for
    thrust faulting (compression); g = surface gravity of the body"""
    z = (np.arange(N_Z) + 0.5) / N_Z * Z_MAX_KM * 1e3
    t = geotherm(q_s_mw, z)
    zm = Z_MOHO_KM * 1e3
    sv = np.where(z <= zm, RHO_CONT * z, RHO_CONT * zm + RHO_MANTLE * (z - zm)) * g
    brittle = beta * (sv - mf.RP_RHO_W * g * z)
    ductile = np.where(z <= Z_FELSIC_KM * 1e3, _creep(QUARTZ, t),
                       np.where(z <= zm, _creep(DIABASE, t), _creep(OLIVINE, t) if mantle_strong else 0.0))
    s = np.minimum(np.minimum(brittle, ductile), SIGMA_CAP)
    s = np.where(t >= TP_C + ADIABAT_C_KM * z / 1e3 - 1e-6, 0.0, s)    # below the lithosphere
    return z, s


def strength(q_s_mw, mantle_strong=True, beta=BETA_EXT, g=mf.EARTH.g):
    """N/m: integrated strength of the continental lithosphere (extensional by default, see envelope)"""
    z, s = envelope(q_s_mw, mantle_strong, beta, g)
    return float(s.sum() * (z[1] - z[0]))


if __name__ == "__main__":
    out = {"edot": EDOT, "rows": []}
    print("q_s mW/m2 | Moho T C | strength jelly sandwich (dry olivine) | creme brulee (mantle weak), 1e12 N/m")
    for q in list(Q_SCAN) + [Q_CONT_MEAN]:
        tm = float(geotherm(q, np.array([Z_MOHO_KM * 1e3]))[0])
        a, b = strength(q, True), strength(q, False)
        out["rows"].append({"q_s": q, "T_moho_C": round(tm), "jelly": round(a / 1e12, 2), "creme": round(b / 1e12, 2)})
        print(f"{q:5.0f} | {tm:5.0f} | {a / 1e12:6.2f} | {b / 1e12:6.2f}")
    (HERE / "results/lith_strength.json").write_text(json.dumps(out, indent=1))


RHO_MAGMA = 2750.0               # kg/m3 (Bialas, Buck & Qin 2010, table 1)
H_PH_M = 100.0                   # m, magma pressure head (Bialas et al. 2010 range 100 m - 1 km; the conservative end)
T_LAB_DIKE_C = 1200.0            # base of the lithosphere for diking (Bialas et al. 2010: H_L = H_1200C)


def lithosphere_thickness(q_s_mw, t_base_c=T_LAB_DIKE_C):
    """m: depth of the t_base_c isotherm of the geotherm"""
    z = (np.arange(N_Z) + 0.5) / N_Z * Z_MAX_KM * 1e3
    t = geotherm(q_s_mw, z)
    return float(z[np.argmax(t >= t_base_c)]) if (t >= t_base_c).any() else Z_MAX_KM * 1e3


def dike_force(q_s_mw):
    """N/m: with magma supplied, the force to open a dike through the whole lithosphere (Bialas, Buck & Qin 2010 eq. 8,
    magma column from the surface plus a pressure head; crust as dense as the magma)"""
    hl = lithosphere_thickness(q_s_mw); hc = Z_MOHO_KM * 1e3; g = mf.EARTH.g
    return float((RHO_MANTLE - RHO_MAGMA) * g * max(hl - hc, 0.0) ** 2 / 2 - RHO_MAGMA * g * H_PH_M * hl)
