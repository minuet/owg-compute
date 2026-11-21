"""Integrated extensional strength of the thin-shell model's continuum lithosphere (SHELLS, Bird's Earth5 rheology),
rewritten from ShellSet MOD_Shells.f90 (Diamnd for friction and creep, Mohr/Squeez for the layer geotherms) with the
parameters of INPUT/iEarth5-049.in. Checked against the model's own stress output at the model's own strain rate
(E-1121-15: ratios 0.82-0.99 at the rift sites). The rift criterion takes its critical force from here so that force
and strength come from one rheology; lith_strength stays available for the comparison."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lith_strength as ls

# iEarth5-049.in (Bird's Earth5 parameters, unchanged by our runs except fFric and tauMax, which do not enter here)
C_FRIC, BIOT = 0.85, 1.0
A_CREEP, B_CREEP, C_CREEP = (2.3e9, 9.5e4), (4000.0, 18314.0), (0.0, 0.0171)
D_CREEP, E_CREEP = 5e8, 1.0 / 3.0
ALPHA_T, RHO_BAR, RHO_H2O = (2.4e-5, 3.94e-5), (2889.0, 3332.0), 1032.0
CONDUC, RADIO, T_SURF, TEM_LIM = (2.7, 3.2), (3.5e-7, 3.2e-8), 273.0, (1223.0, 1673.0)
G = 9.8
T_ADIAB, GRADIE = 1412.0, 6.1e-4             # K, K/m: upper-mantle adiabat (line 11); tAsthK at 100 km (MOD_Shells line 3905)
T_ASTH = T_ADIAB + GRADIE * 100.0e3
DZ = 100.0                                   # m, technical (as hopper_buck_fc.py)


def shells_strength(q, zc, zm, edot, creep="shells", geo="shells"):
    """N/m; q W/m2, zc crust and zm mantle-lithosphere thickness (m). creep="ours" uses lith_strength's creep laws and
    cap in place of Bird's; geo="ours" uses lith_strength's geotherm (C) in place of the SHELLS layer geotherm."""
    s_fric = np.sin(np.arctan(C_FRIC)); gam = (1 + s_fric) / (1 - s_fric)
    dsf = 0.5 * (1 - 1 / gam)                # max shear per effective vertical stress, normal faulting (Diamnd)
    z = np.arange(DZ / 2, zc + zm, DZ)
    lay = (z >= zc).astype(int)
    # layer geotherms with the quadratic correction that makes them reach T_ASTH at the base of the lithosphere
    # (MOD_Shells lines 3905-3965; E-1121-15: left out before, which made thick lithosphere far too hot)
    g2, g3, g7 = q / CONDUC[0], -RADIO[0] / (2 * CONDUC[0]), -RADIO[1] / (2 * CONDUC[1])
    def layers(g3, g7):
        tmoho = T_SURF + g2 * zc + g3 * zc ** 2
        return tmoho, (g2 + 2 * g3 * zc) * CONDUC[0] / CONDUC[1]
    tmoho, dtm = layers(g3, g7)
    test = tmoho + dtm * zm + g7 * zm ** 2 if zm > 0 else tmoho
    dq = -(test - T_ASTH) / (zc + zm) ** 2
    g3, g7 = g3 + dq, g7 + dq
    tmoho, dtm = layers(g3, g7)
    zz = z - zc
    tc = T_SURF + g2 * z + g3 * z ** 2
    tm = tmoho + dtm * zz + g7 * zz ** 2
    t = np.minimum(np.where(lay == 0, tc, tm), np.where(lay == 0, TEM_LIM[0], TEM_LIM[1]))
    if geo == "ours":
        t = ls.geotherm(q * 1e3, z) + 273.15
    rho = np.where(lay == 0, RHO_BAR[0] * (1 - ALPHA_T[0] * t), RHO_BAR[1] * (1 - ALPHA_T[1] * t))
    sv = np.cumsum(rho * G * DZ)
    fric = 2 * dsf * (sv - BIOT * RHO_H2O * G * z)          # differential stress at normal faulting
    if creep == "shells":
        a, b, c = (np.where(lay == 0, v[0], v[1]) for v in (A_CREEP, B_CREEP, C_CREEP))
        # Diamnd: shear = 2 visInf eSCrit exp((b + c z)/T), visInf = a/2 (2 sqrt(II))^(e-1); plane strain: II = edot^2,
        # eSCrit = edot -> shear = a (2 edot)^e / 2; differential = 2 shear, capped at 2 dCreep
        cr = np.minimum(a * (2 * edot) ** E_CREEP * np.exp(np.minimum((b + c * z) / t, 87.0)), 2 * D_CREEP)
    else:
        old, ls.EDOT = ls.EDOT, edot
        tcel = t - 273.15
        cr = np.where(z <= ls.Z_FELSIC_KM * 1e3, ls._creep(ls.QUARTZ, tcel),
                      np.where(lay == 0, ls._creep(ls.DIABASE, tcel), ls._creep(ls.OLIVINE, tcel)))
        cr = np.minimum(cr, ls.SIGMA_CAP)
        ls.EDOT = old
    return float(np.minimum(fric, cr).sum() * DZ)
