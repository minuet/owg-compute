"""Python port of Peter Bird's OrbData5 node-data computation (Bird, May & Carafa, ShellSet; GPL v3), for the planet pipeline:
given nodes (lon, lat, elevation, heat flow) it returns the crust thickness, mantle-lithosphere thickness, chemical density
anomaly and cooling curvature that OrbData5 would write into the .feg file.

Ported, line for line and in the same order of operations (SI units throughout):
  - OrbData5.f90 (src/OrbData5.f90): PARAMETERs of the heat-flow / crust / lithosphere limits (lines 148-179), TAsthK
    (line 287), the heat-flow limits loop at nodes (lines 374-378), grid headers and sizes (lines 337-421), CRUST2
    km -> m (lines 403-407), the node loop and its max/min clamps on zMNode and tLNode (lines 432-459), node coordinates
    as stored by GetNet (MOD_Data.f90 lines 1976-1977; the angles are carried as radians, converted back at OrbData5 lines 433-434)
  - MOD_Data.f90: SUBROUTINE Assign (lines 209-740; elevation, sea-floor age, heat-flow grid lookups, ocean branch with
    Stein & Stein GDH1 and the geometric mean with the plate thickness, continental branch from the S-wave anomaly, geotherm,
    Moho/asthenosphere check, cooling curvature and its temperature-maximum correction, chemical density anomaly with limits,
    badP / badT tests) and SUBROUTINE Squeez (lines 43-207; the isostatic stress integral with its 1-km steps)
  - physical constants read from iEarth5-049.in by ReadPm (MOD_ShellSet.f90 lines 2389-2824; plain list-directed READs
    and ReadN for the two-value lines; the file's own lines are used, none is re-typed here)

Every numeric constant below is copied from OrbData5.f90, MOD_Data.f90 or iEarth5-049.in; the file/line is cited at each one.
The warning/log WRITEs (ModelError, iUnitL) are not ported; the two diagnostics badP and badT are returned as flags.
Not ported: nothing in the computation. The heat-flow grid branch (heat flow exactly 0 in the input) is ported but needs
the heat-flow grid (unit 4 in OrbData5; pass `heat_grid`), otherwise NotImplementedError is raised for such nodes.

Usage: .venv/bin/python -I tools/orbdata_nodes.py <ShellSet INPUT dir>"""
import os, sys, math, time
import numpy as np

# --- OrbData5.f90 PARAMETERs -----------------------------------------------------------------------------------------
QLIM0 = 0.000          # W/m^2, lower limit on heat flow at zero elevation (OrbData5.f90 line 148, qLim0)
DQL_DE = 0.00e-06      # (W/m^2)/m, d(qLim0)/d(elevation) (OrbData5.f90 line 155, dQL_dE)
QLIM1 = 0.300          # W/m^2, upper limit on heat flow (OrbData5.f90 line 161, qLim1)
CLIMIT = 6570.0        # m, lower limit on crust thickness (OrbData5.f90 line 164, cLimit)
HCMAX = 75000.0        # m, upper limit on crust thickness (OrbData5.f90 line 168, hCMax)
HLMAX = 400000.0       # m, upper limit on total lithosphere thickness (OrbData5.f90 line 172, hLMax)
DELTA_RHO_LIMIT = 100.0  # kg/m^3, limit of chemical density anomaly (OrbData5.f90 line 179, delta_rho_limit)
RAD_PER_DEG = 0.0174532925199433    # GetNet, MOD_Data.f90 lines 1976-1977
DEG_PER_RAD = 57.2957795130823      # OrbData5.f90 lines 433-434 (pLon, pLat from the node's radians)
TASTH_DEPTH = 100.0e3  # m, depth where the asthenosphere adiabat is evaluated (OrbData5.f90 line 287)
KM_TO_M = 1000.0       # CRUST2.grd km -> m (OrbData5.f90 line 405)

# --- MOD_Data.f90 Assign ---------------------------------------------------------------------------------------------
AGE_MAX_MA = 200.0     # age >= 200 Ma means unknown/continental (Assign lines 417, 493 'IF (ageMa < 200.0D0)')
GDH1_AGE_SPLIT = 55.0  # Ma, GDH1 of Stein & Stein 1992 (Assign lines 427, 504)
GDH1_A = 0.510         # W/m^2 * sqrt(Ma) (Assign lines 428, 505)
GDH1_B = 0.048         # W/m^2 (Assign lines 430, 507)
GDH1_C = 0.096         # W/m^2 (Assign lines 430, 507)
GDH1_K = -0.0278       # 1/Ma (Assign lines 430, 507)
Q_RADIOACTIVITY = 0.007  # W/m^2 (Assign line 516)
H_PLATE = 95000.0      # m, Stein & Stein plate thickness (Assign line 524)
SLOPE_NUM = 1.494      # s (Assign line 580; slope = 1.494/85000 s/m)
SLOPE_DEN = 85000.0    # m (Assign line 580)
RIDGE_DELTA_TS = 1.636  # s (Assign line 582)
BADP_PA = 40.0e6       # Pa (Assign line 724)
BADT_K = 20.0          # K (Assign line 729)
GRID_FRAC_ADD = 1.00001           # index offset in grid lookup (Assign lines 303, 322, 341, 360, 445, 463, 543, 561; double in e/a/c/s lookups)
GRID_FRAC_ADD_Q = np.float32(1.00001)  # heat-flow lookup writes "+ 1.00001" without D0, i.e. single precision (Assign line 399; ir1 at line 380 is double)
GRID_SIZE_ADD = 1.5    # nXX = (X2 - X1)/DX + 1.5 (OrbData5.f90 lines 339-340, 361-362, 386-387, 397-398, 418-419)
GRID_OUT_SLIGHT = 0.01  # (Assign lines 328-331 etc. 'outsid' / 'wayOut'; log only, kept as flags)
GRID_OUT_FAR = 1.01
GRID_OUT_FAR_HI = 2.01

# --- MOD_Data.f90 Squeez ---------------------------------------------------------------------------------------------
NDREF = 300            # km, depth of the reference profile (Squeez line 79)
REF_W1, REF_W2 = 0.7, 0.3   # dRef(3) and dRef(8) mixing (Squeez lines 99, 104: 0.7 and 0.3)

# iEarth5-049.in line numbers of the physical constants (ReadPm reads them in this order)
PARAM_LINES = {"tAdiab,gradie": 11, "rhoH2O": 17, "rhoBar": 18, "rhoAst": 19, "gMean": 20, "oneKm": 21,
               "radius": 22, "alphaT": 23, "conduc": 24, "radio": 25, "tSurf": 26, "temLim": 27}   # line 1 is the title (title3)
PARAM_FILE = "iEarth5-049.in"       # unit 1 parameter file (ihcflow.in lists the run; here the file used for the Earth run)
GRID_FILES = {"elev": "ETOPO20.grd", "age": "age_1p5.grd", "crust": "CRUST2.grd", "dts": "delta_ts.grd"}  # InputFiles.in


def read_params(inp_dir, fname=PARAM_FILE):
    """Physical constants from the parameter file, as ReadPm: first blank-delimited item of each line, commas separate."""
    with open(os.path.join(inp_dir, fname)) as f:
        lines = f.read().split("\n")
    def vals(key):
        tok = lines[PARAM_LINES[key] - 1].split()[0]
        return [float(x.replace("D", "E")) for x in tok.split(",")]
    p = {}
    p["tAdiab"], p["gradie"] = vals("tAdiab,gradie")
    p["rhoH2O"] = vals("rhoH2O")[0]
    p["rhoBar"] = vals("rhoBar")
    p["rhoAst"] = vals("rhoAst")[0]
    p["gMean"] = vals("gMean")[0]
    p["oneKm"] = vals("oneKm")[0]
    p["radius"] = vals("radius")[0]
    p["alphaT"] = vals("alphaT")
    p["conduc"] = vals("conduc")
    p["radio"] = vals("radio")
    p["tSurf"] = vals("tSurf")[0]
    p["temLim"] = vals("temLim")
    return p


class Grid:
    """A .grd file as OrbData5 reads it: 2 header lines (first three numbers: start, step, end), then rows from the north."""
    def __init__(self, path, scale=1.0):
        with open(path) as f:
            h1 = f.readline().split()[:3]
            h2 = f.readline().split()[:3]
            body = f.read()
        self.x1, self.dx, self.x2 = [float(v) for v in h1]
        self.y1, self.dy, self.y2 = [float(v) for v in h2]
        self.nx = int((self.x2 - self.x1) / self.dx + GRID_SIZE_ADD)
        self.ny = int((self.y2 - self.y1) / self.dy + GRID_SIZE_ADD)
        a = np.array(body.split(), dtype=np.float64)[: self.nx * self.ny].reshape(self.ny, self.nx)
        self.a = a * scale if scale != 1.0 else a
        self.outside = self.far = False

    def lookup(self, lon, lat, add=GRID_FRAC_ADD):
        """Bilinear value, exactly as Assign (clamped indices; lon shifted by +-360 into the grid)."""
        ir1 = int(((self.y2 - lat) / self.dy) + add)
        ir1 = min(max(ir1, 1), self.ny - 1)
        ir2 = ir1 + 1
        fr = ((self.y2 - self.dy * (ir1 - 1)) - lat) / self.dy
        if lon < self.x1 or lon > self.x2:
            if self.x1 <= lon + 360.0 <= self.x2:
                ul = lon + 360.0
            elif self.x1 <= lon - 360.0 <= self.x2:
                ul = lon - 360.0
            else:
                ul = lon
        else:
            ul = lon
        ic1 = int(((ul - self.x1) / self.dx) + add)
        ic1 = min(max(ic1, 1), self.nx - 1)
        ic2 = ic1 + 1
        fc = (ul - (self.x1 + self.dx * (ic1 - 1))) / self.dx
        self.outside = self.outside or fr < -GRID_OUT_SLIGHT or fr > GRID_OUT_FAR or fc < -GRID_OUT_SLIGHT or fc > GRID_OUT_FAR
        self.far = self.far or fr < -GRID_OUT_FAR or fr > GRID_OUT_FAR_HI or fc < -GRID_OUT_FAR or fc > GRID_OUT_FAR_HI
        fr = min(1.0, max(0.0, fr))
        fc = min(1.0, max(0.0, fc))
        a = self.a
        top = a[ir1 - 1, ic1 - 1] + fc * (a[ir1 - 1, ic2 - 1] - a[ir1 - 1, ic1 - 1])
        bot = a[ir2 - 1, ic1 - 1] + fc * (a[ir2 - 1, ic2 - 1] - a[ir2 - 1, ic1 - 1])
        return float(top + fr * (bot - top))


def _stein(age):
    """GDH1 heat flow of Stein & Stein 1992 (Assign lines 418-422)."""
    if age <= GDH1_AGE_SPLIT:
        return GDH1_A / math.sqrt(age)
    return GDH1_B + GDH1_C * math.exp(GDH1_K * age)


class Squeezer:
    """SUBROUTINE Squeez. The reference profile depends only on tSurf and the densities, so it is built once (Squeez builds
    it in every call because `called` is never set to .TRUE.; the values are identical)."""
    def __init__(self, p):
        self.p = p
        rb, aT = p["rhoBar"], p["alphaT"]
        rho_top = rb[0] * (1.0 - aT[0] * p["tSurf"])
        d = [0.0] * (NDREF + 1)
        d[1] = p["rhoH2O"]; d[2] = p["rhoH2O"]
        d[3] = REF_W1 * p["rhoH2O"] + REF_W2 * rho_top
        d[4] = d[5] = d[6] = d[7] = rho_top
        d[8] = REF_W1 * rho_top + REF_W2 * p["rhoAst"]
        for j in range(9, NDREF + 1):
            d[j] = p["rhoAst"]
        pr = [0.0] * (NDREF + 1)
        for i in range(1, NDREF + 1):
            pr[i] = pr[i - 1] + d[i] * p["gMean"] * p["oneKm"]
        self.pref = pr

    def sigzzb(self, drho, elevat, g, zm, zstop):
        """g = (geoth1..geoth8); returns (tauZZ, sigZZB)."""
        p = self.p
        rb, aT, tl = p["rhoBar"], p["alphaT"], p["temLim"]
        rhoH2O, oneKm, gMean = p["rhoH2O"], p["oneKm"], p["gMean"]
        g1, g2, g3, g4, g5, g6, g7, g8 = g
        pref = self.pref

        def tempC(h):
            return min(tl[0], g1 + g2 * h + g3 * h ** 2 + g4 * h ** 3)

        def tempM(h):
            return min(tl[1], g5 + g6 * h + g7 * h ** 2 + g8 * h ** 3)

        if elevat > 0.0:
            ztop = -elevat
            zbase = zstop - elevat
            dense1 = rb[0] * (1.0 - g1 * aT[0]) + drho
            h = 0.0
            layer1 = 1
        else:
            ztop = 0.0
            zbase = zstop + (-elevat)
            dense1 = rhoH2O
            h = elevat
            layer1 = 0
        lastdr = int(zbase / oneKm)
        if zbase > oneKm * lastdr:
            lastdr += 1
        if lastdr > NDREF:
            raise RuntimeError("Squeez: nDRef must be increased to at least %d" % lastdr)
        nstep = int((zbase - ztop) / oneKm)
        old_szz = 0.0
        old_pr = 0.0
        sig = 0.0
        tau = 0.0
        z = ztop
        for _ in range(nstep):
            z = z + oneKm
            h = h + oneKm
            if h > 0.0:
                if h <= zm:
                    T = tempC(h)
                    dense2 = rb[0] * (1. - T * aT[0]) + drho
                    layer2 = 1
                else:
                    T = tempM(h - zm)
                    dense2 = rb[1] * (1. - T * aT[1]) + drho
                    layer2 = 2
            else:
                dense2 = rhoH2O
                layer2 = 0
            if layer1 == 0 and layer2 == 1:
                frac2 = h / oneKm
                frac1 = 1.0 - frac2
            elif layer1 == 1 and layer2 == 2:
                frac2 = (h - zm) / oneKm
                frac1 = 1.0 - frac2
            else:
                frac1 = 0.5
                frac2 = 0.5
            dense = frac1 * dense1 + frac2 * dense2
            if z > 0.0:
                n1 = int(z / oneKm)
                frac = (z / oneKm) - n1
                pr = pref[n1] + frac * (pref[n1 + 1] - pref[n1])
            else:
                pr = 0.0
            sig = sig - dense * gMean * oneKm + (pr - old_pr)
            tau = tau + 0.5 * (sig + old_szz) * oneKm
            dense1 = dense2
            old_szz = sig
            old_pr = pr
            layer1 = layer2
        resid = zbase - z
        h = zstop
        z = zbase
        if zstop <= zm:
            T = tempC(h)
            dense2 = rb[0] * (1.0 - T * aT[0]) + drho
        else:
            T = tempM(h - zm)
            dense2 = rb[1] * (1.0 - T * aT[1]) + drho
        dense = 0.5 * (dense1 + dense2)
        if z > 0.0:
            n1 = int(z / oneKm)
            frac = (z / oneKm) - n1
            pr = pref[n1] + frac * (pref[n1 + 1] - pref[n1])
        else:
            pr = 0.0
        sigb = sig - dense * gMean * resid + (pr - old_pr)
        tau = tau + 0.5 * (sigb + old_szz) * resid
        return tau, sigb


def assign_one(plon, plat, elevat, heat, p, tasth, grids, sq, heat_grid=None):
    """SUBROUTINE Assign for one node. Returns (elev, heat, thickC, thickM, chem_drho, cooling_curv, badP, badT)."""
    conduc, radio = p["conduc"], p["radio"]
    tsurf = p["tSurf"]
    gMean = p["gMean"]
    # elevation, if 0
    if elevat == 0.0:
        elevat = grids["elev"].lookup(plon, plat)
    # sea-floor age, always
    age = grids["age"].lookup(plon, plat)
    # heat flow, if 0
    if heat == 0.0:
        if heat_grid is None:
            raise NotImplementedError("node with heat flow 0 needs the heat-flow grid (OrbData5 unit 4); pass heat_grid")
        heat = heat_grid.lookup(plon, plat, add=GRID_FRAC_ADD_Q)
        if age < AGE_MAX_MA:
            if age <= 0.0:
                heat = QLIM1
            else:
                heat = _stein(age)
    qlimit = QLIM0 + DQL_DE * elevat
    heat = max(heat, qlimit)
    heat = min(heat, QLIM1)
    # crust thickness
    thick_c = grids["crust"].lookup(plon, plat)
    thick_c = max(thick_c, CLIMIT)
    thick_c = min(thick_c, HCMAX)
    if age < AGE_MAX_MA:
        q_gdh1 = _stein(age)
        q_gdh1 = max(q_gdh1, qlimit)
        q_gdh1 = min(q_gdh1, QLIM1)
        delta_t = tasth - tsurf
        h_earth3 = (delta_t * conduc[1]) / (q_gdh1 - Q_RADIOACTIVITY)
        h_earth5 = math.sqrt(h_earth3 * H_PLATE)
        thick_m = h_earth5 - thick_c
        thick_m = max(thick_m, 0.0)
        thick_m = min(thick_m, HLMAX - thick_c)
    else:
        delta_ts = grids["dts"].lookup(plon, plat)
        slope = SLOPE_NUM / SLOPE_DEN
        thick_m = (RIDGE_DELTA_TS - delta_ts) / slope
        thick_m = max(thick_m, 0.0)
        thick_m = min(thick_m, HLMAX - thick_c)
    # steady-state geotherm
    geoth1 = tsurf
    geoth2 = heat / conduc[0]
    geoth3 = -radio[0] / (2.0 * conduc[0])
    geoth4 = 0.0
    tmoho = geoth1 + geoth2 * thick_c + geoth3 * thick_c ** 2
    geoth5 = tmoho
    qred = heat - thick_c * radio[0]
    geoth6 = qred / conduc[1]
    geoth7 = -radio[1] / (2.0 * conduc[1])
    geoth8 = 0.0
    test = geoth1 + geoth2 * thick_c + geoth3 * thick_c ** 2
    if test > tasth:
        terr = test - tasth
        dq = -terr * conduc[0] / thick_c
        heat = heat + dq
        qlimit = QLIM0 + DQL_DE * elevat
        heat = max(heat, qlimit)
        heat = min(heat, QLIM1)
        geoth2 = heat / conduc[0]
        tmoho = tasth
        geoth5 = tmoho
        qred = heat - thick_c * radio[0]
        geoth6 = qred / conduc[1]
    # trial cooling curvature
    test = geoth5 + geoth6 * thick_m + geoth7 * thick_m ** 2
    terr = test - tasth
    total_lith = thick_c + thick_m
    delta_quad = -terr / total_lith ** 2
    cooling = -2.0 * delta_quad
    if cooling > 0.0:
        t_geoth1 = tsurf
        t_geoth2 = heat / conduc[0]
        t_geoth3 = delta_quad - radio[0] / (2.0 * conduc[0])
        t_tmoho = t_geoth1 + t_geoth2 * thick_c + t_geoth3 * thick_c ** 2
        t_geoth5 = t_tmoho
        t_qred = heat - thick_c * radio[0] - cooling * thick_c * conduc[1]
        t_geoth6 = qred / conduc[1]          # as written in Assign (uses qRed, not t_qRed)
        t_geoth7 = delta_quad - radio[1] / (2.0 * conduc[1])
        c0 = t_geoth6
        c1 = 2. * t_geoth7
        z_max = c0 / (-c1) if c1 != 0.0 else math.copysign(math.inf, c0)
        if z_max < thick_m and z_max > 0.0:
            z_star = (geoth7 * thick_c ** 2 + geoth5 - geoth6 * thick_c - tasth) / (geoth7 * thick_c - 0.5 * geoth6)
            m_star = z_star - thick_c
            thick_m = max(min(m_star, thick_m), 0.0)
            total_lith = thick_c + thick_m
            terr = (geoth5 + geoth6 * thick_m + geoth7 * thick_m ** 2) - tasth
            delta_quad = -terr / total_lith ** 2
            cooling = -2.0 * delta_quad
    # final geotherm
    geoth1 = tsurf
    geoth2 = heat / conduc[0]
    geoth3 = delta_quad - radio[0] / (2.0 * conduc[0])
    geoth4 = 0.0
    tmoho = geoth1 + geoth2 * thick_c + geoth3 * thick_c ** 2
    geoth5 = tmoho
    qred = heat - thick_c * radio[0] - cooling * thick_c * conduc[1]
    geoth6 = qred / conduc[1]
    geoth7 = delta_quad - radio[1] / (2.0 * conduc[1])
    geoth8 = 0.0
    g = (geoth1, geoth2, geoth3, geoth4, geoth5, geoth6, geoth7, geoth8)
    drho = 0.0
    _, sigb = sq.sigzzb(drho, elevat, g, thick_c, thick_c + thick_m)
    drho = sigb / (gMean * total_lith)
    drho = min(drho, DELTA_RHO_LIMIT)
    drho = max(drho, -DELTA_RHO_LIMIT)
    _, sigb = sq.sigzzb(drho, elevat, g, thick_c, thick_c + thick_m)
    bad_p = abs(sigb) > BADP_PA
    test = geoth5 + geoth6 * thick_m + geoth7 * thick_m ** 2
    bad_t = abs(test - tasth) > BADT_K
    return elevat, heat, thick_c, thick_m, drho, cooling, bad_p, bad_t


def load_grids(inp_dir):
    return {"elev": Grid(os.path.join(inp_dir, GRID_FILES["elev"])),
            "age": Grid(os.path.join(inp_dir, GRID_FILES["age"])),
            "crust": Grid(os.path.join(inp_dir, GRID_FILES["crust"]), scale=KM_TO_M),
            "dts": Grid(os.path.join(inp_dir, GRID_FILES["dts"]))}


def assign_nodes(lon, lat, elev, heat, inp_dir, heat_grid=None, params=None, grids=None):
    """OrbData5 for all nodes. lon/lat in degrees, elev in m, heat in W/m^2 (0 = take from grid, as OrbData).
    Returns dict of arrays: elev, heat (after limits/edits), zM (crust thickness zMNode), tL (mantle lithosphere thickness
    tLNode), chem_drho, cooling_curv, badP, badT."""
    p = params or read_params(inp_dir)
    grids = grids or load_grids(inp_dir)
    tasth = p["tAdiab"] + p["gradie"] * TASTH_DEPTH          # OrbData5.f90 line 287
    sq = Squeezer(p)
    lon, lat, elev, heat = [np.asarray(a, dtype=np.float64) for a in (lon, lat, elev, heat)]
    n = len(lon)
    out = {k: np.zeros(n) for k in ("elev", "heat", "zM", "tL", "chem_drho", "cooling_curv")}
    out["badP"] = np.zeros(n, bool)
    out["badT"] = np.zeros(n, bool)
    for i in range(n):
        # GetNet stores colatitude/longitude in radians; OrbData5 converts back (lines 433-434)
        xn = (90.0 - lat[i]) * RAD_PER_DEG
        yn = lon[i] * RAD_PER_DEG
        plon = yn * DEG_PER_RAD
        plat = 90.0 - xn * DEG_PER_RAD
        q = float(heat[i])
        qlimit = QLIM0 + DQL_DE * elev[i]                       # impose limits at nodes (OrbData5.f90 lines 374-378)
        if q != 0.0:
            q = max(q, qlimit)
        q = min(q, QLIM1)
        e, h, tc, tm, dr, cc, bp, bt = assign_one(plon, plat, float(elev[i]), q, p, tasth, grids, sq, heat_grid)
        out["elev"][i] = e
        out["heat"][i] = h
        out["zM"][i] = max(tc, CLIMIT)
        out["tL"][i] = max(tm, 0.0)
        out["chem_drho"][i] = dr
        out["cooling_curv"][i] = cc
        out["badP"][i] = bp
        out["badT"][i] = bt
    return out


def read_feg_nodes(path):
    """Node block of a .feg (MOD_Data.f90 GetNet/PutNet): id lon lat elev heat zM tL chem_drho cooling_curv."""
    with open(path) as f:
        f.readline()
        n = int(f.readline().split()[0])
        rows = [f.readline().split() for _ in range(n)]
    return np.array(rows, dtype=np.float64)


if __name__ == "__main__":
    inp = sys.argv[1]
    nodes = read_feg_nodes(os.path.join(inp, "Earth5R.feg"))
    t0 = time.time()
    o = assign_nodes(nodes[:, 1], nodes[:, 2], nodes[:, 3], nodes[:, 4], inp)
    dt = time.time() - t0
    print("nodes %d, runtime %.1f s, badP %d, badT %d" % (len(nodes), dt, o["badP"].sum(), o["badT"].sum()))
    cols = [("elev", 3), ("heat", 4), ("zM", 5), ("tL", 6), ("chem_drho", 7), ("cooling_curv", 8)]
    print("%-13s %11s %11s %10s %10s %10s" % ("column", "max|diff|", "max rel", "n>print", "n>1unit", "n str-diff"))
    for name, c in cols:
        s, m = nodes[:, c], o[name]
        d = np.abs(m - s)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(s != 0, d / np.abs(s), np.where(d == 0, 0, np.inf))
        # print precision (ES10.2: 3 significant digits): one unit of the last printed digit of the stored value
        ex = np.where(s != 0, np.floor(np.log10(np.abs(np.where(s == 0, 1, s)))), 0)
        unit = 10.0 ** (ex - 2)
        mfmt = np.array([float("%.2E" % v) for v in m])
        over_half = int((np.abs(m - s) > 0.5 * unit * (1 + 1e-9)).sum())
        over_one = int((np.abs(mfmt - s) > 1.5 * unit).sum())
        strdiff = int((mfmt != s).sum())
        print("%-13s %11.4g %11.4g %10d %10d %10d" % (name, d.max(), rel.max(), over_half, over_one, strdiff))
