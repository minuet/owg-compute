"""Plate motions from mantle flow driven by subducted slabs (Lithgow-Bertelloni & Richards 1998, Rev. Geophys. 36:27;
Ricard & Vigny 1989, JGR 94:17543): the method that predicts Earth's Cenozoic plate motions with global correlations of
0.7-0.9 from the history of subduction alone. 2026-11-03: replaces random draws of plate speeds (user: speeds must come
from physics) - Earth test first (earth_mantle_flow_check.py).

Physics (LBR98 section 3.3): the mantle is a Newtonian viscous shell with radially varying viscosity, no-slip under the
plates and free slip at the core. The flow is instantaneous Stokes flow, solved by HC (Hager & O'Connell 1981 propagator
method as coded by Becker & O'Connell, github.com/geodynamics/hc - the solver is used as published, only driven from
here). Plates are rigid and in torque balance: the basal shear tractions excited by the density anomalies with the
surface held fixed (Q_int) plus those from the plates' own motions (linear in their Euler vectors, Q_res = M omega)
sum to zero on every plate, omega = -M^+ Q_int, net rotation undetermined (LBR98: removed by SVD).

Density anomalies (LBR98 section 4.3): slabs placed vertically below the trenches of the previous 180 Myr; per unit
trench length and time step a mass drho * H(age) * v_c * dt, with drho = 80 kg/m3 (Ricard et al. 1993), the plate
thickness H = 100 km * sqrt(age / 90 Myr), v_c the convergence rate; slabs sink at v_c in the upper mantle and v_c / 4
in the lower mantle (slowing factor s = 4 ~ ln of the viscosity contrast 50) and become inert at the core-mantle
boundary. Viscosity (LBR98 Table 3, best fit to the geoid): lithosphere 0-130 km 10 x upper mantle, upper mantle
(130-670 km) 1e21 Pa s, lower mantle 50 x. The absolute viscosity only scales all speeds (one constant, fitted on Earth
as LBR98 did). Not included: the oceanic lithosphere's own thickening (< 10 % of the driving forces in LBR98).

Spherical harmonics: real, fully normalised (Dahlen & Tromp 1998), with the Condon-Shortley phase; vector fields as
poloidal B = grad Y / sqrt(l(l+1)) and toroidal C = -r x grad Y / sqrt(l(l+1)) coefficients - HC's input/output
convention, checked against HC itself on Kaggle (owg-hc-probe, 2026-11-03: misfit 0 for l = 1, 2, m = 0, 1, both
cos and sin terms). HC runs at degree 31 (it needs 2^n - 1); as in LBR98 the inputs are truncated at degree 25 and the stresses at 20.
"""
import os, math, subprocess, tempfile
import numpy as np
from scipy.special import lpmv, gammaln

LMAX = 31                       # HC's expansion (it needs 2^n - 1)
LMAX_INPUT = 25                 # LBR98: densities and unit rotations expanded to degree 25
LMAX_STRESS = 20                # LBR98: induced shear stresses summed to degree 20 (converged, their Fig. 12)
DRHO_SLAB = 80.0                # kg/m3, slab - ambient mantle (LBR98 after Ricard et al. 1993)
H_SLAB_KM = 100.0               # plate thickness at 90 Myr (LBR98 eq. 13; half-space cooling)
H_SLAB_AGE = 90.0
SLOWING = 4.0                   # lower-mantle sinking slower by s = 4 ~ ln(50) (LBR98; Gurnis & Davies 1986; Richards 1991)
SINK_LM = float(os.environ["SINK_LM"]) if os.environ.get("SINK_LM") else None   # cm/yr lower-mantle slab sinking from slab
                                # tomography (van der Meer et al. 2010, 2018); off = LBR98's v_c / SLOWING
HISTORY_MYR = 180.0             # LBR98: the previous 180 Myr of subduction
STEP_MYR = 5.0                  # LBR98: slablets every 5 Myr
Z_670, Z_CMB_EARTH = 670.0, 2891.0
DENS_LAYER_KM = float(os.environ.get("DENS_LAYER_KM", "100"))   # env only for the convergence test;
                                         # density layers every 100 km (numerical; slabs are placed every v_c * 5 Myr ~ 25-500 km)
HC_COMMIT = "8010aed677752bc437ffea0f503309687832a788"
HC_AVG_DEN = 4448.8             # kg/m3, HC_AVG_DEN_MANTLE: with -dnp HC scales input % by this
HC_RE_KM = 6371.0087714         # HC's Earth radius (hc_constants.h)
G_HC = 10.0                     # m/s2, HC's gravity (hc_constants.h): slab pull uses the same g as the flow


ASTHENO = tuple(float(v) for v in os.environ["ASTHENO"].split(",")) if os.environ.get("ASTHENO") else None   # off


class Body:
    """Geometry of the convecting shell. HC solves the Stokes equations in non-dimensional form (radius 1, its Earth
    radius, gravity and reference viscosity as units: hc_polsol.c alpha = rho_scale * R * g / eta), so another planet is
    the same problem with its own scales: depths map to HC depths at the same fraction of the radius, density anomalies
    are multiplied by (g R)_planet / (g R)_HC (the buoyancy stress scale), plate velocities are handed to HC as the same
    angular velocities (v / R), and torques and forces are summed with the planet's own radius and gravity.
    g is HC's 10 m/s2 for Earth and scaled by the surface gravity ratio for a planet, so the Earth fit carries over."""

    def __init__(self, radius_km, g, z_660, z_cmb, z_lith=130.0, visc=(1e22, 1e21, 5e22)):
        self.R, self.g, self.z660, self.zcmb = radius_km, g, z_660, z_cmb
        self.eta_um = visc[1]                                     # upper-mantle viscosity, one source for HC.solve_plates
        self.visc = [(z_lith, visc[0]), (z_660, visc[1]), (z_cmb, visc[2])]   # (bottom depth km, Pa s), LBR98 Table 3
        if ASTHENO:
            # asthenosphere test (2026-11-09, PARAM_AUDIT): a low-viscosity layer from the lithosphere's base down to
            # ASTHENO's depth (Earth km, scaled like z_660), its viscosity a factor of the upper mantle's (Becker 2017)
            z_a, f_a = ASTHENO
            self.visc.insert(1, (z_a * z_660 / Z_670, visc[1] * f_a))

    def hc_depth(self, z_km):
        return np.asarray(z_km, float) * HC_RE_KM / self.R

    def buoyancy_factor(self):
        return self.g * self.R / (G_HC * HC_RE_KM)


EARTH = Body(6371.0, G_HC, Z_670, Z_CMB_EARTH)
# The planet (CLAUDE.md: radius 6000 km, surface gravity 9.0 m/s2, Earth's composition and core mass fraction):
#  - g: HC's 10 m/s2 x 9.0 / 9.82 (Earth's surface gravity)
#  - 660-km transition at the same pressure as on Earth (~23.5 GPa; it is the ringwoodite breakdown, a pressure-
#    controlled phase change): with upper-mantle densities as on Earth at the same pressure and g nearly constant through
#    the upper mantle (PREM: 9.82 at the surface, 10.0 at 670 km), depth scales as 1 / g: 670 * 9.82 / 9.0 = 731 km
#  - core: the core radius fraction depends on the core mass fraction and hardly on the planet's mass (Zeng et al. 2016,
#    ApJ 819:127, CRF ~ CMF^0.5), so the same fraction as Earth's (3480 / 6371 = 0.546): CMB at 6000 * 0.454 = 2723 km
#    depth. HC itself requires a core radius within 0.02 of 0.55 (hc_init.c), which this is.
#  - lithosphere 130 km, viscosities, slab density excess and thickness: set by temperature and composition (same as
#    Earth), not by the planet's size
G_PLANET_SURFACE, G_EARTH_SURFACE = 9.0, 9.82
PLANET = Body(6000.0, G_HC * G_PLANET_SURFACE / G_EARTH_SURFACE, Z_670 * G_EARTH_SURFACE / G_PLANET_SURFACE,
              6000.0 * (1 - 3480.0 / 6371.0))


def build_hc(dest="/tmp/hc"):
    """Clone the pinned HC version and build it without GMT (Linux / Kaggle). A file lock makes runs started together
    (multi_seed.py) build it once; without it two clones into the same folder collided (2026-11-08, v20str seed 5)."""
    import fcntl
    # off the cloud all traffic goes through the user's proxy (CLAUDE.md "网络"); Kaggle and GitHub Actions use their own network
    proxy = "" if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("GITHUB_ACTIONS") else "-c http.proxy=socks5h://192.168.0.6:1080 "
    with open(f"{dest}.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if not os.path.exists(f"{dest}/bin/hc"):
            subprocess.run(f"rm -rf {dest} && git {proxy}clone https://github.com/geodynamics/hc.git {dest} && "
                           f"git -C {dest} checkout -q {HC_COMMIT} && "
                           f"make -C {dest} all_no_gmt ADD_FLAGS=-DNO_GMT > {dest}.build.log 2>&1", shell=True, check=True)
    return f"{dest}/bin"


# ---------------------------------------------------------------- spherical harmonics on scattered (equal-area) points
SH_FD_H = 1e-5                  # rad, technical: finite-difference step of the harmonic gradients (audit 2026-11-11)


class SH:
    """Real Dahlen & Tromp harmonics and their gradients at points xyz (unit vectors), for analysis by equal-weight
    quadrature (Fibonacci grids: every point represents the same area) and synthesis."""

    def __init__(self, xyz, lmax=LMAX):
        self.L = lmax
        x, y, z = xyz.T
        th = np.arccos(np.clip(z, -1, 1)); ph = np.arctan2(y, x)
        self.th, self.ph = th, ph
        self.e_th = np.c_[np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)]
        self.e_ph = np.c_[-np.sin(ph), np.cos(ph), np.zeros_like(ph)]
        self.idx = [(l, m, k) for l in range(lmax + 1) for m in range(l + 1) for k in ((0,) if m == 0 else (0, 1))]
        n, K = len(xyz), len(self.idx)
        self.Y = np.zeros((K, n)); self.Gt = np.zeros((K, n)); self.Gp = np.zeros((K, n))
        st = np.sin(th).clip(1e-12); h = SH_FD_H
        for i, (l, m, k) in enumerate(self.idx):
            nrm = math.exp(0.5 * (math.log((2 * l + 1) / (4 * math.pi)) + gammaln(l - m + 1) - gammaln(l + m + 1)))
            f = nrm * (1.0 if m == 0 else math.sqrt(2.0))
            X = f * lpmv(m, l, np.cos(th))
            dX = f * (lpmv(m, l, np.cos(th + h)) - lpmv(m, l, np.cos(th - h))) / (2 * h)
            c, s = np.cos(m * ph), np.sin(m * ph)
            trig, dtrig = (c, -m * s) if k == 0 else (s, m * c)
            self.Y[i] = X * trig; self.Gt[i] = dX * trig; self.Gp[i] = X * dtrig / st
        self.w = 4 * math.pi / n
        ll = np.array([l for l, m, k in self.idx], float)
        self.sq = np.sqrt(ll * (ll + 1)); self.sq[0] = 1.0

    def scalar_coefs(self, f):
        return (self.Y @ f) * self.w

    def vector_coefs(self, v):
        """v: (n, 3) tangent vectors -> poloidal, toroidal coefficients (l = 0 terms zero)."""
        vt, vp = np.einsum("ij,ij->i", v, self.e_th), np.einsum("ij,ij->i", v, self.e_ph)
        P = (self.Gt @ vt + self.Gp @ vp) * self.w / self.sq
        T = (self.Gp @ vt - self.Gt @ vp) * self.w / self.sq
        P[0] = T[0] = 0.0
        return P, T

    def vector_synth(self, P, T):
        """poloidal, toroidal coefficients -> (n, 3) tangent vectors."""
        a, b = P / self.sq, T / self.sq
        vt = a @ self.Gt + b @ self.Gp
        vp = a @ self.Gp - b @ self.Gt
        return vt[:, None] * self.e_th + vp[:, None] * self.e_ph

    def _lines(self, sets):
        """HC long-format coefficient lines, (l, m) order, each 'a b' per set (cos, sin terms); degrees above
        LMAX_INPUT are zero (LBR98)."""
        pos = {key: i for i, key in enumerate(self.idx)}
        out = []
        for l in range(self.L + 1):
            if l > LMAX_INPUT:
                out += [" ".join(["0"] * (2 * len(sets)))] * (l + 1)
                continue
            for m in range(l + 1):
                row = []
                for c in sets:
                    row += [c[pos[(l, m, 0)]], c[pos[(l, m, 1)]] if m else 0.0]
                out.append(" ".join("%.9e" % v for v in row))
        return out

    def write_vel(self, fn, P, T):
        open(fn, "w").write(f"{self.L} 0 0 1 2 0\n" + "\n".join(self._lines([P, T])) + "\n")

    def write_dens(self, fn, depths_km, fields_pct):
        """layers deepest first (HC: radius must increase); fields in % of HC's mean mantle density (-dnp -ds 1)."""
        order = np.argsort(depths_km)[::-1]
        txt = []
        for j, i in enumerate(order):
            txt.append(f"{self.L} {j} {depths_km[i]:.1f} {len(order)} 1 0")
            txt += self._lines([self.scalar_coefs(fields_pct[i])])
        open(fn, "w").write("\n".join(txt) + "\n")


def min_plate_share(lmax=None):
    """smallest plate (share of the sphere) the flow solve resolves (merge_unresolved's limit)"""
    half_wl = math.pi / ((LMAX_STRESS if lmax is None else lmax) + 0.5)
    return 2 * math.pi * (1 - math.cos(half_wl / 2)) / (4 * math.pi)


def merge_unresolved(xyz, plate, lmax):
    """Plates smaller than a cap whose diameter is half the shortest wavelength of the expansion (2 pi R / (lmax + 1/2)
    / 2) cannot be resolved by it - their resisting torques fall below numerical precision (LBR98 section 5.3.2) - and
    move with the neighbour they share most boundary with. Returns the new plate array and {small: neighbour}."""
    from fastkd import cKDTree
    half_wl = math.pi / (lmax + 0.5)                  # radians
    a_min = 2 * math.pi * (1 - math.cos(half_wl / 2)) / (4 * math.pi)     # share of the sphere
    plate = plate.copy()
    nb = cKDTree(xyz).query(xyz, k=7)[1][:, 1:]
    merged = {}
    while True:
        ids, cnt = np.unique(plate, return_counts=True)
        small = [(c, int(p)) for p, c in zip(ids, cnt) if c / len(plate) < a_min]
        if not small or len(ids) < 3:      # guard: keep at least 2 plates (one plate has no relative motion to solve)
            break
        c, p = min(small)
        m = plate == p
        other = plate[nb[m]].ravel(); other = other[other != p]
        if not len(other):
            break
        q = int(np.bincount(other - other.min()).argmax() + other.min())
        plate[m] = q
        for a in [a for a, b in merged.items() if b == p]:
            merged[a] = q
        merged[p] = q
    return plate, merged


# ---------------------------------------------------------------- HC runs
KEEL_K = float(os.environ.get("KEEL_K", "1"))   # 1 = off; see HC.torques


class HC:
    def __init__(self, sh, body=None, bindir=None, work=None):
        self.sh = sh
        self.body = body or EARTH
        self.bin = bindir or build_hc()
        self.dir = work or tempfile.mkdtemp(prefix="hcrun_")
        r = lambda z: 1.0 - z / self.body.R                               # non-dimensional radius
        rows = sorted((r(zb), eta) for zb, eta in self.body.visc)         # value applies upward from the layer bottom
        open(f"{self.dir}/visc.dat", "w").write("\n".join(f"{rr:.6f} {eta:.4e}" for rr, eta in rows) + "\n")
        open(f"{self.dir}/zero.sh", "w").write(f"{sh.L} 0 1000 1 1 0\n" + "\n".join(
            "0 0" for l in range(sh.L + 1) for m in range(l + 1)) + "\n")

    def _surface_traction(self, args):
        for f in ("rtrac.sol.bin",):
            if os.path.exists(f"{self.dir}/{f}"):
                os.remove(f"{self.dir}/{f}")
        subprocess.run(f"{self.bin}/hc {args} -prem {os.path.dirname(self.bin)}/prem/prem.dat -vf visc.dat -rtrac -ng > hc.log 2>&1", shell=True, cwd=self.dir)
        out = subprocess.run(f"{self.bin}/hc_extract_spatial rtrac.sol.bin -1 3", shell=True, cwd=self.dir,
                             capture_output=True, text=True).stdout
        a = np.loadtxt(out.splitlines())
        if a.ndim != 2 or len(a) == 0:
            raise RuntimeError(open(f"{self.dir}/hc.log").read()[-3000:])
        return a                                     # lon lat z srr srt srp on HC's Gauss grid, MPa

    def traction_field(self, args):
        """surface shear traction (MPa) re-expanded on the analysis points: HC's Gauss-grid output is band-limited to
        LMAX, analysed exactly by Gauss quadrature, then synthesised where the plates are."""
        a = self._surface_traction(args)
        lon, lat = np.radians(a[:, 0]), np.radians(a[:, 1])
        xyz = np.c_[np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
        g = getattr(self, "_g", None)
        if g is None or len(g.th) != len(xyz):
            g = self._g = SH(xyz, self.sh.L)
            nlat = len(np.unique(a[:, 1])); nlon = len(a) // nlat
            x, wt = np.polynomial.legendre.leggauss(nlat)        # Gauss weights in latitude (HC: Gauss latitudes)
            lat_u = np.degrees(np.arcsin(x))
            wl = np.interp(a[:, 1], lat_u, wt)
            self._gw = wl * 2 * math.pi / nlon
        vt, vp = a[:, 4], a[:, 5]
        P = (g.Gt @ (vt * self._gw) + g.Gp @ (vp * self._gw)) / g.sq
        T = (g.Gp @ (vt * self._gw) - g.Gt @ (vp * self._gw)) / g.sq
        P[0] = T[0] = 0.0
        hi = np.array([l > LMAX_STRESS for l, m, k in g.idx])
        P[hi] = T[hi] = 0.0
        return self.sh.vector_synth(P, T)

    def torques(self, tau, plate, ids):
        """torque (N m) of the basal traction tau (MPa) on each plate: sum of r x tau dA over equal-area points."""
        r_m = self.body.R * 1e3
        dA = 4 * math.pi * r_m ** 2 / len(self.sh_xyz)
        cm_ = getattr(self, "cont_mask", None)
        if KEEL_K != 1.0 and cm_ is not None and len(cm_) == len(tau):
            # continental roots (KEEL_K, sensitivity test 2026-11-07): the basal traction under continents is KEEL_K times
            # that of the radial-viscosity flow, for driving and resisting alike (Conrad & Lithgow-Bertelloni 2006, GRL:
            # roots raise plate-mantle coupling 2-5x); the caller sets cont_mask on the points and clears the cached _M
            tau = tau * np.where(cm_, KEEL_K, 1.0)[:, None]
        tq = np.cross(self.sh_xyz * r_m, tau * 1e6) * dA
        return np.array([tq[plate == p].sum(axis=0) for p in ids]).ravel()

    def solve_plates(self, xyz, plate, dens_depths, dens_fields_pct, pull=None, slab_resist=0.0,
                     eta_um=None, unit_deg_myr=1.0):
        """Euler vectors (rad/Myr, for HC's reference viscosities; the absolute viscosity scales them) of the plates on
        points xyz; dens_depths in km below the body's surface, dens_fields_pct its density anomalies in % of
        HC_AVG_DEN (scaled here to HC's units by Body.buoyancy_factor). pull: rows (x, y, z, nx, ny, nz, plate, force_N, ds_km) of slab-pull forces at trench points (CLB02);
        slab_resist: dimensionless coefficient c of the slab resistance c * eta_um * v_n per unit trench length
        (eta_um defaults to the body's upper-mantle viscosity)."""
        eta_um = self.body.eta_um if eta_um is None else eta_um
        self.sh_xyz = xyz
        plate, merged = merge_unresolved(xyz, plate, LMAX_STRESS)
        if pull is not None and len(pull):
            pull = pull.copy()
            for a, b in merged.items():
                pull[pull[:, 6] == a, 6] = b
        ids = [int(p) for p in np.unique(plate)]
        self.sh.write_dens(f"{self.dir}/dens.sh", self.body.hc_depth(dens_depths),
                           np.asarray(dens_fields_pct) * self.body.buoyancy_factor())
        q_int = self.torques(self.traction_field("-dens dens.sh -dnp -ds 1 -ns"), plate, ids)
        radius_km = self.body.R
        q_pull = np.zeros_like(q_int)
        if pull is not None and len(pull):
            pos = {p: j for j, p in enumerate(ids)}
            tq = np.cross(pull[:, 0:3] * radius_km * 1e3, pull[:, 3:6] * pull[:, 7:8])
            for row, t_ in zip(pull, tq):
                j = pos.get(int(row[6]))
                if j is not None:
                    q_pull[3 * j:3 * j + 3] += t_
        om = math.radians(unit_deg_myr)
        cache = getattr(self, "_M", None)
        if cache is not None and cache[0].shape == plate.shape and np.array_equal(cache[0], plate) and cache[1] == om:
            M = cache[2]                                  # same plates: the resisting matrix does not change
        else:
            M = np.zeros((3 * len(ids), 3 * len(ids)))
            for j, p in enumerate(ids):
                sel = plate == p
                for k in range(3):
                    e = np.zeros(3); e[k] = om
                    # HC's velocity unit is its own radius: the same angular velocity is the same non-dimensional flow
                    v = np.zeros_like(xyz); v[sel] = np.cross(e, xyz[sel]) * HC_RE_KM / 10.0       # cm/yr
                    P, T = self.sh.vector_coefs(v)
                    self.sh.write_vel(f"{self.dir}/pvel.sh", P, T)
                    M[:, 3 * j + k] = self.torques(self.traction_field("-pvel pvel.sh -dens zero.sh"), plate, ids)
            self._M = (plate.copy(), om, M)
        # sign convention of HC's tractions: the diagonal blocks of M are torques of a plate's own motion, which resist
        # it, so their sign says whether HC's traction is the force on the plate (negative) or on the mantle (positive);
        # the slab pull (a force on the plate) and the slab resistance must enter in the same convention
        sgn = float(np.sign(np.mean([np.trace(M[3 * j:3 * j + 3, 3 * j:3 * j + 3]) for j in range(len(ids))])))
        if slab_resist > 0 and pull is not None and len(pull):
            # slab resistance (Forsyth & Uyeda 1975): the mantle resists the attached slab's descent, a viscous force per
            # unit trench length c * eta_um * v_n against the plate's motion normal to the trench - linear in the Euler
            # vector, so it adds to the resisting matrix (same sign convention as HC's resisting torques)
            pos = {p: j for j, p in enumerate(ids)}
            r_m = radius_km * 1e3
            M = M.copy()
            for row in pull:
                j = pos.get(int(row[6]))
                if j is None:
                    continue
                a = np.cross(row[0:3], row[3:6])                      # r x n
                ds_m = row[8] * 1e3 if len(row) > 8 else 0.0
                M[3 * j:3 * j + 3, 3 * j:3 * j + 3] += sgn * slab_resist * eta_um * ds_m * r_m ** 2 * om / 3.15576e13 * np.outer(a, a)
        U, s, Vt = np.linalg.svd(M)
        s_inv = np.where(np.arange(len(s)) < len(s) - 3, 1.0 / np.where(s > 0, s, 1.0), 0.0)   # drop net rotation
        x = -(Vt.T * s_inv) @ (U.T @ (q_int - sgn * q_pull))        # q_pull is on the plate: -sgn puts it in HC's sign
        # operator for extra on-plate torques (contacts, planet_flow): d omega = sgn * Minv @ torque * om (same algebra)
        self.last_op = {"ids": ids, "Minv": (Vt.T * s_inv) @ U.T, "sgn": sgn, "om": om, "merged": dict(merged),
                        "q_int": q_int, "q_pull": q_pull}                  # kept for diagnostics (planet_flow DEBUG_FLOW)
        info = {"sing_min4": s[-4:].tolist(), "sing_max": float(s[0]), "hc_traction_sign": sgn,
                "torque_suction_Nm": float(np.linalg.norm(q_int)), "torque_pull_Nm": float(np.linalg.norm(q_pull))}
        w = {p: x[3 * j:3 * j + 3] * om for j, p in enumerate(ids)}
        for a, b in merged.items():
            w[a] = w[b]
        info["merged"] = {int(a): int(b) for a, b in merged.items()}
        return w, info


# ---------------------------------------------------------------- slabs
# Slablet rows: t_created, x, y, z, v_c (cm/yr), H (km), ds (km), nx, ny, nz (trench normal towards the overriding
# plate), sinking plate id.
PULL_FRACTION = 1.0             # share of the attached upper-mantle slab weight acting as pull (CLB02: the observed
                                # 3.6 x speed ratio of subducting to other plates needs nearly all; fit flat 40-100 %)


def slab_depth(age, vk, body):
    """depth (km) of slablets age Myr after subduction at convergence vk (km/Myr): v_c in the upper mantle (LBR98); below
    670 km v_c / SLOWING (LBR98) or, with SINK_LM, the lower-mantle rate slab tomography finds nearly the same
    everywhere (van der Meer et al. 2010, 2018)."""
    t670 = body.z660 / np.maximum(vk, 1e-9)
    v_lm = vk / SLOWING if SINK_LM is None else SINK_LM * 10.0
    return np.where(age <= t670, vk * age, body.z660 + v_lm * (age - t670))


def slab_state(slablets, t_now, body=None):
    """alive slablets at t_now: rows, depth (km), mass (kg). LBR98: sink at v_c in the upper mantle, v_c / SLOWING
    below 670 km, inert at the CMB (removed by the caller's z_cmb)."""
    body = body or EARTH
    s = slablets[(slablets[:, 0] > t_now) & (slablets[:, 0] <= t_now + HISTORY_MYR)]
    age = s[:, 0] - t_now
    vk = s[:, 4] * 10.0                                         # km/Myr
    z = slab_depth(age, vk, body)
    mass = DRHO_SLAB * s[:, 5] * s[:, 4] * 10.0 * STEP_MYR * s[:, 6] * 1e9     # kg
    return s, z, mass


def attached_pull(s, z, mass, trenches, body=None):
    """Conrad & Lithgow-Bertelloni 2002 (Science 298:207), combined model: upper-mantle slabs attached to the plates
    sinking at present trenches pull them (about the whole upper-mantle slab weight, CLB02 fig. S1; total 1.9e21 N on
    today's Earth); all other slab material drives the mantle flow (suction). The pull per trench point is the textbook
    slab pull (Turcotte & Schubert; Forsyth & Uyeda 1975): weight of a slab of thickness H(age) hanging from the trench
    to the base of the upper mantle, drho g H (Z_670 - H) ds, applied normal to the trench towards the overriding plate.
    Attached material removed from the flow: upper-mantle slablets of plates that are sinking at a present trench.
    trenches: current trench rows (slablet columns). Returns (pull rows for solve_plates, mask of attached slablets)."""
    body = body or EARTH
    if not len(trenches):
        return np.zeros((0, 9)), np.zeros(len(s), bool)
    H = trenches[:, 5]
    force = PULL_FRACTION * DRHO_SLAB * body.g * H * 1e3 * np.clip(body.z660 - H, 0, None) * 1e3 * trenches[:, 6] * 1e3   # N
    pull = np.c_[trenches[:, 1:4], trenches[:, 7:10], trenches[:, 10], force, trenches[:, 6]]
    att = (z < body.z660) & np.isin(s[:, 10], np.unique(trenches[:, 10])) if len(s) else np.zeros(0, bool)
    return pull, att


def body_pull(s, z, mass, live_ids, locate, side_km, body=None):
    """Slab pull from the slab body (2026-11-07; CLB02 combined model as intended): each upper-mantle slablet pulls,
    with its own weight PULL_FRACTION m g, at its own place and along its recorded subduction direction, the plate it
    was subducted from - its recorded plate if that id still exists, else the oceanic plate now on its sinking side
    (locate(points) -> plate ids, oceanic mask; the probe is side_km across towards the sinking side). attached_pull
    instead hangs the textbook slab from the trench line of the moment, whose segments come and go from step to step
    (Earth test: swapping in the next step's trench lines alone turns the ridges 8 deg, the whole jitter). The slab
    resistance's trench width is shared by the slablets in proportion to their length (rate x step / z660), so a full
    upper-mantle slab carries the trench width once. Rows and mask as attached_pull."""
    body = body or EARTH
    um = z < body.z660
    if not um.any():
        return np.zeros((0, 9)), um
    x, n, su = s[um, 1:4], s[um, 7:10], s[um, 10].astype(int)
    probe = x - n * (side_km / body.R)
    probe /= np.linalg.norm(probe, axis=1)[:, None]
    pid, ocn = locate(probe)
    has = np.isin(su, np.asarray(list(live_ids), int))
    pid = np.where(has, su, pid)
    ok = (pid >= 0) & (ocn | has)
    att = np.zeros(len(s), bool)
    att[np.nonzero(um)[0][ok]] = True
    frac = np.clip(s[um, 4][ok] * 10.0 * STEP_MYR / body.z660, 0, 1)
    return np.c_[x[ok], n[ok], pid[ok], PULL_FRACTION * mass[um][ok] * body.g, s[um, 6][ok] * frac], att


# ridge push (2026-11-06): the third textbook plate-driving force (Forsyth & Uyeda 1975; Turcotte & Schubert,
# Geodynamics 6-20), missing from the slab-only flow model (LBR98 left it out as minor for plate speeds). It acts where
# ridges are, so without it nothing resists a ridge closing again: in the planet's Earth-start runs young (< 10 Myr)
# lithosphere made ~0.55 of the slab-free closing boundary length (Earth ~0.2) and ~5-10x Earth's length.
# Half-space cooling: F(t) = g rho_m a_v dT [1 + 2 rho_m a_v dT / (pi (rho_m - rho_w))] kappa t per unit ridge length.
# The depths of the planet follow GDH1 (Stein & Stein 1992), which flattens with age, so kappa t is taken from the GDH1
# subsidence w = 2 rho_m a_v dT / (rho_m - rho_w) sqrt(kappa t / pi) (the same half-space relation) - F(100 Myr) =
# 2.8e12 N/m (textbook half-space value ~3.9e12; quoted range 2-4e12). Distributed as a body force on the aging
# lithosphere, dF/dx (Wiens & Stein 1985: ridge push is the pressure gradient of the cooling plate, not a line force).
RP_RHO_M, RP_RHO_W = 3300.0, 1000.0   # kg/m3, Turcotte & Schubert ch. 4/6 values
RP_ALPHA, RP_DT = 3e-5, 1280.0        # 1/K and K (T1 - T0), Turcotte & Schubert ch. 4/6 values
RP_GDH1_RIDGE_M = 2600.0              # GDH1 ridge depth (Stein & Stein 1992), the zero of the subsidence


def ridge_push_force(age_myr, g):
    """Ridge push F (N per m of ridge) of lithosphere of the given age, GDH1-consistent half-space formula above."""
    a = np.asarray(age_myr, float)
    t = np.maximum(a, 0.0)
    from ocean_depth_check import gdh1
    gd = gdh1(t)
    w = np.maximum(gd - RP_GDH1_RIDGE_M, 0.0)
    c = 2 * RP_RHO_M * RP_ALPHA * RP_DT / (RP_RHO_M - RP_RHO_W)
    kt = math.pi * (w / c) ** 2                                     # kappa t (m2)
    return g * RP_RHO_M * RP_ALPHA * RP_DT * (1 + 2 * RP_RHO_M * RP_ALPHA * RP_DT / (math.pi * (RP_RHO_M - RP_RHO_W))) * kt


def ridge_push_rows(xyz, nb, plate, age_myr, cont, body=None):
    """Ridge push as on-plate forces at the oceanic points of a grid (rows as attached_pull: x, y, z, nx, ny, nz,
    plate, force_N, ds_km = 0 so no slab resistance): force = grad F x cell area, F from ridge_push_force; the gradient
    from the neighbours on the same plate that are oceanic (none across ridges, trenches or coasts)."""
    body = body or EARTH
    n = len(xyz)
    r_m = body.R * 1e3
    ocean = ~cont & np.isfinite(age_myr)
    F = np.where(ocean, ridge_push_force(np.nan_to_num(age_myr, nan=0.0), body.g), 0.0)
    ok = ocean[nb] & ocean[:, None] & (plate[nb] == plate[:, None])
    d = (xyz[nb] - xyz[:, None, :]) * r_m                                  # m
    d -= np.einsum("ikj,ij->ik", d, xyz)[:, :, None] * xyz[:, None, :]      # tangent part
    dF = (F[nb] - F[:, None]) * ok
    den = (np.einsum("ikj,ikj->ik", d, d) * ok).sum(axis=1)
    grad = 2.0 * np.einsum("ik,ikj->ij", dF, d) / np.where(den > 0, den, np.inf)[:, None]   # N/m2 (isotropic stencil)
    mag = np.linalg.norm(grad, axis=1)
    m = mag > 0
    area = 4 * math.pi * r_m ** 2 / n
    return np.c_[xyz[m], grad[m] / mag[m][:, None], plate[m], mag[m] * area, np.zeros(m.sum())]


def slab_density(xyz, s, z, mass, depths_km, body=None, layer_km=DENS_LAYER_KM):
    """Density anomaly (kg/m3) per layer from slablets (slab_state output, attached ones already removed), each
    spread over the grid cell it falls in and its layer."""
    from fastkd import cKDTree
    body = body or EARTH
    radius_km = body.R
    n = len(xyz)
    out = np.zeros((len(depths_km), n))
    keep = z < body.zcmb
    s, z, mass = s[keep], z[keep], mass[keep]
    if not len(s):
        return out
    cell = cKDTree(xyz).query(s[:, 1:4])[1]
    lay = np.clip(np.searchsorted(np.asarray(depths_km) + layer_km / 2, z), 0, len(depths_km) - 1)
    rr = (radius_km - np.asarray(depths_km)[lay]) / radius_km   # area of a cell shrinks with depth
    cell_m3 = 4 * math.pi * (radius_km * 1e3) ** 2 / n * rr ** 2 * layer_km * 1e3
    np.add.at(out, (lay, cell), mass / cell_m3)
    return out


# ---------------------------------------------------------------- warming under long-lived continents (2026-11-14, E-1114-08)
# Coltice et al. 2007 (Geology 35:391-394, original text checked): with a single supercontinent the mantle at the base of
# the continental thermal boundary layer is "on the order of 100 C higher than with dispersed continents" - "an upper
# bound"; in the aggregation run "100 m.y. after continental aggregation ... the temperature has already increased by
# 75 C" and the final value is reached ~300 m.y. after assembly; the warming is heat moved from outside ("subcontinental
# warming is compensated by cooling outside the continental domain"). Heron & Lowman 2010 (GRL 37, L22302) find
# insulation a minor effect - the literature is split, so this runs as a test at the published upper bound.
# Per mantle-frame cell: dT relaxes to INSUL_DT_K under continent and to 0 elsewhere with time constant INSUL_TAU_MYR;
# the global mean is removed (compensation); density -rho alpha (dT - mean) between INSUL_TOP_KM and the 660
# (rho, alpha: RP_RHO_M, RP_ALPHA). Depth: an energy check (blocked mantle heat flux ~70 mW/m2 for 100 Myr spread over
# 75 K) gives ~700 km, the upper mantle's thickness (PARAM_AUDIT).
INSUL_DT_K = 100.0              # K, Coltice et al. 2007 upper bound
INSUL_TAU_MYR = 100.0 / math.log(4.0)   # 72.1 Myr: 1 - exp(-100 / tau) = 75 / 100 (Coltice et al. 2007 aggregation run)
INSUL_TOP_KM = 100.0            # base of the thermal boundary layer where Coltice et al. 2007 report the temperature (Fig. 1)


def insulation_step(dT, cover, dt_myr):
    """dT (K) per cell after dt_myr with continental cover (0..1) in the mantle frame"""
    target = INSUL_DT_K * np.asarray(cover, float)
    return target + (dT - target) * math.exp(-max(dt_myr, 0.0) / INSUL_TAU_MYR)


def insulation_density(dT, depths_km, body=None):
    """density anomaly (kg/m3) per layer: -rho alpha (dT - global mean) from INSUL_TOP_KM to the 660"""
    body = body or EARTH
    a = -RP_RHO_M * RP_ALPHA * (np.asarray(dT, float) - float(np.mean(dT)))
    d = np.asarray(depths_km)
    return np.where(((d >= INSUL_TOP_KM) & (d < body.z660))[:, None], a[None, :], 0.0)


# ---------------------------------------------------------------- MACMA warming under continents (2026-11-14)
# Combes et al. 2012 (G3, supplement section 2): a layer of thickness h under a continent of width a warms and drains
# sideways at U = rho alpha g h^2 dT / (eta (x^-3 + x)), x = a / 2h (their eq. 25, rederived from eqs. 21-22); its heat
# balance dT/dt = Hdot / cp - (4 / a) U dT (eqs. 23-24); the outflow drags the continent base with eta U / h, which
# breaks it once eta U / h * a/2 reaches its strength. Heating: the mantle heat flow the continent blocks spread over the
# layer, MACMA_Q / (rho h) (PARAM_AUDIT: their "Hradio" read this way reproduces their results; mantle radioactivity
# alone gives 10-50 K and no breakup). Width a: diameter of the landmass's equal-area circle; ocean cells reset to 0.
MACMA_H_KM = 350.0              # km, Combes et al. 2012 supplement: 300 < h < 400 km (middle)
MACMA_Q = 0.07                  # W/m2, blocked mantle heat flow (PARAM_AUDIT, INSUL depth row)
MACMA_ETA = 1e21                # Pa s, upper mantle (Combes et al. 2012 table 1; EARTH.visc upper mantle)
MACMA_CP = 1200.0               # J/kg/K, Combes et al. 2012 table 1
MACMA_FLIM = 7e12               # N/m, whole-lithosphere extensional strength (E-1110-06), within MACMA's 7-12e12
SEC_PER_MYR = 3.156e13


def landmass_labels(cont, nb):
    """connected continental cells (neighbour lists nb); -1 on ocean"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    idx = np.nonzero(cont)[0]
    pos = -np.ones(len(cont), int); pos[idx] = np.arange(len(idx))
    ii = np.repeat(idx, nb.shape[1]); jj = nb[idx].ravel()
    ok = pos[jj] >= 0
    g = coo_matrix((np.ones(ok.sum()), (pos[ii[ok]], pos[jj[ok]])), shape=(len(idx), len(idx)))
    _, lab = connected_components(g, directed=False)
    out = -np.ones(len(cont), int); out[idx] = lab
    return out


def macma_step(dT, cont, nb, xyz, R_km, dt_myr, h_km=MACMA_H_KM, g=9.81):
    """advance the layer warming dT (K, per cell) by dt_myr; returns (dT, U per cell m/s, landmass label, landmass
    centre per cell, a per cell m, F per cell N/m)"""
    h = h_km * 1e3
    cell = 4 * math.pi * (R_km * 1e3) ** 2 / len(xyz)
    lab = landmass_labels(cont, nb)
    dT = np.where(cont, dT, 0.0)
    U = np.zeros(len(xyz)); a_c = np.zeros(len(xyz)); F = np.zeros(len(xyz)); ctr = np.zeros((len(xyz), 3))
    for L in np.unique(lab[lab >= 0]):
        m = lab == L
        a = 2 * math.sqrt(m.sum() * cell / math.pi)
        x = a / (2 * h)
        k = RP_RHO_M * RP_ALPHA * g * h ** 2 / (MACMA_ETA * (x ** -3 + x))
        # dT/dt = A - B dT^2 solved exactly over the step (explicit steps go unstable when a warm landmass splits):
        # towards s = sqrt(A/B) along tanh from below, coth from above
        A_, B_ = MACMA_Q / (RP_RHO_M * h) / MACMA_CP, 4.0 / a * k
        s, r = math.sqrt(A_ / B_), math.sqrt(A_ * B_) * dt_myr * SEC_PER_MYR
        d = np.maximum(dT[m], 0.0) / s
        lo = d < 1.0
        d[lo] = np.tanh(r + np.arctanh(d[lo]))
        d[~lo] = 1.0 / np.tanh(r + np.arctanh(1.0 / np.maximum(d[~lo], 1.0 + 1e-12)))
        d = d * s
        dT[m] = d
        U[m] = k * d
        a_c[m] = a
        F[m] = MACMA_ETA * k * d.mean() / h * a / 2
        c = xyz[m].mean(axis=0); ctr[m] = c / np.linalg.norm(c)
    return dT, U, lab, ctr, a_c, F


def macma_torques(xyz, plate, cont, U, ctr, ids, R_km, h_km=MACMA_H_KM):
    """torques (N m, 3 per plate in ids order) of the outflow drag eta U / h on the continent base, directed away from
    the landmass centre; on an unbroken landmass they nearly cancel"""
    cell = 4 * math.pi * (R_km * 1e3) ** 2 / len(xyz)
    away = xyz * (xyz * ctr).sum(axis=1)[:, None] - ctr
    n_ = np.linalg.norm(away, axis=1)
    ok = cont & (n_ > 1e-9) & (U > 0)
    f = np.zeros_like(xyz)
    f[ok] = away[ok] / n_[ok][:, None] * (MACMA_ETA * U[ok] / (h_km * 1e3) * cell)[:, None]
    tq = np.cross(xyz, f) * R_km * 1e3
    pos = {p: j for j, p in enumerate(ids)}
    q = np.zeros(3 * len(ids))
    for p in np.unique(plate[ok]):
        if int(p) in pos:
            j = pos[int(p)]
            q[3 * j:3 * j + 3] += tq[ok & (plate == p)].sum(axis=0)
    return q


# ---------------------------------------------------------------- deep hot upwellings (LLSVPs)
# 2026-11-03: slabs alone hold a supercontinent together (the planet test: rifts closed again within 10-20 Myr). The other
# half of the flow is the buoyancy of the two large low-shear-velocity provinces (LLSVPs) at the base of the mantle under
# Africa and the Pacific, seen in every tomography model, stable for >= 200-300 Myr (Torsvik et al. 2006, 2010; Burke et al.
# 2008), the source of plumes and LIPs, and the upwelling under supercontinents in convection models (Zhong et al. 2007).
# Earth: their density from tomography - slow shear-velocity anomalies of the SMEAN composite model (Becker & Boschi 2002,
# shipped with HC) in the lowermost 1000 km (the LLSVPs' height above the core, Garnero et al. 2016, Nat. Geosci.),
# converted with dln rho / dln Vs = 0.2 (HC's default; thermal mineral physics 0.2-0.4, Karato 1993). Only slow (hot)
# anomalies: the fast ones are slabs, already in the slab model.
LLSVP_SCALE = 0.2               # dln rho / dln Vs
LLSVP_HEIGHT_KM = 1000.0        # above the core-mantle boundary
TOMO_FILE = "tomography/smean.31.m.ab"


def prem_density(z_km):
    """PREM density (kg/m3) of the lower mantle (Dziewonski & Anderson 1981, 3630-5701 km radius)"""
    x = (6371.0 - np.asarray(z_km, float)) / 6371.0
    return 1000.0 * (7.9565 - 6.4761 * x + 5.5283 * x ** 2 - 3.0807 * x ** 3)


def read_tomography(fn):
    """HC short format: number of layers, then per layer depth (km), lmax, (l, m) pairs of a, b (Dahlen & Tromp)."""
    tok = open(fn).read().split()
    k = 0; nl = int(float(tok[k])); k += 1
    layers = []
    for _ in range(nl):
        z = float(tok[k]); L = int(float(tok[k + 1])); k += 2
        n = (L + 1) * (L + 2) // 2
        ab = np.array(tok[k:k + 2 * n], float).reshape(n, 2); k += 2 * n
        layers.append((z, L, ab))
    return layers


def tomo_llsvp_density(sh, depths_km, bindir=None, z_cmb=Z_CMB_EARTH, scale=LLSVP_SCALE, height=LLSVP_HEIGHT_KM):
    """density anomaly (kg/m3, <= 0) per layer at the SH points: slow SMEAN anomalies within `height` of the core"""
    bindir = bindir or build_hc()
    layers = read_tomography(os.path.join(os.path.dirname(bindir), TOMO_FILE))
    pos = {key: i for i, key in enumerate(sh.idx)}
    zs = np.array([z for z, _, _ in layers]); fields = []
    for z, L, ab in layers:
        c = np.zeros(len(sh.idx)); j = 0
        for l in range(L + 1):
            for m in range(l + 1):
                if l <= sh.L:
                    c[pos[(l, m, 0)]] = ab[j, 0]
                    if m:
                        c[pos[(l, m, 1)]] = ab[j, 1]
                j += 1
        fields.append(c @ sh.Y)                         # dln Vs in %
    fields = np.array(fields)
    out = np.zeros((len(depths_km), sh.Y.shape[1]))
    for i, z in enumerate(depths_km):
        if z < z_cmb - height:
            continue
        k = int(np.argmin(np.abs(zs - z)))
        dv = np.minimum(fields[k], 0.0)                  # slow only (hot)
        out[i] = scale * dv / 100.0 * prem_density(z)
    return out
