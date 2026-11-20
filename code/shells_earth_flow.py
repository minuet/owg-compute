"""Earth test, PLATE_PLAN section 6 step 2 (second half): Bird's thin-shell model SHELLS (ShellSet v1.1.0; May, Bird &
Carafa 2024, GMD 17:6153) driven from below by the mantle flow of our slab snapshot, instead of the basal tractions the
authors infer from torque balance.

What is tested. In the authors' Earth model (ListEx1 = Earth5-049 inputs) the subducting plates are moved by velocity
boundary conditions at their trenches (Earth5R-type4A.bcs, type 4 = the plate's PB2002 velocity) and the other plates by
inferred basal tractions (iConve 0 -> 6 iteration). In our plan the plate history gives the subducting plates' motions
(the same role as the type-4 conditions), so the open question is the second part: can the flow computed from the
planet's own slabs (layer 3) replace the inferred tractions? Here it is fed through SHELLS' own entry for an external
flow (iConve = 2, read as in Convec: nearest of <= 1000 vectors, each a start point and its position after 11 Myr) and
scored with the authors' own data and misfits, for the same 9 (tauMax, fFric) models as their ListEx1:
  - hcflow: our flow (below)
  - baum88: the flow file shipped with SHELLS (Baumgardner 1988 convection, Bird 1998's choice), as a control
and compared with the authors' reference scores (EXAMPLES/ListEx1/Models.txt, inferred tractions).

Our flow (mantle_flow.py, the method of E-1103-02): HC (radial viscosity, LBR98 Table 3) with the present-day plate
motions of Müller et al. 2022 as surface velocities (no net rotation: HC's free-slip core makes degree-1 toroidal flow a
rigid mode) and the density of all slabs of the previous 180 Myr (LBR98, the same slablets as earth_mantle_flow_check.py).
The velocity at the depth of SHELLS' zBAsth (400 km in iEarth5-049.in, the base of the layer whose strength couples plate
and mantle) is converted to the same angular velocity at the surface (SHELLS compares it with the plate's surface
velocity on one sphere), put in the Africa-fixed frame Convec expects, resampled to 1000 points and written so that
Convec's formula returns it exactly.

Usage (Kaggle, via cloud/submit_tool.sh shells_earth_flow shells_earth_flow.py): needs Linux for HC and Intel oneAPI.
Writes results/shells_earth_flow.json, results/hcflow_400km.dig and results/shells_runs.zip (Models.txt and logs)."""
import glob, json, math, os, subprocess, sys, time, zipfile
import numpy as np
import pygplates
from fastkd import cKDTree
from plate_checks import MODEL, TOPOLOGY_FILES, ROTATION, HERE
from plate_engine import fibonacci_xyz
from plate_checks import partition_ids
import mantle_flow as mf
from earth_mantle_flow_check import slablets, euler, nr_vector, R

AFRICA = 701                    # Müller et al. 2022 plate id of Africa (SHELLS' flow frame is Africa-fixed)
Z_BASTH_KM = 400.0              # iEarth5-049.in zBAsth, the depth whose flow Convec supplies
N_VEC = 1000                    # Convec's array limit (MOD_Shells.f90: baum88(5, 1000), DO 220 jVec = 1, 1000)
NODES = os.environ.get("SHELLS_NODES") == "1"   # variant (E-1119-11): the flow given at every real node of the SHELLS
                                                # grid (Earth5R.feg), Convec's array limit raised to that node count by
                                                # patching the two lines above, so its nearest-vector lookup is exact
BAUM_MYR = 11.0                 # Convec: the second point is the position after 11 m.y. (11.D6 * 3.15576D7 s)
SEC_PER_YR = 3.15576e7          # Convec's year
LLSVP = os.environ.get("SHELLS_LLSVP") == "1"   # variant: slabs + the two LLSVPs' buoyancy (mantle_flow.tomo_llsvp_density)
TOMO = os.environ.get("SHELLS_TOMO") == "1"   # upper-bound variant (E-1119-10): the whole mantle's density from SMEAN
                                              # tomography (slow and fast) instead of our slab history, i.e. as if we knew
                                              # Earth's mantle perfectly
TOMO_TOP_KM = 220.0             # tomography above this depth is left out: there it shows cratonic roots, fast but not dense
                                # (common practice in tomography-driven flow, e.g. Becker & O'Connell 2001; PARAM_AUDIT 11.9)
CLB = os.environ.get("SHELLS_CLB") == "1"     # variant (E-1119-11): as the layer-3 chain (E-1103-02, Conrad &
                                              # Lithgow-Bertelloni 2002 combined model), upper-mantle slabs still attached
                                              # at present trenches pull their plates directly (here: the authors' trench
                                              # velocity conditions) and are left out of the flow-driving density
RESID = os.environ.get("SHELLS_RESID") == "1"   # combined method (E-1119-12): our flow + the residual the authors' way
                                                # infers (patch_resid); SHELLS_SMOKE=1: first model of the list only
SMOKE = os.environ.get("SHELLS_SMOKE") == "1"
FREESLIP = os.environ.get("SHELLS_FREESLIP") == "1"   # diagnostic variant (E-1119-07): HC surface free-slip instead of
                                                      # the plate velocities, i.e. the buoyancy-driven flow alone
TRHMAX = 20.0e6                 # Pa, the authors' basal-traction limit after the first iteration (INPUT/UpVar.in)
RUN_TIMEOUT_S = 5 * 3600        # technical: each of the 2 ShellSet runs stops after 5 h, so both fit Kaggle's 12 h
S = "/tmp/ShellSet"
RES = HERE / "results"
LOG = None                      # step log, opened by main()


def sh_run(cmd, cwd="/tmp", timeout=None):
    t = time.time()
    LOG.write(f"\n$ {cmd}\n"); LOG.flush()
    try:
        r = subprocess.run(["bash", "-c", cmd], cwd=cwd, capture_output=True, text=True, timeout=timeout)
        out, err, code = r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        out, err, code = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), "", "timeout"
    LOG.write(out[-6000:] + "\n--stderr--\n" + err[-3000:] + f"\n[exit {code}, {time.time() - t:.0f} s]\n"); LOG.flush()
    return code, out


def gauss_to_points(a, xyz_out, lmax):
    """HC's spatial output (lon lat z v_r v_theta v_phi on its Gauss grid) -> tangent vectors at xyz_out, by exact
    Gauss-quadrature analysis to degree lmax and synthesis (as mantle_flow.HC.traction_field)."""
    lon, lat = np.radians(a[:, 0]), np.radians(a[:, 1])
    g = mf.SH(np.c_[np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], lmax)
    nlat = len(np.unique(a[:, 1])); nlon = len(a) // nlat
    x, wt = np.polynomial.legendre.leggauss(nlat)
    gw = np.interp(a[:, 1], np.degrees(np.arcsin(x)), wt) * 2 * math.pi / nlon
    vt, vp = a[:, 4], a[:, 5]
    P = (g.Gt @ (vt * gw) + g.Gp @ (vp * gw)) / g.sq
    T = (g.Gp @ (vt * gw) - g.Gt @ (vp * gw)) / g.sq
    P[0] = T[0] = 0.0
    return mf.SH(xyz_out, lmax).vector_synth(P, T)


def write_baum(fn, xyz, v_cm_yr, radius_m):
    """Convec reads v = (h - t) * radius / (11 Myr) projected on the plane at t (t, h unit vectors). With
    h = (t + a') / |t + a'| and a' = a / sqrt(1 - |a|^2), the projection is exactly a = v * 11 Myr / radius."""
    a = v_cm_yr * 0.01 / SEC_PER_YR * BAUM_MYR * 1e6 * SEC_PER_YR / radius_m
    a2 = (a ** 2).sum(1, keepdims=True)
    h = xyz + a / np.sqrt(1 - a2)
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    ll = lambda p: (np.degrees(np.arctan2(p[:, 1], p[:, 0])), np.degrees(np.arcsin(np.clip(p[:, 2], -1, 1))))
    (lo1, la1), (lo2, la2) = ll(xyz), ll(h)
    with open(fn, "w") as f:
        for i in range(len(xyz)):
            f.write(f" {lo1[i]:+.8E},{la1[i]:+.8E}\n {lo2[i]:+.8E},{la2[i]:+.8E}\n*** end of line segment ***\n")


def read_baum_like_convec(fn, radius_m):
    """Python copy of Convec's reading (MOD_Shells.f90, iConve == 2) for the self-check: points and (n, 3) velocity
    vectors in cm/yr."""
    L = [l for l in open(fn) if "***" not in l]
    p = np.array([[float(s) for s in l.split(",")] for l in L])
    ll2x = lambda lo, la: np.c_[np.cos(np.radians(la)) * np.cos(np.radians(lo)),
                                np.cos(np.radians(la)) * np.sin(np.radians(lo)), np.sin(np.radians(la))]
    t, h = ll2x(p[0::2, 0], p[0::2, 1]), ll2x(p[1::2, 0], p[1::2, 1])
    v = (h - t) * radius_m / (BAUM_MYR * 1e6 * SEC_PER_YR)
    v -= (v * t).sum(1, keepdims=True) * t                         # only the theta, phi components are kept
    return t, v / 0.01 * SEC_PER_YR


def tomo_density(sh, depths_km, bindir):
    """density anomaly (kg/m3) per layer from all SMEAN anomalies below TOMO_TOP_KM, dln rho = 0.2 dln Vs (the same
    scaling as mantle_flow.tomo_llsvp_density), referred to HC's mean density so that HC receives exactly 0.2 dln Vs"""
    layers = mf.read_tomography(os.path.join(os.path.dirname(bindir), mf.TOMO_FILE))
    pos = {key: i for i, key in enumerate(sh.idx)}
    zs = np.array([zz for zz, _, _ in layers]); fields = []
    for zz, L, ab in layers:
        c = np.zeros(len(sh.idx)); j = 0
        for l in range(L + 1):
            for m in range(l + 1):
                if l <= sh.L:
                    c[pos[(l, m, 0)]] = ab[j, 0]
                    if m:
                        c[pos[(l, m, 1)]] = ab[j, 1]
                j += 1
        fields.append(c @ sh.Y)                                   # dln Vs in %
    out = np.zeros((len(depths_km), sh.Y.shape[1]))
    for i, zz in enumerate(depths_km):
        if zz >= TOMO_TOP_KM:
            out[i] = mf.LLSVP_SCALE * fields[int(np.argmin(np.abs(zs - zz)))] / 100.0 * mf.HC_AVG_DEN
    return out


def feg_nodes(fn):
    """unit vectors of the real nodes of a SHELLS .feg grid (line 2: numNod, nRealN, ...; then index, lon, lat, ...)"""
    L = open(fn).read().splitlines()
    n_real = int(L[1].split()[1])
    ll = np.array([[float(x) for x in l.split()[1:3]] for l in L[2:2 + n_real]])
    lo, la = np.radians(ll[:, 0]), np.radians(ll[:, 1])
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def hc_flow():
    t0 = time.time()
    rot = pygplates.RotationModel(str(ROTATION))
    topo = [pygplates.FeatureCollection(str(MODEL / "Topologies" / f)) for f in TOPOLOGY_FILES]
    ages = np.load(HERE / "results/earth_age_series.npz")
    g_age = fibonacci_xyz(int(round(60000 * (R / 6000.0) ** 2)))
    xyz = g_age                                                   # the check's default analysis grid
    S_ = slablets(rot, topo, mf.STEP_MYR, mf.HISTORY_MYR, ages, cKDTree(g_age))
    sl, z, mass = mf.slab_state(S_, 0.0)
    depths = np.arange(mf.DENS_LAYER_KM / 2, mf.EARTH.zcmb, mf.DENS_LAYER_KM)
    sh = mf.SH(xyz)
    hc = mf.HC(sh)
    drive = np.ones(len(sl), bool)
    if CLB:
        _, att = mf.attached_pull(sl, z, mass, slablets(rot, topo, 0.0, 0.0, ages, cKDTree(g_age)))
        drive = ~att
    dens = mf.slab_density(xyz, sl[drive], z[drive], mass[drive], depths) if not TOMO else tomo_density(sh, depths, hc.bin)
    if LLSVP:
        dens = dens + mf.tomo_llsvp_density(sh, depths, hc.bin)    # E-1103-02 llsvp variant: hot piles from tomography
    sh.write_dens(f"{hc.dir}/dens.sh", mf.EARTH.hc_depth(depths), dens * 100.0 / mf.HC_AVG_DEN * mf.EARTH.buoyancy_factor())
    plate = partition_ids(pygplates.PlatePartitioner(topo, rot, 0.0), xyz)
    W = np.array([euler(rot, 0.0, p) for p in plate])              # rad/Myr, mantle frame
    v = np.cross(W, xyz) * R / 10.0                                 # cm/yr (R km / Myr = R / 10 cm/yr per rad)
    nr = nr_vector(xyz, v)
    v -= np.cross(nr, xyz) * R / 10.0
    P, T = sh.vector_coefs(v)
    sh.write_vel(f"{hc.dir}/pvel.sh", P, T)
    prem = f"{os.path.dirname(hc.bin)}/prem/prem.dat"
    bc = "-fs" if FREESLIP else "-pvel pvel.sh"
    subprocess.run(f"{hc.bin}/hc -dens dens.sh -dnp -ds 1 {bc} -prem {prem} -vf visc.dat > hc_flow.log 2>&1",
                   shell=True, cwd=hc.dir, check=True)
    lev = subprocess.run(f"{hc.bin}/hc_extract_spatial vel.sol.bin -2 4", shell=True, cwd=hc.dir, capture_output=True,
                         text=True).stdout.split()
    lev = np.array([float(s) for s in lev]).reshape(-1, 2)          # 0-based layer index, depth km (HC radius)
    zs = lev[:, 1]
    out = feg_nodes(f"{S}/INPUT/Earth5R.feg") if NODES else fibonacci_xyz(N_VEC)
    # the two HC layers around zBAsth, linearly interpolated in depth (HC's layers are 100 km apart around 400 km)
    lo = lev[zs <= Z_BASTH_KM][np.argmax(zs[zs <= Z_BASTH_KM])]; hi = lev[zs >= Z_BASTH_KM][np.argmin(zs[zs >= Z_BASTH_KM])]
    vf, z_used = 0.0, []
    for (i, zl), wgt in ((lo, 1.0 if hi[1] == lo[1] else (hi[1] - Z_BASTH_KM) / (hi[1] - lo[1])),
                         (hi, 0.0 if hi[1] == lo[1] else (Z_BASTH_KM - lo[1]) / (hi[1] - lo[1]))):
        a = np.loadtxt(subprocess.run(f"{hc.bin}/hc_extract_spatial vel.sol.bin {int(i) + 1} 3", shell=True, cwd=hc.dir,
                                      capture_output=True, text=True).stdout.splitlines())
        if not abs(float(a[0, 2]) - zl) < 1.0:
            raise RuntimeError(f"layer {i} extracted at {a[0, 2]} km, listed at {zl}")
        # same angular velocity at the surface (SHELLS compares it with the plate's surface velocity)
        vf = vf + wgt * gauss_to_points(a, out, mf.LMAX) * mf.HC_RE_KM / (mf.HC_RE_KM - zl)
        z_used.append((float(zl), round(wgt, 4)))
    w_af = W[plate == AFRICA][0] - nr                               # Africa, no-net-rotation frame
    vf -= np.cross(w_af, out) * R / 10.0                            # Africa-fixed
    # the plates' own motion at the flow points (Africa-fixed), for the variant without drag under slab plates
    p_out = partition_ids(pygplates.PlatePartitioner(topo, rot, 0.0), out)
    vplate_out = np.cross(np.array([euler(rot, 0.0, p) for p in p_out]) - nr - w_af, out) * R / 10.0
    fn = RES / "hcflow_400km.dig"
    write_baum(fn, out, vf, R * 1e3)
    t_chk, v_chk = read_baum_like_convec(fn, R * 1e3)
    err = float(np.abs(v_chk - vf).max())
    vp_af = v - np.cross(w_af, xyz) * R / 10.0                      # plates, Africa-fixed (for the summary only)
    info = {"hc_layers_km": zs.tolist(), "layers_used_km_weight": z_used, "n_vectors": len(out), "attached_slab_mass_share_left_out": round(float(mass[~drive].sum() / mass.sum()), 3),
            "flow_rms_cm_yr_africa_fixed": round(float(np.sqrt((vf ** 2).sum(1).mean())), 3),
            "plates_rms_cm_yr_africa_fixed": round(float(np.sqrt((vp_af ** 2).sum(1).mean())), 3),
            "net_rotation_removed_deg_myr": round(float(np.degrees(np.linalg.norm(nr))), 4),
            "convec_readback_max_err_cm_yr": err, "slablets": int(len(sl)), "seconds": round(time.time() - t0)}
    LOG.write(f"flow: {json.dumps(info)}\n"); LOG.flush()
    if not err < 1e-4:                                           # cm/yr; 8-digit coordinates give ~1e-6
        raise RuntimeError(f"flow file does not read back: {err}")
    return fn, info, (out, vf, vplate_out)


SLAB_PLATES = ("AU", "CL", "CO", "IN", "JF", "NZ", "PA", "PS", "RI", "SS")   # SHELLS_v5.0.f90: slab_q = .TRUE.


def pb2002_codes(xyz):
    """PB2002 plate code at each point, from the outlines SHELLS itself reads (INPUT/PB2002_plates.dig)."""
    polys, name, pts = {}, None, []
    for l in open(f"{S}/INPUT/PB2002_plates.dig"):
        s = l.strip()
        if not s:
            continue
        if s.startswith("***"):
            polys[name] = pygplates.PolygonOnSphere([(la, lo) for lo, la in pts]); pts = []
        elif s[0] in "+-0123456789":
            lo, la = (float(x) for x in s.split(","))
            pts.append((lo, la))
        else:
            name = s
    codes = []
    for p in xyz:
        q = pygplates.PointOnSphere(p / np.linalg.norm(p))
        codes.append(next((n for n, g in polys.items() if g.is_point_in_polygon(q)), ""))
    return np.array(codes)


def write_noslab_flow(out, vf, vplate):
    """variant: no basal drag under the plates the authors treat as slab-driven (their slab_q plates receive no inferred
    traction): the flow there is set to the plate's own velocity, so the drag (proportional to the velocity difference)
    is zero; elsewhere our flow unchanged."""
    codes = pb2002_codes(out)
    m = np.isin(codes, SLAB_PLATES)
    v2 = np.where(m[:, None], vplate, vf)
    fn = RES / "hcflow_noslab_400km.dig"
    write_baum(fn, out, v2, R * 1e3)
    return fn, {"points_under_slab_plates": int(m.sum()), "points_without_pb2002_plate": int((codes == "").sum())}


def shellset_build():
    """Intel oneAPI and ShellSet v1.1.0, as cloud/shellset_earth_kernel.py (same pinned packages, same fallback)."""
    sh_run("wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor > /usr/share/keyrings/oneapi.gpg && "
           "echo 'deb [signed-by=/usr/share/keyrings/oneapi.gpg] https://apt.repos.intel.com/oneapi all main' > /etc/apt/sources.list.d/oneAPI.list && "
           "apt-get update -qq")
    sh_run("DEBIAN_FRONTEND=noninteractive apt-get install -y -qq intel-oneapi-compiler-fortran-2024.2 "
           "intel-oneapi-mkl-devel-2024.2 intel-oneapi-mpi-devel-2021.13 2>&1 | tail -5")
    fc, mpi = "ifort", "mpiifort"
    if sh_run(ONEAPI + "which ifort")[0] != 0:
        sh_run("DEBIAN_FRONTEND=noninteractive apt-get install -y -qq intel-oneapi-compiler-fortran intel-oneapi-mkl-devel "
               "intel-oneapi-mpi-devel 2>&1 | tail -5")
        fc, mpi = "ifx", "mpiifx"
    if NODES:
        n = len(feg_nodes(f"{S}/INPUT/Earth5R.feg"))
        src = open(f"{S}/src/MOD_Shells.f90").read()
        for a in ("baum88(5, 1000)", "DO 220 jVec = 1, 1000"):
            assert src.count(a) == 1, a
            src = src.replace(a, a.replace("1000", str(n)))
        open(f"{S}/src/MOD_Shells.f90", "w").write(src)
    if RESID:
        patch_resid()
    sh_run(ONEAPI + f"make ShellSet FC={fc} MPI={mpi} 2>&1 | tail -40", cwd=S)
    return os.path.exists(f"{S}/ShellSet.exe")


ONEAPI = "source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1; "


def patch_resid():
    """The combined method (PLATE_PLAN section 6 step 2): our flow drags the plates (iConve = 2) and only what it leaves
    unexplained is inferred, the authors' way. Iteration 1 is run as the authors' first iteration (Earth5R-type4AplusA.bcs:
    the slabless plates held at their PB2002 velocities) but with our flow's drag on; the torque the velocity conditions
    then apply to each plate (torqVB) is the residual, turned by SHELLS' -Twist- into a traction pole and written into
    the torque report in place of the usual pole from torqBS (= torqMD + torqVB; with the authors' trHMax = 0 in their
    first iteration torqMD = 0, so their pole is the same torqVB pole). The final run (rpeat = 2, Earth5R-type4A.bcs)
    reads it back through -Tract- (zero on slab_q plates, as in iConve 6) and applies it as a fixed basal traction added
    to the flow's drag; that run (resSig allocated) writes the usual pole from torqBS, the whole basal shear (physical drag plus the residual, which SHELLS books under torqVB as the remainder of its force balance), since OrbScore scores the seismic anisotropy against the poles of the final report (Tractor); the first full run wrote the residual pole there too and scored the residual's direction alone (E-1119-14). Its magnitude is limited at trHMax, as the authors limit their inferred tractions (THOnB,
    iConve 6: shear2 = trHMax); the smoke run (E-1119-12) gave up to 213 MPa on small plates without the limit. One step, as the drag is held the same in both runs (trHMax unchanged by UpVar.in)."""
    def sub(fn, pairs):
        src = open(f"{S}/src/{fn}").read()
        for a, b in pairs:
            assert src.count(a) == 1, (fn, a)
            src = src.replace(a, b)
        open(f"{S}/src/{fn}", "w").write(src)
    sub("MOD_Shells.f90", [
        ("integer,parameter :: iUnitT = 6",
         "logical :: resMode = .FALSE.                     ! residual traction poles from torqVB (SHELLS_RESID)\n"
         "double precision, allocatable :: resSig(:, :, :) ! residual basal traction at integration points\n"
         "integer,parameter :: iUnitT = 6"),
        ("sHy = oVB(2, m, i) * eta(m, i)\n",
         "sHy = oVB(2, m, i) * eta(m, i)\n"
         "\t\t\t\t   IF (ALLOCATED(resSig)) THEN\n"
         "\t\t\t\t\t\tsHx = sHx + resSig(1, m, i)\n"
         "\t\t\t\t\t\tsHy = sHy + resSig(2, m, i)\n"
         "\t\t\t\t   END IF\n"),
        ("IF (iConve /= 6) THEN\n!           older code, for defined lower-mantle velocity field.",
         "IF (ALLOCATED(resSig)) DEALLOCATE(resSig)\n"
         "IF (resMode .AND. ANY(basal /= 0.0D0)) THEN\n"
         "\tALLOCATE(resSig(2, 7, mxEl))\n"
         "\tCALL Flow (fPSfer, mxEl, mxNode, nodes, numEl, basal, resSig)\n"
         "\tDO i = 1, numEl\n"      # the authors' limit on inferred tractions (iConve 6: shear2 = trHMax)
         "\t\tDO m = 1, 7\n"
         "\t\t\tshear1 = SQRT(resSig(1, m, i)**2 + resSig(2, m, i)**2)\n"
         "\t\t\tIF (shear1 > trHMax) resSig(1:2, m, i) = resSig(1:2, m, i) * (trHMax / shear1)\n"
         "\t\tEND DO\n"
         "\tEND DO\n"
         "END IF\n"
         "IF (iConve /= 6) THEN\n!           older code, for defined lower-mantle velocity field."),
        ("&                 radius, torqBS, whichP, xNode, yNode, &",
         "&                 radius, MERGE(torqVB, torqBS, resMode .AND. .NOT. ALLOCATED(resSig)), whichP, xNode, yNode, &")])
    sub("SHELLS_v5.0.f90", [
        ("if(rpeat>1 .and. (iConve/=0 .and. iConve/=6)) then",
         "resMode = (iConve == 2)\n       if(rpeat>1 .and. (iConve/=0 .and. iConve/=6 .and. iConve/=2)) then"),
        ("IF (iConve == 6) THEN\n!           Note that -Tract-",
         "IF (iConve == 6 .OR. (iConve == 2 .AND. rpeat > 1)) THEN\n!           Note that -Tract-")])
    sub("ShellSetMain.f90", [("(iConve == 1 .or. iConve == 2 .or. iConve == 3", "(iConve == 1 .or. iConve == 3")])


def setup_variant(name, flow_file):
    """parameter file = iEarth5-049.in with iConve = 2 and the authors' trHMax; InputFiles.in = the authors' with this
    parameter file and the flow file in the final (non-iterating) Shells section."""
    par = open(f"{S}/INPUT/iEarth5-049.in").read().splitlines()
    for k, l in enumerate(par):
        if "iConve, vTimes" in l:
            par[k] = "2,1.00 " + l.split(None, 1)[1]
        elif l.split()[1:2] == ["trHMax"]:
            par[k] = f"{TRHMAX:.1E} " + l.split(None, 1)[1]
    open(f"{S}/INPUT/i{name}.in", "w").write("\n".join(par) + "\n")
    inp = open(f"{S}/EXAMPLES/ListEx1/InputFiles.in").read().splitlines()
    sec = None
    for k, l in enumerate(inp):
        if l.startswith("iEarth5-049.in"):
            inp[k] = l.replace("iEarth5-049.in", f"i{name}.in")
        if l.startswith("Shells (Final"):
            sec = "final"
        if l.startswith("Shells (Main") and RESID:
            sec = "main"
        if sec in ("final", "main") and "Mantle Flow" in l:
            inp[k] = os.path.basename(flow_file) + " " + l.split(None, 1)[1]
            sec = None
    open(f"{S}/INPUT/InputFiles.in", "w").write("\n".join(inp) + "\n")
    os.system(f"cp {S}/EXAMPLES/ListEx1/ListInput.in {S}/INPUT/ListInput.in")
    if SMOKE:   # "2,9" header, then each variable's 9 values: keep the first model
        li = open(f"{S}/INPUT/ListInput.in").read().splitlines()
        out, n = [], None
        for l in li:
            if n is None and "," in l:
                out.append(l.split(",")[0] + ",1"); n = 0
            elif l.startswith("----------"):
                out.append(l); n = 0
            elif n is not None and l.strip() and l.strip()[0] in "0123456789.-" and len(l.split()) == 1:
                if n == 0:
                    out.append(l)
                n += 1
            else:
                out.append(l)
        open(f"{S}/INPUT/ListInput.in", "w").write("\n".join(out) + "\n")
    if os.path.abspath(flow_file) != os.path.abspath(f"{S}/INPUT/{os.path.basename(flow_file)}"):
        os.system(f"cp {flow_file} {S}/INPUT/")


def models_table(fn):
    rows = []
    for l in open(fn):
        f = l.split()
        if len(f) == 11 and f[0].isdigit():
            rows.append([float(x) for x in f])
    return rows


def main():
    global LOG
    LOG = open(RES / "shells_earth_flow_steps.log", "w")
    summary = {"source": "ShellSet v1.1.0 + HC " + mf.HC_COMMIT[:8] + "; Müller et al. 2022 present-day plates and slabs"}
    summary["llsvp"] = LLSVP
    summary["freeslip"] = FREESLIP
    summary["tomo"] = TOMO
    summary["nodes"] = NODES
    summary["clb"] = CLB
    summary["resid"] = RESID
    summary["smoke"] = SMOKE
    sh_run(f"git clone --depth 1 --branch v1.1.0 https://github.com/JonBMay/ShellSet.git {S}")   # its inputs are needed first
    flow_fn, summary["flow"], (out, vf, vplate) = hc_flow()
    noslab_fn, summary["flow_noslab"] = write_noslab_flow(out, vf, vplate)
    json.dump(summary, open(RES / "shells_earth_flow.json", "w"), indent=1)
    summary["build_ok"] = shellset_build()
    if summary["build_ok"]:
        # 2026-11-19 v2: baum88 (Baumgardner's flow) was run in v1 (results/kaggle/shells_earth_flow_v1), not repeated
        variants = (("hcflow", str(flow_fn)), ("hcnoslab", str(noslab_fn)))
        for name, fl in variants[:1] if SMOKE else variants:
            setup_variant(name, fl)
            t = time.time()
            code, _ = sh_run(ONEAPI + "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; "
                             f"set -o pipefail; mpirun -np {2 if SMOKE else 4} ./ShellSet.exe -InOpt List -Dir L{name} {'-Iter 1 ' if RESID else ''}-GM 5,SSR,GV,SD,SA,FSR 2>&1 | tail -60",
                             cwd=S, timeout=RUN_TIMEOUT_S)
            m = glob.glob(f"{S}/L{name}/**/Models.txt", recursive=True)
            summary[name] = {"exit": code, "seconds": round(time.time() - t), "models": models_table(m[0]) if m else None}
            json.dump(summary, open(RES / "shells_earth_flow.json", "w"), indent=1)
            # per-model torque reports (qEarth), node velocities (vEarth) and logs of the final Shells runs, for the
            # plate-by-plate diagnosis (v1 lost them: its one zip of everything exceeded the kernel's 20 MB limit)
            with zipfile.ZipFile(RES / f"shells_{name}_shellsF.zip", "w", zipfile.ZIP_DEFLATED) as zf:
                for f in glob.glob(f"{S}/L{name}/ThID_*_ShellsF_output/*") + \
                        glob.glob(f"{S}/L{name}/ThID_*_Shells_output/qEarth_*"):   # RESID: iteration 1's residual poles
                    zf.write(f, os.path.relpath(f, S))
    summary["reference"] = models_table(f"{S}/EXAMPLES/ListEx1/Models.txt") if os.path.exists(S) else None
    json.dump(summary, open(RES / "shells_earth_flow.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "flow"}, indent=1))
    print(json.dumps(summary["flow"], indent=1))


if __name__ == "__main__":
    main()
