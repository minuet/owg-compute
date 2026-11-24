"""GitHub Actions: 2D constant-force rift run in ASPECT v3.0.0 with the thin-shell model's own rheology, to find the
2D force that breaks a lithosphere of given structure in 20 Myr and compare it with the 1D critical force of the rift
criterion (tools/shells_rheology.py at the critical strain rate; E-1121-18: the 1D force is 12-34 % too high and the
bias depends on the structure, so it is measured here structure by structure).
The set-up is that of tools/aspect_brune.py, which reproduced Brune et al. 2016 M1 (E-1121-26): 500 km wide box, 2 km
cells, free surface, left side free slip, right side a rigid plate end pulled with the force F (plugin
tools/aspect_force_controlled.cc), bottom inflow v H / L, 1350 C-isotherm seed (here the base of the lithosphere,
raised 10 km in a 20 km wide triangle), friction softening to 0.1 of the initial friction coefficient over plastic
strain 0..1 (Brune M1). Changed from M1 so that strength and force come from one rheology (PARAM_AUDIT 11.17):
  - crust and mantle lithosphere: Bird's Earth5 thin-shell creep, differential stress A (2 edot)^(1/3) exp((B + C z)/T),
    written in ASPECT's dislocation form (prefactor 4 / A^3, n = 3, E = 3 R B; the depth term C z as an activation
    volume fitted to the lithostatic pressure of the structure); friction 0.85 with hydrostatic pore pressure (as a
    reduced friction angle per layer), no cohesion; stress cap 5e8 Pa (Bird's dCreep); densities, expansivities and
    conductivities of the thin-shell model;
  - initial geotherm: the thin-shell layer geotherm (surface heat flow q, crust zc, mantle lithosphere zm, quadratic
    correction to 1473 K at the base), made steady by giving each layer the heat production of that profile;
  - below the lithosphere: 1473 K, wet olivine as in M1 (the thin-shell model has no asthenosphere); box depth is the
    base of the lithosphere plus 30 km, as in M1 (120 + 30 km).
Segments: as aspect_brune.py (checkpoint at ASPECT_WALL_H, restart.zip in the code folder resumes; chain.yml continues
until tools/diag/rift2d_eval.py has a verdict).
Env: R2D_Q (mW/m2), R2D_ZC, R2D_ZM (km), R2D_F (TN/m), ASPECT_END_MYR, ASPECT_WALL_H, ASPECT_SMOKE, ASPECT_INPUTS_ONLY,
ASPECT_RESTART (restart file, default restart.zip).
Output: results/<tag>_{run.json,out.txt,statistics,log.txt,restart.zip,viz.zip}"""
import glob, json, math, os, re, shutil, signal, subprocess, sys, threading, time, zipfile
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "diag")]       # in the cloud the diag modules sit next to this file (pack.sh)
import shells_rheology as sr
from brune_m1_eval import V_FAST                         # 10 mm/yr, the verdict of tools/diag/rift2d_eval.py

A = "/tmp/aspect"; B = "/tmp/aspect-build"; D = os.environ.get("ASPECT_DIR", "/tmp/run"); OUT = f"{D}/output"
Q = float(os.environ.get("R2D_Q", "54"))                 # mW/m2
ZC_KM = float(os.environ.get("R2D_ZC", "41"))
ZM_KM = float(os.environ.get("R2D_ZM", "113"))
F = float(os.environ.get("R2D_F", "2.0"))                # TN/m
END_MYR = float(os.environ.get("ASPECT_END_MYR", "30"))
WALL_H = float(os.environ.get("ASPECT_WALL_H", "4.0"))
KILL_GRACE_S = 600                 # technical: hard stop this long after WALL_H, so packing and upload fit (PARAM_AUDIT 11.17)
SMOKE = os.environ.get("ASPECT_SMOKE") == "1"
SMOKE_WALL_H = 0.1                 # technical: each smoke piece stops after 6 min
TAG = f"r2d_q{Q:g}_zc{ZC_KM:g}_zm{ZM_KM:g}_F{F:g}"
ENV_KEYS = ("R2D_Q", "R2D_ZC", "R2D_ZM", "R2D_F", "ASPECT_END_MYR", "ASPECT_WALL_H")

R_GAS = 8.314
N_CREEP = 3.0                                 # 1 / E_CREEP of the thin-shell model
X_KM, CELL_KM = 500.0, 2.0                    # M1
ASTH_KM = 30.0                                # M1: 150 km box below a 120 km lithosphere
LAB_KM = ZC_KM + ZM_KM
Y_KM = math.ceil((LAB_KM + ASTH_KM) / CELL_KM) * CELL_KM
SEED_KM, SEED_HALF_KM = 10.0, 10.0            # M1
SOFT = 0.1                                    # M1: friction coefficient 0.5 -> 0.05 over plastic strain 0..1
T_TOP_K = sr.T_SURF
RHO = dict(c=sr.RHO_BAR[0], m=sr.RHO_BAR[1], a=sr.RHO_BAR[1])
ALPHA = dict(c=sr.ALPHA_T[0], m=sr.ALPHA_T[1], a=sr.ALPHA_T[1])
K = dict(c=sr.CONDUC[0], m=sr.CONDUC[1], a=sr.CONDUC[1])
CP = 1200.0                                   # M1 (heat capacity does not enter the thin-shell strength)
G = sr.G
DX_T = 1e3                                    # m, technical (M1)
ETA_MIN, ETA_MAX = 1e18, 1e26                 # M1
DT_MAX_YR = 20e3; CFL = 0.5; NL_TOL, NL_MAX = 1e-4, 100     # M1
CHEAP_STEPS = 4000                 # ASPECT continental_extension cookbook after v3.0.0 (M1 had 0: E-1122-03, PARAM_AUDIT 11.17)
EXPENSIVE_STEPS, GMRES_RESTART = 5000, 200   # ASPECT crustal_deformation cookbook: the default 1000/50 stopped near convergence (PARAM_AUDIT 11.20)
VIZ_YR = 1e6
CHECKPOINT_S = 1800
WATCH_S = 60                       # technical: how often the statistics are read for the fast phase (PARAM_AUDIT 11.17)
FC_V0_MM_YR, FC_EXPONENT, FC_MAX_FACTOR = 1.0, 1.0, 2.0     # M1 (technical)
PLUGIN = "/tmp/aspect-plugin/libaspect_force_controlled.so"

os.makedirs("results", exist_ok=True)
res = {"script": "aspect_rift2d.py", "F_TN_m": F, "q_mW": Q, "zc_km": ZC_KM, "zm_km": ZM_KM, "end_myr": END_MYR,
       "smoke": SMOKE, "env": {k: os.environ[k] for k in ENV_KEYS if k in os.environ}}


def run(cmd, cwd="/tmp", timeout=None):
    t = time.time()
    print(f"$ {cmd[:150]}", flush=True)
    p = subprocess.Popen(["bash", "-c", cmd], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         start_new_session=True)
    try:
        out, code = p.communicate(timeout=timeout)[0], p.returncode
    except subprocess.TimeoutExpired:      # kill the whole group: mpirun's ranks keep the pipe open (r2d s6, 2026-11-23)
        os.killpg(p.pid, signal.SIGKILL)
        out, code = p.communicate()[0], "timeout"
    print(out[-3000:], f"\n[exit {code}, {time.time() - t:.0f} s]", flush=True)
    return code, out, time.time() - t


def geotherm_coeffs():
    """thin-shell layer geotherm with its base correction (shells_rheology.shells_strength): T = T_SURF + g2 z + g3 z^2
    in the crust, T_moho + dtm zz + g7 zz^2 in the mantle lithosphere; the heat productions that make it steady"""
    q, zc, zm = Q * 1e-3, ZC_KM * 1e3, ZM_KM * 1e3
    g2, g3, g7 = q / K["c"], -sr.RADIO[0] / (2 * K["c"]), -sr.RADIO[1] / (2 * K["m"])
    tmoho = sr.T_SURF + g2 * zc + g3 * zc ** 2
    dtm = (g2 + 2 * g3 * zc) * K["c"] / K["m"]
    dq = -(tmoho + dtm * zm + g7 * zm ** 2 - sr.T_ASTH) / (zc + zm) ** 2
    g3, g7 = g3 + dq, g7 + dq
    return g2, g3, g7, -2 * K["c"] * g3, -2 * K["m"] * g7


def column(z):
    """thin-shell geotherm (K), layer (0 crust, 1 mantle lithosphere, 2 below) and lithostatic pressure on depths z (m)"""
    g2, g3, g7, _, _ = geotherm_coeffs()
    zc = ZC_KM * 1e3
    tmoho, dtm = sr.T_SURF + g2 * zc + g3 * zc ** 2, (g2 + 2 * g3 * zc) * K["c"] / K["m"]
    lay = np.where(z < zc, 0, np.where(z < LAB_KM * 1e3, 1, 2))
    zz = np.minimum(z, LAB_KM * 1e3) - zc
    t = np.where(lay == 0, sr.T_SURF + g2 * z + g3 * z ** 2, tmoho + dtm * zz + g7 * zz ** 2)
    rho = np.where(lay == 0, RHO["c"] * (1 - ALPHA["c"] * t), RHO["m"] * (1 - ALPHA["m"] * t))
    p = np.concatenate([[0.0], np.cumsum(0.5 * (rho[1:] + rho[:-1]) * G * np.diff(z))])
    return t, lay, p


def t_lab():
    """temperature of the thin-shell geotherm at the base of the lithosphere: used for the base and below, so that
    the initial state is exactly the thin-shell geotherm (its quadratic correction reaches T_ASTH only when crust and
    mantle conductivities are equal; 1473-1540 K for the structures of the batch)"""
    return float(column(np.array([LAB_KM * 1e3]))[0][0])


def rheology():
    """ASPECT parameters equal to the thin-shell rheology on this structure (PARAM_AUDIT 11.17)"""
    z = np.arange(0.0, LAB_KM * 1e3 + 1, 100.0)
    t, lay, p = column(z)
    out = {}
    for i, key in ((0, "c"), (1, "m")):
        a, b, c = sr.A_CREEP[i], sr.B_CREEP[i], sr.C_CREEP[i]
        k = lay == i
        # depth term: C z / T = p V / (n R T); least squares over the layer, weighted by 1/T as it enters the exponent
        v = float(np.sum(c * z[k] * p[k] / t[k] ** 2) / np.sum(p[k] ** 2 / t[k] ** 2)) * N_CREEP * R_GAS if c else 0.0
        err = (p[k] * v / (N_CREEP * R_GAS) - c * z[k]) / t[k]
        # friction: thin shell (1 - 1/gamma)(s_v - p_w) = 2 s (s_v - p_w) / (1 + s) at normal faulting, s = sin(atan
        # 0.85), s_v lithostatic, p_w hydrostatic; ASPECT 2 s' P with the mean stress P = s_v - (differential) / 2,
        # i.e. 2 s' s_v / (1 + s'). Equal when s' / (1 + s') = X = s r / (1 + s), r = 1 - p_w / s_v (layer mean)
        ratio = float(np.mean(1 - sr.RHO_H2O * G * z[k][1:] / p[k][1:]))
        s = math.sin(math.atan(sr.C_FRIC)); x = s * ratio / (1 + s)
        phi0 = math.degrees(math.asin(x / (1 - x)))
        phi1 = math.degrees(math.atan(SOFT * math.tan(math.radians(phi0))))
        out[key] = {"prefactor": 4.0 / a ** N_CREEP, "E": N_CREEP * R_GAS * b, "V": v, "phi": phi0,
                    "soft": phi1 / phi0, "max_exponent_error": float(np.abs(err).max()), "pore_ratio": ratio}
    return out


def strength_aspect(rh, edot):
    """integrated differential stress (N/m) of the ASPECT form, to compare with shells_rheology.shells_strength"""
    z = np.arange(50.0, LAB_KM * 1e3, 100.0)
    t, lay, p = column(z)
    s = np.zeros_like(z)
    for i, key in ((0, "c"), (1, "m")):
        k = lay == i; r = rh[key]
        creep = 2 * r["prefactor"] ** (-1 / N_CREEP) * edot ** (1 / N_CREEP) * \
            np.exp(np.minimum((r["E"] + p[k] * r["V"]) / (N_CREEP * R_GAS * t[k]), 87.0))
        sp = math.sin(math.radians(r["phi"]))
        fric = 2 * sp * p[k] / (1 + sp)                    # mean stress at normal faulting (see rheology)
        s[k] = np.minimum(np.minimum(creep, fric), 2 * sr.D_CREEP)
    return float(s.sum() * 100.0)


def initial_temperature():
    """steady conduction on a DX_T grid with the layer heat productions of the thin-shell geotherm: T_TOP at the
    surface, T_LAB at and below the (seeded) base of the lithosphere, insulated sides"""
    _, _, _, hc, hm = geotherm_coeffs()
    x = np.arange(0, X_KM * 1e3 + 1, DX_T); d = np.arange(0, Y_KM * 1e3 + 1, DX_T)
    nx, nd = len(x), len(d)
    lay = np.where(d < ZC_KM * 1e3, 0, np.where(d < LAB_KM * 1e3, 1, 2))
    k = np.array([K["c"], K["m"], K["a"]])[lay]; h = np.array([hc, hm, 0.0])[lay]
    lab = LAB_KM * 1e3 - SEED_KM * 1e3 * np.clip(1 - np.abs(x - X_KM * 5e2) / (SEED_HALF_KM * 1e3), 0, 1)
    fixed = (d[:, None] >= lab[None, :]) | (d[:, None] == 0)
    tfix = np.where(d[:, None] == 0, T_TOP_K, t_lab()) * np.ones((nd, nx))
    idx = np.arange(nd * nx).reshape(nd, nx)
    rows, cols, vals = [], [], []
    rhs = np.zeros(nd * nx)
    kf_d = 2 * k[1:] * k[:-1] / (k[1:] + k[:-1])
    for j in range(nd):
        for i in range(nx):
            p = idx[j, i]
            if fixed[j, i]:
                rows.append(p); cols.append(p); vals.append(1.0); rhs[p] = tfix[j, i]
                continue
            diag = 0.0
            for jj, ii, kk in ((j - 1, i, kf_d[j - 1]), (j + 1, i, kf_d[j] if j + 1 < nd else 0.0),
                               (j, i - 1, k[j]), (j, i + 1, k[j])):
                if 0 <= jj < nd and 0 <= ii < nx and kk > 0:
                    rows.append(p); cols.append(idx[jj, ii]); vals.append(kk); diag += kk
            rows.append(p); cols.append(p); vals.append(-diag)
            rhs[p] = -h[j] * DX_T ** 2
    t = spsolve(coo_matrix((vals, (rows, cols)), shape=(nd * nx, nd * nx)).tocsr(), rhs).reshape(nd, nx)
    return x, d, t


def write_inputs(rh):
    x, d, t = initial_temperature()
    y = Y_KM * 1e3 - d
    with open(f"{D}/r2d_T.txt", "w") as f:
        f.write(f"# POINTS: {len(x)} {len(d)}\n# x y T\n")
        for j in np.argsort(y):
            for i in range(len(x)):
                f.write(f"{x[i]:.0f} {y[j]:.0f} {t[j, i]:.3f}\n")
    tc, _, _ = column(d)
    moho = int(ZC_KM * 1e3 / DX_T)
    fs = {f"{e:g}": round(sr.shells_strength(Q * 1e-3, ZC_KM * 1e3, ZM_KM * 1e3, e) / 1e12, 3) for e in (1e-17, 1e-16)}
    fa = {f"{e:g}": round(strength_aspect(rh, e) / 1e12, 3) for e in (1e-17, 1e-16)}
    _, _, _, hc, hm = geotherm_coeffs()
    return {"box_depth_km": Y_KM, "T_lab_K": round(t_lab(), 1), "T_moho_far_K": round(float(t[moho, -1]), 1),
            "T_misfit_far_K": round(float(np.abs(t[:, -1] - tc).max()), 2),
            "heat_production_uW_m3": [round(hc * 1e6, 4), round(hm * 1e6, 4)],
            "strength_shells_TN_m": fs, "strength_aspect_form_TN_m": fa, "rheology": rh}


def prm(rh, end_yr, wall_h, resume):
    def comp(vals):                                    # background, plastic_strain placeholder, crust, mantle lith.
        return ", ".join(f"{v:.6g}" for v in vals)
    c, m = rh["c"], rh["m"]
    nx, ny = int(X_KM / CELL_KM), int(Y_KM / CELL_KM)
    _, _, _, hc, hm = geotherm_coeffs()
    phi_a = m["phi"]                                   # below the lithosphere friction never governs; mantle value
    return f"""
set Dimension = 2
set Additional shared libraries = {PLUGIN}
set Start time = 0
set End time = {end_yr:g}
set Use years in output instead of seconds = true
set Resume computation = {'true' if resume else 'false'}
set Nonlinear solver scheme = single Advection, iterated defect correction Stokes
set Nonlinear solver tolerance = {NL_TOL:g}
set Max nonlinear iterations = {NL_MAX}
set CFL number = {CFL}
set Maximum time step = {DT_MAX_YR:g}
set Output directory = output
set Pressure normalization = no
subsection Formulation
  set Formulation = Boussinesq approximation
end
subsection Geometry model
  set Model name = box
  subsection Box
    set X repetitions = {nx}
    set Y repetitions = {ny}
    set X extent = {X_KM * 1e3:g}
    set Y extent = {Y_KM * 1e3:g}
  end
end
subsection Mesh refinement
  set Initial global refinement = 0
  set Initial adaptive refinement = 0
  set Time steps between mesh refinement = 0
end
subsection Solver parameters
  subsection Stokes solver parameters
    set Number of cheap Stokes solver steps = {CHEAP_STEPS}
    set Stokes solver type = block AMG
    set Maximum number of expensive Stokes solver steps = {EXPENSIVE_STEPS}
    set GMRES solver restart length = {GMRES_RESTART}
  end
  subsection Newton solver parameters
    set Maximum linear Stokes solver tolerance = 1e-2
    set Use Eisenstat Walker method for Picard iterations = true
  end
end
subsection Mesh deformation
  set Mesh deformation boundary indicators = top: free surface
  subsection Free surface
    set Surface velocity projection = vertical
  end
end
subsection Boundary velocity model
  set Tangential velocity boundary indicators = left
  set Prescribed velocity boundary indicators = right x: force controlled, bottom y: force controlled
  subsection Force controlled
    set Boundary = right
    set Force = {F * 1e12:g}
    set Initial velocity = {FC_V0_MM_YR / 1e3:g}
    set Exponent = {FC_EXPONENT:g}
    set Maximum factor = {FC_MAX_FACTOR:g}
  end
end
subsection Compositional fields
  set Number of fields = 3
  set Names of fields = plastic_strain, crust, mantle_lith
end
subsection Initial composition model
  set Model name = function
  subsection Function
    set Variable names = x,y
    set Function constants = H={Y_KM * 1e3:g}, moho={ZC_KM * 1e3:g}, lab={LAB_KM * 1e3:g}
    set Function expression = 0; if(H-y<moho, 1, 0); if(H-y>=moho && H-y<lab, 1, 0)
  end
end
subsection Boundary composition model
  set Fixed composition boundary indicators = bottom
  set List of model names = initial composition
end
subsection Initial temperature model
  set Model name = ascii data
  subsection Ascii data model
    set Data directory = {D}/
    set Data file name = r2d_T.txt
  end
end
subsection Boundary temperature model
  set Fixed temperature boundary indicators = bottom, top
  set List of model names = box
  subsection Box
    set Bottom temperature = {t_lab()}
    set Top temperature = {T_TOP_K}
  end
end
subsection Heating model
  set List of model names = compositional heating
  subsection Compositional heating
    set Use compositional field for heat production averaging = 1, 0, 1, 1
    set Compositional heating values = {comp((0.0, 0.0, hc, hm))}
  end
end
subsection Material model
  set Model name = visco plastic
  subsection Visco Plastic
    set Reference temperature = 0
    set Minimum strain rate = 1.e-20
    set Reference strain rate = 1.e-16
    set Minimum viscosity = {ETA_MIN:g}
    set Maximum viscosity = {ETA_MAX:g}
    set Define thermal conductivities = true
    set Thermal conductivities = {comp((K['a'], K['a'], K['c'], K['m']))}
    set Heat capacities = {CP}
    set Densities = {comp((RHO['a'], 1.0, RHO['c'], RHO['m']))}
    set Thermal expansivities = {comp((ALPHA['a'], 0.0, ALPHA['c'], ALPHA['m']))}
    set Viscosity averaging scheme = harmonic
    set Viscous flow law = composite
    set Grain size = 1
    set Grain size exponents for diffusion creep = 0
    set Prefactors for diffusion creep = 1.5e-9, 1e-50, 1e-50, 1e-50
    set Activation energies for diffusion creep = 335e3, 0, 0, 0
    set Activation volumes for diffusion creep = 4e-6, 0, 0, 0
    set Prefactors for dislocation creep = {comp((2.12e-15, 1e-50, c['prefactor'], m['prefactor']))}
    set Stress exponents for dislocation creep = {comp((3.5, 1, N_CREEP, N_CREEP))}
    set Activation energies for dislocation creep = {comp((480e3, 0, c['E'], m['E']))}
    set Activation volumes for dislocation creep = {comp((11e-6, 0, c['V'], m['V']))}
    set Angles of internal friction = {comp((phi_a, phi_a, c['phi'], m['phi']))}
    set Cohesions = 0
    set Maximum yield stress = {sr.D_CREEP:g}
    set Strain weakening mechanism = plastic weakening with plastic strain only
    set Start plasticity strain weakening intervals = 0
    set End plasticity strain weakening intervals = 1
    set Cohesion strain weakening factors = 1
    set Friction strain weakening factors = {comp((m['soft'], m['soft'], c['soft'], m['soft']))}
  end
end
subsection Gravity model
  set Model name = vertical
  subsection Vertical
    set Magnitude = {G}
  end
end
subsection Termination criteria
  set Termination criteria = end time, wall time, user request
  set Wall time = {wall_h:g}
  set Checkpoint on termination = true
end
subsection Checkpointing
  set Time between checkpoint = {CHECKPOINT_S}
end
subsection Postprocess
  set List of postprocessors = velocity boundary statistics, topography, temperature statistics, visualization
  subsection Visualization
    set List of output variables = material properties, strain rate
    set Output format = vtu
    set Time between graphical output = {0 if SMOKE else VIZ_YR:g}
    set Interpolate output = false
    subsection Material properties
      set List of material properties = density, viscosity
    end
  end
end
"""


def watch(stop):
    """Stop ASPECT once the opening velocity exceeds V_FAST: the verdict is then RIFT and later steps are not used, so
    the rest of the segment would only cost machine time. ASPECT's "user request" criterion ends the run (with a
    checkpoint) when the file terminate-aspect appears in the output directory."""
    col = None
    while not stop.wait(WATCH_S):
        try:
            lines = open(f"{OUT}/statistics").read().splitlines()
            if col is None:
                head = [l for l in lines if l.startswith("#")]
                col = next(int(l[1:].split(":")[0]) - 1 for l in head
                           if "Maximum velocity magnitude on boundary with indicator 1" in l)
            if any(float(l.split()[col]) > V_FAST for l in lines if l and not l.startswith("#") and len(l.split()) > col):
                open(f"{OUT}/terminate-aspect", "w").close()
                print("fast phase reached: terminate-aspect written", flush=True)
                return
        except (OSError, StopIteration, ValueError):        # file not written yet or a line half written
            pass


def aspect(rh, end_yr, wall_h, resume):
    open(f"{D}/r2d.prm", "w").write(prm(rh, end_yr, wall_h, resume))
    if os.path.exists(f"{OUT}/terminate-aspect"):
        os.remove(f"{OUT}/terminate-aspect")
    stop = threading.Event()
    threading.Thread(target=watch, args=(stop,), daemon=True).start()
    try:
        return _aspect(wall_h)
    finally:
        stop.set()


def _aspect(wall_h):
    return run(f"export OMP_NUM_THREADS=1 OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1; $(ls /usr/bin/mpirun.openmpi 2>/dev/null || echo mpirun) --use-hwthread-cpus "
               f"-np 4 {EXE} r2d.prm >> out.txt 2>&1; e=$?; tail -15 out.txt; exit $e", cwd=D,
               timeout=int(wall_h * 3600 + KILL_GRACE_S))


os.makedirs(OUT, exist_ok=True)
t0 = time.time()
RH = rheology()
res["inputs"] = write_inputs(RH)
res["inputs_s"] = round(time.time() - t0, 1)
print(json.dumps(res, indent=1), flush=True)
RESTART = os.environ.get("ASPECT_RESTART", "restart.zip")   # Kaggle: a run's own restart file packed into the code
resume = os.path.exists(RESTART)
if resume:
    zipfile.ZipFile(RESTART).extractall(OUT)
    res["resumed_from"] = sorted(os.listdir(OUT))
if os.environ.get("ASPECT_INPUTS_ONLY") == "1":          # local check of the inputs, no ASPECT
    open(f"{D}/r2d.prm", "w").write(prm(RH, END_MYR * 1e6, WALL_H, resume))
    json.dump(res, open(f"results/{TAG}_run.json", "w"), indent=1)
    raise SystemExit
run("lscpu | grep -E '^CPU\\(s\\)|Model name'; free -g")
SUDO = "sudo " if os.geteuid() != 0 else ""          # GitHub runs as a user, Kaggle as root without sudo
run(f"{SUDO}apt-get update -qq && {SUDO}DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libdeal.ii-dev cmake "
    "ninja-build 2>&1 | tail -3")
run(f"git clone -q --depth 1 --branch v3.0.0 https://github.com/geodynamics/aspect.git {A}")
run(f"cmake -S {A} -B {B} -G Ninja -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3")
_, _, res["build_s"] = run(f"ninja -C {B} -j 4 2>&1 | tail -3")
os.makedirs(os.path.dirname(PLUGIN), exist_ok=True)
shutil.copy("aspect_force_controlled.cc", os.path.dirname(PLUGIN))
open(os.path.dirname(PLUGIN) + "/CMakeLists.txt", "w").write(
    "CMAKE_MINIMUM_REQUIRED(VERSION 3.13.4)\nFIND_PACKAGE(Aspect 3.0.0 REQUIRED HINTS " + B + ")\n"
    "DEAL_II_INITIALIZE_CACHED_VARIABLES()\nPROJECT(aspect_force_controlled)\n"
    "ADD_LIBRARY(aspect_force_controlled SHARED aspect_force_controlled.cc)\nASPECT_SETUP_PLUGIN(aspect_force_controlled)\n")
run(f"cd {os.path.dirname(PLUGIN)} && cmake -G Ninja . 2>&1 | tail -3 && ninja 2>&1 | tail -5 && ls *.so")
so = glob.glob(os.path.dirname(PLUGIN) + "/*.so")
if so and so[0] != PLUGIN:
    shutil.copy(so[0], PLUGIN)
EXE = f"{B}/aspect-release" if os.path.exists(f"{B}/aspect-release") else f"{B}/aspect"
pieces = [(SMOKE_WALL_H, resume), (SMOKE_WALL_H, True)] if SMOKE else [(WALL_H, resume)]
res["pieces"] = []
for wall_h, rs in pieces:
    code, _, s = aspect(RH, END_MYR * 1e6, wall_h, rs)
    log = open(f"{D}/out.txt").read()
    steps = re.findall(r"\*\*\* Timestep (\d+):\s+t=([\d.e+-]+) years", log)
    res["pieces"].append({"exit": str(code), "run_s": round(s), "last_step": int(steps[-1][0]) if steps else None,
                          "model_yr": float(steps[-1][1]) if steps else None})
    json.dump(res, open(f"results/{TAG}_run.json", "w"), indent=1)
cells = re.findall(r"Number of active cells: ([\d,]+)", log)
dofs = re.findall(r"Number of degrees of freedom: ([\d,]+)", log)
res.update(cells=cells[-1] if cells else None, dofs=dofs[-1] if dofs else None)
json.dump(res, open(f"results/{TAG}_run.json", "w"), indent=1)
shutil.copy(f"{D}/out.txt", f"results/{TAG}_out.txt")
for f in ("statistics", "log.txt"):
    if os.path.exists(f"{OUT}/{f}"):
        shutil.copy(f"{OUT}/{f}", f"results/{TAG}_{f}")
with zipfile.ZipFile(f"results/{TAG}_restart.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(f"{OUT}/restart*") + [f"{OUT}/statistics", f"{OUT}/log.txt"]:
        if os.path.isfile(f):
            z.write(f, os.path.basename(f))
with zipfile.ZipFile(f"results/{TAG}_viz.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(f"{OUT}/solution/*") + glob.glob(f"{OUT}/solution.pvd"):
        z.write(f, os.path.relpath(f, OUT))
print(json.dumps(res, indent=1))
