"""GitHub Actions: reference model M1 of Brune et al. 2016 (Nature 536:201; Methods, Extended Data Table 1 and Fig. 7)
in ASPECT v3.0.0, to test whether our 2D set-up reproduces their constant-force rift evolution before it is used to
calibrate the 1D criterion of tools/diag/brune_force_time.py (E-1121-18).
Set-up (Methods): 500 x 150 km box, 2 km cells; 25 km wet-quartzite upper crust, 10 km wet-anorthite lower crust,
dry-olivine mantle lithosphere to 120 km, wet-olivine asthenosphere; diffusion + dislocation creep, Drucker-Prager
plasticity with friction softening 0.5 -> 0.05 over plastic strain 0..1; free surface; steady initial geotherm 0 C at the
surface, 1350 C below the lithosphere, with the 1350 C isotherm raised 10 km in a 20 km wide triangle at the centre.
Force: left side fixed (zero velocity: horizontal by symmetry, vertical because with an open bottom nothing else
stops the box from moving up or down as a whole - ASPECT's free-surface stabilization scales with the time step, which
is 0 at step 0, so with free slip the first Stokes solve fails with NaN), right side the lithostatic pressure of the far-field column minus a uniform
tension F / 120 km over the lithosphere, so the force in excess of lithostatic is F; bottom open with the lithostatic
pressure (in place of their Winkler foundation). Not included: elasticity, compressibility, shear heating (see
PARAM_AUDIT 11.15). The opening velocity is the maximum velocity on the right boundary (ASPECT statistics).
Segments: a run stops at ASPECT_WALL_H hours with a checkpoint; if restart.zip from the previous segment is present in
the code folder, the run resumes from it.
Env: BRUNE_F (TN/m), ASPECT_END_MYR, ASPECT_WALL_H, ASPECT_SMOKE (1 = two short pieces, the second resumed from the
first, to test the checkpoint chain). Output: results/brune_<F>_{run.json,out.txt,statistics,restart.zip,viz.zip}"""
import glob, json, os, re, shutil, subprocess, time, zipfile
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

A = "/tmp/aspect"; B = "/tmp/aspect-build"; D = os.environ.get("ASPECT_DIR", "/tmp/run"); OUT = f"{D}/output"
F = float(os.environ.get("BRUNE_F", "9.0"))            # TN/m
END_MYR = float(os.environ.get("ASPECT_END_MYR", "15"))
WALL_H = float(os.environ.get("ASPECT_WALL_H", "5.3"))
SMOKE = os.environ.get("ASPECT_SMOKE") == "1"
SMOKE_WALL_H = 0.1                 # technical: each smoke piece stops after 6 min
TAG = f"brune_{F:g}"

# Brune et al. 2016 Methods and Extended Data Table 1 (bg = weak mantle; then the fields upper crust, lower crust,
# strong mantle). Order in ASPECT lists: background, plastic_strain (placeholder, excluded from the volume fractions),
# upper crust, lower crust, strong mantle.
X_KM, Y_KM, CELL_KM = 500.0, 150.0, 2.0
UC_KM, MOHO_KM, LAB_KM = 25.0, 35.0, 120.0
SEED_KM, SEED_HALF_KM = 10.0, 10.0          # 1350 C isotherm raised 10 km, triangle 20 km wide
T_TOP_C, T_LAB_C = 0.0, 1350.0
RHO = dict(uc=2700.0, lc=2850.0, sm=3280.0, wm=3300.0)
ALPHA = dict(uc=2.7e-5, lc=2.7e-5, sm=3.0e-5, wm=3.0e-5)
K = dict(uc=2.5, lc=2.5, sm=3.3, wm=3.3)
HP = dict(uc=1.5e-6, lc=0.2e-6, sm=0.0, wm=0.0)
CP = 1200.0
COHESION = 5e6
MU0, MU1 = 0.5, 0.05                        # friction before and after softening over plastic strain 0..1
G = 9.81
T_REF_K = 273.15                            # density reference temperature = surface temperature (PARAM_AUDIT 11.15)
DX_T = 1e3                                  # m, technical: spacing of the initial-temperature grid (half a cell)
ETA_MIN, ETA_MAX = 1e18, 1e26               # Pa s, technical cut-offs of the ASPECT cookbook
DT_MAX_YR = 20e3                            # technical: the cookbook's maximum time step
CFL = 0.5
NL_TOL, NL_MAX = 1e-4, 100                  # cookbook nonlinear solver settings
VIZ_YR = 1e6
CHECKPOINT_S = 1800

os.makedirs("results", exist_ok=True)
res = {"F_TN_m": F, "end_myr": END_MYR, "smoke": SMOKE}


def run(cmd, cwd="/tmp", timeout=None):
    t = time.time()
    print(f"$ {cmd[:150]}", flush=True)
    try:
        r = subprocess.run(["bash", "-c", cmd], cwd=cwd, capture_output=True, text=True, timeout=timeout)
        out, code = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        out, code = (e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")), "timeout"
    print(out[-3000:], f"\n[exit {code}, {time.time() - t:.0f} s]", flush=True)
    return code, out, time.time() - t


def layer(depth_km):
    return np.where(depth_km < UC_KM, 0, np.where(depth_km < MOHO_KM, 1, np.where(depth_km < LAB_KM, 2, 3)))


def initial_temperature():
    """steady conduction on a DX_T grid: T_TOP at the surface, T_LAB at and below the (seeded) LAB, insulated sides"""
    x = np.arange(0, X_KM * 1e3 + 1, DX_T); d = np.arange(0, Y_KM * 1e3 + 1, DX_T)     # d = depth
    nx, nd = len(x), len(d)
    names = ("uc", "lc", "sm", "wm")
    lay = layer(d / 1e3)
    k = np.array([K[names[i]] for i in lay]); h = np.array([HP[names[i]] for i in lay])
    lab = LAB_KM * 1e3 - SEED_KM * 1e3 * np.clip(1 - np.abs(x - X_KM * 5e2) / (SEED_HALF_KM * 1e3), 0, 1)
    fixed = (d[:, None] >= lab[None, :]) | (d[:, None] == 0)
    tfix = np.where(d[:, None] == 0, T_TOP_C, T_LAB_C) * np.ones((nd, nx))
    idx = np.arange(nd * nx).reshape(nd, nx)
    rows, cols, vals = [], [], []
    rhs = np.zeros(nd * nx)
    kf_d = 2 * k[1:] * k[:-1] / (k[1:] + k[:-1])           # harmonic k between depth nodes j, j+1
    for j in range(nd):
        for i in range(nx):
            p = idx[j, i]
            if fixed[j, i]:
                rows.append(p); cols.append(p); vals.append(1.0); rhs[p] = tfix[j, i]
                continue
            diag = 0.0
            for jj, ii, kk in ((j - 1, i, kf_d[j - 1]), (j + 1, i, kf_d[j] if j + 1 < nd else 0.0),
                               (j, i - 1, k[j]), (j, i + 1, k[j])):
                if 0 <= jj < nd and 0 <= ii < nx and kk > 0:   # outside in x: insulated (no flux)
                    rows.append(p); cols.append(idx[jj, ii]); vals.append(kk); diag += kk
            rows.append(p); cols.append(p); vals.append(-diag)
            rhs[p] = -h[j] * DX_T ** 2
    t = spsolve(coo_matrix((vals, (rows, cols)), shape=(nd * nx, nd * nx)).tocsr(), rhs).reshape(nd, nx)
    return x, d, t


def lithostatic(d, tcol):
    names = ("uc", "lc", "sm", "wm")
    lay = layer(d / 1e3)
    rho = np.array([RHO[names[i]] * (1 - ALPHA[names[i]] * tc) for i, tc in zip(lay, tcol)])
    return np.concatenate([[0.0], np.cumsum(0.5 * (rho[1:] + rho[:-1]) * G * np.diff(d))])


def write_inputs():
    x, d, t = initial_temperature()
    y = Y_KM * 1e3 - d                                     # ASPECT y: up from the bottom
    order = np.argsort(y)
    with open(f"{D}/brune_T.txt", "w") as f:
        f.write(f"# POINTS: {len(x)} {len(d)}\n# x y T\n")
        for j in order:
            for i in range(len(x)):
                f.write(f"{x[i]:.0f} {y[j]:.0f} {t[j, i] + 273.15:.3f}\n")
    p = lithostatic(d, t[:, -1])                           # far-field column at the right side
    # ASPECT interpolates linearly between the file points, so the tension ends in a ramp half a grid step wide either
    # side of the LAB node; dividing by LAB + DX_T / 2 makes that profile integrate to F exactly
    tension = np.where(d <= LAB_KM * 1e3, F * 1e12 / (LAB_KM * 1e3 + DX_T / 2), 0.0)
    with open(f"{D}/brune_traction_right.txt", "w") as f:
        f.write(f"# POINTS: {len(d)}\n# y pressure\n")
        for j in order:
            f.write(f"{y[j]:.0f} {p[j] - tension[j]:.6e}\n")
    with open(f"{D}/brune_traction_bottom.txt", "w") as f:
        f.write(f"# POINTS: 2\n# x pressure\n0 {p[-1]:.6e}\n{X_KM * 1e3:.0f} {p[-1]:.6e}\n")
    names = ("uc", "lc", "sm", "wm")
    moho = int(MOHO_KM * 1e3 / DX_T)
    return {"T_moho_far_C": round(float(t[moho, -1]), 1), "T_moho_centre_C": round(float(t[moho, len(x) // 2]), 1),
            "surface_heat_flow_mW": round(float(K["uc"] * (t[1, -1] - t[0, -1]) / DX_T * 1e3), 1),
            "p_bottom_Pa": float(f"{p[-1]:.4e}"),
            "check_force_TN_m": round(float(np.sum(0.5 * (tension[1:] + tension[:-1]) * np.diff(d))) / 1e12, 3)}


def prm(end_yr, wall_h, resume):
    def comp(vals):                                    # background, plastic_strain placeholder, uc, lc, sm
        return ", ".join(str(v) for v in vals)
    phi0, phi1 = np.degrees(np.arctan(MU0)), np.degrees(np.arctan(MU1))
    nx, ny = int(X_KM / CELL_KM), int(Y_KM / CELL_KM)
    return f"""
set Dimension = 2
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
    set Number of cheap Stokes solver steps = 0
    set Stokes solver type = block AMG
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
  set Zero velocity boundary indicators = left
end
subsection Boundary traction model
  set Prescribed traction boundary indicators = right: ascii data, bottom: ascii data
  subsection Ascii data model
    set Data directory = {D}/
    set Data file name = brune_traction_%s.txt
  end
end
subsection Compositional fields
  set Number of fields = 4
  set Names of fields = plastic_strain, crust_upper, crust_lower, mantle_strong
end
subsection Initial composition model
  set Model name = function
  subsection Function
    set Variable names = x,y
    set Function constants = H={Y_KM * 1e3:g}, uc={UC_KM * 1e3:g}, moho={MOHO_KM * 1e3:g}, lab={LAB_KM * 1e3:g}
    set Function expression = 0; if(H-y<uc, 1, 0); if(H-y>=uc && H-y<moho, 1, 0); if(H-y>=moho && H-y<lab, 1, 0)
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
    set Data file name = brune_T.txt
  end
end
subsection Boundary temperature model
  set Fixed temperature boundary indicators = bottom, top
  set List of model names = box
  subsection Box
    set Bottom temperature = {T_LAB_C + 273.15}
    set Top temperature = {T_TOP_C + 273.15}
  end
end
subsection Heating model
  set List of model names = compositional heating
  subsection Compositional heating
    set Use compositional field for heat production averaging = 1, 0, 1, 1, 1
    set Compositional heating values = {comp((HP['wm'], 0.0, HP['uc'], HP['lc'], HP['sm']))}
  end
end
subsection Material model
  set Model name = visco plastic
  subsection Visco Plastic
    set Reference temperature = {T_REF_K}
    set Minimum strain rate = 1.e-20
    set Reference strain rate = 1.e-16
    set Minimum viscosity = {ETA_MIN:g}
    set Maximum viscosity = {ETA_MAX:g}
    set Define thermal conductivities = true
    set Thermal conductivities = {comp((K['wm'], K['wm'], K['uc'], K['lc'], K['sm']))}
    set Heat capacities = {CP}
    set Densities = {comp((RHO['wm'], 1.0, RHO['uc'], RHO['lc'], RHO['sm']))}
    set Thermal expansivities = {comp((ALPHA['wm'], 0.0, ALPHA['uc'], ALPHA['lc'], ALPHA['sm']))}
    set Viscosity averaging scheme = harmonic
    set Viscous flow law = composite
    set Grain size = 1
    set Grain size exponents for diffusion creep = 0
    set Prefactors for diffusion creep = 1.5e-9, 1e-50, 1e-50, 1e-50, 2.25e-9
    set Activation energies for diffusion creep = 335e3, 0, 0, 0, 375e3
    set Activation volumes for diffusion creep = 4e-6, 0, 0, 0, 6e-6
    set Prefactors for dislocation creep = 2.12e-15, 1e-50, 8.57e-28, 1.79e-15, 6.51e-16
    set Stress exponents for dislocation creep = 3.5, 1, 4.0, 3.0, 3.5
    set Activation energies for dislocation creep = 480e3, 0, 223e3, 356e3, 530e3
    set Activation volumes for dislocation creep = 11e-6, 0, 0, 0, 13e-6
    set Angles of internal friction = {phi0:.4f}
    set Cohesions = {COHESION:g}
    set Strain weakening mechanism = plastic weakening with plastic strain only
    set Start plasticity strain weakening intervals = 0
    set End plasticity strain weakening intervals = 1
    set Cohesion strain weakening factors = 1
    set Friction strain weakening factors = {phi1 / phi0:.5f}
  end
end
subsection Gravity model
  set Model name = vertical
  subsection Vertical
    set Magnitude = {G}
  end
end
subsection Termination criteria
  set Termination criteria = end time, wall time
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
    set Time between graphical output = {VIZ_YR:g}
    set Interpolate output = false
    subsection Material properties
      set List of material properties = density, viscosity
    end
  end
end
"""


def aspect(end_yr, wall_h, resume):
    open(f"{D}/brune.prm", "w").write(prm(end_yr, wall_h, resume))
    return run(f"export OMP_NUM_THREADS=1; $(ls /usr/bin/mpirun.openmpi 2>/dev/null || echo mpirun) --use-hwthread-cpus "
               f"-np 4 {EXE} brune.prm >> out.txt 2>&1; e=$?; tail -15 out.txt; exit $e", cwd=D,
               timeout=int(wall_h * 3600 + 1200))


os.makedirs(OUT, exist_ok=True)
t0 = time.time()
res["inputs"] = write_inputs()
res["inputs_s"] = round(time.time() - t0, 1)
print(json.dumps(res, indent=1), flush=True)
resume = os.path.exists("restart.zip")
if resume:
    zipfile.ZipFile("restart.zip").extractall(OUT)
    res["resumed_from"] = sorted(os.listdir(OUT))
if os.environ.get("ASPECT_INPUTS_ONLY") == "1":          # local check of the inputs, no ASPECT
    open(f"{D}/brune.prm", "w").write(prm(END_MYR * 1e6, WALL_H, resume))
    json.dump(res, open(f"results/{TAG}_run.json", "w"), indent=1)
    raise SystemExit
run("lscpu | grep -E '^CPU\\(s\\)|Model name'; free -g")
run("sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libdeal.ii-dev cmake "
    "ninja-build 2>&1 | tail -3")
run(f"git clone -q --depth 1 --branch v3.0.0 https://github.com/geodynamics/aspect.git {A}")
run(f"cmake -S {A} -B {B} -G Ninja -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3")
_, _, res["build_s"] = run(f"ninja -C {B} -j 4 2>&1 | tail -3")
EXE = f"{B}/aspect-release" if os.path.exists(f"{B}/aspect-release") else f"{B}/aspect"
pieces = [(SMOKE_WALL_H, resume), (SMOKE_WALL_H, True)] if SMOKE else [(WALL_H, resume)]
res["pieces"] = []
for wall_h, rs in pieces:
    code, _, s = aspect(END_MYR * 1e6, wall_h, rs)
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
