"""GitHub Actions: 2D induced subduction initiation at an oceanic transform fault under compression in ASPECT v3.0.0,
to find which plate subducts (polarity) when both sides are normal oceanic lithosphere (E-1122-04, E-1123-01..03).
The set-up is that of Zhang, Zlotnik & Li 2021 (G-cubed 22, doi:10.1029/2020GC009549; ZZL), so that ZZL's own results
check it first (benchmark cases), then the transform fault gets the strength measured on Earth (Hall & Gurnis 2005):
  - box 360 km x 110 km below the surface (ZZL 120 km with 10 km sticky water; here a free surface), 1 km cells;
  - young plate x < 180 km, old plate 180..330 km, a weak contact 330..360 km detaching the old plate from the right
    side (ZZL Fig. 2); 7 km crust on both;
  - half-space cooling geotherms of the two ages, 273 K at the surface, 1573 K + 0.5 K/km below (1628 K at the base);
  - one dislocation-creep law for all rocks (ZZL Table 1 after Gurnis et al. 2004): eta0 = 5e19 Pa s at 1e-15 s^-1 and
    1400 C, n = 3.05, E = 540 kJ/mol; viscosity 1e19..1e24 Pa s;
  - Drucker-Prager plasticity tau = C + mu P, mu 0.3 -> 0 and C 44 -> 4.4 MPa over plastic strain 0..0.8 (ZZL reference);
  - the old plate pushed leftwards at 1.6 cm/yr inside a 10 km x 13 km block at x 300..310 km, depth 7..20 km, which
    does not yield (ZZL Fig. 2), with ASPECT's prescribed_velocity cookbook plugin.
Transform fault (SI_TF):
  none   - as strong as the plates (ZZL Table 2);
  strain - ZZL's transition zone (x 177..183 km, depth 3..52 km) fully weakened: initial plastic strain
           0.8 (ZZL W7 had random strain up to 0.8; W5 with up to 0.24 already gives the same polarity);
  <MPa>  - Hall & Gurnis 2005: a 7 km wide, 16 km deep fault zone of constant yield strength (MPa), their best-fitting
           depth-averaged strengths of Pacific fracture zones being 1..10 MPa, around 4.
Output: point values of the two crusts on a grid around the fault every 0.25 Myr (tools/diag/si2d_eval.py gives the
polarity), statistics, visualization every 0.5 Myr.
Env: SI_AY, SI_AO (Myr), SI_TF, SI_SCHEME (ASPECT nonlinear solver scheme), SI_NEWTON_SWITCH, SI_DAMPER, SI_CELL_KM
(solver tests), SI_NP (MPI processes, default 4), ASPECT_END_MYR, ASPECT_WALL_H, ASPECT_SMOKE, ASPECT_INPUTS_ONLY,
ASPECT_RESTART, ASPECT_BUILD (all: build and run; only: build and stop; skip: use the build of an earlier call, as
tools/si2d_batch.py does to run several cases at once on one large machine).
Output: results/<tag>_{run.json,out.txt,statistics,point_values.txt,log.txt,restart.zip,viz.zip}
All numbers: PARAM_AUDIT 11.19."""
import glob, json, math, os, re, shutil, signal, subprocess, sys, threading, time, zipfile
import numpy as np
from scipy.special import erf

A = "/tmp/aspect"; B = "/tmp/aspect-build"; D = os.environ.get("ASPECT_DIR", "/tmp/run"); OUT = f"{D}/output"
AY = float(os.environ.get("SI_AY", "27.5"))               # Myr, ZZL reference case
AO = float(os.environ.get("SI_AO", "49"))
TF = os.environ.get("SI_TF", "none")
END_MYR = float(os.environ.get("ASPECT_END_MYR", "5"))
WALL_H = float(os.environ.get("ASPECT_WALL_H", "5.3"))
KILL_GRACE_S = 600                 # technical: hard stop this long after WALL_H, so packing and upload fit (PARAM_AUDIT 11.17)
SMOKE = os.environ.get("ASPECT_SMOKE") == "1"
SMOKE_WALL_H = 0.1                 # technical: each smoke piece stops after 6 min (as aspect_rift2d.py)
TAG = f"si2d_ay{AY:g}_ao{AO:g}_tf{TF}"
SCHEME = os.environ.get("SI_SCHEME", "single Advection, iterated defect correction Stokes")   # as aspect_rift2d.py; Newton test (PARAM_AUDIT 11.20)
NEWTON_SWITCH = os.environ.get("SI_NEWTON_SWITCH")   # solver test: Picard until this residual, then Newton (PARAM_AUDIT 11.20)
DAMPER = os.environ.get("SI_DAMPER")                 # solver test: plastic damper viscosity, Pa s (PARAM_AUDIT 11.20)
NP = int(os.environ.get("SI_NP", "4"))            # technical: the 4 cores of a GitHub/Kaggle CPU machine (PARAM_AUDIT 11.20)
BUILD = os.environ.get("ASPECT_BUILD", "all")
ENV_KEYS = ("SI_AY", "SI_AO", "SI_TF", "SI_SCHEME", "SI_NEWTON_SWITCH", "SI_DAMPER", "SI_CELL_KM", "SI_NP", "ASPECT_END_MYR", "ASPECT_WALL_H")

MYR = 3.15576e13                   # s (Julian year, ASPECT's year_in_seconds)
R_GAS = 8.31                       # ZZL Table 1
X_KM, Y_KM, CELL_KM = 360.0, 110.0, 1.0       # ZZL: 360 x 120 km with 10 km sticky water, 1 km grid
CELL_KM = float(os.environ.get("SI_CELL_KM", CELL_KM))   # solver test: coarser grid (PARAM_AUDIT 11.20)
X_TF, X_END = 180.0, 330.0                    # ZZL Fig. 2: plate contact, right end of the old plate
CRUST_KM = 7.0                                # ZZL
PUSH_X, PUSH_Z = (300.0, 310.0), (7.0, 20.0)  # ZZL Fig. 2 (read off the figure)
V_PUSH = 1.6e-2                               # m/yr, ZZL
TZ_X, TZ_Z = (177.0, 183.0), (3.0, 52.0)     # km: ZZL Fig. 2 transition zone (read off the figure)
TZ_STRAIN = 0.8                               # ZZL W7 maximum = end of the weakening interval
HG_WIDTH_KM, HG_DEPTH_KM = 7.0, 16.0          # Hall & Gurnis 2005 fault zone
T_SURF, T_POT, GRAD = 273.0, 1573.0, 0.5      # K, K, K/km: ZZL (1628 K at 110 km)
KAPPA = 1e-6                                  # m2/s, Turcotte & Schubert half-space cooling
RHO_C, RHO_M, ALPHA = 2800.0, 3200.0, 3e-5    # ZZL Table 1
CP = 1250.0                        # J/kg/K, technical: with the diffusivity given it only sets k = kappa rho cp
ETA0, EDOT0, N_CR, E_CR, T0 = 5e19, 1e-15, 3.05, 540e3, 1673.15     # ZZL Table 1
ETA_MIN, ETA_MAX = 1e19, 1e24                 # ZZL Table 1
MU0, C0, C_MIN, EPS_F = 0.3, 44e6, 4.4e6, 0.8  # ZZL Table 1 and reference case
C_RIGID = 1e12                     # Pa, technical: "no plasticity" in the push block (ASPECT's default yield cap)
G = 9.81
DT_MAX_YR = 20e3; CFL = 0.5; NL_TOL, NL_MAX = 1e-4, 100     # as aspect_rift2d.py (Brune M1)
CHEAP_STEPS = 4000                 # as aspect_rift2d.py (E-1122-03)
EXPENSIVE_STEPS, GMRES_RESTART = 5000, 200   # ASPECT crustal_deformation cookbook: the default 1000/50 stopped near convergence (PARAM_AUDIT 11.20)
PV_YR, VIZ_YR = 0.25e6, 0.5e6      # technical: output intervals
PV_X, PV_Z = (100.0, 300.0, 4.0), (10.0, 100.0, 3.0)   # km: point-value grid (start, end, step), technical
CHECKPOINT_S = 1800
SMOKE_CHECKPOINT_S = 60          # technical: smoke pieces are killed after minutes, so the resume test needs earlier checkpoints
PACK_POLL_S = 30                 # technical: look for a new checkpoint this often (PARAM_AUDIT 11.17)
PACK_SETTLE_S = 20               # technical: pack once the checkpoint files have been still this long
NET_TIMEOUT_S = 900               # technical: give up and retry a hung apt-get or git clone (PARAM_AUDIT 11.17)
PLUGIN = "/tmp/aspect-plugin/libprescribed_velocity.so"

os.makedirs("results", exist_ok=True)
res = {"script": "aspect_si2d.py", "age_young_myr": AY, "age_old_myr": AO, "tf": TF, "end_myr": END_MYR,
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
        left = subprocess.run(["pgrep", "-c", "-x", "aspect-release"], capture_output=True, text=True).stdout.strip()
        out += f"\nprocesses left after the hard stop: {left or 0}\n"
    print(out[-3000:], f"\n[exit {code}, {time.time() - t:.0f} s]", flush=True)
    return code, out, time.time() - t


def geotherm(d_km, age_myr):
    """half-space cooling towards the adiabat T_POT + GRAD z (K) on depths d_km"""
    t_ad = T_POT + GRAD * d_km
    return T_SURF + (t_ad - T_SURF) * erf(d_km * 1e3 / (2 * math.sqrt(KAPPA * age_myr * MYR)))


def lith_km(age_myr):
    """depth of the 1573 K isotherm (ZZL's plate thickness; reported only), None if below the box"""
    d = np.arange(0.0, Y_KM + 1e-9, 0.1)
    k = geotherm(d, age_myr) >= 1573.0
    return float(d[np.argmax(k)]) if k.any() else None


def write_temperature():
    x = np.arange(0.0, X_KM + 1e-9, CELL_KM); d = np.arange(0.0, Y_KM + 1e-9, CELL_KM)
    with open(f"{D}/si2d_T.txt", "w") as f:
        f.write(f"# POINTS: {len(x)} {len(d)}\n# x y T\n")
        for dd in d[::-1]:                          # ascending y = Y - depth
            ty, to = geotherm(dd, AY), geotherm(dd, AO)
            for xx in x:
                f.write(f"{xx * 1e3:.0f} {(Y_KM - dd) * 1e3:.0f} {ty if xx < X_TF else to:.3f}\n")


def creep_prefactor():
    """ZZL eta = eta0 (edot/edot0)^(1/n - 1) exp(E/(nR) (1/T - 1/T0)) as ASPECT's dislocation creep
    eta = 1/2 A^(-1/n) edot^(1/n - 1) exp(E/(nRT)): A = (2 eta0)^-n edot0^(1-n) exp(E/(R T0))"""
    return (2 * ETA0) ** -N_CR * EDOT0 ** (1 - N_CR) * math.exp(E_CR / (R_GAS * T0))


def prm(end_yr, wall_h, resume):
    """fields: plastic_strain, crust_y, crust_o, tf_c (fault in the crust), tf_m (fault below), push, contact"""
    phi0 = math.degrees(math.asin(MU0))                  # 2D Drucker-Prager C cos(phi) + P sin(phi) = C + mu P
    c0 = C0 / math.cos(math.radians(phi0))
    tf_c = float(TF) * 1e6 if TF not in ("none", "strain") else C_MIN
    #            background  strain  crust_y  crust_o  tf_c   tf_m   push     contact
    rho = (RHO_M, RHO_M, RHO_C, RHO_C, RHO_C, RHO_M, RHO_M, RHO_M)
    phi = (phi0, phi0, phi0, phi0, 0.0, 0.0, 0.0, 0.0)
    coh = (c0, c0, c0, c0, tf_c, tf_c, C_RIGID, C_MIN)
    cw = (C_MIN / c0,) * 4 + (1.0,) * 4             # weakening only for the plate rocks
    fw = (0.0,) * 4 + (1.0,) * 4
    lst = lambda v: ", ".join(f"{a:.6g}" for a in v)
    hgx = (X_TF - HG_WIDTH_KM / 2, X_TF + HG_WIDTH_KM / 2)
    in_tf = (f"x>={hgx[0] * 1e3:g} && x<={hgx[1] * 1e3:g} && d<={HG_DEPTH_KM * 1e3:g}"
             if TF not in ("none", "strain") else "0")
    strain = (f"if(x>={TZ_X[0] * 1e3:g} && x<={TZ_X[1] * 1e3:g} && d>={TZ_Z[0] * 1e3:g} && d<={TZ_Z[1] * 1e3:g}, "
              f"{TZ_STRAIN:g}, 0)"
              if TF == "strain" else "0")
    contact = f"x>={X_END * 1e3:g}"
    crust = f"d<{CRUST_KM * 1e3:g} && ({in_tf})==0 && ({contact})==0"   # muparser has no "!"
    push = (f"x>={PUSH_X[0] * 1e3:g} && x<={PUSH_X[1] * 1e3:g} && d>={PUSH_Z[0] * 1e3:g} && "
            f"d<={PUSH_Z[1] * 1e3:g}")
    pts = ";".join(f"{x * 1e3:g},{(Y_KM - z) * 1e3:g}" for z in np.arange(PV_Z[0], PV_Z[1] + 1e-9, PV_Z[2])
                   for x in np.arange(PV_X[0], PV_X[1] + 1e-9, PV_X[2]))
    return f"""
set Dimension = 2
set Additional shared libraries = {PLUGIN}
set Start time = 0
set End time = {end_yr:g}
set Use years in output instead of seconds = true
set Resume computation = {'true' if resume else 'false'}
set Nonlinear solver scheme = {SCHEME}
set Nonlinear solver tolerance = {NL_TOL:g}
set Max nonlinear iterations = {NL_MAX}
set CFL number = {CFL}
set Maximum time step = {DT_MAX_YR:g}
set Output directory = output
set Pressure normalization = no
set Prescribe internal velocities = true
subsection Prescribed velocities
  subsection Indicator function
    set Variable names = x,y
    set Function constants = H={Y_KM * 1e3:g}
    set Function expression = if(x>={PUSH_X[0] * 1e3:g} && x<={PUSH_X[1] * 1e3:g} && H-y>={PUSH_Z[0] * 1e3:g} && H-y<={PUSH_Z[1] * 1e3:g}, 1, 0); 0
  end
  subsection Velocity function
    set Variable names = x,y,t
    set Function expression = {-V_PUSH / (MYR / 1e6):.6g}; 0
  end
end
subsection Formulation
  set Formulation = Boussinesq approximation
end
subsection Geometry model
  set Model name = box
  subsection Box
    set X repetitions = {int(X_KM / CELL_KM)}
    set Y repetitions = {int(Y_KM / CELL_KM)}
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
{f"    set Nonlinear Newton solver switch tolerance = {NEWTON_SWITCH}{chr(10)}    set Max pre-Newton nonlinear iterations = {NL_MAX}" if NEWTON_SWITCH else ""}
  end
end
subsection Mesh deformation
  set Mesh deformation boundary indicators = top: free surface
  subsection Free surface
    set Surface velocity projection = vertical
  end
end
subsection Boundary velocity model
  set Tangential velocity boundary indicators = left, right, bottom
end
subsection Compositional fields
  set Number of fields = 7
  set Names of fields = plastic_strain, crust_y, crust_o, tf_c, tf_m, push, contact
end
subsection Initial composition model
  set Model name = function
  subsection Function
    set Variable names = x,y
    set Function constants = H={Y_KM * 1e3:g}
    set Function expression = {';'.join(e.replace('d', '(H-y)') for e in (
        strain, f'if({crust} && x<{X_TF * 1e3:g}, 1, 0)', f'if({crust} && x>={X_TF * 1e3:g}, 1, 0)',
        f'if(({in_tf}) && d<{CRUST_KM * 1e3:g}, 1, 0)', f'if(({in_tf}) && d>={CRUST_KM * 1e3:g}, 1, 0)',
        f'if({push}, 1, 0)', f'if({contact}, 1, 0)'))}
  end
end
subsection Initial temperature model
  set Model name = ascii data
  subsection Ascii data model
    set Data directory = {D}/
    set Data file name = si2d_T.txt
  end
end
subsection Boundary temperature model
  set Fixed temperature boundary indicators = bottom, top
  set List of model names = box
  subsection Box
    set Bottom temperature = {T_POT + GRAD * Y_KM:g}
    set Top temperature = {T_SURF:g}
  end
end
subsection Material model
  set Model name = visco plastic
  set Material averaging = harmonic average only viscosity   # ASPECT cookbooks (PARAM_AUDIT 11.20)
  subsection Visco Plastic
    set Reference temperature = {T_SURF:g}
    set Minimum strain rate = 1.e-20
    set Reference strain rate = {EDOT0:g}
    set Minimum viscosity = {ETA_MIN:g}
    set Maximum viscosity = {ETA_MAX:g}
    set Thermal diffusivities = {KAPPA:g}
    set Heat capacities = {CP:g}
    set Densities = {lst(rho)}
    set Thermal expansivities = {ALPHA:g}
    set Viscosity averaging scheme = harmonic
    set Use adiabatic pressure in plasticity = true   # ZZL eq. 2: yield stress C + mu rho g z (PARAM_AUDIT 11.20)
    set Viscous flow law = dislocation
    set Prefactors for dislocation creep = {creep_prefactor():.6g}
    set Stress exponents for dislocation creep = {N_CR:g}
    set Activation energies for dislocation creep = {E_CR:g}
    set Activation volumes for dislocation creep = 0
    set Angles of internal friction = {lst(phi)}
    set Cohesions = {lst(coh)}
    set Strain weakening mechanism = plastic weakening with plastic strain only
    set Start plasticity strain weakening intervals = 0
    set End plasticity strain weakening intervals = {EPS_F:g}
    set Cohesion strain weakening factors = {lst(cw)}
    set Friction strain weakening factors = {lst(fw)}
{f"    set Use plastic damper = true{chr(10)}    set Plastic damper viscosity = {DAMPER}" if DAMPER else ""}
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
  set Time between checkpoint = {SMOKE_CHECKPOINT_S if SMOKE else CHECKPOINT_S}
end
subsection Postprocess
  set List of postprocessors = point values, velocity statistics, topography, visualization
  subsection Point values
    set Time between point values output = {PV_YR:g}
    set Evaluation points = {pts}
  end
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


def _aspect(wall_h):
    return run(f"export OMP_NUM_THREADS=1 OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1; $(ls /usr/bin/mpirun.openmpi 2>/dev/null || echo mpirun) --use-hwthread-cpus --bind-to none "
               f"-np {NP} {EXE} si2d.prm >> out.txt 2>&1; e=$?; tail -15 out.txt; exit $e", cwd=D,
               timeout=int(os.environ.get("ASPECT_KILL_AFTER_S") or wall_h * 3600 + KILL_GRACE_S))   # test: force the hard stop



def pack_restart():
    """The latest checkpoint into results/, written whole and then renamed, so a kill never leaves half a zip."""
    t = time.time()
    with zipfile.ZipFile(f"results/{TAG}_restart.zip.part", "w", zipfile.ZIP_DEFLATED) as z:
        for f in glob.glob(f"{OUT}/restart*") + [f"{OUT}/statistics", f"{OUT}/point_values.txt", f"{OUT}/log.txt"]:
            if os.path.isfile(f):
                z.write(f, os.path.basename(f))
    os.replace(f"results/{TAG}_restart.zip.part", f"results/{TAG}_restart.zip")
    return time.time() - t


def pack_each_checkpoint(stop):
    """Pack every new checkpoint while ASPECT runs: Kaggle stops the whole machine without warning (TPU idle 2 h,
    si2d-tpu1 2026-11-24) and keeps only the files already in the output folder."""
    seen = None
    while not stop.wait(PACK_POLL_S):
        m = max((os.path.getmtime(f) for f in glob.glob(f"{OUT}/restart*")), default=None)
        if m is not None and m != seen and time.time() - m > PACK_SETTLE_S:
            print(f"checkpoint packed in {pack_restart():.1f} s", flush=True)
            seen = m


os.makedirs(OUT, exist_ok=True)
write_temperature()
res["inputs"] = {"lith_young_km": lith_km(AY), "lith_old_km": lith_km(AO), "creep_prefactor": creep_prefactor(),
                 "phi0_deg": math.degrees(math.asin(MU0)), "cohesion0_aspect": C0 / math.sqrt(1 - MU0 ** 2)}
print(json.dumps(res, indent=1), flush=True)
RESTART = os.environ.get("ASPECT_RESTART", "restart.zip")
resume = os.path.exists(RESTART)
if resume:
    zipfile.ZipFile(RESTART).extractall(OUT)
    res["resumed_from"] = sorted(os.listdir(OUT))
if os.environ.get("ASPECT_INPUTS_ONLY") == "1":          # local check of the inputs, no ASPECT
    open(f"{D}/si2d.prm", "w").write(prm(END_MYR * 1e6, WALL_H, resume))
    json.dump(res, open(f"results/{TAG}_run.json", "w"), indent=1)
    raise SystemExit
run("lscpu | grep -E '^CPU\\(s\\)|Model name'; free -g")
if BUILD != "skip":
    SUDO = "sudo " if os.geteuid() != 0 else ""          # GitHub runs as a user, Kaggle as root without sudo
    run(f"set -o pipefail; for i in 1 2; do {SUDO}timeout -k 30 {NET_TIMEOUT_S} apt-get update -qq && {SUDO}DEBIAN_FRONTEND=noninteractive timeout -k 30 {NET_TIMEOUT_S} apt-get install -y -qq libdeal.ii-dev cmake "
        "ninja-build 2>&1 | tail -3 && break; done")
    run(f"timeout -k 30 {NET_TIMEOUT_S} git clone -q --depth 1 --branch v3.0.0 https://github.com/geodynamics/aspect.git {A}")
    run(f"cmake -S {A} -B {B} -G Ninja -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3")
    _, _, res["build_s"] = run(f"ninja -C {B} -j {os.cpu_count()} 2>&1 | tail -3")
    P = os.path.dirname(PLUGIN)
    os.makedirs(P, exist_ok=True)
    shutil.copy(f"{A}/cookbooks/prescribed_velocity/prescribed_velocity.cc", P)
    open(P + "/CMakeLists.txt", "w").write(
        "CMAKE_MINIMUM_REQUIRED(VERSION 3.13.4)\nFIND_PACKAGE(Aspect 3.0.0 REQUIRED HINTS " + B + ")\n"
        "DEAL_II_INITIALIZE_CACHED_VARIABLES()\nPROJECT(prescribed_velocity)\n"
        "ADD_LIBRARY(prescribed_velocity SHARED prescribed_velocity.cc)\nASPECT_SETUP_PLUGIN(prescribed_velocity)\n")
    run(f"cd {P} && cmake -G Ninja . 2>&1 | tail -3 && ninja 2>&1 | tail -5 && ls *.so")
    so = glob.glob(P + "/*.so")
    if so and so[0] != PLUGIN:
        shutil.copy(so[0], PLUGIN)
if BUILD == "only":
    raise SystemExit
EXE = f"{B}/aspect-release" if os.path.exists(f"{B}/aspect-release") else f"{B}/aspect"
pieces = [(SMOKE_WALL_H, resume), (SMOKE_WALL_H, True)] if SMOKE else [(WALL_H, resume)]
res["pieces"] = []
packer = threading.Event()
packer_t = threading.Thread(target=pack_each_checkpoint, args=(packer,), daemon=True)
packer_t.start()
json.dump(res, open(f"results/{TAG}_run.json", "w"), indent=1)   # read by cloud/github/rescue.py if the job is cut
for wall_h, rs in pieces:
    open(f"{D}/si2d.prm", "w").write(prm(END_MYR * 1e6, wall_h, rs))
    code, _, s = _aspect(wall_h)
    log = open(f"{D}/out.txt").read()
    steps = re.findall(r"\*\*\* Timestep (\d+):\s+t=([\d.e+-]+) years", log)
    res["pieces"].append({"exit": str(code), "run_s": round(s), "last_step": int(steps[-1][0]) if steps else None,
                          "model_yr": float(steps[-1][1]) if steps else None})
    json.dump(res, open(f"results/{TAG}_run.json", "w"), indent=1)
packer.set()
packer_t.join()                  # a packing in progress finishes first
cells = re.findall(r"Number of active cells: ([\d,]+)", log)
res.update(cells=cells[-1] if cells else None)
json.dump(res, open(f"results/{TAG}_run.json", "w"), indent=1)
shutil.copy(f"{D}/out.txt", f"results/{TAG}_out.txt")
for f in ("statistics", "point_values.txt", "log.txt"):
    if os.path.exists(f"{OUT}/{f}"):
        shutil.copy(f"{OUT}/{f}", f"results/{TAG}_{f}")
pack_restart()
with zipfile.ZipFile(f"results/{TAG}_viz.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(f"{OUT}/solution/*") + glob.glob(f"{OUT}/solution.pvd"):
        z.write(f, os.path.relpath(f, OUT))
print(json.dumps(res, indent=1))
