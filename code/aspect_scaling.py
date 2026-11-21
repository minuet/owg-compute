"""GitHub Actions: cost of ASPECT 2D models vs resolution, to size step 3 (criteria table) of PLATE_PLAN section 6
before asking about paid compute. Builds ASPECT v3.0.0 on Ubuntu's deal.II as cloud/aspect_probe_kernel.py, then runs
its continental_extension cookbook unchanged except: End time ASPECT_END years, and every refinement level raised by
ASPECT_REF (0 = as shipped, 2.5 km / 1.25 km cells; 1 = half the spacing). Time per model year from the two runs gives
the resolution scaling; the cookbook's own physics is not changed.
Env: ASPECT_REF (int), ASPECT_END (years). Output: results/aspect_scaling_<ref>.json and the run's statistics/out."""
import json, os, re, shutil, subprocess, time
A = "/tmp/aspect"; B = "/tmp/aspect-build"; D = "/tmp/run"
REF = int(os.environ.get("ASPECT_REF", "0")); END = float(os.environ.get("ASPECT_END", "1e6"))
RUN_TIMEOUT_S = 5 * 3600         # technical: build (~15 min) + run stay under the workflow's 350 min
os.makedirs("results", exist_ok=True)
res = {"ref": REF, "end_yr": END}


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


run("lscpu | grep -E '^CPU\\(s\\)|Thread|Core|Socket|Model name'; free -g")
run("sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libdeal.ii-dev cmake "
    "ninja-build 2>&1 | tail -3")
run(f"git clone -q --depth 1 --branch v3.0.0 https://github.com/geodynamics/aspect.git {A}")
run(f"cmake -S {A} -B {B} -G Ninja -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3")
_, _, res["build_s"] = run(f"ninja -C {B} -j 4 2>&1 | tail -3")
exe = f"{B}/aspect-release" if os.path.exists(f"{B}/aspect-release") else f"{B}/aspect"
shutil.copytree(f"{A}/cookbooks/continental_extension", D)
p = open(f"{D}/continental_extension.prm").read()
p = re.sub(r"set End time\s*=.*", f"set End time = {END:g}", p)
p = re.sub(r"(Initial global refinement\s*=\s*)(\d+)", lambda m: m.group(1) + str(int(m.group(2)) + REF), p)
p = p.replace("x<=160.e3, 4, 3)", f"x<=160.e3, {4 + REF}, {3 + REF})")
open(f"{D}/continental_extension.prm", "w").write(p)
code, out, res["run_s"] = run(
    f"export OMP_NUM_THREADS=1; $(ls /usr/bin/mpirun.openmpi 2>/dev/null || echo mpirun) --use-hwthread-cpus -np 4 {exe} continental_extension.prm > out.txt 2>&1; e=$?; "
    "grep -E 'Number of active cells|Number of degrees' out.txt | tail -2; tail -5 out.txt; exit $e", cwd=D,
    timeout=RUN_TIMEOUT_S)
log = open(f"{D}/out.txt").read()
steps = re.findall(r"\*\*\* Timestep (\d+):\s+t=([\d.e+]+) years", log)
cells = re.findall(r"Number of active cells: ([\d,]+)", log)
dofs = re.findall(r"Number of degrees of freedom: ([\d,]+)", log)
res.update(exit=str(code), steps=int(steps[-1][0]) + 1 if steps else 0, model_yr=float(steps[-1][1]) if steps else 0,
           cells=cells[-1] if cells else None, dofs=dofs[-1] if dofs else None)
res["s_per_myr"] = res["run_s"] / res["model_yr"] * 1e6 if res["model_yr"] else None
json.dump(res, open(f"results/aspect_scaling_{REF}.json", "w"), indent=1)
shutil.copy(f"{D}/out.txt", f"results/aspect_scaling_{REF}_out.txt")
st = f"{D}/output-continental_extension/statistics"
if os.path.exists(st):
    shutil.copy(st, f"results/aspect_scaling_{REF}_statistics")
print(json.dumps(res, indent=1))
