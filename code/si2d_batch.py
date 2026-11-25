"""Several tools/aspect_si2d.py cases at once on one large machine (Kaggle TPU v5e-8 host: 224 threads, 384 GB; the
TPU itself is not used): ASPECT is built once, then each case runs in its own folder with an equal share of the
physical cores. A case resumes from <tag>_restart.zip when that file lies in the code folder (packed by the previous
session, as for the Kaggle rift continuations).
Env: SI_CASES "AY/AO/TF,AY/AO/TF,..." (Myr, Myr, SI_TF); the other aspect_si2d.py variables (ASPECT_END_MYR,
ASPECT_WALL_H, ASPECT_SMOKE, ...) pass through unchanged.
Output: results/ of every case (aspect_si2d.py), results/si2d_batch.json, results/si2d_build.log. Stdout (the Kaggle log
beside a companion job, cloud/tool_kernel.py) gets only a few lines: build, start, every case's model time every
REPORT_S, each case's stop, the end."""
import json, os, re, subprocess, sys, time

REPORT_S = 60 if os.environ.get("ASPECT_SMOKE") == "1" else 1800   # technical: the checkpoint interval of aspect_si2d.py (CHECKPOINT_S, SMOKE_CHECKPOINT_S), one report per new checkpoint (PARAM_AUDIT 11.20)
POLL_S = 10          # technical: a case's stop shows in the log within this, the check itself costs nothing (PARAM_AUDIT 11.20)


def say(s):
    print(f"{time.strftime('%H:%M', time.gmtime())} UTC {s}", flush=True)


def model_myr(i):
    """model time of case i from its ASPECT log (None before the first step)"""
    try:
        t = re.findall(r"\*\*\* Timestep \d+:\s+t=([\d.e+-]+) years", open(f"/tmp/run{i}/out.txt").read())
    except OSError:
        return None
    return round(float(t[-1]) / 1e6, 3) if t else None


cases = [c.split("/") for c in os.environ["SI_CASES"].split(",")]
cores = len({l for l in subprocess.run(["lscpu", "-p=CORE,SOCKET"], capture_output=True, text=True).stdout.split("\n")
             if l and not l.startswith("#")})
OTHER_JOB_CORES = int(os.environ.get("SI_OTHER_JOB_CORES", 0))   # cores kept out for another job; 0: the TPU training job uses the idle hyperthreads (PARAM_AUDIT 11.20)
np_case = max(1, (cores - OTHER_JOB_CORES) // len(cases))   # one MPI process per physical core, shared equally (PARAM_AUDIT 11.20)
here = os.path.dirname(os.path.abspath(__file__))
res = {"cases": cases, "physical_cores": cores, "other_job_cores": OTHER_JOB_CORES, "np_per_case": np_case,
       "cores_left": cores - np_case * len(cases)}
print(json.dumps(res), flush=True)
t = time.time()
with open(f"{here}/results/si2d_build.log", "w") as bl:
    b = subprocess.run([sys.executable, "aspect_si2d.py"], cwd=here, env=dict(os.environ, ASPECT_BUILD="only"),
                       stdout=bl, stderr=subprocess.STDOUT)
res["build"] = {"exit": b.returncode, "s": round(time.time() - t)}
say(f"build exit {b.returncode}, {res['build']['s'] / 60:.0f} min")
procs = []
for i, (ay, ao, tf) in enumerate(cases):
    tag = f"si2d_ay{float(ay):g}_ao{float(ao):g}_tf{tf}"
    env = dict(os.environ, SI_AY=ay, SI_AO=ao, SI_TF=tf, SI_NP=str(np_case), ASPECT_BUILD="skip",
               ASPECT_DIR=f"/tmp/run{i}", ASPECT_RESTART=f"{here}/{tag}_restart.zip")
    log = open(f"{here}/results/{tag}_batch.log", "w")
    procs.append((tag, subprocess.Popen([sys.executable, "aspect_si2d.py"], cwd=here, env=env, stdout=log,
                                        stderr=subprocess.STDOUT)))
say(f"{len(procs)} cases started, {np_case} cores each: " + ", ".join(tag for tag, _ in procs))
res["exit"], last = {}, time.time()
while len(res["exit"]) < len(procs):
    time.sleep(POLL_S)
    for i, (tag, p) in enumerate(procs):
        if tag not in res["exit"] and p.poll() is not None:
            res["exit"][tag] = p.returncode
            say(f"{tag} stopped: exit {p.returncode}, {model_myr(i)} Myr, checkpoint packed "
                f"{'yes' if os.path.exists(f'{here}/results/{tag}_restart.zip') else 'NO'}")
    if time.time() - last >= REPORT_S and len(res["exit"]) < len(procs):
        last = time.time()
        say("Myr: " + ", ".join(f"{tag[5:]} {'stopped' if tag in res['exit'] else model_myr(i)}"
                                for i, (tag, _) in enumerate(procs)))
res["run_s"] = round(time.time() - t)
json.dump(res, open(f"{here}/results/si2d_batch.json", "w"), indent=1)
print(json.dumps(res), flush=True)
say(f"all cases done after {res['run_s'] / 3600:.2f} h")
