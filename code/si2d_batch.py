"""Several tools/aspect_si2d.py cases at once on one large machine (Kaggle TPU v5e-8 host: 224 threads, 384 GB; the
TPU itself is not used): ASPECT is built once, then each case runs in its own folder with an equal share of the
physical cores. A case resumes from <tag>_restart.zip when that file lies in the code folder (packed by the previous
session, as for the Kaggle rift continuations).
Env: SI_CASES "AY/AO/TF,AY/AO/TF,..." (Myr, Myr, SI_TF); the other aspect_si2d.py variables (ASPECT_END_MYR,
ASPECT_WALL_H, ASPECT_SMOKE, ...) pass through unchanged.
Output: results/ of every case (aspect_si2d.py), results/si2d_batch.json."""
import json, os, subprocess, sys, time

cases = [c.split("/") for c in os.environ["SI_CASES"].split(",")]
cores = len({l for l in subprocess.run(["lscpu", "-p=CORE,SOCKET"], capture_output=True, text=True).stdout.split("\n")
             if l and not l.startswith("#")})
np_case = max(1, cores // len(cases))      # one MPI process per physical core, shared equally (PARAM_AUDIT 11.20)
here = os.path.dirname(os.path.abspath(__file__))
res = {"cases": cases, "physical_cores": cores, "np_per_case": np_case}
print(json.dumps(res), flush=True)
t = time.time()
b = subprocess.run([sys.executable, "aspect_si2d.py"], cwd=here, env=dict(os.environ, ASPECT_BUILD="only"))
res["build"] = {"exit": b.returncode, "s": round(time.time() - t)}
procs = []
for i, (ay, ao, tf) in enumerate(cases):
    tag = f"si2d_ay{float(ay):g}_ao{float(ao):g}_tf{tf}"
    env = dict(os.environ, SI_AY=ay, SI_AO=ao, SI_TF=tf, SI_NP=str(np_case), ASPECT_BUILD="skip",
               ASPECT_DIR=f"/tmp/run{i}", ASPECT_RESTART=f"{here}/{tag}_restart.zip")
    log = open(f"{here}/results/{tag}_batch.log", "w")
    procs.append((tag, subprocess.Popen([sys.executable, "aspect_si2d.py"], cwd=here, env=env, stdout=log,
                                        stderr=subprocess.STDOUT)))
res["exit"] = {tag: p.wait() for tag, p in procs}
res["run_s"] = round(time.time() - t)
json.dump(res, open(f"{here}/results/si2d_batch.json", "w"), indent=1)
print(json.dumps(res), flush=True)
