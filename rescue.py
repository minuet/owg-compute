"""Last guard of an ASPECT run on GitHub (run.yml, after the tool step): when the tool step was cut by its time limit
before runner.py packed out/NAME.zip, stop every MPI process, then pack the latest checkpoint (half-hourly) and the
tool's small results the way the tool and runner.py would, so the next segment can resume (2026-11-24: r2d s6 lost
a whole segment when the job hit GitHub's limit). Does nothing when out/NAME.zip exists.
Usage: python rescue.py NAME"""
import glob, json, os, re, subprocess, sys, time, zipfile

NAME = sys.argv[1]
R, D, O = "/tmp/w/results", "/tmp/run", "out"     # as runner.py and the ASPECT tools
if os.path.exists(f"{O}/{NAME}.zip"):
    raise SystemExit
for p in ("mpirun", "mpirun.openmpi", "prterun", "orterun", "aspect-release", "aspect"):
    subprocess.run(["pkill", "-9", "-x", p])
subprocess.run(["pkill", "-9", "-f", "python runner.py"])
time.sleep(3)
left = subprocess.run(["pgrep", "-c", "-x", "aspect-release"], capture_output=True, text=True).stdout.strip() or "0"
runs = glob.glob(f"{R}/*_run.json")
tag = os.path.basename(runs[0])[:-len("_run.json")] if runs else NAME
files = [f for f in glob.glob(f"{D}/output/restart*") + [f"{D}/output/statistics", f"{D}/output/log.txt"] if os.path.isfile(f)]
with zipfile.ZipFile(f"{R}/{tag}_restart.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, os.path.basename(f))
for src, dst in ((f"{D}/output/statistics", "statistics"), (f"{D}/output/log.txt", "log.txt"), (f"{D}/out.txt", "out.txt")):
    if os.path.isfile(src):
        open(f"{R}/{tag}_{dst}", "wb").write(open(src, "rb").read())
steps = re.findall(r"\*\*\* Timestep (\d+):\s+t=([\d.e+-]+) years", open(f"{D}/out.txt").read()) \
    if os.path.isfile(f"{D}/out.txt") else []
if runs:
    res = json.load(open(runs[0]))
    res.setdefault("pieces", []).append({"exit": "timeout", "rescued": True, "processes_left": int(left),
                                         "last_step": int(steps[-1][0]) if steps else None,
                                         "model_yr": float(steps[-1][1]) if steps else None})
    json.dump(res, open(runs[0], "w"), indent=1)
with open(f"{R}/{NAME}.log", "a") as f:
    f.write(f"\nRESCUED by cloud/github/rescue.py: processes left {left}, checkpoint files {len(files)}\n")
os.makedirs(O, exist_ok=True)
with zipfile.ZipFile(f"{O}/{NAME}.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(f"{R}/{tag}_*") + [f"{R}/{NAME}.log"]:
        if os.path.isfile(f) and os.path.getsize(f) < 2e8:     # the size limit of runner.py
            z.write(f, os.path.basename(f))
print(f"rescued {tag}: processes left {left}, checkpoint files {len(files)}")
